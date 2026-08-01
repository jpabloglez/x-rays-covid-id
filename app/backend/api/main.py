"""The HTTP surface: upload an image, and score it with both tracks.

`/predict` returns both models rather than one. The gap between them is the
result this project produced -- 0.9925 on a corpus where COVID came from a
single collection, 0.7460 within one hospital network -- and a response that
picked one would be quoting a number without the thing that gives it meaning.

Every prediction carries the gate report, the ablation retention and the
calibration temperature of the model that produced it. That is not decoration.
Track 1 keeps 91.5% of its discriminative signal with the lung fields blanked
out and Track 2 keeps 83.7%, so neither probability is a diagnostic finding,
and a client that renders one without the caveats is reproducing the failure
mode this repository was built to measure.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api import inference, storage
from api.config import Settings, settings

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Research artefact, not a diagnostic device. Both models were trained to measure a "
    "documented shortcut-learning failure mode, and both exhibit it. Do not use these "
    "outputs for clinical decisions."
)


class UploadResponse(BaseModel):
    detail: str
    imageUrl: str = ""


class PredictionOut(BaseModel):
    track: str
    classes: list[str]
    probabilities: dict[str, float]
    predicted: str
    confidence: float
    macro_auc: float | None = None
    lungs_removed_retention: float | None = None
    caveats: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    detail: str
    imageUrl: str = ""
    predictions: list[PredictionOut]
    comparison: str
    disclaimer: str = DISCLAIMER


def _comparison(predictions: list[inference.Prediction]) -> str:
    """Say what the pair means, derived rather than hardcoded.

    Two models trained on different corpora answering different questions is
    not a consensus mechanism, and a reader shown two numbers will look for
    agreement unless told what they actually are.
    """
    if len(predictions) < 2:
        return (
            "Only one model is loaded, so there is nothing to compare. The value of this "
            "endpoint is the contrast between a model trained on a confounded corpus and "
            "one trained within a single hospital network."
        )
    parts = []
    for prediction in sorted(predictions, key=lambda p: p.track):
        retention = prediction.retention
        share = "unmeasured" if retention is None else f"{retention:.0%}"
        parts.append(
            f"{prediction.track} says {prediction.predicted} "
            f"({prediction.confidence:.0%} confidence, {share} of its signal survives "
            "having the lungs removed)"
        )
    return (
        " and ".join(parts)
        + ". These are different questions over different corpora, not a second opinion: "
        "agreement between them is not corroboration, and disagreement is not a tie to break."
    )


def create_app(config: Settings | None = None) -> FastAPI:
    config = config or settings()
    app = FastAPI(
        title="X-rays COVID / non-COVID",
        description=DISCLAIMER,
        version="0.2.0",
    )
    app.state.config = config

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def _models() -> dict[str, inference.LoadedModel]:
        return inference.registry(config.model_dir)

    @app.get("/health")
    def health() -> dict:
        models = _models()
        return {
            "status": "ok",
            "models": sorted(models),
            # Reported so a deployment that quietly lost its exports is visible
            # from the outside rather than only when someone uploads an image.
            "inference_available": bool(models),
        }

    @app.get("/models")
    def describe_models() -> dict:
        """What is loaded and what each one is worth, without needing an image."""
        return {
            "disclaimer": DISCLAIMER,
            "models": [
                {
                    "track": name,
                    "classes": model.classes,
                    "sources": model.metadata.get("sources", []),
                    "metrics": model.metadata.get("metrics", {}),
                    "gates": model.metadata.get("gates", {}),
                    "ablation": model.metadata.get("ablation", {}),
                    "notes": model.metadata.get("notes", ""),
                }
                for name, model in sorted(_models().items())
            ],
        }

    @app.post("/files/", response_model=UploadResponse, status_code=201)
    async def upload(image: UploadFile = File(...)) -> JSONResponse:
        payload = await image.read()
        try:
            extension = storage.validate(payload, max_bytes=config.max_upload_bytes)
        except storage.RejectedUpload as rejected:
            logger.info("Rejected upload: %s", rejected.detail)
            return JSONResponse(
                {"detail": rejected.detail, "imageUrl": ""}, status_code=rejected.status
            )

        filename = storage.store(payload, extension, config.media_root)
        return JSONResponse(
            {
                "detail": "Image uploaded successfully",
                "imageUrl": f"{config.media_url}{filename}",
            },
            status_code=201,
        )

    @app.post("/predict/", response_model=PredictResponse)
    async def predict(image: UploadFile = File(...)) -> JSONResponse:
        payload = await image.read()
        try:
            extension = storage.validate(payload, max_bytes=config.max_upload_bytes)
        except storage.RejectedUpload as rejected:
            return JSONResponse(
                {"detail": rejected.detail, "imageUrl": "", "predictions": [],
                 "comparison": "", "disclaimer": DISCLAIMER},
                status_code=rejected.status,
            )

        models = _models()
        if not models:
            # 503 rather than 500: the service is fine, it just has nothing to
            # serve, and that is an operator problem rather than a bad request.
            return JSONResponse(
                {"detail": "No models are loaded; run `cxr export` and set MODEL_DIR",
                 "imageUrl": "", "predictions": [], "comparison": "",
                 "disclaimer": DISCLAIMER},
                status_code=503,
            )

        decoded = inference.decode(payload)
        results = [inference.predict(model, decoded) for _, model in sorted(models.items())]
        filename = storage.store(payload, extension, config.media_root)

        return JSONResponse(
            {
                "detail": "Scored by every loaded model",
                "imageUrl": f"{config.media_url}{filename}",
                "predictions": [
                    {
                        "track": result.track,
                        "classes": result.classes,
                        "probabilities": result.probabilities,
                        "predicted": result.predicted,
                        "confidence": result.confidence,
                        "macro_auc": result.metadata.get("metrics", {}).get("macro_auc"),
                        "lungs_removed_retention": result.retention,
                        "caveats": result.caveats(),
                    }
                    for result in results
                ],
                "comparison": _comparison(results),
                "disclaimer": DISCLAIMER,
            },
            status_code=200,
        )

    return app


app = create_app()

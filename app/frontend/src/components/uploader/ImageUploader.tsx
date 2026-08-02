import React, { ChangeEvent, FormEvent, useEffect, useState } from "react";
import { PredictError, PredictResponse, fetchHealth, predict } from "../../api/predict";
import PredictionPanel from "../prediction/PredictionPanel";
import SetupNeededBanner from "../status/SetupNeededBanner";

const ACCEPTED_TYPES = ["image/jpeg", "image/png"];
const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

const SETUP_MESSAGE =
  "No models are loaded yet, so scoring is unavailable. This is a deployment issue, not " +
  "something wrong with your file — ask the operator to run `cxr export` and point " +
  "MODEL_DIR at the result.";

const ImageUpload: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState("");
  const [setupMessage, setSetupMessage] = useState<string | null>(null);
  const [isScoring, setIsScoring] = useState(false);
  // Checked proactively, not just discovered when a submit 503s. Defaults to
  // available so a slow or failed health probe never hides the form on its
  // own -- a real submit attempt is what actually proves the service is
  // unusable, and that path is covered separately (the 503 branch below).
  const [inferenceAvailable, setInferenceAvailable] = useState(true);

  useEffect(() => {
    let live = true;
    fetchHealth()
      .then((health) => {
        if (live) setInferenceAvailable(health.inference_available);
      })
      .catch(() => {
        // A failed probe is not itself proof of misconfiguration; leave the
        // form up and let an actual submit surface the real error.
      });
    return () => {
      live = false;
    };
  }, []);

  // Revokes the *previous* object URL whenever previewUrl changes, and the
  // last one on unmount -- the only place revocation needs to happen.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setError("");
    setSetupMessage(null);
    setResult(null);

    // Checked here as well as on the server. This is a courtesy to save a
    // round trip, not a control: the server re-decodes every upload and never
    // trusts the declared type.
    if (file && !ACCEPTED_TYPES.includes(file.type)) {
      setSelectedImage(null);
      setPreviewUrl(null);
      setError("Please choose a JPEG or PNG image.");
      return;
    }
    if (file && file.size > MAX_UPLOAD_BYTES) {
      setSelectedImage(null);
      setPreviewUrl(null);
      setError("That image is larger than the 20 MB limit.");
      return;
    }
    setSelectedImage(file);
    setPreviewUrl(file ? URL.createObjectURL(file) : null);
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!selectedImage) {
      setError("Choose an image first.");
      return;
    }

    setIsScoring(true);
    setError("");
    setSetupMessage(null);
    try {
      setResult(await predict(selectedImage));
    } catch (thrown) {
      setResult(null);
      // A 503 here is the server saying it has no models loaded, not that
      // this file is a problem. Rendering it through the same red alert as a
      // validation error is exactly the confusion this branch exists to
      // avoid -- so it gets the setup banner instead.
      if (thrown instanceof PredictError && thrown.status === 503) {
        setSetupMessage(thrown.message || SETUP_MESSAGE);
      } else {
        setError(thrown instanceof Error ? thrown.message : "Could not score that image.");
      }
    } finally {
      setIsScoring(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <h2 className="text-2xl font-semibold text-slate-900">Score a chest radiograph</h2>
      <p className="mt-2 max-w-2xl text-sm text-slate-700">
        Two models are run: one trained on pooled public collections, one trained within a
        single hospital network. They disagree often, and the disagreement is the point —
        each answer arrives with a measurement of what it is actually reading.
      </p>

      {inferenceAvailable ? (
        <form onSubmit={handleSubmit} className="mt-5">
          <label htmlFor="radiograph" className="block text-sm font-medium text-slate-800">
            Radiograph (JPEG or PNG, up to 20 MB)
          </label>
          <input
            id="radiograph"
            type="file"
            accept="image/jpeg, image/png"
            onChange={handleFileSelect}
            className="mt-2 block w-full text-sm text-slate-700 file:mr-4 file:rounded file:border-0 file:bg-slate-800 file:px-4 file:py-2 file:text-white hover:file:bg-slate-700"
          />

          {previewUrl && !result && (
            <figure className="mt-4">
              <img
                src={previewUrl}
                alt="Selected radiograph, not yet scored"
                className="max-h-72 rounded border border-slate-300"
              />
              <figcaption className="mt-1 text-xs text-slate-600">
                Selected — not yet scored.
              </figcaption>
            </figure>
          )}

          <button
            type="submit"
            disabled={isScoring}
            className="mt-4 rounded bg-slate-800 px-4 py-2 text-white disabled:opacity-50"
          >
            {isScoring ? "Scoring…" : "Score"}
          </button>
        </form>
      ) : (
        <div className="mt-5">
          <SetupNeededBanner testId="setup-needed" message={SETUP_MESSAGE} />
        </div>
      )}

      {setupMessage && (
        <div className="mt-4">
          <SetupNeededBanner testId="setup-needed-submit" message={setupMessage} />
        </div>
      )}

      {error.length > 0 && (
        <p role="alert" className="mt-4 rounded border border-red-300 bg-red-50 px-4 py-3 text-red-800">
          {error}
        </p>
      )}

      {result && <PredictionPanel result={result} />}
    </div>
  );
};

export default ImageUpload;

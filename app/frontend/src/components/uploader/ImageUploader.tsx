import React, { ChangeEvent, FormEvent, useState } from "react";
import { PredictResponse, predict } from "../../api/predict";
import PredictionPanel from "../prediction/PredictionPanel";

const ACCEPTED_TYPES = ["image/jpeg", "image/png"];
const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

const ImageUpload: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState("");
  const [isScoring, setIsScoring] = useState(false);

  const handleFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setError("");
    setResult(null);

    // Checked here as well as on the server. This is a courtesy to save a
    // round trip, not a control: the server re-decodes every upload and never
    // trusts the declared type.
    if (file && !ACCEPTED_TYPES.includes(file.type)) {
      setSelectedImage(null);
      setError("Please choose a JPEG or PNG image.");
      return;
    }
    if (file && file.size > MAX_UPLOAD_BYTES) {
      setSelectedImage(null);
      setError("That image is larger than the 20 MB limit.");
      return;
    }
    setSelectedImage(file);
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!selectedImage) {
      setError("Choose an image first.");
      return;
    }

    setIsScoring(true);
    setError("");
    try {
      setResult(await predict(selectedImage));
    } catch (thrown) {
      setResult(null);
      setError(thrown instanceof Error ? thrown.message : "Could not score that image.");
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

        <button
          type="submit"
          disabled={isScoring}
          className="mt-4 rounded bg-slate-800 px-4 py-2 text-white disabled:opacity-50"
        >
          {isScoring ? "Scoring…" : "Score"}
        </button>
      </form>

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

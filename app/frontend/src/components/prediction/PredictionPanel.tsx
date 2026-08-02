import React from "react";
import { PredictResponse } from "../../api/predict";
import PredictionCard from "./PredictionCard";

interface Props {
  result: PredictResponse;
}

/**
 * Both models, side by side, with the comparison between them.
 *
 * Deliberately not a single verdict. The two models answer different questions
 * over different corpora, so combining them into one number -- averaging,
 * voting, or picking the confident one -- would invent an agreement that does
 * not exist. The layout puts them beside each other and states what the pair
 * is for, because a reader shown two numbers will look for a winner unless
 * told there is not one.
 */
const PredictionPanel: React.FC<Props> = ({ result }) => (
  <section className="mt-6" aria-label="Model predictions">
    <p
      role="note"
      className="rounded border border-slate-400 bg-slate-800 px-4 py-3 text-sm text-slate-100"
    >
      {result.disclaimer}
    </p>

    <div className="mt-4 grid gap-4 md:grid-cols-2">
      {result.predictions.map((prediction) => (
        <PredictionCard key={prediction.track} prediction={prediction} />
      ))}
    </div>

    {result.comparison && (
      <p
        className="mt-4 rounded border border-slate-300 bg-slate-50 px-4 py-3 text-sm text-slate-800"
        data-testid="comparison"
      >
        {result.comparison}
      </p>
    )}

    {result.imageUrl && (
      <figure className="mt-4">
        <img
          src={result.imageUrl}
          alt="The radiograph that was scored"
          className="max-h-96 rounded border border-slate-300"
        />
        <figcaption className="mt-1 text-xs text-slate-600">
          Stored under the hash of its own bytes. No record is kept of who uploaded it.
        </figcaption>
      </figure>
    )}
  </section>
);

export default PredictionPanel;

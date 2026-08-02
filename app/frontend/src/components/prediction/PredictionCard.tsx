import React from "react";
import { Prediction, retentionVerdict, severityOf } from "../../api/predict";

// The retention badge sits next to the probability rather than below the fold.
// A reader who takes the number and stops reading should still have walked
// past the sentence saying what the number is made of.
const SEVERITY_STYLES: Record<string, string> = {
  high: "bg-red-50 border-red-300 text-red-900",
  moderate: "bg-amber-50 border-amber-300 text-amber-900",
  low: "bg-emerald-50 border-emerald-300 text-emerald-900",
  unmeasured: "bg-slate-100 border-slate-300 text-slate-700",
};

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

interface Props {
  prediction: Prediction;
}

const PredictionCard: React.FC<Props> = ({ prediction }) => {
  const severity = severityOf(prediction.lungs_removed_retention);
  const ordered = [...prediction.classes].sort(
    (a, b) => (prediction.probabilities[b] ?? 0) - (prediction.probabilities[a] ?? 0),
  );

  return (
    <article
      className="rounded-lg border border-slate-300 bg-white p-5 text-slate-900"
      aria-labelledby={`${prediction.track}-heading`}
    >
      <header className="flex items-baseline justify-between gap-3">
        <h3 id={`${prediction.track}-heading`} className="text-lg font-semibold">
          {prediction.track}
        </h3>
        <span className="text-sm text-slate-600">
          {prediction.classes.length}-class
          {prediction.macro_auc !== null && (
            <> · test AUC {prediction.macro_auc.toFixed(4)}</>
          )}
        </span>
      </header>

      <p className="mt-3 text-sm text-slate-700">
        Most likely: <strong className="font-semibold">{prediction.predicted}</strong> at{" "}
        {percent(prediction.confidence)}
      </p>

      <ul className="mt-3 space-y-2" aria-label={`${prediction.track} class probabilities`}>
        {ordered.map((name) => {
          const value = prediction.probabilities[name] ?? 0;
          return (
            <li key={name}>
              <div className="flex justify-between text-sm">
                <span>{name}</span>
                <span className="tabular-nums">{percent(value)}</span>
              </div>
              <div className="mt-1 h-2 w-full rounded bg-slate-200">
                <div
                  className="h-2 rounded bg-slate-600"
                  style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>

      <p
        className={`mt-4 rounded border px-3 py-2 text-sm ${SEVERITY_STYLES[severity]}`}
        data-testid={`retention-${prediction.track}`}
      >
        {retentionVerdict(prediction.lungs_removed_retention)}
      </p>

      {prediction.caveats.length > 0 && (
        <details className="mt-3 text-sm text-slate-700" open>
          <summary className="cursor-pointer font-medium">
            What this number is and is not ({prediction.caveats.length})
          </summary>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            {prediction.caveats.map((caveat) => (
              <li key={caveat}>{caveat}</li>
            ))}
          </ul>
        </details>
      )}
    </article>
  );
};

export default PredictionCard;

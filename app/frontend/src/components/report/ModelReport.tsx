import React, { useEffect, useState } from "react";
import {
  ModelCard,
  ModelsResponse,
  fetchModels,
  retentionVerdict,
  severityOf,
} from "../../api/predict";

const SEVERITY_STYLES: Record<string, string> = {
  high: "bg-red-50 border-red-300 text-red-900",
  moderate: "bg-amber-50 border-amber-300 text-amber-900",
  low: "bg-emerald-50 border-emerald-300 text-emerald-900",
  unmeasured: "bg-slate-100 border-slate-300 text-slate-700",
};

const show = (value: number | null, digits = 4) =>
  value === null || Number.isNaN(value) ? "—" : value.toFixed(digits);

const percent = (value: number | null) =>
  value === null || Number.isNaN(value) ? "not measured" : `${(value * 100).toFixed(1)}%`;

const ModelRow: React.FC<{ model: ModelCard }> = ({ model }) => {
  const severity = severityOf(model.ablation.lungs_removed_retention);
  return (
    <article
      className="rounded-lg border border-slate-300 bg-white p-5"
      aria-labelledby={`report-${model.track}`}
    >
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 id={`report-${model.track}`} className="text-lg font-semibold text-slate-900">
          {model.track}
        </h3>
        <span className="text-sm text-slate-600">
          {model.classes.join(" · ")}
          {model.sources.length > 0 && <> — {model.sources.join(", ")}</>}
        </span>
      </header>

      <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
        <div>
          <dt className="text-slate-600">Test macro AUC</dt>
          <dd className="tabular-nums font-semibold">{show(model.metrics.macro_auc)}</dd>
        </div>
        <div>
          <dt className="text-slate-600">Balanced accuracy</dt>
          <dd className="tabular-nums">{show(model.metrics.balanced_accuracy)}</dd>
        </div>
        <div>
          <dt className="text-slate-600">ECE after calibration</dt>
          <dd className="tabular-nums">{show(model.metrics.ece_after_calibration)}</dd>
        </div>
        <div>
          <dt className="text-slate-600">Ablation images</dt>
          <dd className="tabular-nums">{model.ablation.images ?? "—"}</dd>
        </div>
      </dl>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <p className={`rounded border px-3 py-2 text-sm ${SEVERITY_STYLES[severity]}`}>
          <strong className="font-semibold">Lungs removed:</strong>{" "}
          {percent(model.ablation.lungs_removed_retention)} of the signal survives.
        </p>
        <p className="rounded border border-slate-300 bg-slate-50 px-3 py-2 text-sm text-slate-800">
          <strong className="font-semibold">Lungs only:</strong>{" "}
          {percent(model.ablation.lungs_only_retention)} of the signal survives.
        </p>
      </div>

      <p
        className="mt-3 text-sm text-slate-700"
        data-testid={`report-verdict-${model.track}`}
      >
        {retentionVerdict(model.ablation.lungs_removed_retention)}
      </p>

      <section className="mt-4" aria-label={`${model.track} gates`}>
        <h4 className="text-sm font-semibold text-slate-900">Leakage gates</h4>
        {Object.entries(model.gates.findings).length > 0 && (
          <ul className="mt-2 space-y-1 text-sm">
            {Object.entries(model.gates.findings).map(([gate, finding]) => (
              <li key={gate} className="rounded border border-red-300 bg-red-50 px-3 py-2">
                <strong className="font-semibold">{gate} fails</strong>
                {model.gates.acknowledged.includes(gate) && " (acknowledged)"}: {finding}
              </li>
            ))}
          </ul>
        )}
        {model.gates.skipped.length > 0 && (
          <p
            className="mt-2 rounded border border-slate-300 bg-slate-50 px-3 py-2 text-sm text-slate-700"
            data-testid={`report-skipped-${model.track}`}
          >
            <strong className="font-semibold">Did not run:</strong>{" "}
            {model.gates.skipped.join(", ")}. A skipped gate measured nothing, so it is
            not a passed gate.
          </p>
        )}
      </section>

      {model.notes && <p className="mt-4 text-sm text-slate-600">{model.notes}</p>}
    </article>
  );
};

/**
 * The report, built from what the server has loaded.
 *
 * Every number here comes from `/models`, which reads the metadata sealed
 * inside each exported model. Nothing is typed into this page. A results page
 * that carries its own copy of the figures is the same failure this project
 * kept finding in its own documentation -- a claim that was true when written
 * and quietly stopped being true.
 */
const ModelReport: React.FC = () => {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let live = true;
    fetchModels()
      .then((result) => live && setData(result))
      .catch((thrown) => live && setError(thrown.message));
    return () => {
      live = false;
    };
  }, []);

  if (error) {
    return (
      <p role="alert" className="rounded border border-red-300 bg-red-50 px-4 py-3 text-red-800">
        {error}
      </p>
    );
  }
  if (!data) {
    return <p className="text-slate-600">Loading the report…</p>;
  }

  return (
    <section aria-label="Model report">
      <h2 className="text-2xl font-semibold text-slate-900">What these models are</h2>
      <p className="mt-2 max-w-3xl text-sm text-slate-700">
        Every figure below is read from the models the server currently has loaded, not
        written into this page. If a model is replaced, this changes with it.
      </p>

      <p
        role="note"
        className="mt-4 rounded border border-slate-400 bg-slate-800 px-4 py-3 text-sm text-slate-100"
      >
        {data.disclaimer}
      </p>

      {data.models.length === 0 ? (
        <p
          className="mt-4 rounded border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900"
          data-testid="no-models"
        >
          No models are loaded, so there is nothing to report. Run <code>cxr export</code> and
          point <code>MODEL_DIR</code> at the result.
        </p>
      ) : (
        <div className="mt-4 space-y-4">
          {data.models.map((model) => (
            <ModelRow key={model.track} model={model} />
          ))}
        </div>
      )}

      <section className="mt-8" aria-label="How to read these numbers">
        <h3 className="text-lg font-semibold text-slate-900">How to read these numbers</h3>
        <div className="mt-2 max-w-3xl space-y-3 text-sm text-slate-700">
          <p>
            <strong>Retention is the number that matters.</strong> Blank out the lung fields
            and score again: a model reading pathology should collapse towards chance, and one
            reading acquisition signature barely notices. Both ablations are run, because
            either alone is ambiguous.
          </p>
          <p>
            <strong>A high AUC is not evidence of reading a chest.</strong> Where COVID comes
            from a single collection, provenance predicts the class and the model can score
            near-perfectly without looking at the anatomy at all.
          </p>
          <p>
            <strong>A skipped gate is not a passed gate.</strong> Where a corpus has one
            source, class–source association is undefined, so that check reports nothing
            rather than clearing anything.
          </p>
          <p>
            <strong>Reported scores describe held-out images from the training
            collections.</strong> An image uploaded here came from somewhere else, so those
            figures do not describe the accuracy of a prediction made on it.
          </p>
        </div>
      </section>
    </section>
  );
};

export default ModelReport;

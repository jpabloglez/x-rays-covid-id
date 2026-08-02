import React, { useEffect, useState } from "react";
import { ModelCard, ModelsResponse, Severity, fetchModels, severityOf } from "../../api/predict";
import BarComparisonChart, { BarComparisonDatum } from "../charts/BarComparisonChart";
import { RetentionVerdict, SEVERITY_FILL, SEVERITY_STYLES } from "../severity/SeverityBadge";
import SetupNeededBanner from "../status/SetupNeededBanner";

const show = (value: number | null, digits = 4) =>
  value === null || Number.isNaN(value) ? "—" : value.toFixed(digits);

const percent = (value: number | null) =>
  value === null || Number.isNaN(value) ? "not measured" : `${(value * 100).toFixed(1)}%`;

const ModelRow: React.FC<{ model: ModelCard }> = ({ model }) => {
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
          <dd className="font-mono font-semibold tabular-nums">{show(model.metrics.macro_auc)}</dd>
        </div>
        <div>
          <dt className="text-slate-600">Balanced accuracy</dt>
          <dd className="font-mono tabular-nums">{show(model.metrics.balanced_accuracy)}</dd>
        </div>
        <div>
          <dt className="text-slate-600">ECE after calibration</dt>
          <dd className="font-mono tabular-nums">{show(model.metrics.ece_after_calibration)}</dd>
        </div>
        <div>
          <dt className="text-slate-600">Ablation images</dt>
          <dd className="font-mono tabular-nums">{model.ablation.images ?? "—"}</dd>
        </div>
      </dl>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <SeverityDetail
          severity={severityOf(model.ablation.lungs_removed_retention)}
          label="Lungs removed"
          percentText={percent(model.ablation.lungs_removed_retention)}
        />
        <p className="rounded border border-slate-300 bg-slate-50 px-3 py-2 text-sm text-slate-800">
          <strong className="font-semibold">Lungs only:</strong>{" "}
          <span className="font-mono tabular-nums">
            {percent(model.ablation.lungs_only_retention)}
          </span>{" "}
          of the signal survives.
        </p>
      </div>

      <div className="mt-3">
        <RetentionVerdict
          retention={model.ablation.lungs_removed_retention}
          testId={`report-verdict-${model.track}`}
        />
      </div>

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

/** The "Lungs removed: X% survives" summary box. Same severity meaning and
 * the same shared style map as RetentionVerdict, just a shorter sentence
 * template -- so it reuses SEVERITY_STYLES directly rather than keeping a
 * second copy of it. */
const SeverityDetail: React.FC<{ severity: Severity; label: string; percentText: string }> = ({
  severity,
  label,
  percentText,
}) => (
  <p className={`rounded border px-3 py-2 text-sm ${SEVERITY_STYLES[severity]}`}>
    <strong className="font-semibold">{label}:</strong>{" "}
    <span className="font-mono tabular-nums">{percentText}</span> of the signal survives.
  </p>
);

function eceDomainMax(models: ModelCard[]): number {
  const values = models
    .map((model) => model.metrics.ece_after_calibration)
    .filter((value): value is number => value !== null && !Number.isNaN(value));
  if (values.length === 0) return 0.1;
  const max = Math.max(...values);
  // Rounded up to the next 0.05 and fixed for this render -- an axis that
  // silently rescales between interactions is worse than a little headroom.
  return Math.max(0.05, Math.ceil(max / 0.05) * 0.05);
}

/**
 * The three numbers that are this project's actual result, side by side,
 * before the per-model drill-down. AUC and ECE are drawn neutral: neither has
 * a "good" threshold defined anywhere in the contract, and coloring them as
 * if they did would invent a claim. Retention is the one place color is
 * earned -- it reuses the same severity map as the badge below it, so the bar
 * and the sentence can never disagree.
 */
const ComparisonSection: React.FC<{ models: ModelCard[] }> = ({ models }) => {
  const aucData: BarComparisonDatum[] = models.map((model) => ({
    key: model.track,
    label: model.track,
    value: model.metrics.macro_auc,
    detail: `${model.track}: test macro AUC ${show(model.metrics.macro_auc)}`,
  }));

  const retentionData: BarComparisonDatum[] = models.map((model) => {
    const retention = model.ablation.lungs_removed_retention;
    return {
      key: model.track,
      label: model.track,
      value: retention,
      barClassName: SEVERITY_FILL[severityOf(retention)],
      detail: retentionVerdictFor(model),
    };
  });

  const eceData: BarComparisonDatum[] = models.map((model) => ({
    key: model.track,
    label: model.track,
    value: model.metrics.ece_after_calibration,
    detail: `${model.track}: expected calibration error ${show(model.metrics.ece_after_calibration)}`,
  }));

  return (
    <section aria-label="Track comparison" className="mt-4 grid gap-4 sm:grid-cols-3">
      <div className="rounded-lg border border-slate-300 bg-white p-4">
        <BarComparisonChart
          title="Test macro AUC"
          data={aucData}
          domainMax={1}
          formatValue={(value) => value.toFixed(4)}
        />
      </div>
      <div className="rounded-lg border border-slate-300 bg-white p-4">
        <BarComparisonChart
          title="Retention, lungs removed"
          data={retentionData}
          domainMax={1}
          formatValue={(value) => `${(value * 100).toFixed(1)}%`}
        />
      </div>
      <div className="rounded-lg border border-slate-300 bg-white p-4">
        <BarComparisonChart
          title="ECE after calibration"
          data={eceData}
          domainMax={eceDomainMax(models)}
          formatValue={(value) => value.toFixed(4)}
        />
      </div>
    </section>
  );
};

// Kept out of the severity module: this needs the full model, not just a
// retention value, to name which track the sentence belongs to.
function retentionVerdictFor(model: ModelCard): string {
  const retention = model.ablation.lungs_removed_retention;
  if (retention === null) return `${model.track}: no lung ablation was run for this model.`;
  return `${model.track}: ${(retention * 100).toFixed(1)}% of its signal survives with the lung fields blanked out.`;
}

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
        <div className="mt-4">
          <SetupNeededBanner
            testId="no-models"
            message={
              <>
                No models are loaded, so there is nothing to report. Run <code>cxr export</code>{" "}
                and point <code>MODEL_DIR</code> at the result.
              </>
            }
          />
        </div>
      ) : (
        <>
          <ComparisonSection models={data.models} />
          <div className="mt-4 space-y-4">
            {data.models.map((model) => (
              <ModelRow key={model.track} model={model} />
            ))}
          </div>
        </>
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

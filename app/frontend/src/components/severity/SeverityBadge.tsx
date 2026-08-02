import React from "react";
import { Severity, retentionVerdict, severityOf } from "../../api/predict";

/**
 * How loudly a severity should look, in one place.
 *
 * This used to be defined separately in PredictionCard.tsx and
 * ModelReport.tsx -- two copies that could silently drift apart on exactly
 * the thing that decides whether a reader notices a shortcut. Both consumers
 * import this instead.
 */
export const SEVERITY_STYLES: Record<Severity, string> = {
  high: "bg-red-50 border-red-300 text-red-900",
  moderate: "bg-amber-50 border-amber-300 text-amber-900",
  low: "bg-emerald-50 border-emerald-300 text-emerald-900",
  unmeasured: "bg-slate-100 border-slate-300 text-slate-700",
};

/** Matching SVG fill classes, for the bar charts. Same meaning, same map. */
export const SEVERITY_FILL: Record<Severity, string> = {
  high: "fill-red-500",
  moderate: "fill-amber-500",
  low: "fill-emerald-500",
  unmeasured: "fill-slate-400",
};

interface SeverityBadgeProps {
  severity: Severity;
  testId?: string;
  children: React.ReactNode;
}

export const SeverityBadge: React.FC<SeverityBadgeProps> = ({ severity, testId, children }) => (
  <p className={`rounded border px-3 py-2 text-sm ${SEVERITY_STYLES[severity]}`} data-testid={testId}>
    {children}
  </p>
);

interface RetentionVerdictProps {
  retention: number | null;
  testId?: string;
}

/**
 * The specific case that was duplicated twice: a retention number rendered as
 * its verdict sentence, colored by what that verdict means. Both call sites
 * pass their existing testid through so PredictionPanel.test.tsx and
 * ModelReport.test.tsx keep passing unmodified.
 */
export const RetentionVerdict: React.FC<RetentionVerdictProps> = ({ retention, testId }) => (
  <SeverityBadge severity={severityOf(retention)} testId={testId}>
    {retentionVerdict(retention)}
  </SeverityBadge>
);

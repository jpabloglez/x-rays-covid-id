// The /predict contract, mirrored from app/backend/api/main.py.
//
// `caveats` is not decoration and the UI is not free to drop it. Track 1 keeps
// 91.5% of its discriminative signal with the lung fields blanked out and
// Track 2 keeps 83.7%, so a probability rendered on its own is the exact
// artefact this project exists to argue against.

export interface Prediction {
  track: string;
  classes: string[];
  probabilities: Record<string, number>;
  predicted: string;
  confidence: number;
  macro_auc: number | null;
  lungs_removed_retention: number | null;
  caveats: string[];
}

export interface PredictResponse {
  detail: string;
  imageUrl: string;
  predictions: Prediction[];
  comparison: string;
  disclaimer: string;
}

// Requests go through Vite's dev proxy, so the browser stays same-origin.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

export class PredictError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "PredictError";
    this.status = status;
  }
}

export async function predict(image: File): Promise<PredictResponse> {
  const body = new FormData();
  body.append("image", image);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/predict/`, { method: "POST", body });
  } catch {
    throw new PredictError("Could not reach the server. Is the backend running?", 0);
  }

  let payload: Partial<PredictResponse> = {};
  try {
    payload = await response.json();
  } catch {
    // A non-JSON body from a proxy or gateway. Fall through to the status.
  }

  if (!response.ok) {
    // 503 means the service is healthy but has no exports loaded. That is an
    // operator problem, and saying so beats "prediction failed".
    const fallback =
      response.status === 503
        ? "No models are loaded on the server yet."
        : "Could not score that image.";
    throw new PredictError(payload.detail || fallback, response.status);
  }

  return {
    detail: payload.detail ?? "",
    imageUrl: payload.imageUrl ?? "",
    predictions: payload.predictions ?? [],
    comparison: payload.comparison ?? "",
    disclaimer: payload.disclaimer ?? "",
  };
}

/** How loudly to render a retention figure. */
export type Severity = "unmeasured" | "high" | "moderate" | "low";

export function severityOf(retention: number | null): Severity {
  if (retention === null || Number.isNaN(retention)) return "unmeasured";
  if (retention >= 0.8) return "high";
  if (retention >= 0.5) return "moderate";
  return "low";
}

/** What the retention means, in a sentence, derived rather than looked up. */
export function retentionVerdict(retention: number | null): string {
  const share = retention === null ? null : `${Math.round(retention * 100)}%`;
  switch (severityOf(retention)) {
    case "unmeasured":
      return "No lung ablation was run for this model, so there is no evidence either way about what it reads.";
    case "high":
      return `${share} of this model's signal survives with the lung fields blanked out. It is mostly not reading the anatomy.`;
    case "moderate":
      return `${share} of this model's signal survives with the lung fields blanked out. A majority of what it uses lies outside them.`;
    default:
      return `${share} of this model's signal survives with the lung fields blanked out, which rules out one shortcut but not every shortcut.`;
  }
}


// --------------------------------------------------------------- /models

export interface ModelGates {
  blocking: string[];
  acknowledged: string[];
  skipped: string[];
  /** What each failing gate measured, keyed by gate id. */
  findings: Record<string, string>;
}

export interface ModelAblation {
  lungs_removed_retention: number | null;
  lungs_only_retention: number | null;
  images: number | null;
  interpretation: string | null;
}

export interface ModelMetrics {
  macro_auc: number | null;
  balanced_accuracy: number | null;
  per_class_auc: Record<string, number>;
  ece_after_calibration: number | null;
}

export interface ModelCard {
  track: string;
  classes: string[];
  sources: string[];
  metrics: ModelMetrics;
  gates: ModelGates;
  ablation: ModelAblation;
  notes: string;
}

export interface ModelsResponse {
  disclaimer: string;
  models: ModelCard[];
}

/**
 * What the server actually has loaded.
 *
 * The report is built from this rather than from figures typed into the page,
 * so it cannot describe a model that is not there or quote a score the running
 * service would not produce. A page that hardcodes its own results is the
 * stale-claim failure this project keeps finding in its own documentation.
 */
export async function fetchModels(): Promise<ModelsResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/models`);
  } catch {
    throw new PredictError("Could not reach the server. Is the backend running?", 0);
  }
  if (!response.ok) {
    throw new PredictError("Could not load the model report.", response.status);
  }
  const payload = (await response.json()) as Partial<ModelsResponse>;
  return { disclaimer: payload.disclaimer ?? "", models: payload.models ?? [] };
}

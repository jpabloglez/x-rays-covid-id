import React from "react";

interface SetupNeededBannerProps {
  message: React.ReactNode;
  testId?: string;
}

/**
 * "This deployment isn't configured yet," never "you made a mistake."
 *
 * Deliberately not `role="alert"` and deliberately not the red error styling
 * used everywhere else in this app: an empty MODEL_DIR is an operator
 * problem, not something wrong with the file someone picked, and rendering it
 * identically to a validation error was the exact bug this component fixes.
 * Shared between the score view (checked proactively via /health, and as a
 * fallback on a 503 from /predict) and the report view (already had this
 * styling; now it just imports it instead of keeping its own copy).
 */
export const SetupNeededBanner: React.FC<SetupNeededBannerProps> = ({ message, testId }) => (
  <p
    className="rounded border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900"
    data-testid={testId}
  >
    {message}
  </p>
);

export default SetupNeededBanner;

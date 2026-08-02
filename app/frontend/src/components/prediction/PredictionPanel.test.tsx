import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PredictResponse } from "../../api/predict";
import PredictionPanel from "./PredictionPanel";

// Track 1 and Track 2 as they actually come back, including the numbers that
// make this project's point: 0.9925 with 91.5% of the signal surviving the
// lungs being blanked out, against 0.7460 with 83.7%.
const RESULT: PredictResponse = {
  detail: "Scored by every loaded model",
  imageUrl: "/media/abc.png",
  disclaimer: "Research artefact, not a diagnostic device.",
  comparison:
    "track1 says covid (79% confidence) and track2 says non_covid (56% confidence). " +
    "These are different questions over different corpora, not a second opinion.",
  predictions: [
    {
      track: "track1",
      classes: ["normal", "pneumonia", "covid"],
      probabilities: { normal: 0.065, pneumonia: 0.148, covid: 0.787 },
      predicted: "covid",
      confidence: 0.787,
      macro_auc: 0.9925,
      lungs_removed_retention: 0.9152,
      caveats: [
        "92% of this model's discriminative signal survives having the lung fields blanked out.",
        "Gate G4 fails and is acknowledged: COVID is drawn from a single collection.",
      ],
    },
    {
      track: "track2",
      classes: ["non_covid", "covid"],
      probabilities: { non_covid: 0.565, covid: 0.435 },
      predicted: "non_covid",
      confidence: 0.565,
      macro_auc: 0.746,
      lungs_removed_retention: 0.837,
      caveats: [
        "Gate G5 fails and is acknowledged: Cramer's V 0.474 between scanner_model and class.",
      ],
    },
  ],
};

describe("PredictionPanel", () => {
  it("renders every model rather than choosing one", () => {
    render(<PredictionPanel result={RESULT} />);
    expect(screen.getByRole("heading", { name: "track1" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "track2" })).toBeInTheDocument();
  });

  it("states that the two models are not a second opinion", () => {
    // A reader shown two numbers looks for a winner. The panel has to say
    // there is not one, or the layout implies a consensus that does not exist.
    render(<PredictionPanel result={RESULT} />);
    expect(screen.getByTestId("comparison")).toHaveTextContent("not a second opinion");
  });

  it("shows the disclaimer without needing an interaction", () => {
    render(<PredictionPanel result={RESULT} />);
    expect(screen.getByRole("note")).toHaveTextContent("not a diagnostic device");
  });

  it("puts the retention verdict on screen for every model", () => {
    // The guard that matters. A probability rendered without the measurement
    // of what produced it is the artefact this repository argues against, so
    // the UI must not be able to drop it silently.
    render(<PredictionPanel result={RESULT} />);
    expect(screen.getByTestId("retention-track1")).toHaveTextContent(
      "mostly not reading the anatomy",
    );
    expect(screen.getByTestId("retention-track2")).toHaveTextContent(
      "mostly not reading the anatomy",
    );
  });

  it("renders every caveat the server sent, not a truncated selection", () => {
    render(<PredictionPanel result={RESULT} />);
    for (const prediction of RESULT.predictions) {
      for (const caveat of prediction.caveats) {
        expect(screen.getByText(caveat)).toBeInTheDocument();
      }
    }
  });

  it("says the stored image is not linked to whoever uploaded it", () => {
    render(<PredictionPanel result={RESULT} />);
    expect(screen.getByText(/No record is kept of who uploaded it/)).toBeInTheDocument();
  });

  it("copes with a single model without implying a comparison", () => {
    const solo: PredictResponse = {
      ...RESULT,
      predictions: [RESULT.predictions[1]],
      comparison: "Only one model is loaded, so there is nothing to compare.",
    };
    render(<PredictionPanel result={solo} />);
    expect(screen.getByTestId("comparison")).toHaveTextContent("nothing to compare");
  });
});

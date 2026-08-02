import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ModelsResponse } from "../../api/predict";
import ModelReport from "./ModelReport";

// Both models exactly as /models returns them, including the two findings that
// are the project's result: Track 1 keeping 91.5% of its signal with the lungs
// blanked out, and Track 2's scanner confound.
const MODELS: ModelsResponse = {
  disclaimer: "Research artefact, not a diagnostic device.",
  models: [
    {
      track: "track1",
      classes: ["normal", "pneumonia", "covid"],
      sources: ["covid_radiography", "rsna_pneumonia"],
      metrics: {
        macro_auc: 0.9925,
        balanced_accuracy: 0.9653,
        per_class_auc: { normal: 0.9894, pneumonia: 0.988, covid: 1.0 },
        ece_after_calibration: 0.0069,
      },
      gates: {
        blocking: [],
        acknowledged: ["G3", "G4"],
        skipped: [],
        findings: { G4: "COVID is drawn from a single collection." },
      },
      ablation: {
        lungs_removed_retention: 0.9152,
        lungs_only_retention: 0.8922,
        images: 4168,
        interpretation: "…",
      },
      notes: "3-class, pooled across 2 sources.",
    },
    {
      track: "track2",
      classes: ["non_covid", "covid"],
      sources: ["bimcv_covid19"],
      metrics: {
        macro_auc: 0.746,
        balanced_accuracy: 0.6479,
        per_class_auc: { non_covid: 0.746, covid: 0.746 },
        ece_after_calibration: 0.0854,
      },
      gates: {
        blocking: [],
        acknowledged: ["G5"],
        skipped: ["G1b", "G3", "G4"],
        findings: {
          G5: "Cramer's V 0.474 between scanner_model and class over 42 devices.",
        },
      },
      ablation: {
        lungs_removed_retention: 0.837,
        lungs_only_retention: 0.4653,
        images: 292,
        interpretation: "…",
      },
      notes: "2-class, single source.",
    },
  ],
};

function mockFetch(payload: unknown, ok = true) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({ ok, status: ok ? 200 : 500, json: async () => payload }),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("ModelReport", () => {
  it("reports every loaded model", async () => {
    mockFetch(MODELS);
    render(<ModelReport />);
    expect(await screen.findByRole("heading", { name: "track1" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "track2" })).toBeInTheDocument();
  });

  it("shows the figures the server reports rather than any typed into the page", async () => {
    mockFetch(MODELS);
    render(<ModelReport />);
    // Scoped to each model's own card: the comparison chart now also renders
    // these same figures elsewhere on the page, which is intentional -- an
    // unscoped query would find both and correctly fail as ambiguous.
    const track1Card = (await screen.findByRole("heading", { name: "track1" })).closest(
      "article",
    )!;
    const track2Card = screen.getByRole("heading", { name: "track2" }).closest("article")!;
    expect(within(track1Card).getByText("0.9925")).toBeInTheDocument();
    expect(within(track2Card).getByText("0.7460")).toBeInTheDocument();
  });

  it("states what each failing gate measured, not just that it failed", async () => {
    mockFetch(MODELS);
    render(<ModelReport />);
    expect(
      await screen.findByText(/Cramer's V 0.474 between scanner_model and class/),
    ).toBeInTheDocument();
  });

  it("says a skipped gate is not a passed gate", async () => {
    mockFetch(MODELS);
    render(<ModelReport />);
    await waitFor(() =>
      expect(screen.getByTestId("report-skipped-track2")).toHaveTextContent(
        "not a passed gate",
      ),
    );
  });

  it("puts the retention verdict beside every model", async () => {
    mockFetch(MODELS);
    render(<ModelReport />);
    await waitFor(() =>
      expect(screen.getByTestId("report-verdict-track1")).toHaveTextContent(
        "mostly not reading the anatomy",
      ),
    );
    expect(screen.getByTestId("report-verdict-track2")).toHaveTextContent(
      "mostly not reading the anatomy",
    );
  });

  it("reports an empty server as empty rather than showing nothing", async () => {
    // The failure mode a hardcoded report would hide: no models loaded, but the
    // page still displaying a set of impressive numbers.
    mockFetch({ disclaimer: "d", models: [] });
    render(<ModelReport />);
    expect(await screen.findByTestId("no-models")).toHaveTextContent("nothing to report");
  });

  it("surfaces a failure to reach the server", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("boom")));
    render(<ModelReport />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not reach the server");
  });

  it("charts the AUC comparison with both tracks' real figures", async () => {
    mockFetch(MODELS);
    render(<ModelReport />);
    const chart = await screen.findByRole("img", { name: /Test macro AUC/ });
    expect(chart).toHaveAccessibleName("Test macro AUC: track1 0.9925, track2 0.7460");
  });

  it("charts the retention comparison, colored by the same severity map as the badge", async () => {
    mockFetch(MODELS);
    render(<ModelReport />);
    const chart = await screen.findByRole("img", { name: /Retention, lungs removed/ });
    expect(chart).toHaveAccessibleName(
      "Retention, lungs removed: track1 91.5%, track2 83.7%",
    );
  });

  it("charts the calibration comparison", async () => {
    mockFetch(MODELS);
    render(<ModelReport />);
    const chart = await screen.findByRole("img", { name: /ECE after calibration/ });
    expect(chart).toHaveAccessibleName("ECE after calibration: track1 0.0069, track2 0.0854");
  });

  it("renders no comparison charts when no models are loaded", async () => {
    // An axis with nothing plotted underneath a "no models" banner would
    // contradict that banner. The section must not render at all.
    mockFetch({ disclaimer: "d", models: [] });
    render(<ModelReport />);
    await screen.findByTestId("no-models");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});

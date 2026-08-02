import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RetentionVerdict, SeverityBadge } from "./SeverityBadge";

describe("SeverityBadge", () => {
  it("renders its children inside a bordered box carrying the severity's styling", () => {
    render(<SeverityBadge severity="high">custom content</SeverityBadge>);
    const box = screen.getByText("custom content");
    expect(box.className).toContain("bg-red-50");
  });
});

describe("RetentionVerdict", () => {
  // Pinned against the project's real, reported numbers: Track 1 keeps 91.5%
  // of its signal with the lungs blanked out (reads the collection), Track 2
  // keeps 83.7% (reads the scanner). Both are "high" severity -- the
  // regression this whole project exists to catch is a UI that stops making
  // that obvious.
  it("renders Track 1's real retention as high severity", () => {
    render(<RetentionVerdict retention={0.9152} testId="t1" />);
    const box = screen.getByTestId("t1");
    expect(box).toHaveTextContent("92% of this model's signal survives");
    expect(box.className).toContain("bg-red-50");
  });

  it("renders Track 2's real retention as high severity", () => {
    render(<RetentionVerdict retention={0.837} testId="t2" />);
    const box = screen.getByTestId("t2");
    expect(box).toHaveTextContent("84% of this model's signal survives");
    expect(box.className).toContain("bg-red-50");
  });

  it("crosses into moderate severity below 0.8", () => {
    render(<RetentionVerdict retention={0.79} testId="mod" />);
    expect(screen.getByTestId("mod").className).toContain("bg-amber-50");
  });

  it("crosses into low severity below 0.5", () => {
    render(<RetentionVerdict retention={0.49} testId="low" />);
    expect(screen.getByTestId("low").className).toContain("bg-emerald-50");
  });

  it("renders a missing ablation as unmeasured, not as zero", () => {
    // None and 0.0 mean opposite things: zero retention would be the best
    // possible result (the model collapsed to chance without its lungs), and
    // "unmeasured" says no ablation was run at all.
    render(<RetentionVerdict retention={null} testId="none" />);
    const box = screen.getByTestId("none");
    expect(box).toHaveTextContent("No lung ablation was run");
    expect(box.className).toContain("bg-slate-100");
  });
});

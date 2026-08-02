import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import BarComparisonChart, { BarComparisonDatum } from "./BarComparisonChart";

const DATA: BarComparisonDatum[] = [
  { key: "track1", label: "track1", value: 0.9925, detail: "track1 macro AUC 0.9925" },
  { key: "track2", label: "track2", value: 0.746, detail: "track2 macro AUC 0.7460" },
];

describe("BarComparisonChart", () => {
  it("renders one bar per datum, proportional to value/domainMax", () => {
    const { container } = render(
      <BarComparisonChart title="Test macro AUC" data={DATA} domainMax={1} width={400} />,
    );
    const bars = container.querySelectorAll("rect.fill-slate-600");
    expect(bars).toHaveLength(2);

    // Both bars share the same track width; track1's fraction (0.9925) must
    // produce a wider bar than track2's (0.7460).
    const widths = Array.from(bars).map((bar) => Number(bar.getAttribute("width")));
    expect(widths[0]).toBeGreaterThan(widths[1]);
    expect(widths[1] / widths[0]).toBeCloseTo(0.746 / 0.9925, 2);
  });

  it("renders a null value as unmeasured, with no foreground bar for that row", () => {
    const data: BarComparisonDatum[] = [
      { key: "a", label: "a", value: 0.5 },
      { key: "b", label: "b", value: null },
    ];
    const { container } = render(
      <BarComparisonChart title="X" data={data} domainMax={1} unmeasuredLabel="not measured" />,
    );
    expect(container.querySelectorAll("rect.fill-slate-600")).toHaveLength(1);
    expect(screen.getByText("not measured")).toBeInTheDocument();
  });

  it("names every label and formatted value in the accessible name", () => {
    render(
      <BarComparisonChart
        title="Test macro AUC"
        data={DATA}
        domainMax={1}
        formatValue={(value) => value.toFixed(4)}
      />,
    );
    const svg = screen.getByRole("img");
    expect(svg).toHaveAccessibleName(
      "Test macro AUC: track1 0.9925, track2 0.7460",
    );
  });

  it("puts the full sentence behind a bar in its tooltip, not just the number", () => {
    const { container } = render(
      <BarComparisonChart title="Retention" data={DATA} domainMax={1} />,
    );
    const titles = Array.from(container.querySelectorAll("title")).map((el) => el.textContent);
    expect(titles).toContain("track1 macro AUC 0.9925");
    expect(titles).toContain("track2 macro AUC 0.7460");
  });
});

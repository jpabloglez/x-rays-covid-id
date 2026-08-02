import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ChestIcon from "./ChestIcon";

describe("ChestIcon", () => {
  it("renders a decorative SVG with a sensible icon viewBox", () => {
    const { container } = render(<ChestIcon />);
    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
    expect(svg).toHaveAttribute("viewBox", "0 0 24 24");
    // Decorative, beside visible text (the page title) -- must not be
    // announced a second time by assistive tech.
    expect(svg).toHaveAttribute("aria-hidden", "true");
  });

  it("takes sizing and color from the caller via className, not a hardcoded one", () => {
    const { container } = render(<ChestIcon className="h-8 w-8 text-slate-700" />);
    expect(container.querySelector("svg")).toHaveClass("h-8", "w-8", "text-slate-700");
  });
});

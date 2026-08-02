import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import SetupNeededBanner from "./SetupNeededBanner";

describe("SetupNeededBanner", () => {
  it("renders the given message", () => {
    render(<SetupNeededBanner message="test message" testId="banner" />);
    expect(screen.getByTestId("banner")).toHaveTextContent("test message");
  });

  it("is not an alert and does not carry the red error styling", () => {
    // The bug this component fixes: an operator misconfiguration (no models
    // loaded) rendered identically to a user mistake (file too large). The
    // two must be structurally distinguishable, not just differently worded.
    render(<SetupNeededBanner message="x" testId="banner" />);
    const banner = screen.getByTestId("banner");
    expect(banner).not.toHaveAttribute("role", "alert");
    expect(banner.className).not.toContain("bg-red-50");
    expect(banner.className).toContain("bg-amber-50");
  });
});

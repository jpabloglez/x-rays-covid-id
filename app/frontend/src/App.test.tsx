import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

// App mounts ImageUpload by default, which checks /health on mount. Mocked so
// the test is not making a real network call to a server that is not running.
vi.mock("./api/predict", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api/predict")>();
  return {
    ...actual,
    fetchHealth: vi.fn().mockResolvedValue({ status: "ok", models: [], inference_available: true }),
  };
});

beforeEach(() => {
  vi.clearAllMocks();
});

describe("App", () => {
  it("shows the shortened title, without the old subtitle appended to it", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "Chest radiograph classifier" })).toBeInTheDocument();
    expect(screen.queryByText(/research artefact$/)).not.toBeInTheDocument();
  });

  it("shows the chest icon beside the title", () => {
    const { container } = render(<App />);
    expect(container.querySelector("header svg")).toBeInTheDocument();
  });

  it("switches between the score and report views", async () => {
    render(<App />);
    expect(await screen.findByText(/Score a chest radiograph/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "What these models are" }));
    expect(await screen.findByText(/What these models are/)).toBeInTheDocument();
  });
});

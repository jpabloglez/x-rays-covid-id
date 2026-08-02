import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PredictError, PredictResponse } from "../../api/predict";
import ImageUploader from "./ImageUploader";

vi.mock("../../api/predict", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/predict")>();
  return { ...actual, predict: vi.fn(), fetchHealth: vi.fn() };
});

// Re-imported after the mock so these are the mocked functions, typed as such.
import { fetchHealth, predict } from "../../api/predict";

const mockedPredict = predict as unknown as ReturnType<typeof vi.fn>;
const mockedFetchHealth = fetchHealth as unknown as ReturnType<typeof vi.fn>;

const RESULT: PredictResponse = {
  detail: "Scored by every loaded model",
  imageUrl: "/media/abc123.png",
  disclaimer: "Research artefact, not a diagnostic device.",
  comparison: "track1 says covid.",
  predictions: [
    {
      track: "track1",
      classes: ["non_covid", "covid"],
      probabilities: { non_covid: 0.435, covid: 0.565 },
      predicted: "covid",
      confidence: 0.565,
      macro_auc: 0.746,
      lungs_removed_retention: 0.837,
      caveats: ["84% of this model's signal survives with the lung fields blanked out."],
    },
  ],
};

function pngFile(name = "chest.png", size = 1024): File {
  const file = new File([new Uint8Array(size)], name, { type: "image/png" });
  return file;
}

function selectFile(file: File | File[]) {
  const input = screen.getByLabelText(/radiograph/i) as HTMLInputElement;
  const files = Array.isArray(file) ? file : [file];
  Object.defineProperty(input, "files", { value: files, configurable: true });
  fireEvent.change(input);
  return input;
}

beforeEach(() => {
  mockedFetchHealth.mockResolvedValue({ status: "ok", models: ["track1"], inference_available: true });
  mockedPredict.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("ImageUploader — preview", () => {
  it("shows a preview of the selected file before it is submitted", async () => {
    render(<ImageUploader />);
    selectFile(pngFile());
    const preview = await screen.findByAltText("Selected radiograph, not yet scored");
    expect(preview).toBeInTheDocument();
    // Not scored yet: predict() must not have been called just by selecting.
    expect(mockedPredict).not.toHaveBeenCalled();
  });

  it("swaps the preview when a different file is chosen before submitting", async () => {
    let counter = 0;
    vi.spyOn(URL, "createObjectURL").mockImplementation(() => `blob:mock-${counter++}`);

    render(<ImageUploader />);
    selectFile(pngFile("a.png"));
    const first = await screen.findByAltText("Selected radiograph, not yet scored");
    const firstSrc = first.getAttribute("src");

    selectFile(pngFile("b.png"));
    const second = await screen.findByAltText("Selected radiograph, not yet scored");
    expect(second.getAttribute("src")).not.toBe(firstSrc);
  });

  it("shows only the server's stored image after a successful score, not both", async () => {
    mockedPredict.mockResolvedValue(RESULT);
    render(<ImageUploader />);
    selectFile(pngFile());
    await screen.findByAltText("Selected radiograph, not yet scored");

    fireEvent.click(screen.getByRole("button", { name: /score/i }));

    const scoredImage = await screen.findByAltText("The radiograph that was scored");
    expect(scoredImage).toHaveAttribute("src", RESULT.imageUrl);
    expect(screen.queryByAltText("Selected radiograph, not yet scored")).not.toBeInTheDocument();
    expect(screen.getAllByRole("img")).toHaveLength(1);
  });
});

describe("ImageUploader — setup needed", () => {
  it("shows the setup banner and hides the form when /health reports no models", async () => {
    mockedFetchHealth.mockResolvedValue({ status: "ok", models: [], inference_available: false });
    render(<ImageUploader />);

    const banner = await screen.findByTestId("setup-needed");
    expect(banner).toHaveTextContent("deployment issue");
    expect(screen.queryByLabelText(/radiograph/i)).not.toBeInTheDocument();
  });

  it("shows the setup banner, not the generic red alert, when predict() 503s", async () => {
    mockedPredict.mockRejectedValue(
      new PredictError("No models are loaded on the server yet.", 503),
    );
    render(<ImageUploader />);
    selectFile(pngFile());
    fireEvent.click(screen.getByRole("button", { name: /score/i }));

    const banner = await screen.findByTestId("setup-needed-submit");
    expect(banner).toHaveTextContent("No models are loaded");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("ImageUploader — ordinary errors stay ordinary errors", () => {
  it("rejects an oversized file with the plain red alert, not the setup banner", async () => {
    render(<ImageUploader />);
    selectFile(pngFile("big.png", 21 * 1024 * 1024));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("20 MB limit");
    expect(screen.queryByTestId("setup-needed-submit")).not.toBeInTheDocument();
  });

  it("rejects a non-image file type with the plain red alert", async () => {
    render(<ImageUploader />);
    const file = new File(["x"], "doc.pdf", { type: "application/pdf" });
    selectFile(file);

    expect(await screen.findByRole("alert")).toHaveTextContent("JPEG or PNG");
  });

  it("shows the plain red alert on a network failure, not the setup banner", async () => {
    mockedPredict.mockRejectedValue(new PredictError("Could not reach the server. Is the backend running?", 0));
    render(<ImageUploader />);
    selectFile(pngFile());
    fireEvent.click(screen.getByRole("button", { name: /score/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Could not reach the server");
    expect(screen.queryByTestId("setup-needed-submit")).not.toBeInTheDocument();
  });
});

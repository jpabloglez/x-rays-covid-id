import "@testing-library/jest-dom/vitest";

// jsdom does not implement the object URL API. ImageUploader uses it for the
// local file preview; without a stub every test that selects a file throws.
if (typeof URL.createObjectURL === "undefined") {
  URL.createObjectURL = () => "blob:mock";
}
if (typeof URL.revokeObjectURL === "undefined") {
  URL.revokeObjectURL = () => undefined;
}

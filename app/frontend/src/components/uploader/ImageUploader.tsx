import React, { useState, ChangeEvent, FormEvent } from "react";

interface UploadResponse {
  detail: string;
  imageUrl: string;
}

// Requests go through Vite's dev proxy, so the browser stays same-origin.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

const ACCEPTED_TYPES = ["image/jpeg", "image/png"];
const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

const ImageUpload: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState("");
  const [error, setError] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  // Function to handle file selection
  const handleFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setError("");
    setUploadedImageUrl("");

    if (file && !ACCEPTED_TYPES.includes(file.type)) {
      setSelectedImage(null);
      setError("Please choose a JPEG or PNG image.");
      return;
    }
    if (file && file.size > MAX_UPLOAD_BYTES) {
      setSelectedImage(null);
      setError("That image is larger than the 20 MB limit.");
      return;
    }
    setSelectedImage(file);
  };

  // Function to handle form submission
  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!selectedImage) {
      setError("Choose an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedImage);

    setIsUploading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE_URL}/files/`, {
        method: "POST",
        body: formData,
      });

      const data: UploadResponse = await response.json();
      if (!response.ok) {
        setError(data.detail || "Upload failed.");
        return;
      }
      setUploadedImageUrl(data.imageUrl);
    } catch {
      setError("Could not reach the server. Is the backend running?");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="container">
      <div className="row">
        <h2 className="text-2xl text-black mb-4">Image Upload</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="file"
            accept="image/jpeg, image/png"
            onChange={handleFileSelect}
            className="bg-blue-400 mb-2 content-center"
          />
          <div className="row text-black">Accepts jpg/jpeg and png files</div>

          <button
            type="submit"
            disabled={isUploading}
            className="bg-blue-400 text-black mt-4 py-2 px-4 rounded disabled:opacity-50"
          >
            {isUploading ? "Uploading…" : "Upload"}
          </button>

          {error.length > 0 && (
            <p role="alert" className="text-red-600 mt-2">
              {error}
            </p>
          )}
          {uploadedImageUrl.length > 0 && (
            <img src={uploadedImageUrl} alt="Uploaded file" className="mt-2" />
          )}
        </form>
      </div>
    </div>
  );
};

export default ImageUpload;

import React, { useState, ChangeEvent, FormEvent } from "react";

interface UploadResponse {
  detail: string;
  imageUrl: string;
}

const ImageUpload: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState("");

  // Function to handle file selection
  const handleFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setSelectedImage(file || null);
  };

  // Function to handle form submission
  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (selectedImage) {
      const formData = new FormData();
      formData.append("image", selectedImage);

      try {
        // Send the formData to the server for image upload
        const response = await fetch("http://localhost:3080/files/", {
          method: "POST",
          body: formData,
        });

        console.log("Response from API:", response);
        const data: UploadResponse = await response.json();
        console.log("Data from API:", data.imageUrl);
        setUploadedImageUrl(data.imageUrl);

        //}
      } catch (error) {
        console.error("Error uploading image =>=>=>", error);
      }
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
        <div className="row text-black">
          Accepts jpg/jpeg and png files
        </div>

        <button
          type="submit"
          className="bg-blue-400 text-black mt-4 py-2 px-4 rounded"
        >
          Upload
        </button>        
        {uploadedImageUrl.length > 0 && (
          <img src={uploadedImageUrl} alt="Uploaded file" className="mt-2" />
        )}
      </form>
      </div>
      </div>);
};

export default ImageUpload;

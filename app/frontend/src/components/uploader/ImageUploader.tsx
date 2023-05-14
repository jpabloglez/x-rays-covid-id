import React, { useState, ChangeEvent, FormEvent } from 'react';
const ImageUpload: React.FC = () => {
    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [uploadedImageUrl, setUploadedImageUrl] = useState('');

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
        formData.append('image', selectedImage);
  
        try {
          // Send the formData to the server for image upload
          const response = await fetch('http://localhost:3080/files/', {
            method: 'POST',
            body: formData,
          });
  
          // Handle the response from the server
          if (response.ok) {
            const responseData = await response.json();
            alert('Image uploaded successfully!' + responseData['imageUrl']);
            setUploadedImageUrl(responseData.imageUrl);
            console.log('Image uploaded successfully!');
          } else {
            console.error('Error uploading image');
          }
        } catch (error) {
          console.error('Error uploading image:', error);
        }
      }
    };
  
    return (
      <div className="container">
        <h2 className="text-2xl font-bold mb-4">Image Upload</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="file"
            accept="image/jpeg, image/png"
            onChange={handleFileSelect}
            className="mb-2"
          />
          <button type="submit" className="bg-blue-500 text-white py-2 px-4 rounded">
            Upload
          </button>
        </form>
      </div>
    );
  };
  
  export default ImageUpload;
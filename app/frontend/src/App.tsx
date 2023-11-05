import { useState } from "react";
import "./App.css";

import Navbar from "./components/navbars/Navbar";
import Footer from "./components/footers/Footer";
import ImageUpload from './components/uploader/ImageUploader';

function App() {
  const [count, setCount] = useState(0);

  return (
    <div className="container mx-auto border-solid border-2 rounded-lg">
      <header>
        <Navbar></Navbar>
      </header>


      {/* Image Upload */}
      <div className="flex items p-4 bg-transparent min-h-screen self-auto">
        <div className="bg-white shadow rounded self-auto  p-4 m-4">
          <section id="imageupload">
            <ImageUpload />
          </section>ç
        </div>
      </div>


      <Footer></Footer>
    </div>
  );
}

export default App;

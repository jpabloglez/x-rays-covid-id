import { useState } from "react";
import "./App.css";

import Contact from "./components/Contact";

import Navbar from "./components/navbars/Navbar";
import ProfileCard from "./components/profiles/Profile";
import About from "./components/about/About";
import Publications from "./components/publications/Publications";
import { publications } from "./components/publications/PublicationsList";
import Projects from "./components/projects/Projects";
import { projects } from "./components/projects/ProjectsList";
import Footer from "./components/footers/Footer";
import ImageUpload from './components/uploader/ImageUploader';

function App() {
  const [count, setCount] = useState(0);

  return (
    <div className="container mx-auto border-solid border-2 rounded-lg bg-white">
      <header>
        <Navbar></Navbar>
      </header>

      {/* Profile */}
      <div className="bg-gray-100 flex justify-left p-4 min-w-fit">
        <div className="m-4">
          <ProfileCard
            name="John Doe"
            title="Software Engineer"
            image="https://via.placeholder.com/150"
            description="John is a passionate software engineer with 5+ years of experience in web and mobile app development."
            github="jpablogglez"
            linkedin="jpablogglez"
            twitter="jpablogglez"
            urlcv="https://via.placeholder.com/150"
          />
        </div>

        <div className="bg-gray-100 flex items-right justify-right p-4 mt-8 m-4 max-w-4xl">
          <section id="about">
            <About title="About me" />
          </section>
        </div>
      </div>

      {/* Projects */}
      <div className="bg-gray-100 flex items-center justify-center p-4 min-w-fit">
        <section id="projects">
          <Projects projects={projects} />
        </section>
      </div>

      {/* Publications */}

      <div className="bg-gray-100 flex items-center justify-center p-4 min-w-fit">
        <section id="publications">
          <Publications publications={publications} />
        </section>
      </div>

      {/* Image Upload */}
      <div className="bg-gray-100 flex items-center justify-center p-4 min-w-fit">
        <section id="imageupload">
          <ImageUpload />
        </section>
      </div>


      {/* Contact */}

      <Footer></Footer>
    </div>
  );
}

export default App;

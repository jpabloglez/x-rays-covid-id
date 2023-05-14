import React from "react";

export default function Footer({}) {
  const [navbarOpen, setNavbarOpen] = React.useState(false);
  return (
    <footer className="text-gray-701 body-font">
      <div className="container px-5 py-8 mx-auto flex items-center sm:flex-row flex-col">
        <a className="flex title-font font-medium items-center md:justify-start justify-center text-gray-900">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="currentColor"
            className="w-10 h-10 text-white p-2 bg-indigo-500 rounded-full"
            viewBox="0 0 975.036 975.036"
          ></svg>
          <span className="ml-3 text-xl">Portofolio</span>
        </a>
        <p className="text-sm text-gray-500 sm:ml-6 sm:mt-0 mt-4">
          © John Doe —
          <a
            href="https://twitter.com/knyttneve"
            className="text-gray-600 ml-1"
            rel="noopener noreferrer"
            target="_blank"
          >
            @johndoe
          </a>
        </p>
      </div>
    </footer>
  );
}

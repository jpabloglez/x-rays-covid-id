import React from "react";

export default function Footer({}) {
  const [navbarOpen, setNavbarOpen] = React.useState(false);
  return (
    <footer className="text-gray-500 body-font bg-gray-100">
      <div className="container px-2 py-2 mx-auto flex rounded items-center sm:flex-row flex-col bg-gradient-to-r from-sky-100 to-indigo-100">
        <a className="flex title-font font-medium items-center md:justify-start justify-center text-gray-900">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="currentColor"
            className="w-10 h-10 text-white p-2 bg-indigo-100 rounded-2"
            viewBox="0 0 975.036 975.036"
          ></svg>
          <span className="ml-3 text-xl">COVID-xr</span>
        </a>
        {/*}
        <p className="text-sm text-gray-500 sm:ml-6 sm:mt-0 mt-4 justify-end">
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
  */}
      </div>
    </footer>
  );
}

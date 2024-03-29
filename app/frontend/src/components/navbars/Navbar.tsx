import React from "react";

export default function Navbar({ }) {
  const [navbarOpen, setNavbarOpen] = React.useState(false);
  return (
    <nav className="flex items-center justify-between flex-wrap bg-neutral-100 p-6 rounded-lg shadow-md">
            <div className="flex items-center flex-shrink-0 text-gray-500 mr-6">
                <span className="font-semibold text-xl tracking-tight">
                COVID-xr
                </span>
            </div>
            <div className="block lg:hidden">
                <button className="flex items-center px-3 py-2 border rounded text-teal-200 border-teal-400 hover:text-white hover:border-white">
                <svg
                    className="fill-current h-3 w-3"
                    viewBox="0 0 20 20"
                    xmlns="http://www.w3.org/2000/svg"
                >
                    <title>Menu</title>
                    <path d="M0 3h20v2H0V3zm0 6h20v2H0V9zm0 6h20v2H0v-2z" />
                </svg>
                </button>
            </div>
            <div className="w-full block flex-grow lg:flex lg:items-center lg:w-auto">
                <div className="text-sm lg:flex-grow">
                <a
                    href="#about"
                    className="block mt-4 lg:inline-block lg:mt-0 text-gray-500 hover:text-white mr-4"
                >
                    About
                </a>
                <a
                    href="#projects"
                    className="block mt-4 lg:inline-block lg:mt-0 text-gray-500 hover:text-white mr-4"
                >
                    References
                </a>
                </div>
                <div>
                <a
                    href="#"
                    className="inline-block text-sm px-4 py-2 leading-none border rounded text-gray-500 border-white hover:border-transparent hover:text-gray-500 hover:bg-neutral-100
                     mt-4 lg:mt-0"
                >
                    Download
                </a>
                </div>
            </div>
            </nav>
  );
}
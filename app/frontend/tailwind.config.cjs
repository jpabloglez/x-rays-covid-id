/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  //darkMode: false,
  theme: {
    extend: {
      backgroundImage: (theme) => ({
        'gradient': 'linear-gradient(135deg, #f1edfe 0%, rgb(194, 225, 254) 100%)',
      }),
    },
  },
  plugins: [

    //require('@tailwindcss/forms'),
    //require('@tailwindcss/typography'),
    require('@tailwindcss/aspect-ratio'),
  ],
}
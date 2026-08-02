const defaultTheme = require('tailwindcss/defaultTheme')

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
      fontFamily: {
        // Tailwind's own default stacks, named explicitly rather than left
        // implicit. Costs nothing -- no webfont, no network request, no
        // licensing review -- and gives every numeric value in the app one
        // named class (`font-mono`) instead of a system fallback nobody chose
        // on purpose. A real webfont is a one-line swap here if ever wanted.
        sans: [...defaultTheme.fontFamily.sans],
        mono: [...defaultTheme.fontFamily.mono],
      },
    },
  },
  plugins: [

    //require('@tailwindcss/forms'),
    //require('@tailwindcss/typography'),
    require('@tailwindcss/aspect-ratio'),
  ],
}
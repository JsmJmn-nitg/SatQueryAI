/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        satBgDark: '#090D1A',
        satCardDark: '#0B1021',
        satBorderDark: '#1A233D',
      }
    },
  },
  plugins: [],
}

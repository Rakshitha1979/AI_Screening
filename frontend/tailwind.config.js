/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter"', '"Segoe UI"', "sans-serif"],
      },
      colors: {
        canvas: "#F5F7FC",
        heading: "#243B60",
        subheading: "#31527F",
        body: "#263449",
        brandBlue: "#2563EB",
        brandTeal: "#0D9488",
      },
    },
  },
  plugins: [],
};

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      keyframes: {
        scan: { '0%, 100%': { top: '0%' }, '50%': { top: 'calc(100% - 2px)' } },
        'fade-in': { from: { opacity: '0', transform: 'translateY(-4px)' }, to: { opacity: '1', transform: 'none' } },
      },
      animation: {
        scan: 'scan 1.8s ease-in-out infinite',
        'fade-in': 'fade-in 150ms ease-out',
      },
    },
  },
  plugins: [],
}

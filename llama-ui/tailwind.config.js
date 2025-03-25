/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        terminal: {
          black: '#000000',
          green: '#33FF33',
          cyan: '#00CCCC',
          blue: '#0000FF',
          darkBlue: '#000033',
          purple: '#800080',
          yellow: '#FFFF00',
          orange: '#FF8800',
          red: '#FF0000',
          background: '#0C0C0C',
          foreground: '#33FF33',
          commandPrompt: '#33FF33',
          highlight: '#0055FF',
          dimmed: '#777777',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'Courier New', 'monospace'],
      },
      boxShadow: {
        'terminal': '0 0 10px rgba(51, 255, 51, 0.5)',
        'glow': '0 0 8px rgba(51, 255, 51, 0.8)',
      },
      animation: {
        'cursor-blink': 'blink 1s step-end infinite',
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        }
      }
    },
  },
  plugins: [],
  darkMode: 'class',
} 
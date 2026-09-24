import type { Config } from "tailwindcss";

// Design tokens for QueryMind — see README for the design rationale.
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#171522",       // dark surface (sidebar, header)
        parchment: "#F3EEE2", // light surface (cards, main chat bg)
        slate: "#2A2833",     // body text on light surfaces
        signal: "#2F8F82",    // AI responses / active states
        citation: "#C98A3B",  // source citation markers
      },
      fontFamily: {
        display: ["var(--font-fraunces)", "serif"],
        sans: ["var(--font-inter)", "sans-serif"],
      },
      maxWidth: {
        chat: "720px",
        "chat-wide": "1000px",
      },
    },
  },
  plugins: [],
};

export default config;

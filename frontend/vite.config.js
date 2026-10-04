import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In development, /api is forwarded to the FastAPI backend (uvicorn on port 8000).
// In Docker, nginx does the same forwarding (Phase 7), so the code always calls relative /api URLs.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { "/api": process.env.VITE_API_TARGET || "http://localhost:8000" },
  },
});

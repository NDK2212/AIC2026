import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // FastAPI mounts the production bundle at /static. During development,
  // Vite owns the root URL and proxies API calls to FastAPI.
  base: command === "build" ? "/static/" : "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    proxy: {
      "/api": process.env.VITE_API_TARGET || "http://127.0.0.1:7860",
    },
  },
}));

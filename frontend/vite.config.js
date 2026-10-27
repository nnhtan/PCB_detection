import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.PCB_API_URL || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    watch: {
      usePolling: true,
      interval: 500,
    },
    proxy: {
      "/api": apiTarget,
      "/health": apiTarget,
      "/docs": apiTarget,
      "/redoc": apiTarget,
      "/openapi.json": apiTarget,
    },
  },
});

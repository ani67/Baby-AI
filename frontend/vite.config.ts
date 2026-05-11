import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ingest":          "http://127.0.0.1:8765",  // /ingest + /ingest_runtime
      "/idle":            "http://127.0.0.1:8765",
      "/sleep":           "http://127.0.0.1:8765",
      "/seed":            "http://127.0.0.1:8765",
      "/state":           "http://127.0.0.1:8765",
      "/graph":           "http://127.0.0.1:8765",
      "/save":            "http://127.0.0.1:8765",
      "/load":            "http://127.0.0.1:8765",
      "/runtime_status":  "http://127.0.0.1:8765",  // v1.1
      "/ws":              { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
});

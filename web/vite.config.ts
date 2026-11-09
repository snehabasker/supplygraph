import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` proxies the API to `supplygraph serve` (port 8000).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Cytoscape is most of the bundle; a separate chunk caches across app releases.
    rollupOptions: { output: { manualChunks: { cytoscape: ["cytoscape"] } } },
    chunkSizeWarningLimit: 600,
  },
});

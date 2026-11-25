import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const target = process.env.PREDICTX_API_TARGET || 'http://127.0.0.1:8000';
const proxy = {
  '/api': {
    target,
    rewrite: (path) => path.replace(/^\/api/, ''),
  },
  '/docs': { target },
  '/openapi.json': { target },
};

export default defineConfig({
  plugins: [react()],
  server: { proxy },
  preview: { proxy },
});

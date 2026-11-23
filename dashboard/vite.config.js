import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const proxy = {
  '/api': {
    target: process.env.PREDICTX_API_TARGET || 'http://127.0.0.1:8000',
    rewrite: (path) => path.replace(/^\/api/, ''),
  },
};

export default defineConfig({
  plugins: [react()],
  server: { proxy },
  preview: { proxy },
});

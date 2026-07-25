import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  // Inside compose the backend is reachable by service name; on a laptop it is
  // on localhost. Proxying keeps the browser same-origin, so no CORS in dev.
  const target = env.VITE_PROXY_TARGET || 'http://localhost:3080'

  return {
    plugins: [react()],
    server: {
      port: 3000,
      host: '0.0.0.0',
      proxy: {
        '/api': {
          target,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
        // Uploaded images are served by the backend, not from public/.
        '/media': {
          target,
          changeOrigin: true,
        },
      },
    },
  }
})

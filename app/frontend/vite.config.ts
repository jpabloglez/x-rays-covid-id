import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    /*
    hmr: {
      protocol: 'ws',
      port: 3000
    }
    */
    //port: 3000,
    //host: '127.0.0.1'
    /*
    proxy: {
      '/api': {
        target: 'http://localhost:3080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }*/
  }
})

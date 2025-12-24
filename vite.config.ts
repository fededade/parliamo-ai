import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Questo assicura che i percorsi siano relativi ./ invece di assoluti /
  base: './'
})
/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static exports for better deployment compatibility
  output: 'standalone',
  
  // Environment variables for API connection
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'
  },
  
  // Allow external images and resources
  images: {
    domains: ['localhost'],
    unoptimized: true
  }
}

module.exports = nextConfig 
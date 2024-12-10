// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false, // Disable strict mode temporarily
  webpack: (config) => {
    config.ignoreWarnings = [/Failed to parse source map/]
    return config
  }
}

export default nextConfig

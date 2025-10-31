import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  eslint: {
    // Warnings in content files (unused imports, any types) don't affect functionality
    // Disable during build, enable for development linting
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Type check during build (keep type safety)
    ignoreBuildErrors: false,
  },
  webpack: (config, { isServer }) => {
    // Fixes npm packages that depend on `fs` module
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }
    return config;
  },
};

export default nextConfig;

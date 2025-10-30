import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Link from 'next/link';
import ExportImportMenu from '@/components/ExportImportMenu';
import StorageInitializer from '@/components/StorageInitializer';
import Script from 'next/script';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'CodeBlanket',
  description:
    'Learn and practice binary search algorithms with interactive Python coding challenges',
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 5,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className} suppressHydrationWarning>
        <Script
          src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"
          strategy="lazyOnload"
        />

        <StorageInitializer />
        <nav className="sticky top-0 z-50 border-b border-gray-700 bg-[#282a36] text-white shadow-lg">
          <div className="container mx-auto px-4 py-3 sm:py-4">
            <div className="flex items-center justify-between">
              <Link
                href="/"
                className="flex items-center space-x-2 transition-opacity hover:opacity-90"
              >
                <div className="text-xl font-bold text-[#bd93f9] sm:text-2xl">
                  🛌 CodeBlanket
                </div>
              </Link>
              <ExportImportMenu />
            </div>
          </div>
        </nav>

        <main className="min-h-[calc(100vh-4rem)] overflow-y-auto bg-[#282a36] sm:min-h-[calc(100vh-4.5rem)]">
          {children}
        </main>
      </body>
    </html>
  );
}

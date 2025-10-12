import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Link from 'next/link';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'CodeBlanket - Master Binary Search Algorithms',
  description:
    'Learn and practice binary search algorithms with interactive Python coding challenges',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"
          async
        />
      </head>
      <body className={inter.className} suppressHydrationWarning>
        <nav className="border-b border-gray-700 bg-[#282a36] text-white shadow-lg">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <Link
                href="/"
                className="flex items-center space-x-2 transition-opacity hover:opacity-90"
              >
                <div className="text-2xl font-bold text-[#bd93f9]">
                  CodeBlanket
                </div>
              </Link>
              <div className="flex items-center space-x-6">
                <Link
                  href="/"
                  className="font-medium text-[#f8f8f2] transition-colors hover:text-[#bd93f9]"
                >
                  Topics
                </Link>
                <Link
                  href="/problems"
                  className="font-medium text-[#f8f8f2] transition-colors hover:text-[#bd93f9]"
                >
                  Problems
                </Link>
              </div>
            </div>
          </div>
        </nav>

        <main className="min-h-screen bg-[#282a36]">{children}</main>
      </body>
    </html>
  );
}

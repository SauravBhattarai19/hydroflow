import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Link from 'next/link';
import 'katex/dist/katex.min.css';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'OPM Hydrology Course',
  description: 'Interactive open-source hydrological modelling textbook',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-slate-50 text-slate-900 antialiased`}>
        {/* Top nav */}
        <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur-sm">
          <div className="mx-auto max-w-5xl px-6 h-14 flex items-center justify-between">
            <Link href="/" className="font-bold text-sky-700 tracking-tight text-lg">
              OPM · Hydrology Course
            </Link>
            <nav className="flex items-center gap-6 text-sm text-slate-600">
              <Link href="/" className="hover:text-sky-700 transition-colors">Chapters</Link>
              <a href="https://github.com/SauravBhattarai19/hydroflow" target="_blank" rel="noopener"
                className="hover:text-sky-700 transition-colors">GitHub</a>
            </nav>
          </div>
        </header>

        {/* Main */}
        <main className="mx-auto max-w-5xl px-6 py-10">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-200 mt-20 py-8 text-center text-sm text-slate-400">
          Open-source hydrology education · Built with Next.js + KaTeX + Tailwind
        </footer>
      </body>
    </html>
  );
}

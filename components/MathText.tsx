'use client';

import { useEffect, useState } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface MathTextProps {
  children: string;
  className?: string;
}

/**
 * Component that renders text with inline and display LaTeX math using KaTeX
 *
 * Supports:
 * - Inline math: $x^2 + y^2 = z^2$
 * - Display math: $$\frac{a}{b}$$
 * - Unicode symbols: ℤ, ℚ, ℝ, π, √, etc.
 */
export function MathText({ children, className = '' }: MathTextProps) {
  const [rendered, setRendered] = useState<string>('');

  useEffect(() => {
    try {
      let text = children;

      // First, handle display math ($$...$$)
      text = text.replace(/\$\$([\s\S]+?)\$\$/g, (match, math) => {
        try {
          const html = katex.renderToString(math.trim(), {
            displayMode: true,
            throwOnError: false,
            strict: false,
          });
          return `<div class="katex-display-block my-4">${html}</div>`;
        } catch (e) {
          console.error('KaTeX error (display):', e);
          return match;
        }
      });

      // Then, handle inline math ($...$)
      text = text.replace(/\$([^$\n]+?)\$/g, (match, math) => {
        try {
          const html = katex.renderToString(math.trim(), {
            displayMode: false,
            throwOnError: false,
            strict: false,
          });
          return `<span class="katex-inline">${html}</span>`;
        } catch (e) {
          console.error('KaTeX error (inline):', e);
          return match;
        }
      });

      // Handle bold text (**text**)
      text = text.replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>');

      // Handle italic text (*text* or _text_)
      text = text.replace(/\*([^*]+?)\*/g, '<em>$1</em>');
      text = text.replace(/_([^_]+?)_/g, '<em>$1</em>');

      // Handle inline code (`code`)
      text = text.replace(
        /`([^`]+?)`/g,
        '<code class="rounded bg-[#44475a] px-1.5 py-0.5 font-mono text-sm text-[#f8f8f2]">$1</code>',
      );

      setRendered(text);
    } catch (e) {
      console.error('Error rendering math:', e);
      setRendered(children);
    }
  }, [children]);

  if (!rendered) {
    return <span className={className}>{children}</span>;
  }

  return (
    <span
      className={className}
      dangerouslySetInnerHTML={{ __html: rendered }}
    />
  );
}

/**
 * Inline helper to render just inline math
 */
export function InlineMath({ children }: { children: string }) {
  const [html, setHtml] = useState('');

  useEffect(() => {
    try {
      const rendered = katex.renderToString(children, {
        displayMode: false,
        throwOnError: false,
        strict: false,
      });
      setHtml(rendered);
    } catch (e) {
      console.error('KaTeX inline error:', e);
      setHtml(children);
    }
  }, [children]);

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

/**
 * Display helper to render display math
 */
export function DisplayMath({ children }: { children: string }) {
  const [html, setHtml] = useState('');

  useEffect(() => {
    try {
      const rendered = katex.renderToString(children, {
        displayMode: true,
        throwOnError: false,
        strict: false,
      });
      setHtml(rendered);
    } catch (e) {
      console.error('KaTeX display error:', e);
      setHtml(children);
    }
  }, [children]);

  return <div className="my-4" dangerouslySetInnerHTML={{ __html: html }} />;
}

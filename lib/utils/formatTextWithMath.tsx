/**
 * Enhanced text formatting utilities that support mathematical notation via KaTeX
 */

import { ReactNode } from 'react';
import { ClientOnlySyntaxHighlighter } from '@/components/ClientOnlySyntaxHighlighter';
import { MathText } from '@/components/MathText';

/**
 * Formats text with code blocks, inline code, bold, and LaTeX math
 *
 * Supports:
 * - Code blocks: ```language\ncode```
 * - Inline code: `code`
 * - Bold: **text**
 * - Inline math: $x^2$
 * - Display math: $$\frac{a}{b}$$
 * - Unicode symbols: ℤ, ℚ, ℝ, π, √, etc.
 *
 * @param text - The text to format
 * @returns Array of React nodes with formatted text and rendered math
 */
export function formatTextWithMath(text: string): ReactNode[] {
  // First check if this contains a code block (triple backticks)
  const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
  const hasCodeBlock = codeBlockRegex.test(text);

  if (hasCodeBlock) {
    // Reset regex
    codeBlockRegex.lastIndex = 0;
    const parts: ReactNode[] = [];
    let lastIndex = 0;
    let match;
    let keyIndex = 0;

    while ((match = codeBlockRegex.exec(text)) !== null) {
      // Add text before the code block (with math support)
      if (match.index > lastIndex) {
        const beforeText = text.slice(lastIndex, match.index);
        parts.push(
          <MathText key={`math-text-${keyIndex}`}>{beforeText}</MathText>,
        );
        keyIndex++;
      }

      // Add the code block
      const language = match[1] || 'python';
      const code = match[2].trim();
      parts.push(
        <div key={`code-${keyIndex}`} className="my-4">
          <ClientOnlySyntaxHighlighter language={language} code={code} />
        </div>,
      );
      keyIndex++;

      lastIndex = match.index + match[0].length;
    }

    // Add remaining text after last code block (with math support)
    if (lastIndex < text.length) {
      const afterText = text.slice(lastIndex);
      parts.push(
        <MathText key={`math-text-${keyIndex}`}>{afterText}</MathText>,
      );
    }

    return parts;
  }

  // No code blocks, return the whole text with math support
  return [<MathText key="math-text-0">{text}</MathText>];
}

/**
 * Check if text contains LaTeX math notation
 */
export function containsMath(text: string): boolean {
  return /\$\$[\s\S]+?\$\$|\$[^$\n]+?\$/.test(text);
}

/**
 * Format a single line with inline code, bold, and inline math
 * This is useful for headers and single-line text
 */
export function formatLineWithMath(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  let keyIndex = 0;

  // Check if contains inline code or backticks
  const inlineCodeRegex = /`([^`]+)`/g;
  const hasMath = containsMath(text);
  const hasInlineCode = inlineCodeRegex.test(text);
  inlineCodeRegex.lastIndex = 0;

  if (hasInlineCode && hasMath) {
    // Handle both inline code and math - need to be careful about order
    let lastIndex = 0;
    let match;

    while ((match = inlineCodeRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        const beforeText = text.slice(lastIndex, match.index);
        parts.push(
          <MathText key={`math-inline-${keyIndex}`}>{beforeText}</MathText>,
        );
        keyIndex++;
      }

      parts.push(
        <code
          key={`code-inline-${keyIndex}`}
          className="rounded bg-[#44475a] px-1.5 py-0.5 font-mono text-sm text-[#f8f8f2]"
        >
          {match[1]}
        </code>,
      );
      keyIndex++;

      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      const afterText = text.slice(lastIndex);
      parts.push(
        <MathText key={`math-inline-${keyIndex}`}>{afterText}</MathText>,
      );
    }

    return parts;
  } else if (hasMath) {
    // Just math, no inline code
    return [<MathText key="math-inline-0">{text}</MathText>];
  } else if (hasInlineCode) {
    // Just inline code, no math
    let lastIndex = 0;
    let match;

    while ((match = inlineCodeRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index));
      }

      parts.push(
        <code
          key={`code-inline-${keyIndex}`}
          className="rounded bg-[#44475a] px-1.5 py-0.5 font-mono text-sm text-[#f8f8f2]"
        >
          {match[1]}
        </code>,
      );
      keyIndex++;

      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }

    return parts;
  }

  // No math or inline code, return as is
  return [text];
}

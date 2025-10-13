/**
 * Text formatting utilities for rendering markdown-like text with inline code and bold.
 */

import { ReactNode } from 'react';
import { ClientOnlySyntaxHighlighter } from '@/components/ClientOnlySyntaxHighlighter';

/**
 * Formats text with code blocks, inline code (backticks) and bold (double asterisks) into React nodes
 * @param text - The text to format
 * @returns Array of React nodes with formatted text
 * @example
 * formatText("Use the `binary_search` function to find **target** value")
 * // Returns: ["Use the ", <code>binary_search</code>, " function to find ", <strong>target</strong>, " value"]
 */
export function formatText(text: string): ReactNode[] {
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
      // Add text before the code block
      if (match.index > lastIndex) {
        const beforeText = text.slice(lastIndex, match.index);
        parts.push(...formatTextWithoutCodeBlocks(beforeText, keyIndex));
        keyIndex += beforeText.length;
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

    // Add remaining text after last code block
    if (lastIndex < text.length) {
      const afterText = text.slice(lastIndex);
      parts.push(...formatTextWithoutCodeBlocks(afterText, keyIndex));
    }

    return parts;
  }

  // No code blocks, just handle inline formatting
  return formatTextWithoutCodeBlocks(text, 0);
}

/**
 * Formats text with inline code and bold (no code blocks)
 */
function formatTextWithoutCodeBlocks(
  text: string,
  startKey: number,
): ReactNode[] {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);

  return parts.map((part, index) => {
    // Handle inline code (single backticks only, not triple)
    if (part.startsWith('`') && part.endsWith('`') && !part.startsWith('```')) {
      const code = part.slice(1, -1);
      return (
        <code
          key={`${startKey}-${index}`}
          className="rounded bg-[#6272a4] px-1.5 py-0.5 font-mono text-sm text-[#8be9fd]"
        >
          {code}
        </code>
      );
    }

    // Handle bold text
    if (part.startsWith('**') && part.endsWith('**')) {
      const boldText = part.slice(2, -2);
      return (
        <span
          key={`${startKey}-${index}`}
          className="font-semibold text-[#f8f8f2]"
        >
          {boldText}
        </span>
      );
    }

    return <span key={`${startKey}-${index}`}>{part}</span>;
  });
}

/**
 * Formats the first line of description text for problem cards
 * @param text - The full description text
 * @returns Formatted React nodes for the first line only
 */
export function formatDescription(text: string): ReactNode[] {
  const firstLine = text.split('\n')[0];
  return formatText(firstLine);
}

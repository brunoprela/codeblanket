/**
 * Text formatting utilities for rendering markdown-like text with inline code and bold.
 */

import { ReactNode } from 'react';

/**
 * Formats text with inline code (backticks) and bold (double asterisks) into React nodes
 * @param text - The text to format
 * @returns Array of React nodes with formatted text
 * @example
 * formatText("Use the `binary_search` function to find **target** value")
 * // Returns: ["Use the ", <code>binary_search</code>, " function to find ", <strong>target</strong>, " value"]
 */
export function formatText(text: string): ReactNode[] {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);

  return parts.map((part, index) => {
    // Handle inline code
    if (part.startsWith('`') && part.endsWith('`')) {
      const code = part.slice(1, -1);
      return (
        <code
          key={index}
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
        <span key={index} className="text-[#f8f8f2]">
          {boldText}
        </span>
      );
    }

    return <span key={index}>{part}</span>;
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

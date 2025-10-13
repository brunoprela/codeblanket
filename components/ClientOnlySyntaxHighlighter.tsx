'use client';

import { useState, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';

// Dracula theme
const draculaTheme = {
  'code[class*="language-"]': {
    color: '#f8f8f2',
    background: '#282a36',
    textShadow: 'none',
    fontFamily: 'ui-monospace, monospace',
    fontSize: '0.875rem',
    lineHeight: '1.5',
  },
  'pre[class*="language-"]': {
    color: '#f8f8f2',
    background: '#282a36',
    textShadow: 'none',
    fontFamily: 'ui-monospace, monospace',
    fontSize: '0.875rem',
    lineHeight: '1.5',
    padding: '1rem',
    margin: '0',
    overflow: 'auto',
    borderRadius: '0.5rem',
  },
  comment: { color: '#6272a4', fontStyle: 'italic' },
  keyword: { color: '#ff79c6' },
  operator: { color: '#ff79c6' },
  string: { color: '#f1fa8c' },
  number: { color: '#bd93f9' },
  boolean: { color: '#bd93f9' },
  function: { color: '#50fa7b' },
  'class-name': { color: '#8be9fd' },
  parameter: { color: '#ffb86c' },
  variable: { color: '#f8f8f2' },
  punctuation: { color: '#f8f8f2' },
};

interface ClientOnlySyntaxHighlighterProps {
  language: string;
  code: string;
  customStyle?: React.CSSProperties;
}

export function ClientOnlySyntaxHighlighter({
  language,
  code,
  customStyle,
}: ClientOnlySyntaxHighlighterProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    // Return a placeholder with the same styling during SSR
    return (
      <pre
        style={{
          margin: 0,
          borderRadius: '0.5rem',
          background: '#282a36',
          color: '#f8f8f2',
          padding: '1rem',
          overflow: 'auto',
          fontFamily: 'ui-monospace, monospace',
          fontSize: '0.875rem',
          lineHeight: '1.5',
          ...customStyle,
        }}
      >
        <code>{code}</code>
      </pre>
    );
  }

  return (
    <SyntaxHighlighter
      language={language}
      style={draculaTheme}
      customStyle={{
        margin: 0,
        borderRadius: '0.5rem',
        background: '#282a36',
        ...customStyle,
      }}
    >
      {code}
    </SyntaxHighlighter>
  );
}

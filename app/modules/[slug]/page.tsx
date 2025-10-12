'use client';

import { notFound } from 'next/navigation';
import Link from 'next/link';
import { use, ReactElement } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';

import { getModuleById } from '@/lib/modules';
import { getProblemById } from '@/lib/problems';
import { formatText } from '@/lib/utils/formatText';

// Custom Dracula theme matching our app's color scheme
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
  },
  comment: { color: '#6272a4', fontStyle: 'italic' },
  prolog: { color: '#6272a4' },
  doctype: { color: '#6272a4' },
  cdata: { color: '#6272a4' },
  punctuation: { color: '#f8f8f2' },
  property: { color: '#50fa7b' },
  tag: { color: '#ff79c6' },
  constant: { color: '#bd93f9' },
  symbol: { color: '#bd93f9' },
  deleted: { color: '#ff5555' },
  boolean: { color: '#bd93f9' },
  number: { color: '#bd93f9' },
  selector: { color: '#50fa7b' },
  'attr-name': { color: '#50fa7b' },
  string: { color: '#f1fa8c' },
  char: { color: '#f1fa8c' },
  builtin: { color: '#8be9fd' },
  inserted: { color: '#50fa7b' },
  operator: { color: '#ff79c6' },
  entity: { color: '#f8f8f2' },
  url: { color: '#f1fa8c' },
  variable: { color: '#f8f8f2' },
  atrule: { color: '#ff79c6' },
  'attr-value': { color: '#f1fa8c' },
  function: { color: '#50fa7b' },
  'class-name': { color: '#8be9fd' },
  keyword: { color: '#ff79c6' },
  regex: { color: '#f1fa8c' },
  important: { color: '#ffb86c', fontWeight: 'bold' },
  bold: { fontWeight: 'bold' },
  italic: { fontStyle: 'italic' },
  namespace: { opacity: 0.7 },
};

export default function ModulePage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = use(params);
  const moduleData = getModuleById(slug);

  if (!moduleData) {
    notFound();
  }

  return (
    <div className="container mx-auto max-w-4xl px-4 py-12">
      {/* Back Link */}
      <Link
        href="/"
        className="mb-6 inline-flex items-center text-sm font-medium text-[#6272a4] transition-colors hover:text-[#bd93f9]"
      >
        <svg
          className="mr-1.5 h-4 w-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 19l-7-7 7-7"
          />
        </svg>
        Back to Topics
      </Link>

      {/* Module Header */}
      <div className="mb-12">
        <div className="mb-4 flex items-center gap-4">
          <span className="text-6xl">{moduleData.icon}</span>
          <div>
            <h1 className="text-4xl font-bold text-[#f8f8f2]">
              {moduleData.title}
            </h1>
            <p className="mt-2 text-lg text-[#6272a4]">
              {moduleData.description}
            </p>
          </div>
        </div>

        {/* Complexity Badges */}
        {(moduleData.timeComplexity || moduleData.spaceComplexity) && (
          <div className="flex gap-3">
            {moduleData.timeComplexity && (
              <div className="rounded-lg border-2 border-[#bd93f9] bg-[#bd93f9]/10 px-4 py-2">
                <span className="text-sm font-semibold text-[#bd93f9]">
                  Time: {moduleData.timeComplexity}
                </span>
              </div>
            )}
            {moduleData.spaceComplexity && (
              <div className="rounded-lg border-2 border-[#50fa7b] bg-[#50fa7b]/10 px-4 py-2">
                <span className="text-sm font-semibold text-[#50fa7b]">
                  Space: {moduleData.spaceComplexity}
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Module Sections */}
      <div className="space-y-12">
        {moduleData.sections.map((section, index) => (
          <section key={section.id} className="scroll-mt-8" id={section.id}>
            {/* Section Number and Title */}
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#bd93f9] font-bold text-[#282a36]">
                {index + 1}
              </div>
              <h2 className="text-2xl font-bold text-[#f8f8f2]">
                {section.title}
              </h2>
            </div>

            {/* Section Content */}
            <div className="rounded-lg border-2 border-[#44475a] bg-[#44475a] p-8">
              <div className="prose prose-invert max-w-none">
                {(() => {
                  // Parse content to handle code blocks properly
                  const elements: ReactElement[] = [];
                  const codeBlockRegex = /```[\s\S]*?```/g;
                  let lastIndex = 0;
                  let elementKey = 0;

                  // Find all code blocks
                  const matches = Array.from(
                    section.content.matchAll(codeBlockRegex),
                  );

                  matches.forEach((match) => {
                    // Add text before code block
                    if (match.index! > lastIndex) {
                      const textBefore = section.content
                        .slice(lastIndex, match.index)
                        .trim();
                      if (textBefore) {
                        // Split by double newlines for paragraphs
                        textBefore.split('\n\n').forEach((para) => {
                          const trimmed = para.trim();
                          if (trimmed) {
                            elements.push(
                              <p
                                key={`text-${elementKey++}`}
                                className="mb-4 leading-relaxed text-[#f8f8f2]"
                              >
                                {formatText(trimmed)}
                              </p>,
                            );
                          }
                        });
                      }
                    }

                    // Add code block
                    const codeBlock = match[0];
                    const lines = codeBlock.split('\n');
                    // Extract language from first line (e.g., ```python)
                    const language =
                      lines[0].replace(/```/g, '').trim() || 'text';
                    const code = lines.slice(1, -1).join('\n');
                    elements.push(
                      <div
                        key={`code-${elementKey++}`}
                        className="my-4 overflow-x-auto rounded-lg"
                      >
                        <SyntaxHighlighter
                          language={language}
                          style={draculaTheme}
                          customStyle={{
                            margin: 0,
                            borderRadius: '0.5rem',
                            background: '#282a36',
                          }}
                        >
                          {code}
                        </SyntaxHighlighter>
                      </div>,
                    );

                    lastIndex = match.index! + match[0].length;
                  });

                  // Add remaining text after last code block
                  if (lastIndex < section.content.length) {
                    const textAfter = section.content.slice(lastIndex).trim();
                    if (textAfter) {
                      textAfter.split('\n\n').forEach((para) => {
                        const trimmed = para.trim();
                        if (trimmed) {
                          elements.push(
                            <p
                              key={`text-${elementKey++}`}
                              className="mb-4 leading-relaxed text-[#f8f8f2]"
                            >
                              {formatText(trimmed)}
                            </p>,
                          );
                        }
                      });
                    }
                  }

                  return elements;
                })()}
              </div>

              {/* Code Example */}
              {section.codeExample && (
                <div className="mt-6">
                  <div className="mb-2 font-semibold text-[#bd93f9]">
                    Code Example:
                  </div>
                  <div className="overflow-x-auto rounded-lg">
                    <SyntaxHighlighter
                      language="python"
                      style={draculaTheme}
                      customStyle={{
                        margin: 0,
                        borderRadius: '0.5rem',
                        background: '#282a36',
                      }}
                    >
                      {section.codeExample}
                    </SyntaxHighlighter>
                  </div>
                </div>
              )}
            </div>
          </section>
        ))}
      </div>

      {/* Key Takeaways */}
      <div className="mt-12 rounded-lg border-2 border-[#50fa7b] bg-[#50fa7b]/10 p-8">
        <h2 className="mb-4 text-2xl font-bold text-[#50fa7b]">
          🎯 Key Takeaways
        </h2>
        <ul className="space-y-3">
          {moduleData.keyTakeaways.map((takeaway, index) => (
            <li key={index} className="flex items-start gap-3">
              <span className="mt-1 text-[#50fa7b]">✓</span>
              <span className="text-[#f8f8f2]">{takeaway}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Related Problems */}
      {moduleData.relatedProblems.length > 0 && (
        <div className="mt-12">
          <h2 className="mb-6 text-2xl font-bold text-[#f8f8f2]">
            📝 Practice Problems
          </h2>
          <p className="mb-6 text-[#6272a4]">
            Now that you&apos;ve learned the concepts, try solving these
            problems to reinforce your understanding:
          </p>
          <div className="space-y-4">
            {moduleData.relatedProblems.map((problemId) => {
              const problem = getProblemById(problemId);
              if (!problem) return null;

              const difficultyColors = {
                Easy: 'bg-[#50fa7b] text-[#282a36] border-[#50fa7b]',
                Medium: 'bg-[#f1fa8c] text-[#282a36] border-[#f1fa8c]',
                Hard: 'bg-[#ff5555] text-[#282a36] border-[#ff5555]',
              };

              return (
                <Link
                  key={problemId}
                  href={`/problems/${problemId}`}
                  className="block"
                >
                  <div className="group cursor-pointer rounded-lg border-2 border-[#44475a] bg-[#44475a] p-6 transition-all hover:border-[#bd93f9] hover:shadow-xl">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="mb-2 flex items-center gap-2">
                          <span
                            className={`rounded-full border px-3 py-1 text-xs font-bold ${difficultyColors[problem.difficulty]}`}
                          >
                            {problem.difficulty}
                          </span>
                        </div>
                        <h3 className="text-lg font-semibold text-[#f8f8f2] transition-colors group-hover:text-[#bd93f9]">
                          {problem.title}
                        </h3>
                      </div>
                      <div className="ml-4 text-[#bd93f9] transition-colors group-hover:text-[#ff79c6]">
                        <svg
                          className="h-6 w-6"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 5l7 7-7 7"
                          />
                        </svg>
                      </div>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      )}

      {/* Bottom Navigation */}
      <div className="mt-12 flex justify-between border-t-2 border-[#44475a] pt-8">
        <Link
          href="/"
          className="rounded-lg bg-[#6272a4] px-6 py-3 font-semibold text-[#f8f8f2] transition-colors hover:bg-[#6272a4]/80"
        >
          ← All Topics
        </Link>
        <Link
          href={`/topics/${moduleData.id}`}
          className="rounded-lg bg-[#bd93f9] px-6 py-3 font-semibold text-[#282a36] transition-colors hover:bg-[#ff79c6]"
        >
          View All {moduleData.title} Problems →
        </Link>
      </div>
    </div>
  );
}

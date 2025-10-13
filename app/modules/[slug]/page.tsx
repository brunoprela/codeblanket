'use client';

import { notFound } from 'next/navigation';
import Link from 'next/link';
import { use, ReactElement, useState, useEffect } from 'react';

import { getModuleById } from '@/lib/modules';
import { formatText } from '@/lib/utils/formatText';
import { InteractiveCodeBlock } from '@/components/InteractiveCodeBlock';
import { ClientOnlySyntaxHighlighter } from '@/components/ClientOnlySyntaxHighlighter';

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

  // State for expanded/collapsed sections (all expanded by default)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(moduleData.sections.map((s) => s.id)),
  );

  // State for completed sections
  const [completedSections, setCompletedSections] = useState<Set<string>>(
    new Set(),
  );

  // State for showing quiz solutions
  const [showSolutions, setShowSolutions] = useState<Set<string>>(new Set());

  // Load completed sections from localStorage
  useEffect(() => {
    const storageKey = `module-${slug}-completed`;
    const stored = localStorage.getItem(storageKey);
    if (stored) {
      try {
        const completed = JSON.parse(stored);
        setCompletedSections(new Set(completed));
      } catch (e) {
        console.error('Failed to load completed sections:', e);
      }
    }
  }, [slug]);

  // Collapse completed sections after loading
  useEffect(() => {
    if (completedSections.size > 0) {
      setExpandedSections((prev) => {
        const newSet = new Set(prev);
        completedSections.forEach((sectionId) => {
          newSet.delete(sectionId);
        });
        return newSet;
      });
    }
  }, [completedSections]);

  // Save completed sections to localStorage
  const toggleSectionComplete = (sectionId: string) => {
    setCompletedSections((prev) => {
      const newSet = new Set(prev);
      const wasCompleted = newSet.has(sectionId);

      if (wasCompleted) {
        newSet.delete(sectionId);
        // When marking as incomplete, expand the section
        setExpandedSections((expanded) => {
          const newExpanded = new Set(expanded);
          newExpanded.add(sectionId);
          return newExpanded;
        });
      } else {
        newSet.add(sectionId);
        // When marking as complete, collapse the section
        setExpandedSections((expanded) => {
          const newExpanded = new Set(expanded);
          newExpanded.delete(sectionId);
          return newExpanded;
        });
      }

      // Save to localStorage
      const storageKey = `module-${slug}-completed`;
      localStorage.setItem(storageKey, JSON.stringify(Array.from(newSet)));

      return newSet;
    });
  };

  const toggleSection = (sectionId: string) => {
    setExpandedSections((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(sectionId)) {
        newSet.delete(sectionId);
      } else {
        newSet.add(sectionId);
      }
      return newSet;
    });
  };

  const toggleSolution = (sectionId: string, questionId: string) => {
    const key = `${sectionId}-${questionId}`;
    setShowSolutions((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(key)) {
        newSet.delete(key);
      } else {
        newSet.add(key);
      }
      return newSet;
    });
  };

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

        {/* Progress Bar */}
        {moduleData.sections.length > 0 && (
          <div className="mb-4 rounded-lg border-2 border-[#6272a4] bg-[#6272a4]/10 p-4">
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="font-semibold text-[#f8f8f2]">
                Progress: {completedSections.size} /{' '}
                {moduleData.sections.length} sections completed
              </span>
              <span className="text-[#6272a4]">
                {Math.round(
                  (completedSections.size / moduleData.sections.length) * 100,
                )}
                %
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-[#44475a]">
              <div
                className="h-full rounded-full bg-gradient-to-r from-[#bd93f9] to-[#50fa7b] transition-all duration-300"
                style={{
                  width: `${(completedSections.size / moduleData.sections.length) * 100}%`,
                }}
              />
            </div>
          </div>
        )}

        {/* Complexity Badges and Practice Link */}
        <div className="flex flex-wrap items-center gap-3">
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
          <Link
            href={`/topics/${moduleData.id}?from=modules/${slug}`}
            className="ml-auto rounded-lg bg-[#bd93f9] px-6 py-2 font-semibold text-[#282a36] transition-colors hover:bg-[#ff79c6]"
          >
            📝 Practice Problems →
          </Link>
        </div>
      </div>

      {/* Module Sections */}
      <div className="space-y-6">
        {moduleData.sections.map((section, index) => {
          const isExpanded = expandedSections.has(section.id);
          const isCompleted = completedSections.has(section.id);

          return (
            <section key={section.id} className="scroll-mt-8" id={section.id}>
              {/* Section Header */}
              <div className="rounded-lg border-2 border-[#44475a] bg-[#44475a]">
                <div
                  className="flex cursor-pointer items-center gap-3 p-6"
                  onClick={() => toggleSection(section.id)}
                >
                  {/* Section Number */}
                  <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-[#bd93f9] font-bold text-[#282a36]">
                    {index + 1}
                  </div>

                  {/* Section Title */}
                  <h2 className="flex-1 text-2xl font-bold text-[#f8f8f2]">
                    {section.title}
                  </h2>

                  {/* Completed Checkbox */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleSectionComplete(section.id);
                    }}
                    className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded border-2 transition-colors ${isCompleted
                      ? 'border-[#50fa7b] bg-[#50fa7b] text-[#282a36]'
                      : 'border-[#6272a4] bg-transparent text-transparent hover:border-[#50fa7b]'
                      }`}
                    title={
                      isCompleted ? 'Mark as incomplete' : 'Mark as completed'
                    }
                  >
                    {isCompleted && (
                      <svg
                        className="h-5 w-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={3}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    )}
                  </button>

                  {/* Expand/Collapse Arrow */}
                  <div
                    className={`flex-shrink-0 text-[#bd93f9] transition-transform ${isExpanded ? 'rotate-180' : ''
                      }`}
                  >
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
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </div>
                </div>

                {/* Section Content */}
                {isExpanded && (
                  <div className="border-t-2 border-[#282a36] p-8">
                    <div className="prose prose-invert max-w-none">
                      {(() => {
                        // Parse content to handle code blocks and tables
                        const elements: ReactElement[] = [];
                        const codeBlockRegex = /```[\s\S]*?```/g;
                        const tableRegex =
                          /\|(.+)\|\n\|[-:\s|]+\|\n((?:\|.+\|\n?)+)/g;
                        let elementKey = 0;

                        // Function to render paragraphs and lists
                        const renderParagraphsAndLists = (text: string) => {
                          const lines = text.split('\n');
                          let i = 0;

                          while (i < lines.length) {
                            const line = lines[i].trim();

                            // Check for numbered list (1. 2. 3. etc)
                            if (/^\d+\.\s/.test(line)) {
                              const listItems: string[] = [];
                              while (
                                i < lines.length &&
                                /^\d+\.\s/.test(lines[i].trim())
                              ) {
                                listItems.push(
                                  lines[i].trim().replace(/^\d+\.\s/, ''),
                                );
                                i++;
                              }
                              elements.push(
                                <ol
                                  key={`ol-${elementKey++}`}
                                  className="mb-4 ml-6 list-decimal space-y-2 text-[#f8f8f2]"
                                >
                                  {listItems.map((item, idx) => (
                                    <li key={idx}>{formatText(item)}</li>
                                  ))}
                                </ol>,
                              );
                            }
                            // Check for bullet list (- or *)
                            else if (/^[-*]\s/.test(line)) {
                              const listItems: string[] = [];
                              while (
                                i < lines.length &&
                                /^[-*]\s/.test(lines[i].trim())
                              ) {
                                listItems.push(
                                  lines[i].trim().replace(/^[-*]\s/, ''),
                                );
                                i++;
                              }
                              elements.push(
                                <ul
                                  key={`ul-${elementKey++}`}
                                  className="mb-4 ml-6 list-disc space-y-2 text-[#f8f8f2]"
                                >
                                  {listItems.map((item, idx) => (
                                    <li key={idx}>{formatText(item)}</li>
                                  ))}
                                </ul>,
                              );
                            }
                            // Regular paragraph
                            else if (line) {
                              // Collect all lines until next list or empty line
                              let para = line;
                              i++;
                              while (
                                i < lines.length &&
                                lines[i].trim() &&
                                !/^(\d+\.|-|\*)\s/.test(lines[i].trim())
                              ) {
                                para += ' ' + lines[i].trim();
                                i++;
                              }
                              elements.push(
                                <p
                                  key={`text-${elementKey++}`}
                                  className="mb-4 leading-relaxed text-[#f8f8f2]"
                                >
                                  {formatText(para)}
                                </p>,
                              );
                            } else {
                              i++;
                            }
                          }
                        };

                        // Function to render text content
                        const renderText = (text: string) => {
                          if (!text.trim()) return;

                          // Check if text contains a table
                          const tableMatches = Array.from(
                            text.matchAll(tableRegex),
                          );

                          if (tableMatches.length > 0) {
                            let textIndex = 0;

                            tableMatches.forEach((tableMatch) => {
                              // Add text before table
                              if (tableMatch.index! > textIndex) {
                                const textBefore = text
                                  .slice(textIndex, tableMatch.index)
                                  .trim();
                                if (textBefore) {
                                  renderParagraphsAndLists(textBefore);
                                }
                              }

                              // Parse and render table
                              const headerLine = tableMatch[1];
                              const bodyLines = tableMatch[2]
                                .trim()
                                .split('\n');

                              const headers = headerLine
                                .split('|')
                                .map((h) => h.trim())
                                .filter((h) => h);
                              const rows = bodyLines.map((line) =>
                                line
                                  .split('|')
                                  .map((cell) => cell.trim())
                                  .filter((cell) => cell),
                              );

                              elements.push(
                                <div
                                  key={`table-${elementKey++}`}
                                  className="my-6 overflow-x-auto"
                                >
                                  <table className="min-w-full border-collapse rounded-lg">
                                    <thead>
                                      <tr className="bg-[#6272a4]/20">
                                        {headers.map((header, i) => (
                                          <th
                                            key={i}
                                            className="border border-[#6272a4] px-4 py-2 text-left font-semibold text-[#bd93f9]"
                                          >
                                            {header}
                                          </th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {rows.map((row, rowIdx) => (
                                        <tr
                                          key={rowIdx}
                                          className="border-b border-[#6272a4] hover:bg-[#6272a4]/10"
                                        >
                                          {row.map((cell, cellIdx) => (
                                            <td
                                              key={cellIdx}
                                              className="border border-[#6272a4] px-4 py-2 text-[#f8f8f2]"
                                            >
                                              {formatText(cell)}
                                            </td>
                                          ))}
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>,
                              );

                              textIndex =
                                tableMatch.index! + tableMatch[0].length;
                            });

                            // Add remaining text after last table
                            if (textIndex < text.length) {
                              const textAfter = text.slice(textIndex).trim();
                              if (textAfter) {
                                renderParagraphsAndLists(textAfter);
                              }
                            }
                          } else {
                            // No tables, just render paragraphs and lists
                            renderParagraphsAndLists(text);
                          }
                        };

                        // Find all code blocks
                        const codeMatches = Array.from(
                          section.content.matchAll(codeBlockRegex),
                        );

                        let lastIndex = 0;

                        codeMatches.forEach((match) => {
                          // Add text/tables before code block
                          if (match.index! > lastIndex) {
                            const textBefore = section.content
                              .slice(lastIndex, match.index)
                              .trim();
                            renderText(textBefore);
                          }

                          // Add code block
                          const codeBlock = match[0];
                          const lines = codeBlock.split('\n');
                          // Extract language from first line (e.g., ```python)
                          const language =
                            lines[0].replace(/```/g, '').trim() || 'text';
                          const code = lines.slice(1, -1).join('\n');
                          elements.push(
                            <InteractiveCodeBlock
                              key={`code-${elementKey++}`}
                              code={code}
                              language={language}
                            />,
                          );

                          lastIndex = match.index! + match[0].length;
                        });

                        // Add remaining text/tables after last code block
                        if (lastIndex < section.content.length) {
                          const textAfter = section.content
                            .slice(lastIndex)
                            .trim();
                          renderText(textAfter);
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
                        <InteractiveCodeBlock
                          code={section.codeExample}
                          language="python"
                        />
                      </div>
                    )}

                    {/* Quiz Section */}
                    {section.quiz && section.quiz.length > 0 && (
                      <div className="mt-8 rounded-lg border-2 border-[#f1fa8c] bg-[#f1fa8c]/5 p-6">
                        <h3 className="mb-4 flex items-center gap-2 text-xl font-bold text-[#f1fa8c]">
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
                              d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                          </svg>
                          Discussion Questions
                        </h3>
                        <p className="mb-6 text-sm text-[#6272a4]">
                          Think through these questions or discuss them on
                          camera. Click to reveal sample answers.
                        </p>

                        <div className="space-y-5">
                          {section.quiz.map((question, qIndex) => {
                            const questionKey = `${section.id}-${question.id}`;
                            const showSolution =
                              showSolutions.has(questionKey);

                            return (
                              <div
                                key={question.id}
                                className="rounded-lg border-2 border-[#44475a] bg-[#282a36] p-5"
                              >
                                <div className="mb-4 text-lg font-semibold leading-relaxed text-[#f8f8f2]">
                                  <span className="mr-2 text-[#bd93f9]">
                                    {qIndex + 1}.
                                  </span>
                                  {question.question}
                                </div>

                                {question.hint && !showSolution && (
                                  <div className="mb-4 rounded-lg border border-[#6272a4] bg-[#6272a4]/10 p-3 text-sm text-[#8be9fd]">
                                    <span className="font-semibold">
                                      💡 Hint:{' '}
                                    </span>
                                    {question.hint}
                                  </div>
                                )}

                                <button
                                  onClick={() =>
                                    toggleSolution(section.id, question.id)
                                  }
                                  className="rounded-lg bg-[#bd93f9] px-4 py-2 text-sm font-semibold text-[#282a36] transition-colors hover:bg-[#ff79c6]"
                                >
                                  {showSolution
                                    ? '🔒 Hide Sample Answer'
                                    : '🔓 Reveal Sample Answer'}
                                </button>

                                {/* Sample Answer & Key Points */}
                                {showSolution && (
                                  <div className="mt-4 space-y-4">
                                    {/* Sample Answer */}
                                    <div className="rounded-lg border-2 border-[#50fa7b] bg-[#50fa7b]/10 p-4">
                                      <div className="mb-2 flex items-center gap-2 font-semibold text-[#50fa7b]">
                                        <svg
                                          className="h-5 w-5"
                                          fill="none"
                                          stroke="currentColor"
                                          viewBox="0 0 24 24"
                                        >
                                          <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                                          />
                                        </svg>
                                        Sample Answer
                                      </div>
                                      <p className="text-sm leading-relaxed text-[#f8f8f2]">
                                        {question.sampleAnswer}
                                      </p>
                                    </div>

                                    {/* Key Points */}
                                    {question.keyPoints.length > 0 && (
                                      <div className="rounded-lg border-2 border-[#8be9fd] bg-[#8be9fd]/10 p-4">
                                        <div className="mb-2 font-semibold text-[#8be9fd]">
                                          ✓ Key Points to Cover:
                                        </div>
                                        <ul className="space-y-1.5 text-sm text-[#f8f8f2]">
                                          {question.keyPoints.map(
                                            (point, index) => (
                                              <li
                                                key={index}
                                                className="flex items-start gap-2"
                                              >
                                                <span className="mt-1 text-[#8be9fd]">
                                                  •
                                                </span>
                                                <span>{point}</span>
                                              </li>
                                            ),
                                          )}
                                        </ul>
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </section>
          );
        })}
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

      {/* Bottom Navigation */}
      <div className="mt-12 flex justify-center gap-4 border-t-2 border-[#44475a] pt-8">
        <Link
          href="/"
          className="rounded-lg bg-[#6272a4] px-6 py-3 font-semibold text-[#f8f8f2] transition-colors hover:bg-[#6272a4]/80"
        >
          ← All Topics
        </Link>
        <Link
          href={`/topics/${moduleData.id}?from=modules/${slug}`}
          className="rounded-lg bg-[#bd93f9] px-6 py-3 font-semibold text-[#282a36] transition-colors hover:bg-[#ff79c6]"
        >
          Practice {moduleData.title} Problems →
        </Link>
      </div>
    </div>
  );
}

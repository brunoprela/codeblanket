'use client';

import { notFound } from 'next/navigation';
import Link from 'next/link';
import { getProblemById } from '@/lib/problems';
import { PythonCodeRunner } from '@/components/PythonCodeRunner';
import { useState, useEffect, use } from 'react';
import { isProblemCompleted } from '@/lib/storage';

export default function ProblemPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const problem = getProblemById(id);
  const [isCompleted, setIsCompleted] = useState(false);

  useEffect(() => {
    if (problem) {
      setIsCompleted(isProblemCompleted(problem.id));
    }

    // Listen for problem reset events
    const handleReset = (event: CustomEvent) => {
      if (event.detail?.problemId === problem?.id) {
        setIsCompleted(false);
      }
    };

    window.addEventListener('problemReset', handleReset as EventListener);

    return () => {
      window.removeEventListener('problemReset', handleReset as EventListener);
    };
  }, [problem]);

  const handleSuccess = () => {
    setIsCompleted(true);
  };

  if (!problem) {
    notFound();
  }

  const difficultyColors = {
    Easy: 'bg-[#50fa7b] text-[#282a36]',
    Medium: 'bg-[#f1fa8c] text-[#282a36]',
    Hard: 'bg-[#ff5555] text-[#282a36]',
  };

  return (
    <div className="flex h-[calc(100vh-64px)] flex-col">
      {/* Top Bar */}
      <div className="border-b border-[#44475a] bg-[#282a36] px-4 py-3">
        <div className="flex items-center justify-between">
          <Link
            href="/problems"
            className="flex items-center text-sm font-medium text-[#bd93f9] transition-colors hover:text-[#ff79c6]"
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
            Problem List
          </Link>
        </div>
      </div>

      {/* Split View */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Panel - Problem Description */}
        <div className="w-1/2 overflow-y-auto border-r border-[#44475a] bg-[#282a36]">
          <div className="p-6">
            {/* Problem Header */}
            <div className="mb-6">
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <span
                  className={`rounded-full px-3 py-1 text-xs font-bold ${difficultyColors[problem.difficulty]}`}
                >
                  {problem.difficulty}
                </span>
                <span className="rounded-full border-2 border-[#bd93f9] bg-[#bd93f9]/10 px-3 py-1 text-xs font-semibold text-[#bd93f9]">
                  {problem.topic}
                </span>
                {isCompleted && (
                  <span className="rounded-full border-2 border-[#50fa7b] bg-[#50fa7b]/10 px-3 py-1 text-xs font-semibold text-[#50fa7b]">
                    ✓ Completed
                  </span>
                )}
              </div>
              <h1 className="mb-4 text-2xl font-bold text-[#f8f8f2]">
                {problem.title}
              </h1>
            </div>

            {/* Description */}
            <div className="prose prose-sm max-w-none">
              <div className="mb-6 text-[#f8f8f2]">
                {problem.description.split('\n\n').map((paragraph, idx) => (
                  <p
                    key={idx}
                    className="mb-4 leading-relaxed whitespace-pre-wrap"
                  >
                    {formatText(paragraph)}
                  </p>
                ))}
              </div>

              {/* Examples */}
              <div className="mb-6">
                <h3 className="mb-4 text-base font-semibold text-[#f8f8f2]">
                  Examples
                </h3>
                {problem.examples.map((example, index) => (
                  <div
                    key={index}
                    className="mb-4 rounded-lg bg-[#44475a] p-4 font-mono text-sm"
                  >
                    <div className="mb-2">
                      <span className="font-semibold text-[#bd93f9]">
                        Input:
                      </span>{' '}
                      <span className="text-[#f8f8f2]">{example.input}</span>
                    </div>
                    <div className="mb-2">
                      <span className="font-semibold text-[#bd93f9]">
                        Output:
                      </span>{' '}
                      <span className="text-[#f8f8f2]">{example.output}</span>
                    </div>
                    {example.explanation && (
                      <div>
                        <span className="font-semibold text-[#bd93f9]">
                          Explanation:
                        </span>{' '}
                        <span className="text-[#6272a4]">
                          {example.explanation}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Constraints */}
              {problem.constraints && problem.constraints.length > 0 && (
                <div className="mb-6">
                  <h3 className="mb-3 text-base font-semibold text-[#f8f8f2]">
                    Constraints
                  </h3>
                  <ul className="list-inside space-y-1.5 text-sm text-[#f8f8f2]">
                    {problem.constraints.map((constraint, index) => (
                      <li key={index} className="font-mono">
                        • {constraint}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Hints */}
              {problem.hints && problem.hints.length > 0 && (
                <details className="mb-6 rounded-lg border-2 border-[#bd93f9] bg-[#44475a] p-4">
                  <summary className="cursor-pointer font-semibold text-[#bd93f9]">
                    💡 Hints
                  </summary>
                  <ul className="mt-3 space-y-2 text-sm text-[#f8f8f2]">
                    {problem.hints.map((hint, index) => (
                      <li key={index} className="ml-4">
                        {index + 1}. {hint}
                      </li>
                    ))}
                  </ul>
                </details>
              )}

              {/* Complexity */}
              {(problem.timeComplexity || problem.spaceComplexity) && (
                <div className="rounded-lg border-2 border-[#44475a] bg-[#44475a] p-4">
                  <h3 className="mb-3 text-base font-semibold text-[#f8f8f2]">
                    Expected Complexity
                  </h3>
                  <div className="space-y-2 text-sm">
                    {problem.timeComplexity && (
                      <div>
                        <span className="font-semibold text-[#bd93f9]">
                          Time:
                        </span>{' '}
                        <span className="font-mono text-[#8be9fd]">
                          {problem.timeComplexity}
                        </span>
                      </div>
                    )}
                    {problem.spaceComplexity && (
                      <div>
                        <span className="font-semibold text-[#bd93f9]">
                          Space:
                        </span>{' '}
                        <span className="font-mono text-[#8be9fd]">
                          {problem.spaceComplexity}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Code Editor */}
        <div className="flex w-1/2 flex-col bg-[#282a36]">
          <div className="flex-1 overflow-hidden">
            <PythonCodeRunner
              starterCode={problem.starterCode}
              testCases={problem.testCases}
              problemId={problem.id}
              onSuccess={handleSuccess}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper function to format text with inline code and bold
function formatText(text: string) {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);

  return parts.map((part, index) => {
    // Handle inline code
    if (part.startsWith('`') && part.endsWith('`')) {
      const code = part.slice(1, -1);
      return (
        <code
          key={index}
          className="rounded bg-[#44475a] px-2 py-0.5 font-mono text-sm text-[#8be9fd]"
        >
          {code}
        </code>
      );
    }
    // Handle bold text
    if (part.startsWith('**') && part.endsWith('**')) {
      const boldText = part.slice(2, -2);
      return (
        <strong key={index} className="font-semibold text-[#f8f8f2]">
          {boldText}
        </strong>
      );
    }
    return <span key={index}>{part}</span>;
  });
}

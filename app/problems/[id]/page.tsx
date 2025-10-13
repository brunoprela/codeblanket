'use client';

import { notFound, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { useState, useEffect, use, Suspense } from 'react';

import { getProblemById } from '@/lib/problems';
import { SimpleCodeEditor } from '@/components/SimpleCodeEditor';
import { isProblemCompleted } from '@/lib/helpers/storage';
import { formatText } from '@/lib/utils/formatText';

function ProblemPageContent({ id }: { id: string }) {
  const searchParams = useSearchParams();
  const from = searchParams.get('from') || 'problems';
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

  // Convert topic to URL slug - handle special cases
  const topicToSlug: Record<string, string> = {
    'Advanced Graphs': 'advanced-graphs',
    'Arrays & Hashing': 'arrays-hashing',
    Backtracking: 'backtracking',
    'Breadth-First Search (BFS)': 'bfs',
    'Binary Search': 'binary-search',
    'Bit Manipulation': 'bit-manipulation',
    'Depth-First Search (DFS)': 'dfs',
    'Dynamic Programming': 'dynamic-programming',
    'Fenwick Tree (Binary Indexed Tree)': 'fenwick-tree',
    'Fenwick Tree': 'fenwick-tree',
    Graphs: 'graphs',
    Greedy: 'greedy',
    'Heap / Priority Queue': 'heap',
    Intervals: 'intervals',
    'Linked List': 'linked-list',
    'Math & Geometry': 'math-geometry',
    'Segment Tree': 'segment-tree',
    'Sliding Window': 'sliding-window',
    Sorting: 'sorting',
    'Sorting Algorithms': 'sorting',
    Stack: 'stack',
    'Time & Space Complexity': 'time-space-complexity',
    Trees: 'trees',
    Tries: 'tries',
    'Two Pointers': 'two-pointers',
    'Python Fundamentals': 'python-fundamentals',
    'Python Intermediate': 'python-intermediate',
    'Python Advanced': 'python-advanced',
    'Python Object-Oriented Programming': 'python-oop',
  };

  const topicSlug =
    topicToSlug[problem.topic] ||
    problem.topic
      .toLowerCase()
      .replace(/\s+/g, '-')
      .replace(/[&/()/]/g, '-')
      .replace(/--+/g, '-');

  // Determine back button URL and text based on 'from' parameter
  let backUrl = '/problems';
  let backText = 'Problem List';

  if (from.startsWith('modules/')) {
    backUrl = `/${from}`;
    backText = 'Back to Module';
  } else if (from.startsWith('topics/')) {
    backUrl = `/${from}`;
    backText = 'Back to Topic';
  } else if (from === 'problems') {
    backUrl = '/problems';
    backText = 'Problem List';
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Top Bar */}
      <div className="flex-shrink-0 border-b border-[#44475a] bg-[#282a36] px-4 py-3">
        <div className="flex items-center justify-between">
          <Link
            href={backUrl}
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
            {backText}
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
                <Link
                  href={`/problems?from=${from}`}
                  className={`rounded-full px-3 py-1 text-xs font-bold transition-opacity hover:opacity-80 ${difficultyColors[problem.difficulty]}`}
                  title={`View all ${problem.difficulty} problems`}
                >
                  {problem.difficulty}
                </Link>
                <Link
                  href={`/topics/${topicSlug}?from=${from}`}
                  className="rounded-full border-2 border-[#bd93f9] bg-[#bd93f9]/10 px-3 py-1 text-xs font-semibold text-[#bd93f9] transition-colors hover:border-[#ff79c6] hover:bg-[#ff79c6]/10 hover:text-[#ff79c6]"
                  title={`View all ${problem.topic} problems`}
                >
                  {problem.topic}
                </Link>
                {isCompleted && (
                  <span className="rounded-full border-2 border-[#50fa7b] bg-[#50fa7b]/10 px-3 py-1 text-xs font-semibold text-[#50fa7b]">
                    ✓ Completed
                  </span>
                )}
              </div>

              <h1 className="mb-4 text-2xl font-bold text-[#f8f8f2]">
                {problem.title}
              </h1>

              {/* External Resources */}
              {(problem.leetcodeUrl || problem.youtubeUrl) && (
                <div className="mb-4 flex flex-wrap gap-2">
                  {problem.leetcodeUrl && (
                    <a
                      href={problem.leetcodeUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1.5 rounded-lg border border-[#f1fa8c] bg-[#f1fa8c]/10 px-3 py-1.5 text-xs font-semibold text-[#f1fa8c] transition-colors hover:bg-[#f1fa8c]/20"
                    >
                      <svg
                        className="h-4 w-4"
                        fill="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path d="M13.483 0a1.374 1.374 0 0 0-.961.438L7.116 6.226l-3.854 4.126a5.266 5.266 0 0 0-1.209 2.104 5.35 5.35 0 0 0-.125.513 5.527 5.527 0 0 0 .062 2.362 5.83 5.83 0 0 0 .349 1.017 5.938 5.938 0 0 0 1.271 1.818l4.277 4.193.039.038c2.248 2.165 5.852 2.133 8.063-.074l2.396-2.392c.54-.54.54-1.414.003-1.955a1.378 1.378 0 0 0-1.951-.003l-2.396 2.392a3.021 3.021 0 0 1-4.205.038l-.02-.019-4.276-4.193c-.652-.64-.972-1.469-.948-2.263a2.68 2.68 0 0 1 .066-.523 2.545 2.545 0 0 1 .619-1.164L9.13 8.114c1.058-1.134 3.204-1.27 4.43-.278l3.501 2.831c.593.48 1.461.387 1.94-.207a1.384 1.384 0 0 0-.207-1.943l-3.5-2.831c-.8-.647-1.766-1.045-2.774-1.202l2.015-2.158A1.384 1.384 0 0 0 13.483 0zm-2.866 12.815a1.38 1.38 0 0 0-1.38 1.382 1.38 1.38 0 0 0 1.38 1.382H20.79a1.38 1.38 0 0 0 1.38-1.382 1.38 1.38 0 0 0-1.38-1.382z" />
                      </svg>
                      LeetCode
                    </a>
                  )}
                  {problem.youtubeUrl && (
                    <a
                      href={problem.youtubeUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1.5 rounded-lg border border-[#ff5555] bg-[#ff5555]/10 px-3 py-1.5 text-xs font-semibold text-[#ff5555] transition-colors hover:bg-[#ff5555]/20"
                    >
                      <svg
                        className="h-4 w-4"
                        fill="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
                      </svg>
                      Video
                    </a>
                  )}
                </div>
              )}
            </div>

            {/* Description */}
            <div className="prose prose-sm max-w-none">
              <div className="mb-6 text-[#f8f8f2]">
                {problem.description.split('\n\n').map((paragraph, idx) => (
                  <div
                    key={idx}
                    className="mb-4 leading-relaxed whitespace-pre-wrap"
                  >
                    {formatText(paragraph)}
                  </div>
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
        <div className="flex w-1/2 flex-col overflow-hidden bg-[#282a36]">
          <SimpleCodeEditor
            starterCode={problem.starterCode}
            testCases={problem.testCases}
            problemId={problem.id}
            onSuccess={handleSuccess}
          />
        </div>
      </div>
    </div>
  );
}

export default function ProblemPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);

  return (
    <Suspense
      fallback={
        <div className="flex h-full items-center justify-center bg-[#282a36] text-[#f8f8f2]">
          Loading...
        </div>
      }
    >
      <ProblemPageContent id={id} />
    </Suspense>
  );
}

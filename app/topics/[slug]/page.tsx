'use client';

import { notFound } from 'next/navigation';
import Link from 'next/link';
import { problemCategories } from '@/lib/problems';
import { Difficulty } from '@/lib/types';
import { useEffect, useState, use } from 'react';
import { getCompletedProblems } from '@/lib/storage';

export default function TopicProblemsPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = use(params);
  const [completedProblems, setCompletedProblems] = useState<Set<string>>(
    new Set(),
  );

  // Load completed problems from localStorage
  useEffect(() => {
    setCompletedProblems(getCompletedProblems());
  }, []);

  const category = problemCategories.find((cat) => cat.id === slug);

  if (!category) {
    notFound();
  }

  const problems = category.problems;

  const easyProblems = problems.filter((p) => p.difficulty === 'Easy');
  const mediumProblems = problems.filter((p) => p.difficulty === 'Medium');
  const hardProblems = problems.filter((p) => p.difficulty === 'Hard');

  const difficultyColors = {
    Easy: 'bg-[#50fa7b] text-[#282a36] border-[#50fa7b]',
    Medium: 'bg-[#f1fa8c] text-[#282a36] border-[#f1fa8c]',
    Hard: 'bg-[#ff5555] text-[#282a36] border-[#ff5555]',
  };

  return (
    <div className="container mx-auto max-w-6xl px-4 py-12">
      {/* Header */}
      <div className="mb-8 space-y-4">
        <Link
          href="/"
          className="inline-flex items-center text-[#bd93f9] transition-colors hover:text-[#ff79c6]"
        >
          ← Back to Topics
        </Link>
        <div className="flex items-center gap-4">
          <span className="text-5xl">{category.icon}</span>
          <div>
            <h1 className="text-4xl font-bold text-[#f8f8f2]">
              {category.title}
            </h1>
            <p className="mt-2 text-lg text-[#6272a4]">
              {category.description}
            </p>
          </div>
        </div>
      </div>

      {/* Problems by Difficulty */}
      <div>
        {easyProblems.length > 0 && (
          <div>
            <h3 className="mb-4 flex items-center gap-3 text-2xl font-bold text-[#50fa7b]">
              <span className="rounded-lg bg-[#50fa7b]/20 px-4 py-2">Easy</span>
              <span className="text-[#6272a4]">({easyProblems.length})</span>
            </h3>
            <div>
              {easyProblems.map((problem) => (
                <ProblemCard
                  key={problem.id}
                  problem={problem}
                  difficultyColors={difficultyColors}
                  isCompleted={completedProblems.has(problem.id)}
                />
              ))}
            </div>
          </div>
        )}

        {mediumProblems.length > 0 && (
          <div>
            <h3 className="mb-4 flex items-center gap-3 text-2xl font-bold text-[#f1fa8c]">
              <span className="rounded-lg bg-[#f1fa8c]/20 px-4 py-2">
                Medium
              </span>
              <span className="text-[#6272a4]">({mediumProblems.length})</span>
            </h3>
            <div>
              {mediumProblems.map((problem) => (
                <ProblemCard
                  key={problem.id}
                  problem={problem}
                  difficultyColors={difficultyColors}
                  isCompleted={completedProblems.has(problem.id)}
                />
              ))}
            </div>
          </div>
        )}

        {hardProblems.length > 0 && (
          <div>
            <h3 className="mb-4 flex items-center gap-3 text-2xl font-bold text-[#ff5555]">
              <span className="rounded-lg bg-[#ff5555]/20 px-4 py-2">Hard</span>
              <span className="text-[#6272a4]">({hardProblems.length})</span>
            </h3>
            <div>
              {hardProblems.map((problem) => (
                <ProblemCard
                  key={problem.id}
                  problem={problem}
                  difficultyColors={difficultyColors}
                  isCompleted={completedProblems.has(problem.id)}
                />
              ))}
            </div>
          </div>
        )}

        {problems.length === 0 && (
          <p className="text-center text-[#6272a4]">
            No problems found for this category.
          </p>
        )}
      </div>
    </div>
  );
}

function ProblemCard({
  problem,
  difficultyColors,
  isCompleted,
}: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  problem: any;
  difficultyColors: Record<Difficulty, string>;
  isCompleted: boolean;
}) {
  // Format description with inline code and bold text
  const formatDescription = (text: string) => {
    const firstLine = text.split('\n')[0];
    const parts = firstLine.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);

    return parts.map((part, index) => {
      if (part.startsWith('`') && part.endsWith('`')) {
        const code = part.slice(1, -1);
        return (
          <code
            key={index}
            className="rounded bg-[#6272a4] px-2 py-0.5 font-mono text-sm text-[#f8f8f2]"
          >
            {code}
          </code>
        );
      }
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
  };

  return (
    <Link href={`/problems/${problem.id}`} className="mb-6 block">
      <div className="group cursor-pointer rounded-lg border-2 border-[#44475a] bg-[#44475a] p-6 transition-all hover:border-[#bd93f9] hover:shadow-xl">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <span
                className={`rounded-full border px-3 py-1 text-xs font-bold ${difficultyColors[problem.difficulty as Difficulty]}`}
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
            <h4 className="mb-3 text-xl font-semibold text-[#f8f8f2] transition-colors group-hover:text-[#bd93f9]">
              {problem.title}
            </h4>
            <p className="line-clamp-2 text-sm leading-relaxed text-[#f8f8f2]">
              {formatDescription(problem.description)}
            </p>
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
}

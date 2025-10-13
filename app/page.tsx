import Link from 'next/link';

import { allProblems } from '@/lib/problems';
import { moduleCategories } from '@/lib/modules';

export default function Home() {
  const totalProblems = allProblems.length;
  const easyCount = allProblems.filter((p) => p.difficulty === 'Easy').length;
  const mediumCount = allProblems.filter(
    (p) => p.difficulty === 'Medium',
  ).length;
  const hardCount = allProblems.filter((p) => p.difficulty === 'Hard').length;

  return (
    <div className="container mx-auto max-w-6xl px-4 py-12">
      {/* Hero Section */}
      <div className="mb-16 text-center">
        <h1 className="mb-4 text-5xl font-bold text-[#f8f8f2]">
          Master Algorithms & Data Structures
        </h1>
        <p className="mx-auto max-w-2xl text-xl text-[#6272a4]">
          Learn through hands-on practice. Write Python code directly in your
          browser and get instant feedback.
        </p>
      </div>

      {/* Stats */}
      <div className="mb-16 grid gap-6 md:grid-cols-4">
        <div className="rounded-lg border-2 border-[#44475a] bg-[#44475a] p-6 text-center">
          <div className="mb-2 text-4xl font-bold text-[#bd93f9]">
            {totalProblems}
          </div>
          <div className="font-medium text-[#f8f8f2]">Total Problems</div>
        </div>
        <div className="rounded-lg border-2 border-[#44475a] bg-[#44475a] p-6 text-center">
          <div className="mb-2 text-4xl font-bold text-[#50fa7b]">
            {easyCount}
          </div>
          <div className="font-medium text-[#f8f8f2]">Easy</div>
        </div>
        <div className="rounded-lg border-2 border-[#44475a] bg-[#44475a] p-6 text-center">
          <div className="mb-2 text-4xl font-bold text-[#f1fa8c]">
            {mediumCount}
          </div>
          <div className="font-medium text-[#f8f8f2]">Medium</div>
        </div>
        <div className="rounded-lg border-2 border-[#44475a] bg-[#44475a] p-6 text-center">
          <div className="mb-2 text-4xl font-bold text-[#ff5555]">
            {hardCount}
          </div>
          <div className="font-medium text-[#f8f8f2]">Hard</div>
        </div>
      </div>

      {/* Learning Path */}
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold text-[#f8f8f2]">
              📚 Learning Path
            </h2>
            <p className="mt-2 text-[#f8f8f2]">
              Follow this structured curriculum from fundamentals to advanced
              topics
            </p>
          </div>
          <Link
            href="/problems"
            className="rounded-lg bg-[#bd93f9] px-6 py-2.5 font-semibold text-[#282a36] transition-colors hover:bg-[#ff79c6]"
          >
            View All Problems →
          </Link>
        </div>

        <div className="space-y-5">
          {moduleCategories.map((moduleCategory, index) => (
            <div
              key={moduleCategory.id}
              className="flex items-center gap-6 rounded-xl border-2 border-[#44475a] bg-[#44475a] p-6 shadow-lg"
            >
              {/* Number */}
              <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-[#bd93f9] text-xl font-bold text-[#282a36]">
                {index + 1}
              </div>

              {/* Icon */}
              <div className="text-4xl">{moduleCategory.icon}</div>

              {/* Content */}
              <div className="flex-1">
                <h3 className="mb-1 text-xl font-bold text-[#f8f8f2]">
                  {moduleCategory.title}
                </h3>
                <p className="text-sm text-[#f8f8f2]">
                  {moduleCategory.description}
                </p>
              </div>

              {/* Metadata and Actions */}
              <div className="flex flex-shrink-0 items-center gap-3">
                <div className="rounded-full border-2 border-[#f8f8f2] bg-[#f8f8f2]/10 px-3 py-1 text-xs font-semibold text-[#f8f8f2]">
                  {moduleCategory.module.sections.length} sections
                </div>
                <div className="rounded-full border-2 border-[#f8f8f2] bg-[#f8f8f2]/10 px-3 py-1 text-xs font-semibold text-[#f8f8f2]">
                  {moduleCategory.problemCount} problems
                </div>
                <Link
                  href={`/modules/${moduleCategory.id}`}
                  className="rounded-lg bg-[#bd93f9] px-4 py-2 text-sm font-semibold text-[#282a36] transition-colors hover:bg-[#ff79c6]"
                >
                  Learn
                </Link>
                <Link
                  href={`/topics/${moduleCategory.id}`}
                  className="rounded-lg border-2 border-[#bd93f9] bg-transparent px-4 py-2 text-sm font-semibold text-[#bd93f9] transition-colors hover:bg-[#bd93f9] hover:text-[#282a36]"
                >
                  Practice
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

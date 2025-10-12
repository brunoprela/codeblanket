import Link from 'next/link';

import { problemCategories, allProblems } from '@/lib/problems';
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

      {/* Learning Modules */}
      <div className="mb-16 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold text-[#f8f8f2]">
              📚 Learning Modules
            </h2>
            <p className="mt-2 text-[#6272a4]">
              Start here to learn the concepts before practicing
            </p>
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {moduleCategories.map((moduleCategory) => (
            <Link
              key={moduleCategory.id}
              href={`/modules/${moduleCategory.id}`}
            >
              <div className="group h-full cursor-pointer rounded-xl border-2 border-[#bd93f9] bg-[#bd93f9]/10 p-8 shadow-lg transition-all hover:border-[#ff79c6] hover:shadow-2xl">
                <div className="mb-4 flex items-center justify-between">
                  <div className="text-5xl">{moduleCategory.icon}</div>
                  <div className="rounded-full border-2 border-[#bd93f9] bg-[#bd93f9]/20 px-4 py-1.5 text-sm font-semibold text-[#bd93f9]">
                    {moduleCategory.module.sections.length} sections
                  </div>
                </div>
                <h3 className="mb-3 text-2xl font-bold text-[#f8f8f2] transition-colors group-hover:text-[#ff79c6]">
                  {moduleCategory.title}
                </h3>
                <p className="mb-4 text-[#f8f8f2]">
                  {moduleCategory.description}
                </p>
                <div className="flex items-center gap-2 text-sm text-[#6272a4]">
                  <span>
                    📝 {moduleCategory.problemCount} practice problems
                  </span>
                </div>
                <div className="mt-6 flex items-center font-semibold text-[#bd93f9] transition-transform group-hover:translate-x-2">
                  Start Learning
                  <svg
                    className="ml-2 h-5 w-5"
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
            </Link>
          ))}
        </div>
      </div>

      {/* Problem Categories */}
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold text-[#f8f8f2]">
              💪 Practice Problems
            </h2>
            <p className="mt-2 text-[#6272a4]">
              Jump straight to solving problems by topic
            </p>
          </div>
          <Link
            href="/problems"
            className="rounded-lg bg-[#bd93f9] px-6 py-2.5 font-semibold text-[#282a36] transition-colors hover:bg-[#ff79c6]"
          >
            View All Problems →
          </Link>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {problemCategories.map((category) => (
            <Link key={category.id} href={`/topics/${category.id}`}>
              <div className="group h-full cursor-pointer rounded-xl border-2 border-[#44475a] bg-[#44475a] p-8 shadow-lg transition-all hover:border-[#bd93f9] hover:shadow-2xl">
                <div className="mb-4 flex items-center justify-between">
                  <div className="text-5xl">{category.icon}</div>
                  <div className="rounded-full bg-[#bd93f9] px-4 py-1.5 text-sm font-semibold text-[#282a36]">
                    {category.problemCount} problems
                  </div>
                </div>
                <h3 className="mb-3 text-2xl font-bold text-[#f8f8f2] transition-colors group-hover:text-[#bd93f9]">
                  {category.title}
                </h3>
                <p className="text-[#f8f8f2]">{category.description}</p>
                <div className="mt-6 flex items-center font-semibold text-[#bd93f9] transition-transform group-hover:translate-x-2">
                  Practice Now
                  <svg
                    className="ml-2 h-5 w-5"
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
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}

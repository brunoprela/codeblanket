'use client';

import { useState, useMemo, useEffect } from 'react';
import Link from 'next/link';

import { allProblems } from '@/lib/problems';
import { Difficulty } from '@/lib/types';
import { getCompletedProblems } from '@/lib/helpers/storage';
import { formatDescription } from '@/lib/utils/formatText';

export default function AllProblemsPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDifficulties, setSelectedDifficulties] = useState<
    Set<Difficulty>
  >(new Set(['Easy', 'Medium', 'Hard']));
  const [sortBy, setSortBy] = useState<'number' | 'title' | 'difficulty'>(
    'number',
  );
  const [completedProblems, setCompletedProblems] = useState<Set<string>>(
    new Set(),
  );
  const [showCompleted, setShowCompleted] = useState(true);
  const [showIncomplete, setShowIncomplete] = useState(true);

  // Load completed problems from localStorage
  useEffect(() => {
    // Initial load
    setCompletedProblems(getCompletedProblems());

    // Reload when tab/window regains focus (after completing problems on other pages)
    const handleFocus = () => {
      setCompletedProblems(getCompletedProblems());
    };

    // Reload when storage changes (e.g., completing problems in another tab)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'codeblanket_completed_problems') {
        setCompletedProblems(getCompletedProblems());
      }
    };

    // Listen for custom event when problems are completed
    const handleProblemCompleted = () => {
      setCompletedProblems(getCompletedProblems());
    };

    window.addEventListener('focus', handleFocus);
    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('problemCompleted', handleProblemCompleted);
    window.addEventListener('problemReset', handleProblemCompleted);

    return () => {
      window.removeEventListener('focus', handleFocus);
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('problemCompleted', handleProblemCompleted);
      window.removeEventListener('problemReset', handleProblemCompleted);
    };
  }, []);

  // Filter and sort problems
  const filteredAndSortedProblems = useMemo(() => {
    let problems = allProblems.filter((problem) => {
      const matchesSearch =
        searchTerm === '' ||
        problem.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        problem.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        problem.topic.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesDifficulty = selectedDifficulties.has(problem.difficulty);

      const isCompleted = completedProblems.has(problem.id);
      const matchesCompletion =
        (showCompleted && isCompleted) || (showIncomplete && !isCompleted);

      return matchesSearch && matchesDifficulty && matchesCompletion;
    });

    // Sort problems
    problems = [...problems].sort((a, b) => {
      if (sortBy === 'number') {
        return (a.order ?? 0) - (b.order ?? 0);
      } else if (sortBy === 'title') {
        return a.title.localeCompare(b.title);
      } else {
        // difficulty
        const difficultyOrder = { Easy: 1, Medium: 2, Hard: 3 };
        return difficultyOrder[a.difficulty] - difficultyOrder[b.difficulty];
      }
    });

    return problems;
  }, [
    searchTerm,
    selectedDifficulties,
    sortBy,
    completedProblems,
    showCompleted,
    showIncomplete,
  ]);

  const toggleDifficulty = (difficulty: Difficulty) => {
    const newDifficulties = new Set(selectedDifficulties);
    if (newDifficulties.has(difficulty)) {
      newDifficulties.delete(difficulty);
    } else {
      newDifficulties.add(difficulty);
    }
    setSelectedDifficulties(newDifficulties);
  };

  const easyCount = allProblems.filter((p) => p.difficulty === 'Easy').length;
  const mediumCount = allProblems.filter(
    (p) => p.difficulty === 'Medium',
  ).length;
  const hardCount = allProblems.filter((p) => p.difficulty === 'Hard').length;

  const difficultyColors = {
    Easy: 'bg-[#50fa7b] text-[#282a36] border-[#50fa7b]',
    Medium: 'bg-[#f1fa8c] text-[#282a36] border-[#f1fa8c]',
    Hard: 'bg-[#ff5555] text-[#282a36] border-[#ff5555]',
  };

  return (
    <div className="container mx-auto max-w-7xl px-4 py-12">
      <h1 className="mb-8 text-4xl font-bold text-[#f8f8f2]">All Problems</h1>

      <div className="flex flex-col gap-8 lg:flex-row">
        {/* Sidebar for Filters and Sort */}
        <aside className="w-full lg:w-1/4">
          <div className="rounded-lg border-2 border-[#44475a] bg-[#44475a] p-6 shadow-lg">
            <h2 className="mb-6 text-xl font-semibold text-[#f8f8f2]">
              Filters & Sort
            </h2>

            {/* Search */}
            <div className="mb-6">
              <label
                htmlFor="search"
                className="mb-2 block text-sm font-medium text-[#f8f8f2]"
              >
                Search
              </label>
              <input
                type="text"
                id="search"
                placeholder="Search problems..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full rounded-md border-2 border-[#6272a4] bg-[#282a36] p-2.5 text-[#f8f8f2] placeholder-[#6272a4] focus:border-[#bd93f9] focus:ring-2 focus:ring-[#bd93f9]/20 focus:outline-none"
              />
            </div>

            {/* Difficulty Filter */}
            <div className="mb-6">
              <h3 className="mb-3 text-base font-medium text-[#f8f8f2]">
                Difficulty
              </h3>
              {(['Easy', 'Medium', 'Hard'] as Difficulty[]).map((diff) => {
                const counts = {
                  Easy: easyCount,
                  Medium: mediumCount,
                  Hard: hardCount,
                };
                const colors = {
                  Easy: 'text-[#50fa7b]',
                  Medium: 'text-[#f1fa8c]',
                  Hard: 'text-[#ff5555]',
                };
                return (
                  <div key={diff} className="mb-2 flex items-center">
                    <input
                      type="checkbox"
                      id={`difficulty-${diff}`}
                      checked={selectedDifficulties.has(diff)}
                      onChange={() => toggleDifficulty(diff)}
                      className="h-4 w-4 rounded border-[#6272a4] bg-[#282a36] text-[#bd93f9] focus:ring-[#bd93f9]"
                    />
                    <label
                      htmlFor={`difficulty-${diff}`}
                      className={`ml-3 text-sm font-medium ${colors[diff]}`}
                    >
                      {diff} ({counts[diff]})
                    </label>
                  </div>
                );
              })}
            </div>

            {/* Completion Status Filter */}
            <div className="mb-6">
              <h3 className="mb-3 text-base font-medium text-[#f8f8f2]">
                Status
              </h3>
              <div className="mb-2 flex items-center">
                <input
                  type="checkbox"
                  id="show-completed"
                  checked={showCompleted}
                  onChange={() => setShowCompleted(!showCompleted)}
                  className="h-4 w-4 rounded border-[#6272a4] bg-[#282a36] text-[#bd93f9] focus:ring-[#bd93f9]"
                />
                <label
                  htmlFor="show-completed"
                  className="ml-3 text-sm font-medium text-[#50fa7b]"
                >
                  Completed ({completedProblems.size})
                </label>
              </div>
              <div className="mb-2 flex items-center">
                <input
                  type="checkbox"
                  id="show-incomplete"
                  checked={showIncomplete}
                  onChange={() => setShowIncomplete(!showIncomplete)}
                  className="h-4 w-4 rounded border-[#6272a4] bg-[#282a36] text-[#bd93f9] focus:ring-[#bd93f9]"
                />
                <label
                  htmlFor="show-incomplete"
                  className="ml-3 text-sm font-medium text-[#f8f8f2]"
                >
                  Not Completed ({allProblems.length - completedProblems.size})
                </label>
              </div>
            </div>

            {/* Sort By */}
            <div>
              <h3 className="mb-3 text-base font-medium text-[#f8f8f2]">
                Sort By
              </h3>
              <select
                value={sortBy}
                onChange={(e) =>
                  setSortBy(e.target.value as 'number' | 'title' | 'difficulty')
                }
                className="w-full rounded-md border-2 border-[#6272a4] bg-[#282a36] p-2.5 text-[#f8f8f2] focus:border-[#bd93f9] focus:ring-2 focus:ring-[#bd93f9]/20 focus:outline-none"
              >
                <option value="number">Problem Number</option>
                <option value="title">Title (A-Z)</option>
                <option value="difficulty">Difficulty</option>
              </select>
            </div>
          </div>
        </aside>

        {/* Problem List */}
        <div className="flex-1">
          <div className="mb-6 flex items-center justify-between">
            <h2 className="text-2xl font-semibold text-[#f8f8f2]">
              {filteredAndSortedProblems.length} Problems Found
            </h2>
          </div>

          {filteredAndSortedProblems.length > 0 ? (
            <div>
              {filteredAndSortedProblems.map((problem) => (
                <ProblemCard
                  key={problem.id}
                  problem={problem}
                  difficultyColors={difficultyColors}
                  isCompleted={completedProblems.has(problem.id)}
                />
              ))}
            </div>
          ) : (
            <div className="rounded-lg border-2 border-[#44475a] bg-[#44475a] p-8 text-center shadow-lg">
              <p className="text-lg text-[#6272a4]">
                No problems match your criteria. Try adjusting your filters!
              </p>
            </div>
          )}
        </div>
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
  return (
    <Link href={`/problems/${problem.id}?from=problems`} className="mb-6 block">
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

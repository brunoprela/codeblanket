'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';

import { allProblems } from '@/lib/problems';
import { moduleCategories } from '@/lib/modules';
import { getCompletedProblems } from '@/lib/helpers/storage';

export default function Home() {
  const totalProblems = allProblems.length;
  const totalModules = moduleCategories.length;

  // Track completed problems
  const [completedProblems, setCompletedProblems] = useState<Set<string>>(
    new Set(),
  );

  // Track completed sections per module
  const [moduleProgress, setModuleProgress] = useState<
    Record<string, { completed: number; total: number }>
  >({});

  // Calculate completed problems per topic
  const getTopicProblemsProgress = (topicTitle: string) => {
    const topicProblems = allProblems.filter((p) => p.topic === topicTitle);
    const completedCount = topicProblems.filter((p) =>
      completedProblems.has(p.id),
    ).length;
    return {
      completed: completedCount,
      total: topicProblems.length,
    };
  };

  // Load completion data
  useEffect(() => {
    // Load completed problems
    setCompletedProblems(getCompletedProblems());

    // Load module progress
    const progress: Record<string, { completed: number; total: number }> = {};
    moduleCategories.forEach((moduleCategory) => {
      const storageKey = `module-${moduleCategory.id}-completed`;
      const stored = localStorage.getItem(storageKey);
      const completedSections = stored ? JSON.parse(stored) : [];
      progress[moduleCategory.id] = {
        completed: completedSections.length,
        total: moduleCategory.module.sections.length,
      };
    });
    setModuleProgress(progress);

    // Listen for completion events
    const handleUpdate = () => {
      setCompletedProblems(getCompletedProblems());

      const newProgress: Record<string, { completed: number; total: number }> =
        {};
      moduleCategories.forEach((moduleCategory) => {
        const storageKey = `module-${moduleCategory.id}-completed`;
        const stored = localStorage.getItem(storageKey);
        const completedSections = stored ? JSON.parse(stored) : [];
        newProgress[moduleCategory.id] = {
          completed: completedSections.length,
          total: moduleCategory.module.sections.length,
        };
      });
      setModuleProgress(newProgress);
    };

    window.addEventListener('focus', handleUpdate);
    window.addEventListener('storage', handleUpdate);
    window.addEventListener('problemCompleted', handleUpdate);
    window.addEventListener('problemReset', handleUpdate);

    return () => {
      window.removeEventListener('focus', handleUpdate);
      window.removeEventListener('storage', handleUpdate);
      window.removeEventListener('problemCompleted', handleUpdate);
      window.removeEventListener('problemReset', handleUpdate);
    };
  }, []);

  // Calculate completed modules (all sections completed)
  const completedModulesCount = Object.values(moduleProgress).filter(
    (progress) =>
      progress.completed > 0 && progress.completed === progress.total,
  ).length;

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
      <div className="mb-16 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {/* Completed Stats */}
        <div className="rounded-lg border-2 border-[#50fa7b] bg-[#50fa7b]/10 p-6 text-center">
          <div className="mb-2 text-4xl font-bold text-[#50fa7b]">
            {completedProblems.size}
          </div>
          <div className="font-medium text-[#f8f8f2]">Problems Solved</div>
        </div>
        <div className="rounded-lg border-2 border-[#bd93f9] bg-[#bd93f9]/10 p-6 text-center">
          <div className="mb-2 text-4xl font-bold text-[#bd93f9]">
            {completedModulesCount}
          </div>
          <div className="font-medium text-[#f8f8f2]">Modules Completed</div>
        </div>

        {/* Total Stats */}
        <div className="rounded-lg border-2 border-[#44475a] bg-[#44475a] p-6 text-center">
          <div className="mb-2 text-4xl font-bold text-[#8be9fd]">
            {totalProblems}
          </div>
          <div className="font-medium text-[#f8f8f2]">Total Problems</div>
        </div>
        <div className="rounded-lg border-2 border-[#44475a] bg-[#44475a] p-6 text-center">
          <div className="mb-2 text-4xl font-bold text-[#ff79c6]">
            {totalModules}
          </div>
          <div className="font-medium text-[#f8f8f2]">Total Modules</div>
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
          {moduleCategories.map((moduleCategory, index) => {
            const progress = moduleProgress[moduleCategory.id] || {
              completed: 0,
              total: moduleCategory.module.sections.length,
            };
            const progressPercent =
              progress.total > 0
                ? Math.round((progress.completed / progress.total) * 100)
                : 0;
            const isCompleted =
              progress.completed > 0 && progress.completed === progress.total;

            // Get problems progress for this topic (match by title, not id)
            const problemsProgress = getTopicProblemsProgress(
              moduleCategory.title,
            );
            const problemsPercent =
              problemsProgress.total > 0
                ? Math.round(
                  (problemsProgress.completed / problemsProgress.total) * 100,
                )
                : 0;

            const hasAnyProgress =
              progress.completed > 0 || problemsProgress.completed > 0;

            return (
              <div
                key={moduleCategory.id}
                className="rounded-xl border-2 border-[#44475a] bg-[#44475a] p-6 shadow-lg"
              >
                <div className="flex items-center gap-6">
                  {/* Number */}
                  <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-[#bd93f9] text-xl font-bold text-[#282a36]">
                    {index + 1}
                  </div>

                  {/* Icon */}
                  <div className="text-4xl">{moduleCategory.icon}</div>

                  {/* Content */}
                  <div className="flex-1">
                    <div className="mb-1 flex items-center gap-2">
                      <h3 className="text-xl font-bold text-[#f8f8f2]">
                        {moduleCategory.title}
                      </h3>
                      {isCompleted && (
                        <span className="rounded-full border-2 border-[#50fa7b] bg-[#50fa7b]/20 px-2 py-0.5 text-xs font-semibold text-[#50fa7b]">
                          ✓ Module Complete
                        </span>
                      )}
                      {problemsProgress.completed === problemsProgress.total &&
                        problemsProgress.total > 0 && (
                          <span className="rounded-full border-2 border-[#8be9fd] bg-[#8be9fd]/20 px-2 py-0.5 text-xs font-semibold text-[#8be9fd]">
                            ✓ All Problems Solved
                          </span>
                        )}
                    </div>
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

                {/* Progress Bars */}
                {hasAnyProgress && (
                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    {/* Module Progress */}
                    {progress.completed > 0 && (
                      <div>
                        <div className="mb-1 flex items-center justify-between text-xs">
                          <span className="font-semibold text-[#bd93f9]">
                            📚 Module: {progress.completed} / {progress.total}
                          </span>
                          <span className="text-[#6272a4]">
                            {progressPercent}%
                          </span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-[#282a36]">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-[#bd93f9] to-[#ff79c6] transition-all duration-300"
                            style={{ width: `${progressPercent}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Problems Progress */}
                    {problemsProgress.completed > 0 && (
                      <div>
                        <div className="mb-1 flex items-center justify-between text-xs">
                          <span className="font-semibold text-[#8be9fd]">
                            💻 Problems: {problemsProgress.completed} /{' '}
                            {problemsProgress.total}
                          </span>
                          <span className="text-[#6272a4]">
                            {problemsPercent}%
                          </span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-[#282a36]">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-[#8be9fd] to-[#50fa7b] transition-all duration-300"
                            style={{ width: `${problemsPercent}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

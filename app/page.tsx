'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';

import { allProblems } from '@/lib/content/problems';
import { moduleCategories, topicSections } from '@/lib/content/topics';
import { getCompletedProblems } from '@/lib/helpers/storage';
import {
  getTotalDiscussionQuestionsCount,
  getTotalMultipleChoiceQuestionsCount,
  getCompletedMultipleChoiceQuestionsCount,
} from '@/lib/helpers/indexeddb';
import { getVideoMetadataForQuestion } from '@/lib/helpers/storage-adapter';
import { getUserStats } from '@/lib/helpers/storage-stats';
import { ModuleSection } from '@/lib/types';

export default function Home() {
  const totalProblems = allProblems.length;

  // Track selected section with localStorage persistence
  // Initialize with null to prevent flash of wrong content
  const [selectedSectionId, setSelectedSectionId] = useState<string | null>(
    null,
  );

  // Track if we've loaded the initial state
  const [isInitialized, setIsInitialized] = useState(false);

  // Track completed problems
  const [completedProblems, setCompletedProblems] = useState<Set<string>>(
    new Set(),
  );

  // Track completed sections per module
  const [moduleProgress, setModuleProgress] = useState<
    Record<string, { completed: number; total: number }>
  >({});

  // Track completed discussion questions
  const [completedDiscussions, setCompletedDiscussions] = useState(0);
  const [totalDiscussions, setTotalDiscussions] = useState(0);

  // Track completed discussion questions per module
  const [moduleDiscussionProgress, setModuleDiscussionProgress] = useState<
    Record<string, { completed: number; total: number }>
  >({});

  // Track completed multiple choice questions
  const [completedMultipleChoice, setCompletedMultipleChoice] = useState(0);
  const [totalMultipleChoice, setTotalMultipleChoice] = useState(0);

  // Track completed multiple choice questions per module
  const [moduleMultipleChoiceProgress, setModuleMultipleChoiceProgress] =
    useState<Record<string, { completed: number; total: number }>>({});

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

  // Load selected section from localStorage on client mount
  useEffect(() => {
    const saved = localStorage.getItem('selected-learning-path');
    if (saved && topicSections.some((s) => s.id === saved)) {
      setSelectedSectionId(saved);
    } else {
      // If no saved selection, use the first section
      setSelectedSectionId(topicSections[0]?.id || '');
    }
    setIsInitialized(true);
  }, []);

  // Save selected learning path to localStorage
  useEffect(() => {
    if (selectedSectionId) {
      localStorage.setItem('selected-learning-path', selectedSectionId);
    }
  }, [selectedSectionId]);

  // Load completion data
  useEffect(() => {
    // Load completed problems (async for authenticated users)
    const loadProblems = async () => {
      const problems = await getCompletedProblems();
      setCompletedProblems(problems);
    };
    loadProblems();

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

    // Load discussion question stats (efficient - uses stats API for authenticated users)
    const loadDiscussionStats = async () => {
      // Try to get efficient stats from API (authenticated users)
      const userStats = await getUserStats();

      if (userStats) {
        // Authenticated user - use efficient stats from API
        const total = getTotalDiscussionQuestionsCount(moduleCategories);
        setCompletedDiscussions(userStats.completedDiscussionQuestions);
        setTotalDiscussions(total);

        // Use module-specific video counts from stats API
        const discussionProgress: Record<
          string,
          { completed: number; total: number }
        > = {};

        moduleCategories.forEach((moduleCategory) => {
          let totalCount = 0;
          moduleCategory.module.sections.forEach((section) => {
            if (section.quiz) {
              totalCount += section.quiz.length;
            }
          });

          discussionProgress[moduleCategory.id] = {
            completed: userStats.moduleVideoCounts[moduleCategory.id] || 0,
            total: totalCount,
          };
        });

        setModuleDiscussionProgress(discussionProgress);
      } else {
        // Anonymous user - use IndexedDB (local, instant)
        const { getCompletedDiscussionQuestionsCount } = await import(
          '@/lib/helpers/indexeddb'
        );
        const completed = await getCompletedDiscussionQuestionsCount();
        const total = getTotalDiscussionQuestionsCount(moduleCategories);
        setCompletedDiscussions(completed);
        setTotalDiscussions(total);

        // For anonymous users, load from IndexedDB (fast, local)
        const discussionProgress: Record<
          string,
          { completed: number; total: number }
        > = {};

        const { getVideosForQuestion } = await import(
          '@/lib/helpers/indexeddb'
        );

        for (const moduleCategory of moduleCategories) {
          let completedCount = 0;
          let totalCount = 0;

          for (const section of moduleCategory.module.sections) {
            if (section.quiz) {
              totalCount += section.quiz.length;

              for (const question of section.quiz) {
                const questionId = `${moduleCategory.id}-${section.id}-${question.id}`;
                const videos = await getVideosForQuestion(questionId);
                if (videos.length > 0) {
                  completedCount++;
                }
              }
            }
          }

          discussionProgress[moduleCategory.id] = {
            completed: completedCount,
            total: totalCount,
          };
        }

        setModuleDiscussionProgress(discussionProgress);
      }
    };
    loadDiscussionStats();

    // Load multiple choice question stats
    const loadMultipleChoiceStats = () => {
      const total = getTotalMultipleChoiceQuestionsCount(moduleCategories);
      const completed =
        getCompletedMultipleChoiceQuestionsCount(moduleCategories);
      setTotalMultipleChoice(total);
      setCompletedMultipleChoice(completed);

      // Load multiple choice progress per module
      const mcProgress: Record<string, { completed: number; total: number }> =
        {};

      for (const moduleCategory of moduleCategories) {
        let completedCount = 0;
        let totalCount = 0;

        for (const section of moduleCategory.module.sections) {
          if (section.multipleChoice) {
            totalCount += section.multipleChoice.length;

            const storageKey = `mc-quiz-${moduleCategory.id}-${section.id}`;
            const stored = localStorage.getItem(storageKey);
            if (stored) {
              try {
                const completedQuestions = JSON.parse(stored);
                // Deduplicate in case of corrupted data
                const uniqueQuestions = [...new Set(completedQuestions)];

                // Fix corrupted data if duplicates found
                if (uniqueQuestions.length !== completedQuestions.length) {
                  localStorage.setItem(
                    storageKey,
                    JSON.stringify(uniqueQuestions),
                  );
                  console.warn(
                    `Fixed duplicates in ${storageKey}: ${completedQuestions.length} → ${uniqueQuestions.length}`,
                  );
                }

                completedCount += uniqueQuestions.length;
              } catch (e) {
                console.error('Failed to parse MC quiz progress:', e);
              }
            }
          }
        }

        mcProgress[moduleCategory.id] = {
          completed: completedCount,
          total: totalCount,
        };
      }

      setModuleMultipleChoiceProgress(mcProgress);
    };
    loadMultipleChoiceStats();

    // Listen for completion events with debouncing
    let updateTimeout: NodeJS.Timeout | null = null;

    const handleUpdate = async () => {
      // Debounce updates to prevent counting the same data multiple times
      if (updateTimeout) {
        clearTimeout(updateTimeout);
      }

      updateTimeout = setTimeout(async () => {
        const problems = await getCompletedProblems();
        setCompletedProblems(problems);

        const newProgress: Record<
          string,
          { completed: number; total: number }
        > = {};
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

        // Update discussion stats (efficient - use stats API)
        const userStats = await getUserStats();

        if (userStats) {
          // Authenticated: Use efficient stats API
          setCompletedDiscussions(userStats.completedDiscussionQuestions);
        } else {
          // Anonymous: Use IndexedDB (local, fast)
          const { getCompletedDiscussionQuestionsCount } = await import(
            '@/lib/helpers/indexeddb'
          );
          const completed = await getCompletedDiscussionQuestionsCount();
          setCompletedDiscussions(completed);
        }

        // Note: Module-specific discussion progress updated lazily, not on every change

        // Update multiple choice stats
        const mcCompleted =
          getCompletedMultipleChoiceQuestionsCount(moduleCategories);
        setCompletedMultipleChoice(mcCompleted);

        // Update multiple choice progress per module
        const mcProgress: Record<string, { completed: number; total: number }> =
          {};

        for (const moduleCategory of moduleCategories) {
          let completedCount = 0;
          let totalCount = 0;

          for (const section of moduleCategory.module.sections) {
            if (section.multipleChoice) {
              totalCount += section.multipleChoice.length;

              const storageKey = `mc-quiz-${moduleCategory.id}-${section.id}`;
              const stored = localStorage.getItem(storageKey);
              if (stored) {
                try {
                  const completedQuestions = JSON.parse(stored);
                  // Deduplicate in case of corrupted data
                  const uniqueQuestions = [...new Set(completedQuestions)];

                  // Fix corrupted data if duplicates found
                  if (uniqueQuestions.length !== completedQuestions.length) {
                    localStorage.setItem(
                      storageKey,
                      JSON.stringify(uniqueQuestions),
                    );
                    console.warn(
                      `Fixed duplicates in ${storageKey}: ${completedQuestions.length} → ${uniqueQuestions.length}`,
                    );
                  }

                  completedCount += uniqueQuestions.length;
                } catch (e) {
                  console.error('Failed to parse MC quiz progress:', e);
                }
              }
            }
          }

          mcProgress[moduleCategory.id] = {
            completed: completedCount,
            total: totalCount,
          };
        }

        setModuleMultipleChoiceProgress(mcProgress);
      }, 100); // 100ms debounce
    };

    window.addEventListener('focus', handleUpdate);
    window.addEventListener('storage', handleUpdate);
    window.addEventListener('problemCompleted', handleUpdate);
    window.addEventListener('problemReset', handleUpdate);
    window.addEventListener('mcQuizUpdated', handleUpdate);

    return () => {
      if (updateTimeout) clearTimeout(updateTimeout);
      window.removeEventListener('focus', handleUpdate);
      window.removeEventListener('storage', handleUpdate);
      window.removeEventListener('problemCompleted', handleUpdate);
      window.removeEventListener('problemReset', handleUpdate);
      window.removeEventListener('mcQuizUpdated', handleUpdate);
    };
  }, []);

  // Calculate completed sub-modules (sections) across all modules
  const completedSectionsCount = Object.values(moduleProgress).reduce(
    (sum, progress) => sum + progress.completed,
    0,
  );
  const totalSectionsCount = Object.values(moduleProgress).reduce(
    (sum, progress) => sum + progress.total,
    0,
  );

  // Calculate overall progress percentage
  const totalCompleted =
    completedSectionsCount +
    completedMultipleChoice +
    completedDiscussions +
    completedProblems.size;
  const totalItems =
    totalSectionsCount + totalMultipleChoice + totalDiscussions + totalProblems;
  const overallProgressPercent =
    totalItems > 0 ? ((totalCompleted / totalItems) * 100).toFixed(2) : '0.00';

  // Get the selected topic section
  const selectedSection = topicSections.find(
    (section) => section.id === selectedSectionId,
  );

  // Handler for section change with smooth scroll
  const handleSectionChange = (sectionId: string) => {
    if (sectionId === selectedSectionId) return;

    setSelectedSectionId(sectionId);

    // Smooth scroll to top of content
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Don't render content until we've loaded the saved selection
  if (!isInitialized || !selectedSectionId) {
    return (
      <div className="container mx-auto max-w-[1400px] px-2 py-6 sm:py-8 lg:py-12">
        <div className="flex items-center justify-center py-12">
          <div className="text-[#6272a4]">Loading...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto max-w-[1400px] px-2 py-6 sm:py-8 lg:py-12">
      {/* Stats */}
      <div className="mb-6 grid grid-cols-2 gap-3 sm:mb-8 sm:gap-4 md:grid-cols-4 md:gap-6">
        {/* Sub-Modules (Sections) Completed */}
        <div className="rounded-lg border-2 border-[#bd93f9] bg-[#bd93f9]/10 p-3 text-center sm:p-4 lg:p-6">
          <div className="mb-1 text-2xl font-bold text-[#bd93f9] sm:mb-2 sm:text-3xl lg:text-4xl">
            {completedSectionsCount} / {totalSectionsCount}
          </div>
          <div className="text-xs font-medium text-[#f8f8f2] sm:text-sm lg:text-base">
            Sections
          </div>
        </div>

        {/* Multiple Choice Questions */}
        <div className="rounded-lg border-2 border-[#8be9fd] bg-[#8be9fd]/10 p-3 text-center sm:p-4 lg:p-6">
          <div className="mb-1 text-2xl font-bold text-[#8be9fd] sm:mb-2 sm:text-3xl lg:text-4xl">
            {completedMultipleChoice} / {totalMultipleChoice}
          </div>
          <div className="text-xs font-medium text-[#f8f8f2] sm:text-sm lg:text-base">
            Multiple Choice
          </div>
        </div>

        {/* Discussion Questions */}
        <div className="rounded-lg border-2 border-[#f1fa8c] bg-[#f1fa8c]/10 p-3 text-center sm:p-4 lg:p-6">
          <div className="mb-1 text-2xl font-bold text-[#f1fa8c] sm:mb-2 sm:text-3xl lg:text-4xl">
            {completedDiscussions} / {totalDiscussions}
          </div>
          <div className="text-xs font-medium text-[#f8f8f2] sm:text-sm lg:text-base">
            Discussions
          </div>
        </div>

        {/* Problems Solved */}
        <div className="rounded-lg border-2 border-[#50fa7b] bg-[#50fa7b]/10 p-3 text-center sm:p-4 lg:p-6">
          <div className="mb-1 text-2xl font-bold text-[#50fa7b] sm:mb-2 sm:text-3xl lg:text-4xl">
            {completedProblems.size} / {totalProblems}
          </div>
          <div className="text-xs font-medium text-[#f8f8f2] sm:text-sm lg:text-base">
            Problems
          </div>
        </div>
      </div>

      {/* Overall Progress Bar */}
      <div className="mb-6 sm:mb-8">
        <div className="mb-2 flex items-center justify-between text-sm sm:text-base">
          <span className="font-semibold text-[#f8f8f2]">
            Overall Progress: {totalCompleted} / {totalItems}
          </span>
          <span className="font-bold text-[#bd93f9]">
            {overallProgressPercent}%
          </span>
        </div>
        <div className="h-3 overflow-hidden rounded-full bg-[#282a36] shadow-inner sm:h-4">
          <div
            className="h-full rounded-full bg-gradient-to-r from-[#bd93f9] via-[#ff79c6] to-[#50fa7b] transition-all duration-500 ease-out"
            style={{ width: `${overallProgressPercent}%` }}
          />
        </div>
      </div>

      {/* Header */}
      <div className="mb-4 flex flex-col items-start justify-between gap-3 sm:mb-6 sm:flex-row sm:items-center">
        <h2 className="text-2xl font-bold text-[#f8f8f2] sm:text-3xl">
          📚 Learning Path
        </h2>
        <Link
          href="/problems"
          className="w-full rounded-lg bg-[#bd93f9] px-4 py-2 text-center text-sm font-semibold text-[#282a36] transition-colors hover:bg-[#ff79c6] sm:w-auto sm:px-6 sm:py-2.5 sm:text-base"
        >
          View All Problems →
        </Link>
      </div>

      {/* Two-Pane Layout */}
      <div className="flex flex-col gap-4 sm:gap-6 lg:flex-row lg:items-start">
        {/* Left Sidebar - Section Navigation */}
        <div className="scrollbar-hide lg:sticky lg:top-4 lg:h-[calc(100vh-8rem)] lg:w-64 lg:flex-shrink-0 lg:overflow-y-auto">
          <div className="flex gap-2 overflow-x-auto pb-2 lg:flex-col lg:space-y-2 lg:overflow-visible lg:pb-0">
            {topicSections.map((topicSection) => {
              const isSelected = topicSection.id === selectedSectionId;
              const sectionModuleCount = topicSection.modules.length;

              return (
                <button
                  key={topicSection.id}
                  onClick={() => handleSectionChange(topicSection.id)}
                  className={`min-w-[160px] flex-shrink-0 rounded-lg border-2 p-3 text-left transition-all sm:min-w-0 sm:p-4 lg:w-full ${
                    isSelected
                      ? 'border-[#bd93f9] bg-[#bd93f9]/20 shadow-lg'
                      : 'border-[#44475a] bg-[#44475a]/50 hover:border-[#bd93f9]/50 hover:bg-[#44475a]'
                  }`}
                >
                  <div className="flex items-center gap-2 sm:gap-3">
                    <div className="text-xl sm:text-2xl">
                      {topicSection.icon}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-bold text-[#f8f8f2] sm:text-base">
                        {topicSection.title}
                      </div>
                      <div className="text-xs text-[#f8f8f2]/60">
                        {sectionModuleCount} module
                        {sectionModuleCount !== 1 ? 's' : ''}
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Content - Modules in Selected Section */}
        <div className="min-w-0 flex-1 lg:overflow-y-auto">
          {selectedSection && (
            <div className="space-y-4 sm:space-y-6">
              {/* Section Header */}
              <div className="rounded-lg border-2 border-[#bd93f9] bg-[#bd93f9]/10 p-4 sm:p-6">
                <h3 className="text-xl font-bold text-[#f8f8f2] sm:text-2xl">
                  {selectedSection.icon} {selectedSection.title}
                </h3>
              </div>

              {/* Modules in this section */}
              {selectedSection.modules.map((moduleCategory, index) => {
                const progress = moduleProgress[moduleCategory.id] || {
                  completed: 0,
                  total: moduleCategory.module.sections.length,
                };
                const progressPercent =
                  progress.total > 0
                    ? Math.round((progress.completed / progress.total) * 100)
                    : 0;
                const isCompleted =
                  progress.completed > 0 &&
                  progress.completed === progress.total;

                // Get problems progress for this topic (match by title, not id)
                const problemsProgress = getTopicProblemsProgress(
                  moduleCategory.title,
                );
                const problemsPercent =
                  problemsProgress.total > 0
                    ? Math.round(
                        (problemsProgress.completed / problemsProgress.total) *
                          100,
                      )
                    : 0;

                // Get discussion progress for this module
                // Calculate total from module data
                let discussionTotal = 0;
                moduleCategory.module.sections.forEach(
                  (section: ModuleSection) => {
                    if (section.quiz) {
                      discussionTotal += section.quiz.length;
                    }
                  },
                );

                const discussionProgress = moduleDiscussionProgress[
                  moduleCategory.id
                ] || {
                  completed: 0,
                  total: discussionTotal,
                };

                // Ensure we use the calculated total if state doesn't have it yet
                const finalDiscussionProgress = {
                  completed: discussionProgress.completed,
                  total: discussionTotal,
                };

                const discussionPercent =
                  finalDiscussionProgress.total > 0
                    ? Math.round(
                        (finalDiscussionProgress.completed /
                          finalDiscussionProgress.total) *
                          100,
                      )
                    : 0;

                // Get multiple choice progress for this module
                // Calculate total from module data
                let mcTotal = 0;
                moduleCategory.module.sections.forEach(
                  (section: ModuleSection) => {
                    if (section.multipleChoice) {
                      mcTotal += section.multipleChoice.length;
                    }
                  },
                );

                const mcProgress = moduleMultipleChoiceProgress[
                  moduleCategory.id
                ] || {
                  completed: 0,
                  total: mcTotal,
                };

                // Ensure we use the calculated total if state doesn't have it yet
                const finalMcProgress = {
                  completed: mcProgress.completed,
                  total: mcTotal,
                };

                const mcPercent =
                  finalMcProgress.total > 0
                    ? Math.round(
                        (finalMcProgress.completed / finalMcProgress.total) *
                          100,
                      )
                    : 0;

                return (
                  <div
                    key={moduleCategory.id}
                    className="rounded-xl border-2 border-[#44475a] bg-[#44475a] p-4 shadow-lg sm:p-6"
                  >
                    <div className="flex flex-col gap-3 sm:gap-4 lg:flex-row lg:items-center lg:gap-6">
                      {/* Number and Icon */}
                      <div className="flex items-center gap-3 sm:gap-4">
                        <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-[#bd93f9] text-lg font-bold text-[#282a36] sm:h-12 sm:w-12 sm:text-xl">
                          {index + 1}
                        </div>
                        <div className="text-3xl sm:text-4xl">
                          {moduleCategory.icon}
                        </div>
                      </div>

                      {/* Content */}
                      <div className="min-w-0 flex-1">
                        <div className="mb-1 flex flex-wrap items-center gap-2">
                          <h3 className="text-lg font-bold text-[#f8f8f2] sm:text-xl">
                            {moduleCategory.title}
                          </h3>
                          {isCompleted && (
                            <span className="rounded-full border-2 border-[#50fa7b] bg-[#50fa7b]/20 px-2 py-0.5 text-xs font-semibold text-[#50fa7b]">
                              ✓ Module
                            </span>
                          )}
                          {finalMcProgress.completed ===
                            finalMcProgress.total &&
                            finalMcProgress.total > 0 && (
                              <span className="rounded-full border-2 border-[#8be9fd] bg-[#8be9fd]/20 px-2 py-0.5 text-xs font-semibold text-[#8be9fd]">
                                ✓ MC
                              </span>
                            )}
                          {finalDiscussionProgress.completed ===
                            finalDiscussionProgress.total &&
                            finalDiscussionProgress.total > 0 && (
                              <span className="rounded-full border-2 border-[#f1fa8c] bg-[#f1fa8c]/20 px-2 py-0.5 text-xs font-semibold text-[#f1fa8c]">
                                ✓ Discuss
                              </span>
                            )}
                          {/* Hide "All Problems Solved" badge for system design modules */}
                          {selectedSection.id !== 'system-design' &&
                            problemsProgress.completed ===
                              problemsProgress.total &&
                            problemsProgress.total > 0 && (
                              <span className="rounded-full border-2 border-[#50fa7b] bg-[#50fa7b]/20 px-2 py-0.5 text-xs font-semibold text-[#50fa7b]">
                                ✓ Problems
                              </span>
                            )}
                        </div>
                        <p className="text-xs text-[#f8f8f2] sm:text-sm">
                          {moduleCategory.description}
                        </p>
                      </div>

                      {/* Metadata and Actions */}
                      <div className="flex flex-wrap items-center gap-2 sm:gap-3 lg:flex-shrink-0">
                        <div className="rounded-full border-2 border-[#f8f8f2] bg-[#f8f8f2]/10 px-2 py-1 text-xs font-semibold text-[#f8f8f2] sm:px-3">
                          {moduleCategory.module.sections.length} sections
                        </div>
                        {/* Hide problems count for system design modules */}
                        {selectedSection.id !== 'system-design' && (
                          <div className="rounded-full border-2 border-[#f8f8f2] bg-[#f8f8f2]/10 px-2 py-1 text-xs font-semibold text-[#f8f8f2] sm:px-3">
                            {problemsProgress.total} problems
                          </div>
                        )}
                        <Link
                          href={`/modules/${moduleCategory.id}`}
                          className="rounded-lg bg-[#bd93f9] px-3 py-1.5 text-xs font-semibold text-[#282a36] transition-colors hover:bg-[#ff79c6] sm:px-4 sm:py-2 sm:text-sm"
                        >
                          Learn
                        </Link>
                        {/* Hide Practice button for system design modules */}
                        {selectedSection.id !== 'system-design' && (
                          <Link
                            href={`/topics/${moduleCategory.id}`}
                            className="rounded-lg border-2 border-[#bd93f9] bg-transparent px-3 py-1.5 text-xs font-semibold text-[#bd93f9] transition-colors hover:bg-[#bd93f9] hover:text-[#282a36] sm:px-4 sm:py-2 sm:text-sm"
                          >
                            Practice
                          </Link>
                        )}
                      </div>
                    </div>

                    {/* Progress Bars - Always show */}
                    <div
                      className={`mt-4 grid gap-3 ${selectedSection.id === 'system-design' ? 'md:grid-cols-3' : 'md:grid-cols-2 lg:grid-cols-4'}`}
                    >
                      {/* Module Progress */}
                      <div>
                        <div className="mb-1 flex items-center justify-between text-xs">
                          <span className="font-semibold text-[#bd93f9]">
                            📚 Modules: {progress.completed} / {progress.total}
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

                      {/* Multiple Choice Progress */}
                      <div>
                        <div className="mb-1 flex items-center justify-between text-xs">
                          <span className="font-semibold text-[#8be9fd]">
                            📝 Multiple Choice: {finalMcProgress.completed} /{' '}
                            {finalMcProgress.total}
                          </span>
                          <span className="text-[#6272a4]">{mcPercent}%</span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-[#282a36]">
                          <div
                            className="h-full rounded-full bg-[#8be9fd] transition-all duration-300"
                            style={{ width: `${mcPercent}%` }}
                          />
                        </div>
                      </div>

                      {/* Discussion Progress */}
                      <div>
                        <div className="mb-1 flex items-center justify-between text-xs">
                          <span className="font-semibold text-[#f1fa8c]">
                            🎥 Discussions: {finalDiscussionProgress.completed}{' '}
                            / {finalDiscussionProgress.total}
                          </span>
                          <span className="text-[#6272a4]">
                            {discussionPercent}%
                          </span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-[#282a36]">
                          <div
                            className="h-full rounded-full bg-[#f1fa8c] transition-all duration-300"
                            style={{ width: `${discussionPercent}%` }}
                          />
                        </div>
                      </div>

                      {/* Problems Progress - Hide for system design modules */}
                      {selectedSection.id !== 'system-design' && (
                        <div>
                          <div className="mb-1 flex items-center justify-between text-xs">
                            <span className="font-semibold text-[#50fa7b]">
                              💻 Problems: {problemsProgress.completed} /{' '}
                              {problemsProgress.total}
                            </span>
                            <span className="text-[#6272a4]">
                              {problemsPercent}%
                            </span>
                          </div>
                          <div className="h-1.5 overflow-hidden rounded-full bg-[#282a36]">
                            <div
                              className="h-full rounded-full bg-[#50fa7b] transition-all duration-300"
                              style={{ width: `${problemsPercent}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

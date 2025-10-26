'use client';

import { notFound } from 'next/navigation';
import Link from 'next/link';
import { use, ReactElement, ReactNode, useState, useEffect } from 'react';

import { getModuleById } from '@/lib/content/topics';
import { allProblems } from '@/lib/content/problems';
import { getCompletedProblems } from '@/lib/helpers/storage';
import { formatTextWithMath } from '@/lib/utils/formatTextWithMath';
import { InteractiveCodeBlock } from '@/components/InteractiveCodeBlock';
import { VideoRecorder } from '@/components/VideoRecorder';
import { MultipleChoiceQuiz } from '@/components/MultipleChoiceQuiz';
import {
  saveVideo,
  getVideosForQuestion,
  deleteVideo,
} from '@/lib/helpers/indexeddb';

/**
 * Formats sample answer text with markdown support (headers, lists, code blocks, bold, etc.)
 */
function formatSampleAnswer(text: string | undefined): ReactNode[] {
  if (!text) return [];
  const elements: ReactNode[] = [];
  let elementKey = 0;

  // Extract code blocks first
  const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
  const codeBlocks: Array<{ language: string; code: string }> = [];
  let match;

  while ((match = codeBlockRegex.exec(text)) !== null) {
    codeBlocks.push({
      language: match[1] || 'javascript',
      code: match[2].trim(),
    });
  }

  // Replace code blocks with placeholders
  let processedText = text;
  codeBlocks.forEach((_, index) => {
    processedText = processedText.replace(
      /```\w+?\n[\s\S]*?```/,
      `__CODE_BLOCK_${index}__`,
    );
  });

  // Split by double newlines to get paragraphs
  const paragraphs = processedText.split('\n\n');

  paragraphs.forEach((paragraph) => {
    const trimmed = paragraph.trim();
    if (!trimmed) return;

    // Check if this is a code block placeholder
    const codeBlockMatch = trimmed.match(/^__CODE_BLOCK_(\d+)__$/);
    if (codeBlockMatch) {
      const blockIndex = parseInt(codeBlockMatch[1]);
      const block = codeBlocks[blockIndex];
      elements.push(
        <div key={`code-${elementKey++}`} className="my-4">
          {formatTextWithMath(`\`\`\`${block.language}\n${block.code}\`\`\``)}
        </div>,
      );
      return;
    }

    // Process line by line for other markdown elements
    const lines = paragraph.split('\n');
    let listItems: string[] = [];
    let numberedListItems: string[] = [];

    const flushList = () => {
      if (listItems.length > 0) {
        elements.push(
          <ul
            key={`list-${elementKey++}`}
            className="mb-4 ml-6 list-disc space-y-2"
          >
            {listItems.map((item, idx) => (
              <li key={idx} className="text-[#f8f8f2]">
                {formatTextWithMath(item)}
              </li>
            ))}
          </ul>,
        );
        listItems = [];
      }
      if (numberedListItems.length > 0) {
        elements.push(
          <ol
            key={`numlist-${elementKey++}`}
            className="mb-4 ml-6 list-decimal space-y-2"
          >
            {numberedListItems.map((item, idx) => (
              <li key={idx} className="text-[#f8f8f2]">
                {formatTextWithMath(item)}
              </li>
            ))}
          </ol>,
        );
        numberedListItems = [];
      }
    };

    lines.forEach((line) => {
      const trimmedLine = line.trim();

      // Handle headers
      if (trimmedLine.startsWith('### ')) {
        flushList();
        elements.push(
          <h4
            key={`h4-${elementKey++}`}
            className="mt-3 mb-2 text-base font-semibold text-[#bd93f9]"
          >
            {formatTextWithMath(trimmedLine.substring(4))}
          </h4>,
        );
      } else if (trimmedLine.startsWith('## ')) {
        flushList();
        elements.push(
          <h3
            key={`h3-${elementKey++}`}
            className="mt-4 mb-3 text-lg font-semibold text-[#50fa7b] first:mt-0"
          >
            {formatTextWithMath(trimmedLine.substring(3))}
          </h3>,
        );
      } else if (trimmedLine.startsWith('# ')) {
        flushList();
        elements.push(
          <h2
            key={`h2-${elementKey++}`}
            className="mt-5 mb-3 text-xl font-bold text-[#8be9fd] first:mt-0"
          >
            {formatTextWithMath(trimmedLine.substring(2))}
          </h2>,
        );
      }
      // Handle horizontal rules
      else if (trimmedLine === '---' || trimmedLine === '***') {
        flushList();
        elements.push(
          <hr key={`hr-${elementKey++}`} className="my-4 border-[#6272a4]" />,
        );
      }
      // Handle unordered list items (-, *, or •)
      else if (trimmedLine.match(/^[-*•]\s+/)) {
        const content = trimmedLine.replace(/^[-*•]\s+/, '');
        listItems.push(content);
      }
      // Handle numbered lists
      else if (trimmedLine.match(/^\d+\.\s+/)) {
        // If we have unordered items, flush them first
        if (listItems.length > 0) {
          flushList();
        }
        const content = trimmedLine.replace(/^\d+\.\s+/, '');
        numberedListItems.push(content);
      }
      // Handle indented sub-items (for nested lists)
      else if (trimmedLine.match(/^\s{2,}[-*•]\s+/) && listItems.length > 0) {
        const content = trimmedLine.replace(/^\s+[-*•]\s+/, '  • ');
        listItems.push(content);
      }
      // Handle regular text
      else if (trimmedLine !== '') {
        flushList();
        elements.push(
          <div
            key={`line-${elementKey++}`}
            className="mb-2 leading-relaxed text-[#f8f8f2]"
          >
            {formatTextWithMath(trimmedLine)}
          </div>,
        );
      }
    });

    // Flush any remaining list items
    flushList();
  });

  return elements;
}

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

  // State for selected section (first section by default)
  const [selectedSectionId, setSelectedSectionId] = useState<string>(
    moduleData.sections[0]?.id || '',
  );

  // State for completed sections
  const [completedSections, setCompletedSections] = useState<Set<string>>(
    new Set(),
  );

  // State for showing quiz solutions
  const [showSolutions, setShowSolutions] = useState<Set<string>>(new Set());

  // State for video recordings - maps questionId to array of {id, url}
  const [videoUrls, setVideoUrls] = useState<
    Record<string, Array<{ id: string; url: string }>>
  >({});

  // State for tracking various progress types
  const [completedProblems, setCompletedProblems] = useState<Set<string>>(
    new Set(),
  );
  const [completedMultipleChoice, setCompletedMultipleChoice] = useState(0);
  const [completedDiscussions, setCompletedDiscussions] = useState(0);

  // Load videos from IndexedDB
  useEffect(() => {
    const loadVideos = async () => {
      const urlsMap: Record<string, Array<{ id: string; url: string }>> = {};
      for (const [sectionIndex, section] of moduleData.sections.entries()) {
        const sectionId = section.id || `section-${sectionIndex}`;
        if (section.quiz) {
          for (const question of section.quiz) {
            const questionId = `${slug}-${sectionId}-${question.id}`;
            const videos = await getVideosForQuestion(questionId);
            if (videos.length > 0) {
              urlsMap[questionId] = videos.map((video) => ({
                id: video.id,
                url: URL.createObjectURL(video.blob),
              }));
            }
          }
        }
      }
      setVideoUrls(urlsMap);
    };
    loadVideos();

    // Cleanup video URLs on unmount
    return () => {
      Object.values(videoUrls).forEach((videos) => {
        videos.forEach((video) => URL.revokeObjectURL(video.url));
      });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slug]);

  // Handle video save
  const handleVideoSave = async (
    questionId: string,
    videoBlob: Blob,
    videoId: string,
  ) => {
    try {
      await saveVideo(videoId, videoBlob);
      const url = URL.createObjectURL(videoBlob);
      setVideoUrls((prev) => ({
        ...prev,
        [questionId]: [...(prev[questionId] || []), { id: videoId, url }],
      }));
    } catch (error) {
      console.error('Failed to save video:', error);
    }
  };

  // Handle video delete
  const handleVideoDelete = async (questionId: string, videoId: string) => {
    try {
      await deleteVideo(videoId);

      setVideoUrls((prev) => {
        const videos = prev[questionId] || [];
        const videoToDelete = videos.find((v) => v.id === videoId);
        if (videoToDelete) {
          URL.revokeObjectURL(videoToDelete.url);
        }

        const newVideos = videos.filter((v) => v.id !== videoId);
        const newUrls = { ...prev };

        if (newVideos.length === 0) {
          delete newUrls[questionId];
        } else {
          newUrls[questionId] = newVideos;
        }

        return newUrls;
      });
    } catch (error) {
      console.error('Failed to delete video:', error);
    }
  };

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

  // Load progress data for all types
  useEffect(() => {
    const loadProgress = async () => {
      // Load completed problems
      setCompletedProblems(getCompletedProblems());

      // Load multiple choice progress
      let mcCompleted = 0;
      for (const [sectionIndex, section] of moduleData.sections.entries()) {
        const sectionId = section.id || `section-${sectionIndex}`;
        if (section.multipleChoice && section.multipleChoice.length > 0) {
          const storageKey = `mc-quiz-${slug}-${sectionId}`;
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

              mcCompleted += uniqueQuestions.length;
            } catch (e) {
              console.error('Failed to parse MC quiz progress:', e);
            }
          }
        }
      }
      setCompletedMultipleChoice(mcCompleted);

      // Load discussion progress (count videos)
      let discussionCompleted = 0;
      for (const [sectionIndex, section] of moduleData.sections.entries()) {
        const sectionId = section.id || `section-${sectionIndex}`;
        if (section.quiz) {
          for (const question of section.quiz) {
            const questionId = `${slug}-${sectionId}-${question.id}`;
            const videos = await getVideosForQuestion(questionId);
            if (videos.length > 0) {
              discussionCompleted++;
            }
          }
        }
      }
      setCompletedDiscussions(discussionCompleted);
    };

    loadProgress();

    // Listen for updates with debouncing to prevent multiple rapid updates
    let updateTimeout: NodeJS.Timeout | null = null;

    const handleUpdate = async () => {
      // Debounce updates to prevent counting the same data multiple times
      if (updateTimeout) {
        clearTimeout(updateTimeout);
      }

      updateTimeout = setTimeout(async () => {
        setCompletedProblems(getCompletedProblems());

        // Update multiple choice
        let mcCompleted = 0;
        for (const [sectionIndex, section] of moduleData.sections.entries()) {
          const sectionId = section.id || `section-${sectionIndex}`;
          if (section.multipleChoice && section.multipleChoice.length > 0) {
            const storageKey = `mc-quiz-${slug}-${sectionId}`;
            const stored = localStorage.getItem(storageKey);
            if (stored) {
              try {
                const completedQuestions = JSON.parse(stored);
                // Deduplicate to ensure clean count
                const uniqueQuestions = [...new Set(completedQuestions)];
                mcCompleted += uniqueQuestions.length;
              } catch (e) {
                console.error('Failed to parse MC quiz progress:', e);
              }
            }
          }
        }
        setCompletedMultipleChoice(mcCompleted);

        // Update discussion progress
        let discussionCompleted = 0;
        for (const [sectionIndex, section] of moduleData.sections.entries()) {
          const sectionId = section.id || `section-${sectionIndex}`;
          if (section.quiz) {
            for (const question of section.quiz) {
              const questionId = `${slug}-${sectionId}-${question.id}`;
              const videos = await getVideosForQuestion(questionId);
              if (videos.length > 0) {
                discussionCompleted++;
              }
            }
          }
        }
        setCompletedDiscussions(discussionCompleted);
      }, 100); // 100ms debounce
    };

    // Set up less aggressive polling (1 second instead of 300ms)
    // This catches changes from other tabs/windows
    const pollInterval = setInterval(handleUpdate, 1000);

    window.addEventListener('focus', handleUpdate);
    window.addEventListener('storage', handleUpdate);
    window.addEventListener('problemCompleted', handleUpdate);
    window.addEventListener('problemReset', handleUpdate);
    window.addEventListener('mcQuizUpdated', handleUpdate);

    return () => {
      if (updateTimeout) clearTimeout(updateTimeout);
      clearInterval(pollInterval);
      window.removeEventListener('focus', handleUpdate);
      window.removeEventListener('storage', handleUpdate);
      window.removeEventListener('problemCompleted', handleUpdate);
      window.removeEventListener('problemReset', handleUpdate);
      window.removeEventListener('mcQuizUpdated', handleUpdate);
    };
  }, [slug, moduleData.sections]);

  // Get selected section
  const selectedSection = moduleData.sections.find(
    (s, index) => (s.id || `section-${index}`) === selectedSectionId,
  );

  const selectedSectionIndex = moduleData.sections.findIndex(
    (s, index) => (s.id || `section-${index}`) === selectedSectionId,
  );

  const selectedSectionIdResolved = selectedSection
    ? selectedSection.id || `section-${selectedSectionIndex}`
    : selectedSectionId;

  // Save completed sections to localStorage
  const toggleSectionComplete = (sectionId: string) => {
    setCompletedSections((prev) => {
      const newSet = new Set(prev);
      const wasCompleted = newSet.has(sectionId);

      if (wasCompleted) {
        newSet.delete(sectionId);
      } else {
        newSet.add(sectionId);
      }

      // Save to localStorage
      const storageKey = `module-${slug}-completed`;
      localStorage.setItem(storageKey, JSON.stringify(Array.from(newSet)));

      return newSet;
    });
  };

  const toggleSolution = (sectionId: string, questionId: string | number) => {
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

  // Calculate totals for progress tracking
  const totalSections = moduleData.sections.length;
  const totalMultipleChoice = moduleData.sections.reduce(
    (sum, section) => sum + (section.multipleChoice?.length || 0),
    0,
  );
  const totalDiscussions = moduleData.sections.reduce(
    (sum, section) => sum + (section.quiz?.length || 0),
    0,
  );
  const totalProblems = allProblems.filter(
    (p) => p.topic === moduleData.title,
  ).length;
  const completedProblemsCount = allProblems.filter(
    (p) => p.topic === moduleData.title && completedProblems.has(p.id),
  ).length;

  // Calculate percentages
  const sectionsPercent =
    totalSections > 0
      ? Math.round((completedSections.size / totalSections) * 100)
      : 0;
  const mcPercent =
    totalMultipleChoice > 0
      ? Math.round((completedMultipleChoice / totalMultipleChoice) * 100)
      : 0;
  const discussionsPercent =
    totalDiscussions > 0
      ? Math.round((completedDiscussions / totalDiscussions) * 100)
      : 0;
  const problemsPercent =
    totalProblems > 0
      ? Math.round((completedProblemsCount / totalProblems) * 100)
      : 0;

  return (
    <div className="container mx-auto max-w-7xl px-4 py-4 sm:py-8">
      {/* Back Link */}
      <Link
        href="/"
        className="mb-4 inline-flex items-center text-sm font-medium text-[#6272a4] transition-colors hover:text-[#bd93f9] sm:mb-6"
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
      <div className="mb-6 sm:mb-8">
        <div className="mb-4 flex flex-col items-start gap-3 sm:flex-row sm:items-center sm:gap-4">
          <span className="text-4xl sm:text-5xl">{moduleData.icon}</span>
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-[#f8f8f2] sm:text-3xl">
              {moduleData.title}
            </h1>
            <p className="mt-1 text-sm text-[#6272a4] sm:text-base">
              {moduleData.description}
            </p>
          </div>
          {/* Hide Practice Problems button for system design modules */}
          {!slug.startsWith('system-design') && (
            <Link
              href={`/topics/${moduleData.id}?from=modules/${slug}`}
              className="w-full rounded-lg bg-[#bd93f9] px-4 py-2 text-center font-semibold text-[#282a36] transition-colors hover:bg-[#ff79c6] sm:w-auto sm:px-6"
            >
              📝 Practice Problems →
            </Link>
          )}
        </div>

        {/* Progress Bars */}
        <div className="rounded-lg border-2 border-[#6272a4] bg-[#6272a4]/10 p-4">
          <div
            className={`grid gap-3 ${slug.startsWith('system-design') ? 'md:grid-cols-3' : 'md:grid-cols-2 lg:grid-cols-4'}`}
          >
            {/* Module Sections Progress */}
            <div>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="font-semibold text-[#bd93f9]">
                  📚 Modules: {completedSections.size} / {totalSections}
                </span>
                <span className="text-[#6272a4]">{sectionsPercent}%</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-[#282a36]">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-[#bd93f9] to-[#ff79c6] transition-all duration-300"
                  style={{ width: `${sectionsPercent}%` }}
                />
              </div>
            </div>

            {/* Multiple Choice Progress */}
            <div>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="font-semibold text-[#8be9fd]">
                  📝 Multiple Choice: {completedMultipleChoice} /{' '}
                  {totalMultipleChoice}
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
                  🎥 Discussions: {completedDiscussions} / {totalDiscussions}
                </span>
                <span className="text-[#6272a4]">{discussionsPercent}%</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-[#282a36]">
                <div
                  className="h-full rounded-full bg-[#f1fa8c] transition-all duration-300"
                  style={{ width: `${discussionsPercent}%` }}
                />
              </div>
            </div>

            {/* Problems Progress - Hide for system design modules */}
            {!slug.startsWith('system-design') && (
              <div>
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span className="font-semibold text-[#50fa7b]">
                    💻 Problems: {completedProblemsCount} / {totalProblems}
                  </span>
                  <span className="text-[#6272a4]">{problemsPercent}%</span>
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
      </div>

      {/* Two-Pane Layout */}
      <div className="flex flex-col gap-4 sm:gap-6 lg:flex-row">
        {/* Left Pane: Section Navigation */}
        <div className="w-full flex-shrink-0 lg:w-80">
          <div className="rounded-lg border-2 border-[#44475a] bg-[#282a36] lg:sticky lg:top-4">
            <div className="border-b-2 border-[#44475a] p-3 sm:p-4">
              <h2 className="text-base font-bold text-[#f8f8f2] sm:text-lg">
                📚 Sections
              </h2>
            </div>
            <div className="max-h-[300px] overflow-y-auto p-2 lg:max-h-[calc(100vh-200px)]">
              {moduleData.sections.map((section, index) => {
                const sectionId = section.id || `section-${index}`;
                const isSelected = selectedSectionId === sectionId;
                const isCompleted = completedSections.has(sectionId);

                return (
                  <button
                    key={sectionId}
                    onClick={() => setSelectedSectionId(sectionId)}
                    className={`mb-2 w-full rounded-lg border-2 p-2 text-left transition-all sm:p-3 ${
                      isSelected
                        ? 'border-[#bd93f9] bg-[#bd93f9]/20'
                        : 'border-[#44475a] bg-[#44475a] hover:border-[#6272a4]'
                    }`}
                  >
                    <div className="flex items-center gap-2 sm:gap-3">
                      {/* Section Number */}
                      <div
                        className={`flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full text-sm font-bold sm:h-8 sm:w-8 sm:text-base ${
                          isSelected
                            ? 'bg-[#bd93f9] text-[#282a36]'
                            : 'bg-[#6272a4] text-[#f8f8f2]'
                        }`}
                      >
                        {index + 1}
                      </div>

                      {/* Section Title */}
                      <div className="min-w-0 flex-1">
                        <h3 className="truncate text-xs font-semibold text-[#f8f8f2] sm:text-sm">
                          {section.title}
                        </h3>
                      </div>

                      {/* Completed Indicator */}
                      {isCompleted && (
                        <svg
                          className="h-5 w-5 flex-shrink-0 text-[#50fa7b]"
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
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Pane: Section Content */}
        <div className="min-w-0 flex-1">
          {selectedSection && (
            <div className="rounded-lg border-2 border-[#44475a] bg-[#282a36] p-4 sm:p-6 lg:p-8">
              {/* Section Header */}
              <div className="mb-4 flex items-start justify-between gap-3 sm:mb-6">
                <div className="min-w-0 flex-1">
                  <h2 className="mb-2 text-xl font-bold text-[#f8f8f2] sm:text-2xl">
                    {selectedSection.title}
                  </h2>
                  {(moduleData.timeComplexity ||
                    moduleData.spaceComplexity) && (
                    <div className="flex flex-wrap gap-2">
                      {moduleData.timeComplexity && (
                        <span className="rounded-lg border border-[#bd93f9] bg-[#bd93f9]/10 px-2 py-1 text-xs font-semibold text-[#bd93f9]">
                          Time: {moduleData.timeComplexity}
                        </span>
                      )}
                      {moduleData.spaceComplexity && (
                        <span className="rounded-lg border border-[#50fa7b] bg-[#50fa7b]/10 px-2 py-1 text-xs font-semibold text-[#50fa7b]">
                          Space: {moduleData.spaceComplexity}
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {/* Completed Checkbox */}
                <button
                  onClick={() =>
                    toggleSectionComplete(selectedSectionIdResolved)
                  }
                  className={`flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg border-2 transition-colors sm:h-10 sm:w-10 ${
                    completedSections.has(selectedSectionIdResolved)
                      ? 'border-[#50fa7b] bg-[#50fa7b] text-[#282a36]'
                      : 'border-[#6272a4] bg-transparent text-transparent hover:border-[#50fa7b]'
                  }`}
                  title={
                    completedSections.has(selectedSectionIdResolved)
                      ? 'Mark as incomplete'
                      : 'Mark as completed'
                  }
                >
                  {completedSections.has(selectedSectionIdResolved) && (
                    <svg
                      className="h-5 w-5 sm:h-6 sm:w-6"
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
              </div>

              {/* Section Content */}
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

                      // Check for horizontal rule (---)
                      if (/^---+$/.test(line)) {
                        elements.push(
                          <hr
                            key={`hr-${elementKey++}`}
                            className="my-8 border-t-2 border-[#44475a]"
                          />,
                        );
                        i++;
                      }
                      // Check for headers (# ## ### #### etc)
                      else if (/^#{1,6}\s/.test(line)) {
                        const headerMatch = line.match(/^(#{1,6})\s+(.+)/);
                        if (headerMatch) {
                          const level = headerMatch[1].length;
                          const text = headerMatch[2];

                          // Render appropriate header based on level
                          if (level === 1) {
                            elements.push(
                              <h1
                                key={`header-${elementKey++}`}
                                className="mt-8 mb-6 text-3xl font-bold text-[#ff79c6]"
                              >
                                {formatTextWithMath(text)}
                              </h1>,
                            );
                          } else if (level === 2) {
                            elements.push(
                              <h2
                                key={`header-${elementKey++}`}
                                className="mt-6 mb-4 text-2xl font-bold text-[#bd93f9]"
                              >
                                {formatTextWithMath(text)}
                              </h2>,
                            );
                          } else if (level === 3) {
                            elements.push(
                              <h3
                                key={`header-${elementKey++}`}
                                className="mt-4 mb-3 text-xl font-bold text-[#8be9fd]"
                              >
                                {formatTextWithMath(text)}
                              </h3>,
                            );
                          } else if (level === 4) {
                            elements.push(
                              <h4
                                key={`header-${elementKey++}`}
                                className="mt-3 mb-2 text-lg font-semibold text-[#50fa7b]"
                              >
                                {formatTextWithMath(text)}
                              </h4>,
                            );
                          } else if (level === 5) {
                            elements.push(
                              <h5
                                key={`header-${elementKey++}`}
                                className="mt-2 mb-2 text-base font-semibold text-[#f1fa8c]"
                              >
                                {formatTextWithMath(text)}
                              </h5>,
                            );
                          } else {
                            elements.push(
                              <h6
                                key={`header-${elementKey++}`}
                                className="mt-2 mb-2 text-sm font-semibold text-[#ffb86c]"
                              >
                                {formatTextWithMath(text)}
                              </h6>,
                            );
                          }
                        }
                        i++;
                      }
                      // Check for numbered list (1. 2. 3. etc)
                      else if (/^\d+\.\s/.test(line)) {
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
                              <li key={idx}>{formatTextWithMath(item)}</li>
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
                              <li key={idx}>{formatTextWithMath(item)}</li>
                            ))}
                          </ul>,
                        );
                      }
                      // Regular paragraph
                      else if (line) {
                        // Collect all lines until next list, header, or empty line
                        let para = line;
                        i++;
                        while (
                          i < lines.length &&
                          lines[i].trim() &&
                          !/^(\d+\.|-|\*|#{1,6})\s/.test(lines[i].trim())
                        ) {
                          para += ' ' + lines[i].trim();
                          i++;
                        }
                        elements.push(
                          <p
                            key={`text-${elementKey++}`}
                            className="mb-4 leading-relaxed text-[#f8f8f2]"
                          >
                            {formatTextWithMath(para)}
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
                    const tableMatches = Array.from(text.matchAll(tableRegex));

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
                        const bodyLines = tableMatch[2].trim().split('\n');

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
                                        {formatTextWithMath(cell)}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>,
                        );

                        textIndex = tableMatch.index! + tableMatch[0].length;
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
                    selectedSection.content.matchAll(codeBlockRegex),
                  );

                  let lastIndex = 0;

                  codeMatches.forEach((match) => {
                    // Add text/tables before code block
                    if (match.index! > lastIndex) {
                      const textBefore = selectedSection.content
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
                  if (lastIndex < selectedSection.content.length) {
                    const textAfter = selectedSection.content
                      .slice(lastIndex)
                      .trim();
                    renderText(textAfter);
                  }

                  return elements;
                })()}
              </div>

              {/* Code Example */}
              {selectedSection.codeExample && (
                <div className="mt-4 sm:mt-6">
                  <div className="mb-2 text-sm font-semibold text-[#bd93f9] sm:text-base">
                    Code Example:
                  </div>
                  <InteractiveCodeBlock
                    code={selectedSection.codeExample}
                    language="python"
                  />
                </div>
              )}

              {/* Clear separation before quizzes */}
              {((selectedSection.multipleChoice &&
                selectedSection.multipleChoice.length > 0) ||
                (selectedSection.quiz && selectedSection.quiz.length > 0)) && (
                <div className="my-6 border-t-2 border-[#44475a] sm:my-8" />
              )}

              {/* Multiple Choice Quiz */}
              {selectedSection.multipleChoice &&
                selectedSection.multipleChoice.length > 0 && (
                  <div className="mb-6 sm:mb-8">
                    <div className="mb-4 flex items-center gap-2 sm:gap-3">
                      <div className="h-1 w-1 rounded-full bg-[#8be9fd]" />
                      <h3 className="text-lg font-bold text-[#8be9fd] sm:text-xl">
                        📝 Test Your Knowledge
                      </h3>
                      <div className="h-1 flex-1 bg-gradient-to-r from-[#8be9fd] to-transparent" />
                    </div>
                    <MultipleChoiceQuiz
                      questions={selectedSection.multipleChoice}
                      sectionId={selectedSectionIdResolved}
                      moduleId={slug}
                    />
                  </div>
                )}

              {/* Separator between quizzes */}
              {selectedSection.multipleChoice &&
                selectedSection.multipleChoice.length > 0 &&
                selectedSection.quiz &&
                selectedSection.quiz.length > 0 && (
                  <div className="my-6 border-t-2 border-[#44475a] sm:my-8" />
                )}

              {/* Discussion Questions Section */}
              {selectedSection.quiz && selectedSection.quiz.length > 0 && (
                <div>
                  <div className="mb-4 flex items-center gap-2 sm:gap-3">
                    <div className="h-1 w-1 rounded-full bg-[#f1fa8c]" />
                    <h3 className="text-lg font-bold text-[#f1fa8c] sm:text-xl">
                      💬 Discussion & Practice
                    </h3>
                    <div className="h-1 flex-1 bg-gradient-to-r from-[#f1fa8c] to-transparent" />
                  </div>
                  <div className="rounded-lg border-2 border-[#f1fa8c] bg-[#f1fa8c]/5 p-4 sm:p-6">
                    <p className="mb-4 text-xs text-[#6272a4] sm:mb-6 sm:text-sm">
                      Think through these questions or discuss them on camera.
                      Click to reveal sample answers.
                    </p>

                    <div className="space-y-4 sm:space-y-5">
                      {selectedSection.quiz.map((question, qIndex) => {
                        const questionKey = `${selectedSectionIdResolved}-${question.id}`;
                        const questionId = `${slug}-${selectedSectionIdResolved}-${question.id}`;
                        const showSolution = showSolutions.has(questionKey);

                        return (
                          <div
                            key={questionKey}
                            className="rounded-lg border-2 border-[#44475a] bg-[#282a36] p-3 sm:p-5"
                          >
                            <div className="mb-3 text-base leading-relaxed font-semibold text-[#f8f8f2] sm:mb-4 sm:text-lg">
                              <span className="mr-2 text-[#bd93f9]">
                                {qIndex + 1}.
                              </span>
                              {question.question}
                            </div>

                            {question.hint && !showSolution && (
                              <div className="mb-3 rounded-lg border border-[#6272a4] bg-[#6272a4]/10 p-2 text-xs text-[#8be9fd] sm:mb-4 sm:p-3 sm:text-sm">
                                <span className="font-semibold">💡 Hint: </span>
                                {question.hint}
                              </div>
                            )}

                            <button
                              onClick={() =>
                                toggleSolution(
                                  selectedSectionIdResolved,
                                  question.id,
                                )
                              }
                              className="rounded-lg bg-[#bd93f9] px-3 py-2 text-xs font-semibold text-[#282a36] transition-colors hover:bg-[#ff79c6] sm:px-4 sm:text-sm"
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
                                  <div className="mb-3 flex items-center gap-2 font-semibold text-[#50fa7b]">
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
                                  <div className="space-y-2 text-sm">
                                    {formatSampleAnswer(
                                      question.sampleAnswer || question.answer,
                                    )}
                                  </div>
                                </div>

                                {/* Key Points */}
                                {question.keyPoints &&
                                  question.keyPoints.length > 0 && (
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

                            {/* Video Recorder */}
                            <VideoRecorder
                              questionId={questionId}
                              existingVideos={videoUrls[questionId] || []}
                              onSave={(videoBlob, videoId) =>
                                handleVideoSave(questionId, videoBlob, videoId)
                              }
                              onDelete={(videoId) =>
                                handleVideoDelete(questionId, videoId)
                              }
                            />
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Key Takeaways */}
      <div className="mt-8 rounded-lg border-2 border-[#50fa7b] bg-[#50fa7b]/10 p-4 sm:mt-12 sm:p-6 lg:p-8">
        <h2 className="mb-3 text-xl font-bold text-[#50fa7b] sm:mb-4 sm:text-2xl">
          🎯 Key Takeaways
        </h2>
        <ul className="space-y-2 sm:space-y-3">
          {moduleData.keyTakeaways?.map((takeaway, index) => (
            <li key={index} className="flex items-start gap-2 sm:gap-3">
              <span className="mt-1 flex-shrink-0 text-[#50fa7b]">✓</span>
              <span className="text-sm text-[#f8f8f2] sm:text-base">
                {takeaway}
              </span>
            </li>
          ))}
        </ul>
      </div>

      {/* Bottom Navigation */}
      <div className="mt-8 flex flex-col justify-center gap-3 border-t-2 border-[#44475a] pt-6 sm:mt-12 sm:flex-row sm:gap-4 sm:pt-8">
        <Link
          href="/"
          className="rounded-lg bg-[#6272a4] px-4 py-3 text-center font-semibold text-[#f8f8f2] transition-colors hover:bg-[#6272a4]/80 sm:px-6"
        >
          ← All Topics
        </Link>
        {!slug.startsWith('system-design') && (
          <Link
            href={`/topics/${moduleData.id}?from=modules/${slug}`}
            className="rounded-lg bg-[#bd93f9] px-4 py-3 text-center font-semibold text-[#282a36] transition-colors hover:bg-[#ff79c6] sm:px-6"
          >
            Practice {moduleData.title} Problems →
          </Link>
        )}
      </div>
    </div>
  );
}

'use client';

import { useState, useEffect } from 'react';
import { MultipleChoiceQuestion } from '@/lib/types';

interface MultipleChoiceQuizProps {
  questions: MultipleChoiceQuestion[];
  sectionId: string;
  moduleId: string;
}

export function MultipleChoiceQuiz({
  questions,
  sectionId,
  moduleId,
}: MultipleChoiceQuizProps) {
  const [selectedAnswers, setSelectedAnswers] = useState<
    Record<string, number>
  >({});
  const [showResults, setShowResults] = useState<Record<string, boolean>>({});
  const [completedQuestions, setCompletedQuestions] = useState<Set<string>>(
    new Set(),
  );

  // Load completed questions from localStorage
  useEffect(() => {
    const storageKey = `mc-quiz-${moduleId}-${sectionId}`;
    const stored = localStorage.getItem(storageKey);
    if (stored) {
      try {
        const completed = JSON.parse(stored);
        setCompletedQuestions(new Set(completed));
      } catch (e) {
        console.error('Failed to load completed questions:', e);
      }
    }
  }, [moduleId, sectionId]);

  // Save completed questions to localStorage
  const saveProgress = (questionId: string) => {
    const newCompleted = new Set(completedQuestions);
    newCompleted.add(questionId);
    setCompletedQuestions(newCompleted);

    const storageKey = `mc-quiz-${moduleId}-${sectionId}`;
    localStorage.setItem(storageKey, JSON.stringify(Array.from(newCompleted)));
  };

  const handleSelectAnswer = (
    question: MultipleChoiceQuestion,
    optionIndex: number,
  ) => {
    // Set the selected answer
    setSelectedAnswers((prev) => ({ ...prev, [question.id]: optionIndex }));

    // Immediately show results
    setShowResults((prev) => ({ ...prev, [question.id]: true }));

    // If correct, mark as completed
    if (optionIndex === question.correctAnswer) {
      saveProgress(question.id);
    }
  };

  const handleTryAgain = (questionId: string) => {
    setShowResults((prev) => ({ ...prev, [questionId]: false }));
    setSelectedAnswers((prev) => {
      const newAnswers = { ...prev };
      delete newAnswers[questionId];
      return newAnswers;
    });
  };

  const handleResetQuestion = (questionId: string) => {
    // Remove from completed
    const newCompleted = new Set(completedQuestions);
    newCompleted.delete(questionId);
    setCompletedQuestions(newCompleted);

    const storageKey = `mc-quiz-${moduleId}-${sectionId}`;
    localStorage.setItem(storageKey, JSON.stringify(Array.from(newCompleted)));

    // Reset UI state
    setShowResults((prev) => ({ ...prev, [questionId]: false }));
    setSelectedAnswers((prev) => {
      const newAnswers = { ...prev };
      delete newAnswers[questionId];
      return newAnswers;
    });
  };

  const handleResetAll = () => {
    if (
      !confirm(
        'Are you sure you want to reset all quiz progress? This cannot be undone.',
      )
    ) {
      return;
    }

    // Clear all state
    setCompletedQuestions(new Set());
    setShowResults({});
    setSelectedAnswers({});

    // Clear localStorage
    const storageKey = `mc-quiz-${moduleId}-${sectionId}`;
    localStorage.removeItem(storageKey);
  };

  const correctCount = questions.filter((q) =>
    completedQuestions.has(q.id),
  ).length;
  const totalCount = questions.length;

  return (
    <div className="mt-6 rounded-lg border-2 border-[#8be9fd] bg-[#282a36] p-6">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <svg
            className="h-6 w-6 text-[#8be9fd]"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"
            />
          </svg>
          <h3 className="text-xl font-bold text-[#f8f8f2]">
            Multiple Choice Quiz
          </h3>
        </div>
        <div className="flex items-center gap-3">
          <div className="rounded-full bg-[#8be9fd]/20 px-3 py-1 text-sm font-semibold text-[#8be9fd]">
            {correctCount} / {totalCount} Correct
          </div>
          {correctCount > 0 && (
            <button
              onClick={handleResetAll}
              className="rounded-lg bg-[#6272a4] px-3 py-1 text-sm font-semibold text-[#f8f8f2] transition-colors hover:bg-[#6272a4]/80"
              title="Reset entire quiz"
            >
              Reset All
            </button>
          )}
        </div>
      </div>

      <div className="space-y-6">
        {questions.map((question, qIndex) => {
          const selected = selectedAnswers[question.id];
          const showResult = showResults[question.id];
          const isCompleted = completedQuestions.has(question.id);
          const isCorrect = selected === question.correctAnswer;

          return (
            <div
              key={question.id}
              className={`rounded-lg border-2 p-5 ${isCompleted
                  ? 'border-[#50fa7b] bg-[#50fa7b]/5'
                  : 'border-[#44475a] bg-[#44475a]'
                }`}
            >
              {/* Question */}
              <div className="mb-4 flex items-start gap-3">
                <span className="flex-shrink-0 text-lg font-bold text-[#8be9fd]">
                  {qIndex + 1}.
                </span>
                <p className="flex-1 text-lg font-semibold text-[#f8f8f2]">
                  {question.question}
                </p>
                {isCompleted && (
                  <div className="ml-auto flex flex-shrink-0 items-center gap-2">
                    <span className="text-xl">✅</span>
                    <button
                      onClick={() => handleResetQuestion(question.id)}
                      className="rounded px-2 py-1 text-xs font-semibold text-[#6272a4] transition-colors hover:bg-[#6272a4]/20"
                      title="Reset this question"
                    >
                      Reset
                    </button>
                  </div>
                )}
              </div>

              {/* Options */}
              <div className="mb-4 space-y-2">
                {question.options.map((option, optionIndex) => {
                  const isSelected = selected === optionIndex;
                  const isCorrectOption =
                    optionIndex === question.correctAnswer;
                  const showCorrect = showResult && isCorrectOption;
                  const showIncorrect = showResult && isSelected && !isCorrect;

                  return (
                    <button
                      key={optionIndex}
                      onClick={() =>
                        !showResult && handleSelectAnswer(question, optionIndex)
                      }
                      disabled={showResult}
                      className={`w-full rounded-lg border-2 p-3 text-left transition-all ${showCorrect
                          ? 'border-[#50fa7b] bg-[#50fa7b]/20 text-[#50fa7b]'
                          : showIncorrect
                            ? 'border-[#ff5555] bg-[#ff5555]/20 text-[#ff5555]'
                            : isSelected
                              ? 'border-[#8be9fd] bg-[#8be9fd]/20 text-[#f8f8f2]'
                              : 'border-[#44475a] bg-[#1e1f29] text-[#f8f8f2] hover:border-[#6272a4] hover:bg-[#44475a]'
                        } ${showResult ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                    >
                      <div className="flex items-center gap-3">
                        <span className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full border-2 font-semibold">
                          {String.fromCharCode(65 + optionIndex)}
                        </span>
                        <span className="flex-1">{option}</span>
                        {showCorrect && <span className="text-xl">✓</span>}
                        {showIncorrect && <span className="text-xl">✗</span>}
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Result */}
              {showResult && (
                <div className="space-y-3">
                  {/* Result message */}
                  <div
                    className={`rounded-lg border-2 p-4 ${isCorrect
                        ? 'border-[#50fa7b] bg-[#50fa7b]/10'
                        : 'border-[#ff5555] bg-[#ff5555]/10'
                      }`}
                  >
                    <div
                      className={`mb-2 font-bold ${isCorrect ? 'text-[#50fa7b]' : 'text-[#ff5555]'
                        }`}
                    >
                      {isCorrect ? '✓ Correct!' : '✗ Incorrect'}
                    </div>
                    <p className="text-sm text-[#f8f8f2]">
                      {question.explanation}
                    </p>
                  </div>

                  {/* Try again button for incorrect answers (if not yet completed) */}
                  {!isCorrect && !isCompleted && (
                    <button
                      onClick={() => handleTryAgain(question.id)}
                      className="w-full rounded-lg bg-[#6272a4] px-4 py-2 font-semibold text-[#f8f8f2] transition-colors hover:bg-[#6272a4]/80"
                    >
                      Try Again
                    </button>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

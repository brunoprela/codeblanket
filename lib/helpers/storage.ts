/**
 * LocalStorage helper utilities for tracking problem completion and user code.
 * Provides type-safe access to browser localStorage with error handling.
 */

const COMPLETED_PROBLEMS_KEY = 'codeblanket_completed_problems';
const USER_CODE_KEY_PREFIX = 'codeblanket_code_';

/**
 * Problem Completion Tracking
 */

/**
 * Retrieves all completed problem IDs from localStorage
 * @returns Set of completed problem IDs
 */
export function getCompletedProblems(): Set<string> {
  if (typeof window === 'undefined') return new Set();

  try {
    const stored = localStorage.getItem(COMPLETED_PROBLEMS_KEY);
    return stored ? new Set(JSON.parse(stored)) : new Set();
  } catch (error) {
    console.error('Failed to load completed problems:', error);
    return new Set();
  }
}

/**
 * Marks a problem as completed in localStorage
 * @param problemId - The unique identifier of the problem
 */
export function markProblemCompleted(problemId: string): void {
  if (typeof window === 'undefined') return;

  try {
    const completed = getCompletedProblems();
    completed.add(problemId);
    localStorage.setItem(
      COMPLETED_PROBLEMS_KEY,
      JSON.stringify([...completed]),
    );
  } catch (error) {
    console.error('Failed to save completion status:', error);
  }
}

/**
 * Marks a problem as incomplete in localStorage
 * @param problemId - The unique identifier of the problem
 */
export function markProblemIncomplete(problemId: string): void {
  if (typeof window === 'undefined') return;

  try {
    const completed = getCompletedProblems();
    completed.delete(problemId);
    localStorage.setItem(
      COMPLETED_PROBLEMS_KEY,
      JSON.stringify([...completed]),
    );
  } catch (error) {
    console.error('Failed to remove completion status:', error);
  }
}

/**
 * Checks if a problem is marked as completed
 * @param problemId - The unique identifier of the problem
 * @returns true if problem is completed, false otherwise
 */
export function isProblemCompleted(problemId: string): boolean {
  return getCompletedProblems().has(problemId);
}

/**
 * Clears all completed problems from localStorage
 */
export function clearCompletedProblems(): void {
  if (typeof window === 'undefined') return;

  try {
    localStorage.removeItem(COMPLETED_PROBLEMS_KEY);
  } catch (error) {
    console.error('Failed to clear completion status:', error);
  }
}

/**
 * User Code Storage
 */

/**
 * Saves user's code for a specific problem to localStorage
 * @param problemId - The unique identifier of the problem
 * @param code - The user's code to save
 */
export function saveUserCode(problemId: string, code: string): void {
  if (typeof window === 'undefined') return;

  try {
    localStorage.setItem(`${USER_CODE_KEY_PREFIX}${problemId}`, code);
  } catch (error) {
    console.error('Failed to save user code:', error);
  }
}

/**
 * Retrieves user's saved code for a specific problem
 * @param problemId - The unique identifier of the problem
 * @returns The saved code, or null if no code is saved
 */
export function getUserCode(problemId: string): string | null {
  if (typeof window === 'undefined') return null;

  try {
    return localStorage.getItem(`${USER_CODE_KEY_PREFIX}${problemId}`);
  } catch (error) {
    console.error('Failed to load user code:', error);
    return null;
  }
}

/**
 * Removes saved user code for a specific problem
 * @param problemId - The unique identifier of the problem
 */
export function clearUserCode(problemId: string): void {
  if (typeof window === 'undefined') return;

  try {
    localStorage.removeItem(`${USER_CODE_KEY_PREFIX}${problemId}`);
  } catch (error) {
    console.error('Failed to clear user code:', error);
  }
}

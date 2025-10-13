/**
 * Storage helper utilities for tracking problem completion and user code.
 * Uses localStorage as cache with IndexedDB for persistent backup.
 * Provides type-safe access with error handling.
 */

import { setItem, getItem } from './indexeddb';

const COMPLETED_PROBLEMS_KEY = 'codeblanket_completed_problems';
const USER_CODE_KEY_PREFIX = 'codeblanket_code_';
const CUSTOM_TESTS_KEY_PREFIX = 'codeblanket_tests_';

/**
 * Sync to IndexedDB in the background (fire and forget)
 */
function syncToIndexedDB(key: string, value: unknown): void {
  setItem(key, value).catch((error) => {
    console.error('IndexedDB sync failed:', error);
  });
}

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
 * Marks a problem as completed in localStorage and syncs to IndexedDB
 * @param problemId - The unique identifier of the problem
 */
export function markProblemCompleted(problemId: string): void {
  if (typeof window === 'undefined') return;

  try {
    const completed = getCompletedProblems();
    completed.add(problemId);
    const completedArray = [...completed];
    localStorage.setItem(
      COMPLETED_PROBLEMS_KEY,
      JSON.stringify(completedArray),
    );
    // Sync to IndexedDB in background
    syncToIndexedDB(COMPLETED_PROBLEMS_KEY, completedArray);
  } catch (error) {
    console.error('Failed to save completion status:', error);
  }
}

/**
 * Marks a problem as incomplete in localStorage and syncs to IndexedDB
 * @param problemId - The unique identifier of the problem
 */
export function markProblemIncomplete(problemId: string): void {
  if (typeof window === 'undefined') return;

  try {
    const completed = getCompletedProblems();
    completed.delete(problemId);
    const completedArray = [...completed];
    localStorage.setItem(
      COMPLETED_PROBLEMS_KEY,
      JSON.stringify(completedArray),
    );
    // Sync to IndexedDB in background
    syncToIndexedDB(COMPLETED_PROBLEMS_KEY, completedArray);
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
 * Saves user's code for a specific problem to localStorage and syncs to IndexedDB
 * @param problemId - The unique identifier of the problem
 * @param code - The user's code to save
 */
export function saveUserCode(problemId: string, code: string): void {
  if (typeof window === 'undefined') return;

  try {
    const key = `${USER_CODE_KEY_PREFIX}${problemId}`;
    localStorage.setItem(key, code);
    // Sync to IndexedDB in background
    syncToIndexedDB(key, code);
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

/**
 * Custom Test Cases Storage
 */

/**
 * Saves user's custom test cases for a specific problem to localStorage and syncs to IndexedDB
 * @param problemId - The unique identifier of the problem
 * @param testCases - Array of test cases to save
 */
export function saveCustomTestCases(
  problemId: string,
  testCases: unknown[],
): void {
  if (typeof window === 'undefined') return;

  try {
    const key = `${CUSTOM_TESTS_KEY_PREFIX}${problemId}`;
    const serialized = JSON.stringify(testCases);
    localStorage.setItem(key, serialized);
    // Sync to IndexedDB in background
    syncToIndexedDB(key, testCases);
  } catch (error) {
    console.error('Failed to save custom test cases:', error);
  }
}

/**
 * Retrieves user's saved custom test cases for a specific problem
 * @param problemId - The unique identifier of the problem
 * @returns Array of saved test cases, or empty array if none are saved
 */
export function getCustomTestCases(problemId: string): unknown[] {
  if (typeof window === 'undefined') return [];

  try {
    const key = `${CUSTOM_TESTS_KEY_PREFIX}${problemId}`;
    const stored = localStorage.getItem(key);
    return stored ? JSON.parse(stored) : [];
  } catch (error) {
    console.error('Failed to load custom test cases:', error);
    return [];
  }
}

/**
 * Removes saved custom test cases for a specific problem
 * @param problemId - The unique identifier of the problem
 */
export function clearCustomTestCases(problemId: string): void {
  if (typeof window === 'undefined') return;

  try {
    localStorage.removeItem(`${CUSTOM_TESTS_KEY_PREFIX}${problemId}`);
  } catch (error) {
    console.error('Failed to clear custom test cases:', error);
  }
}

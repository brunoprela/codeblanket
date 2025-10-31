/**
 * Storage helper utilities for tracking problem completion and user code.
 * Uses localStorage as cache with dual-backend storage (IndexedDB or PostgreSQL).
 * Automatically routes to the correct backend based on authentication state.
 * Provides type-safe access with error handling.
 */

import { setItem } from './storage-adapter';

const COMPLETED_PROBLEMS_KEY = 'codeblanket_completed_problems';
const USER_CODE_KEY_PREFIX = 'codeblanket_code_';
const CUSTOM_TESTS_KEY_PREFIX = 'codeblanket_tests_';

/**
 * Sync to storage backend in the background (fire and forget)
 * Automatically uses IndexedDB for anonymous users or PostgreSQL for authenticated users
 */
function syncToStorage(key: string, value: unknown): void {
  setItem(key, value).catch((error) => {
    console.error('Storage sync failed:', error);
  });
}

/**
 * Problem Completion Tracking
 */

/**
 * Retrieves all completed problem IDs from localStorage
 * @returns Set of completed problem IDs
 */
export async function getCompletedProblems(): Promise<Set<string>> {
  if (typeof window === 'undefined') return new Set();

  try {
    // Check authentication
    const authResponse = await fetch('/api/auth/check');
    const authData = await authResponse.json();
    const isAuthenticated = authData.authenticated === true;

    if (isAuthenticated) {
      // Fetch from PostgreSQL via API (ignore localStorage entirely)
      const response = await fetch(
        '/api/progress?key=codeblanket_completed_problems',
      );

      if (response.ok) {
        const data = await response.json();
        return data.value ? new Set(JSON.parse(data.value)) : new Set();
      }

      // If API fails, return empty (don't fall back to localStorage for authenticated users)
      console.warn(
        'Failed to fetch completed problems from API, returning empty',
      );
      return new Set();
    } else {
      // Anonymous user: Use localStorage
      const stored = localStorage.getItem(COMPLETED_PROBLEMS_KEY);
      return stored ? new Set(JSON.parse(stored)) : new Set();
    }
  } catch (error) {
    console.error('Failed to load completed problems:', error);
    return new Set();
  }
}

/**
 * Marks a problem as completed in localStorage and syncs to storage backend
 * For authenticated users, saves to PostgreSQL
 * @param problemId - The unique identifier of the problem
 */
export async function markProblemCompleted(problemId: string): Promise<void> {
  if (typeof window === 'undefined') return;

  try {
    const completed = await getCompletedProblems();
    completed.add(problemId);
    const completedArray = [...completed];

    // Check authentication - authenticated users DON'T use localStorage
    const authResponse = await fetch('/api/auth/check');
    const authData = await authResponse.json();
    const isAuthenticated = authData.authenticated === true;

    if (!isAuthenticated) {
      // Only write to localStorage for anonymous users
      localStorage.setItem(
        COMPLETED_PROBLEMS_KEY,
        JSON.stringify(completedArray),
      );
    }

    // Sync to storage backend (PostgreSQL for authenticated, IndexedDB for anonymous)
    syncToStorage(COMPLETED_PROBLEMS_KEY, completedArray);
  } catch (error) {
    console.error('Failed to save completion status:', error);
  }
}

/**
 * Marks a problem as incomplete in localStorage and syncs to storage backend
 * For authenticated users, saves to PostgreSQL
 * @param problemId - The unique identifier of the problem
 */
export async function markProblemIncomplete(problemId: string): Promise<void> {
  if (typeof window === 'undefined') return;

  try {
    const completed = await getCompletedProblems();
    completed.delete(problemId);
    const completedArray = [...completed];

    // Check authentication - authenticated users DON'T use localStorage
    const authResponse = await fetch('/api/auth/check');
    const authData = await authResponse.json();
    const isAuthenticated = authData.authenticated === true;

    if (!isAuthenticated) {
      // Only write to localStorage for anonymous users
      localStorage.setItem(
        COMPLETED_PROBLEMS_KEY,
        JSON.stringify(completedArray),
      );
    }

    // Sync to storage backend (PostgreSQL for authenticated, IndexedDB for anonymous)
    syncToStorage(COMPLETED_PROBLEMS_KEY, completedArray);
  } catch (error) {
    console.error('Failed to remove completion status:', error);
  }
}

/**
 * Checks if a problem is marked as completed
 * For authenticated users, checks PostgreSQL
 * @param problemId - The unique identifier of the problem
 * @returns true if problem is completed, false otherwise
 */
export async function isProblemCompleted(problemId: string): Promise<boolean> {
  const completed = await getCompletedProblems();
  return completed.has(problemId);
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

    // SECURITY: Authenticated users should NOT write to localStorage
    // Only sync to PostgreSQL (handled by syncToStorage)
    // localStorage is ONLY for anonymous users

    // Note: We'll check auth in syncToStorage, so just always sync
    // The backend will handle routing to correct storage
    syncToStorage(key, code);

    // For anonymous users, also write to localStorage for instant access
    // This check happens inside syncToStorage, but we do it here too
    // to avoid localStorage writes for authenticated users
    fetch('/api/auth/check')
      .then((res) => res.json())
      .then((data) => {
        if (!data.authenticated) {
          localStorage.setItem(key, code);
        }
      })
      .catch(() => {
        // On error, assume anonymous
        localStorage.setItem(key, code);
      });
  } catch (error) {
    console.error('Failed to save user code:', error);
  }
}

/**
 * Retrieves user's saved code for a specific problem
 * For authenticated users, fetches from PostgreSQL
 * For anonymous users, uses localStorage
 * @param problemId - The unique identifier of the problem
 * @returns The saved code, or null if no code is saved
 */
export async function getUserCode(problemId: string): Promise<string | null> {
  if (typeof window === 'undefined') return null;

  try {
    // Check authentication
    const authResponse = await fetch('/api/auth/check');
    const authData = await authResponse.json();
    const isAuthenticated = authData.authenticated === true;

    if (isAuthenticated) {
      // Fetch from PostgreSQL via API
      const key = `${USER_CODE_KEY_PREFIX}${problemId}`;
      const response = await fetch(
        `/api/progress?key=${encodeURIComponent(key)}`,
      );

      if (response.ok) {
        const data = await response.json();
        // SECURITY: DON'T cache authenticated user data in localStorage
        // It will leak to anonymous users after logout
        return data.value || null;
      }

      // If API fails, return empty (don't fall back to localStorage for authenticated users)
      console.warn('Failed to fetch user code from API, returning null');
      return null;
    } else {
      // Anonymous user: Use localStorage
      return localStorage.getItem(`${USER_CODE_KEY_PREFIX}${problemId}`);
    }
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

    // SECURITY: Don't write to localStorage for authenticated users
    // Sync to backend first
    syncToStorage(key, testCases);

    // For anonymous users only, write to localStorage
    fetch('/api/auth/check')
      .then((res) => res.json())
      .then((data) => {
        if (!data.authenticated) {
          localStorage.setItem(key, serialized);
        }
      })
      .catch(() => {
        // On error, assume anonymous
        localStorage.setItem(key, serialized);
      });
  } catch (error) {
    console.error('Failed to save custom test cases:', error);
  }
}

/**
 * Retrieves user's saved custom test cases for a specific problem
 * For authenticated users, fetches from PostgreSQL
 * For anonymous users, uses localStorage
 * @param problemId - The unique identifier of the problem
 * @returns Array of saved test cases, or empty array if none are saved
 */
export async function getCustomTestCases(
  problemId: string,
): Promise<unknown[]> {
  if (typeof window === 'undefined') return [];

  try {
    // Check authentication
    const authResponse = await fetch('/api/auth/check');
    const authData = await authResponse.json();
    const isAuthenticated = authData.authenticated === true;

    if (isAuthenticated) {
      // Fetch from PostgreSQL via API
      const key = `${CUSTOM_TESTS_KEY_PREFIX}${problemId}`;
      const response = await fetch(
        `/api/progress?key=${encodeURIComponent(key)}`,
      );

      if (response.ok) {
        const data = await response.json();
        // SECURITY: DON'T cache authenticated user data in localStorage
        return data.value
          ? Array.isArray(data.value)
            ? data.value
            : JSON.parse(data.value)
          : [];
      }

      // If API fails, return empty (don't fall back to localStorage for authenticated users)
      console.warn(
        'Failed to fetch custom test cases from API, returning empty',
      );
      return [];
    } else {
      // Anonymous user: Use localStorage
      const key = `${CUSTOM_TESTS_KEY_PREFIX}${problemId}`;
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : [];
    }
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

/**
 * Multiple Choice Quiz Progress Storage
 */

/**
 * Saves multiple choice quiz progress for a specific module/section
 * @param moduleId - The module identifier
 * @param sectionId - The section identifier
 * @param completedQuestionIds - Array of completed question IDs
 */
export function saveMultipleChoiceProgress(
  moduleId: string,
  sectionId: string,
  completedQuestionIds: string[],
): void {
  if (typeof window === 'undefined') return;

  try {
    const key = `mc-quiz-${moduleId}-${sectionId}`;
    const serialized = JSON.stringify(completedQuestionIds);

    // SECURITY: Don't write to localStorage for authenticated users
    syncToStorage(key, completedQuestionIds);

    // For anonymous users only, write to localStorage
    fetch('/api/auth/check')
      .then((res) => res.json())
      .then((data) => {
        if (!data.authenticated) {
          localStorage.setItem(key, serialized);
        }
      })
      .catch(() => {
        // On error, assume anonymous
        localStorage.setItem(key, serialized);
      });
  } catch (error) {
    console.error('Failed to save multiple choice progress:', error);
  }
}

/**
 * Retrieves multiple choice quiz progress for a specific module/section
 * For authenticated users, fetches from PostgreSQL
 * For anonymous users, uses localStorage
 * @param moduleId - The module identifier
 * @param sectionId - The section identifier
 * @returns Array of completed question IDs
 */
export async function getMultipleChoiceProgress(
  moduleId: string,
  sectionId: string,
): Promise<string[]> {
  if (typeof window === 'undefined') return [];

  try {
    // Check authentication
    const authResponse = await fetch('/api/auth/check');
    const authData = await authResponse.json();
    const isAuthenticated = authData.authenticated === true;

    if (isAuthenticated) {
      // Fetch from PostgreSQL via API
      const key = `mc-quiz-${moduleId}-${sectionId}`;
      const response = await fetch(
        `/api/progress?key=${encodeURIComponent(key)}`,
      );

      if (response.ok) {
        const data = await response.json();
        return data.value ? (JSON.parse(data.value) as string[]) : [];
      }

      // If API fails, return empty (don't fall back to localStorage for authenticated users)
      console.warn('Failed to fetch MC progress from API, returning empty');
      return [];
    } else {
      // Anonymous user: Use localStorage
      const key = `mc-quiz-${moduleId}-${sectionId}`;
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : [];
    }
  } catch (error) {
    console.error('Failed to load multiple choice progress:', error);
    return [];
  }
}

/**
 * Clears multiple choice quiz progress for a specific module/section
 * @param moduleId - The module identifier
 * @param sectionId - The section identifier
 */
export function clearMultipleChoiceProgress(
  moduleId: string,
  sectionId: string,
): void {
  if (typeof window === 'undefined') return;

  try {
    const key = `mc-quiz-${moduleId}-${sectionId}`;
    localStorage.removeItem(key);
  } catch (error) {
    console.error('Failed to clear multiple choice progress:', error);
  }
}

/**
 * Module Completion Storage
 */

/**
 * Retrieves completed sections for a specific module
 * For authenticated users, fetches from PostgreSQL
 * For anonymous users, uses localStorage
 * @param moduleId - The module identifier
 * @returns Set of completed section IDs
 */
export async function getCompletedSections(
  moduleId: string,
): Promise<Set<string>> {
  if (typeof window === 'undefined') return new Set();

  try {
    // Check authentication
    const authResponse = await fetch('/api/auth/check');
    const authData = await authResponse.json();
    const isAuthenticated = authData.authenticated === true;

    if (isAuthenticated) {
      // Fetch from PostgreSQL via API
      const key = `module-${moduleId}-completed`;
      const response = await fetch(
        `/api/progress?key=${encodeURIComponent(key)}`,
      );

      if (response.ok) {
        const data = await response.json();
        return data.value ? new Set(JSON.parse(data.value)) : new Set();
      }

      // If API fails, return empty (don't fall back to localStorage for authenticated users)
      console.warn(
        'Failed to fetch module completion from API, returning empty',
      );
      return new Set();
    } else {
      // Anonymous user: Use localStorage
      const key = `module-${moduleId}-completed`;
      const stored = localStorage.getItem(key);
      return stored ? new Set(JSON.parse(stored)) : new Set();
    }
  } catch (error) {
    console.error('Failed to load completed sections:', error);
    return new Set();
  }
}

/**
 * Saves completed sections for a specific module
 * @param moduleId - The module identifier
 * @param completedSectionIds - Array of completed section IDs
 */
export function saveCompletedSections(
  moduleId: string,
  completedSectionIds: string[],
): void {
  if (typeof window === 'undefined') return;

  try {
    const key = `module-${moduleId}-completed`;
    const serialized = JSON.stringify(completedSectionIds);

    // SECURITY: Don't write to localStorage for authenticated users
    syncToStorage(key, completedSectionIds);

    // For anonymous users only, write to localStorage
    fetch('/api/auth/check')
      .then((res) => res.json())
      .then((data) => {
        if (!data.authenticated) {
          localStorage.setItem(key, serialized);
        }
      })
      .catch(() => {
        // On error, assume anonymous
        localStorage.setItem(key, serialized);
      });
  } catch (error) {
    console.error('Failed to save completed sections:', error);
  }
}

/**
 * Marks a section as completed in a module
 * For authenticated users, saves to PostgreSQL
 * @param moduleId - The module identifier
 * @param sectionId - The section identifier
 */
export async function markSectionCompleted(
  moduleId: string,
  sectionId: string,
): Promise<void> {
  if (typeof window === 'undefined') return;

  try {
    const completed = await getCompletedSections(moduleId);
    completed.add(sectionId);
    saveCompletedSections(moduleId, [...completed]);
  } catch (error) {
    console.error('Failed to mark section completed:', error);
  }
}

/**
 * Marks a section as incomplete in a module
 * For authenticated users, saves to PostgreSQL
 * @param moduleId - The module identifier
 * @param sectionId - The section identifier
 */
export async function markSectionIncomplete(
  moduleId: string,
  sectionId: string,
): Promise<void> {
  if (typeof window === 'undefined') return;

  try {
    const completed = await getCompletedSections(moduleId);
    completed.delete(sectionId);
    saveCompletedSections(moduleId, [...completed]);
  } catch (error) {
    console.error('Failed to mark section incomplete:', error);
  }
}

/**
 * Checks if a section is marked as completed
 * For authenticated users, checks PostgreSQL
 * @param moduleId - The module identifier
 * @param sectionId - The section identifier
 * @returns true if section is completed, false otherwise
 */
export async function isSectionCompleted(
  moduleId: string,
  sectionId: string,
): Promise<boolean> {
  const completed = await getCompletedSections(moduleId);
  return completed.has(sectionId);
}

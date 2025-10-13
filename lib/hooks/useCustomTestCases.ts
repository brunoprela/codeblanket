/**
 * Custom hook for managing user-defined custom test cases with persistent storage
 */

import { useState, useEffect } from 'react';
import { CustomTestCase } from '@/lib/types';
import {
  getCustomTestCases,
  saveCustomTestCases,
  clearCustomTestCases as clearStoredTestCases,
} from '@/lib/helpers/storage';

interface UseCustomTestCasesReturn {
  /** Custom test cases defined by the user */
  customTestCases: CustomTestCase[];
  /** Function to add a new custom test case */
  addCustomTestCase: (testCase: CustomTestCase) => void;
  /** Function to update an existing custom test case */
  updateCustomTestCase: (id: string, testCase: CustomTestCase) => void;
  /** Function to delete a custom test case */
  deleteCustomTestCase: (id: string) => void;
  /** Function to clear all custom test cases */
  clearAllCustomTestCases: () => void;
}

/**
 * Hook for managing custom test cases persistence in localStorage + IndexedDB
 * @param problemId - Unique identifier for the problem
 * @returns Object with custom test cases state and update functions
 * @example
 * ```tsx
 * const { customTestCases, addCustomTestCase } = useCustomTestCases('binary-search-1');
 *
 * const handleAdd = () => {
 *   addCustomTestCase({ input: [1, 2, 3], expected: true });
 * };
 * ```
 */
export function useCustomTestCases(
  problemId: string | undefined,
): UseCustomTestCasesReturn {
  const [customTestCases, setCustomTestCases] = useState<CustomTestCase[]>([]);

  // Load saved custom test cases on mount
  useEffect(() => {
    if (problemId) {
      const saved = getCustomTestCases(problemId) as CustomTestCase[];
      setCustomTestCases(saved);
    }
  }, [problemId]);

  // Save custom test cases whenever they change
  useEffect(() => {
    if (problemId && customTestCases.length > 0) {
      saveCustomTestCases(problemId, customTestCases);
    } else if (problemId && customTestCases.length === 0) {
      // Clear storage if no custom test cases
      clearStoredTestCases(problemId);
    }
  }, [customTestCases, problemId]);

  const addCustomTestCase = (testCase: CustomTestCase) => {
    setCustomTestCases((prev) => [...prev, testCase]);
  };

  const updateCustomTestCase = (id: string, testCase: CustomTestCase) => {
    setCustomTestCases((prev) =>
      prev.map((tc) => (tc.id === id ? testCase : tc)),
    );
  };

  const deleteCustomTestCase = (id: string) => {
    setCustomTestCases((prev) => prev.filter((tc) => tc.id !== id));
  };

  const clearAllCustomTestCases = () => {
    setCustomTestCases([]);
    if (problemId) {
      clearStoredTestCases(problemId);
    }
  };

  return {
    customTestCases,
    addCustomTestCase,
    updateCustomTestCase,
    deleteCustomTestCase,
    clearAllCustomTestCases,
  };
}

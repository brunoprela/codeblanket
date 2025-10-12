/**
 * Custom hook for managing user code storage in localStorage
 */

import { useState, useEffect } from 'react';
import {
  getUserCode,
  saveUserCode,
  clearUserCode,
} from '@/lib/helpers/storage';

interface UseCodeStorageReturn {
  /** Current code in the editor */
  code: string;
  /** Function to update the code */
  setCode: (code: string) => void;
  /** Function to reset code to starter code */
  resetCode: () => void;
}

/**
 * Hook for managing code persistence in localStorage
 * @param problemId - Unique identifier for the problem
 * @param starterCode - Default starter code for the problem
 * @returns Object with code state and update functions
 * @example
 * ```tsx
 * const { code, setCode, resetCode } = useCodeStorage('binary-search-1', defaultCode);
 *
 * return (
 *   <Editor
 *     value={code}
 *     onChange={setCode}
 *   />
 * );
 * ```
 */
export function useCodeStorage(
  problemId: string | undefined,
  starterCode: string,
): UseCodeStorageReturn {
  const [code, setCodeState] = useState(starterCode);

  // Load saved code on mount
  useEffect(() => {
    if (problemId) {
      const savedCode = getUserCode(problemId);
      if (savedCode) {
        setCodeState(savedCode);
      }
    }
  }, [problemId]);

  // Save code to localStorage when it changes
  useEffect(() => {
    if (problemId && code !== starterCode) {
      saveUserCode(problemId, code);
    }
  }, [code, problemId, starterCode]);

  const setCode = (newCode: string) => {
    setCodeState(newCode);
  };

  const resetCode = () => {
    setCodeState(starterCode);
    if (problemId) {
      clearUserCode(problemId);
    }
  };

  return { code, setCode, resetCode };
}

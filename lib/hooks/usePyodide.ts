/**
 * Custom hook for loading and managing Pyodide (Python runtime in browser)
 */

import { useState, useEffect } from 'react';
import { getPyodide } from '@/lib/pyodide';

interface UsePyodideReturn {
  /** Whether Pyodide is ready to execute Python code */
  isReady: boolean;
  /** Whether Pyodide is currently loading */
  isLoading: boolean;
  /** Error message if Pyodide failed to load */
  error: string | null;
}

/**
 * Hook for loading Pyodide on component mount
 * @returns Object with Pyodide loading state
 * @example
 * ```tsx
 * const { isReady, isLoading, error } = usePyodide();
 *
 * if (isLoading) return <LoadingSpinner />;
 * if (error) return <ErrorDisplay message={error} />;
 * if (!isReady) return null;
 *
 * return <CodeEditor />;
 * ```
 */
export function usePyodide(): UsePyodideReturn {
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadPyodide() {
      try {
        setIsLoading(true);
        await getPyodide();
        setIsReady(true);
        setError(null);
      } catch (err) {
        console.error('Failed to load Pyodide:', err);
        const errorMessage =
          err instanceof Error ? err.message : 'Unknown error';
        setError(errorMessage);
      } finally {
        setIsLoading(false);
      }
    }

    loadPyodide();
  }, []);

  return { isReady, isLoading, error };
}

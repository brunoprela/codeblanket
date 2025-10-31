/**
 * Efficient storage stats functions
 * Returns counts and indicators without downloading all data
 */

/**
 * Check if user is authenticated
 */
async function isUserAuthenticated(): Promise<boolean> {
  if (typeof window === 'undefined') return false;

  try {
    const response = await fetch('/api/auth/check');
    if (!response.ok) return false;
    const data = await response.json();
    return data.authenticated === true;
  } catch {
    return false;
  }
}

// Request deduplication: Prevent multiple simultaneous calls
let inflightRequest: Promise<UserStats | null> | null = null;

export interface UserStats {
  totalProgressItems: number;
  totalVideos: number;
  completedDiscussionQuestions: number;
  hasCompletedProblems: boolean;
  completedProblemsCount: number; // Actual count of completed problems
  completedProblemsList: string[]; // List of completed problem IDs
  multipleChoiceQuizCount: number; // Actual count of completed MC questions
  moduleProgressCount: number;
  moduleVideoCounts: Record<string, number>; // Module-specific video counts
  moduleMCCounts: Record<string, number>; // Module-specific MC counts
  moduleCompletionMap: Record<string, number>; // Module-specific section completion counts
  keys: string[];
}

/**
 * Get efficient stats for authenticated users (counts only, no full data)
 * Falls back to null for anonymous users (use IndexedDB directly)
 */
export async function getUserStats(): Promise<UserStats | null> {
  const isAuthenticated = await isUserAuthenticated();

  console.log('[getUserStats] isAuthenticated:', isAuthenticated);

  if (!isAuthenticated) {
    console.log('[getUserStats] User not authenticated, returning null');
    inflightRequest = null; // Clear any pending request
    return null; // Anonymous users use IndexedDB directly
  }

  // OPTIMIZATION: If there's already a request in flight, return that promise
  // This prevents duplicate API calls when multiple components mount simultaneously
  if (inflightRequest) {
    console.log('[getUserStats] Reusing inflight request');
    return inflightRequest;
  }

  // Start new request and store promise
  inflightRequest = (async () => {
    try {
      console.log('[getUserStats] Fetching fresh stats from /api/stats...');
      const startTime = performance.now();
      const response = await fetch('/api/stats');
      const fetchTime = performance.now() - startTime;

      console.log(
        `[getUserStats] Response status: ${response.status} (${fetchTime.toFixed(0)}ms)`,
      );

      if (!response.ok) {
        console.error(
          '[getUserStats] API returned error status:',
          response.status,
        );
        const errorText = await response.text();
        console.error('[getUserStats] Error response:', errorText);
        return null;
      }

      const stats = await response.json();
      const totalTime = performance.now() - startTime;
      console.log(`[getUserStats] Stats received in ${totalTime.toFixed(0)}ms`);

      return stats as UserStats;
    } catch (error) {
      console.error('[getUserStats] Exception fetching user stats:', error);
      return null;
    } finally {
      // Clear inflight request after 100ms to allow fresh requests
      setTimeout(() => {
        inflightRequest = null;
      }, 100);
    }
  })();

  return inflightRequest;
}

/**
 * Check if a specific key exists in user's progress (efficient)
 * For authenticated users, uses cached keys from stats
 * For anonymous users, checks localStorage
 */
export async function hasProgressKey(
  key: string,
  cachedKeys?: string[],
): Promise<boolean> {
  const isAuthenticated = await isUserAuthenticated();

  if (!isAuthenticated) {
    // Anonymous users: check localStorage
    return !!localStorage.getItem(key);
  }

  // Authenticated users: use cached keys if available
  if (cachedKeys) {
    return cachedKeys.includes(key);
  }

  // Fallback: fetch from API
  try {
    const response = await fetch(
      `/api/progress?key=${encodeURIComponent(key)}`,
    );
    if (!response.ok) return false;
    const data = await response.json();
    return data.value !== null;
  } catch {
    return false;
  }
}

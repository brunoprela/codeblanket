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
    return null; // Anonymous users use IndexedDB directly
  }

  try {
    console.log('[getUserStats] Fetching from /api/stats...');
    const response = await fetch('/api/stats');

    console.log(
      '[getUserStats] Response status:',
      response.status,
      response.statusText,
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
    console.log('[getUserStats] Stats received:', stats);
    return stats as UserStats;
  } catch (error) {
    console.error('[getUserStats] Exception fetching user stats:', error);
    return null;
  }
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

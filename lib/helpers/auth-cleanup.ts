/**
 * Security: Clean up all user data when logging out
 * Prevents authenticated user data from leaking to anonymous users
 */

/**
 * Clear all user progress data from localStorage and IndexedDB
 * CRITICAL: Call this on logout to prevent data leakage
 */
export async function clearAllUserData(): Promise<void> {
  console.log(
    '[Auth Cleanup] Clearing all user data from localStorage and IndexedDB',
  );

  try {
    // Clear ALL localStorage keys related to user progress
    const keysToRemove: string[] = [];

    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (
        key &&
        (key.startsWith('codeblanket_') ||
          key.startsWith('mc-quiz-') ||
          key.startsWith('module-'))
      ) {
        keysToRemove.push(key);
      }
    }

    // Remove all user data keys
    keysToRemove.forEach((key) => localStorage.removeItem(key));
    console.log(
      `[Auth Cleanup] Removed ${keysToRemove.length} localStorage items`,
    );

    // Clear IndexedDB (if it exists)
    if (typeof window !== 'undefined' && 'indexedDB' in window) {
      try {
        const { importData } = await import('./indexeddb');
        await importData({}); // Clear all progress data
        console.log('[Auth Cleanup] Cleared IndexedDB progress data');

        // Clear videos from IndexedDB
        const dbRequest = indexedDB.open('CodeBlanket', 2);
        dbRequest.onsuccess = () => {
          const db = dbRequest.result;
          if (db.objectStoreNames.contains('videos')) {
            const transaction = db.transaction('videos', 'readwrite');
            const store = transaction.objectStore('videos');
            store.clear();
            console.log('[Auth Cleanup] Cleared IndexedDB videos');
          }
          db.close();
        };
      } catch (error) {
        console.error('[Auth Cleanup] Error clearing IndexedDB:', error);
      }
    }

    console.log('[Auth Cleanup] User data cleanup complete');
  } catch (error) {
    console.error('[Auth Cleanup] Error during cleanup:', error);
    throw error;
  }
}

/**
 * Check if user is authenticated
 * @returns true if user is logged in
 */
export async function isAuthenticated(): Promise<boolean> {
  if (typeof window === 'undefined') return false;

  try {
    const response = await fetch('/api/auth/check');
    const data = await response.json();
    return data.authenticated === true;
  } catch {
    return false;
  }
}

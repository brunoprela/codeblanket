/**
 * IndexedDB wrapper for persistent storage
 */

const DB_NAME = 'CodeBlanketDB';
const DB_VERSION = 2;
const STORE_NAME = 'progress';
const VIDEO_STORE_NAME = 'videos';

interface ProgressData {
  id: string;
  data: unknown;
  timestamp: number;
}

let dbInstance: IDBDatabase | null = null;
let dbPromise: Promise<IDBDatabase> | null = null;

/**
 * Initialize and open the IndexedDB database with connection pooling
 */
async function openDB(): Promise<IDBDatabase> {
  // Check if we have a valid cached connection
  if (dbInstance) {
    try {
      // Test if connection is still valid by attempting a transaction
      dbInstance.transaction(STORE_NAME, 'readonly');
      return dbInstance;
    } catch {
      // Connection is closed, clear the cache
      console.warn('IndexedDB connection was closed, reopening...');
      dbInstance = null;
      dbPromise = null;
    }
  }

  // If there's already a connection attempt in progress, wait for it
  if (dbPromise) {
    return dbPromise;
  }

  // Create new connection
  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => {
      dbPromise = null;
      reject(request.error);
    };

    request.onsuccess = () => {
      dbInstance = request.result;

      // Handle unexpected close
      dbInstance.onclose = () => {
        console.warn('IndexedDB connection closed unexpectedly');
        dbInstance = null;
        dbPromise = null;
      };

      // Handle version change (another tab upgraded the database)
      dbInstance.onversionchange = () => {
        console.warn('IndexedDB version change detected, closing connection');
        dbInstance?.close();
        dbInstance = null;
        dbPromise = null;
      };

      resolve(dbInstance);
    };

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(VIDEO_STORE_NAME)) {
        db.createObjectStore(VIDEO_STORE_NAME, { keyPath: 'id' });
      }
    };

    request.onblocked = () => {
      console.warn(
        'IndexedDB open request blocked, may need to close other tabs',
      );
    };
  });

  return dbPromise;
}

/**
 * Set a value in IndexedDB
 */
export async function setItem(key: string, value: unknown): Promise<void> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    const data: ProgressData = {
      id: key,
      data: value,
      timestamp: Date.now(),
    };

    return new Promise((resolve, reject) => {
      const request = store.put(data);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('IndexedDB setItem error:', error);
    // Fallback to localStorage
    localStorage.setItem(key, JSON.stringify(value));
  }
}

/**
 * Get a value from IndexedDB
 */
export async function getItem<T>(key: string): Promise<T | null> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.get(key);
      request.onsuccess = () => {
        const result = request.result as ProgressData | undefined;
        resolve((result?.data as T | null) ?? null);
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('IndexedDB getItem error:', error);
    // Fallback to localStorage
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : null;
  }
}

/**
 * Remove a value from IndexedDB
 */
export async function removeItem(key: string): Promise<void> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.delete(key);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('IndexedDB removeItem error:', error);
    // Fallback to localStorage
    localStorage.removeItem(key);
  }
}

/**
 * Get all data from IndexedDB
 */
export async function getAllData(): Promise<Record<string, unknown>> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => {
        const results = request.result as ProgressData[];
        const data: Record<string, unknown> = {};
        results.forEach((item) => {
          data[item.id] = item.data;
        });
        resolve(data);
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('IndexedDB getAllData error:', error);
    return {};
  }
}

/**
 * Import data into IndexedDB (overwrites existing)
 */
export async function importData(data: Record<string, unknown>): Promise<void> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    // Clear existing data first
    await new Promise<void>((resolve, reject) => {
      const clearRequest = store.clear();
      clearRequest.onsuccess = () => resolve();
      clearRequest.onerror = () => reject(clearRequest.error);
    });

    // Import new data
    const promises = Object.entries(data).map(([key, value]) => {
      return new Promise<void>((resolve, reject) => {
        const progressData: ProgressData = {
          id: key,
          data: value,
          timestamp: Date.now(),
        };
        const request = store.put(progressData);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    });

    await Promise.all(promises);
  } catch (error) {
    console.error('IndexedDB importData error:', error);
    throw error;
  }
}

/**
 * Migrate data from localStorage to IndexedDB
 */
export async function migrateFromLocalStorage(): Promise<void> {
  try {
    // Keys to migrate - include all CodeBlanket-related data
    const keysToMigrate = [
      'codeblanket_completed_problems',
      'codeblanket_code_',
      'codeblanket_tests_', // Custom test cases
      'module-', // Module completion
      'mc-quiz-', // Multiple choice quiz progress
      'codeblanket-',
    ];

    // Skip these keys
    const keysToSkip = [
      'codeblanket_migration_complete',
      'codeblanket-auto-backup',
    ];

    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (!key) continue;

      // Skip certain keys
      if (keysToSkip.includes(key)) continue;

      // Check if key should be migrated
      const shouldMigrate = keysToMigrate.some((prefix) =>
        key.startsWith(prefix),
      );

      if (shouldMigrate) {
        const value = localStorage.getItem(key);
        if (value) {
          try {
            const parsed = JSON.parse(value);
            await setItem(key, parsed);
          } catch {
            // If not JSON, store as string
            await setItem(key, value);
          }
        }
      }
    }
  } catch (error) {
    console.error('Migration error:', error);
  }
}

/**
 * Save video blob to IndexedDB with unique video ID
 */
export async function saveVideo(
  videoId: string,
  videoBlob: Blob,
): Promise<void> {
  try {
    const db = await openDB();
    const transaction = db.transaction(VIDEO_STORE_NAME, 'readwrite');
    const store = transaction.objectStore(VIDEO_STORE_NAME);

    const data = {
      id: videoId,
      video: videoBlob,
      timestamp: Date.now(),
    };

    return new Promise((resolve, reject) => {
      const request = store.put(data);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('IndexedDB saveVideo error:', error);
    throw error;
  }
}

/**
 * Get a single video by ID
 */
export async function getVideo(videoId: string): Promise<Blob | null> {
  try {
    const db = await openDB();
    const transaction = db.transaction(VIDEO_STORE_NAME, 'readonly');
    const store = transaction.objectStore(VIDEO_STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.get(videoId);
      request.onsuccess = () => {
        const result = request.result;
        resolve(result ? result.video : null);
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('IndexedDB getVideo error:', error);
    return null;
  }
}

/**
 * Get all videos for a specific question
 */
export async function getVideosForQuestion(
  questionIdPrefix: string,
): Promise<Array<{ id: string; blob: Blob; timestamp: number }>> {
  try {
    const db = await openDB();
    const transaction = db.transaction(VIDEO_STORE_NAME, 'readonly');
    const store = transaction.objectStore(VIDEO_STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => {
        const allVideos = request.result || [];
        const filtered = allVideos
          .filter((item) => item.id.startsWith(questionIdPrefix))
          .map((item) => ({
            id: item.id,
            blob: item.video,
            timestamp: item.timestamp,
          }))
          .sort((a, b) => a.timestamp - b.timestamp);
        resolve(filtered);
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('IndexedDB getVideosForQuestion error:', error);
    return [];
  }
}

/**
 * Delete a specific video from IndexedDB
 */
export async function deleteVideo(videoId: string): Promise<void> {
  try {
    const db = await openDB();
    const transaction = db.transaction(VIDEO_STORE_NAME, 'readwrite');
    const store = transaction.objectStore(VIDEO_STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.delete(videoId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('IndexedDB deleteVideo error:', error);
    throw error;
  }
}

/**
 * Get total count of unique questions with at least one video
 */
export async function getCompletedDiscussionQuestionsCount(): Promise<number> {
  try {
    const db = await openDB();
    const transaction = db.transaction(VIDEO_STORE_NAME, 'readonly');
    const store = transaction.objectStore(VIDEO_STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => {
        const allVideos = request.result || [];
        // Extract unique question prefixes (before the timestamp)
        const uniqueQuestions = new Set<string>();
        allVideos.forEach((item) => {
          // Video ID format: moduleSlug-sectionId-questionId-timestamp
          // We need to extract: moduleSlug-sectionId-questionId
          const parts = item.id.split('-');
          if (parts.length >= 4) {
            const questionPrefix = parts.slice(0, -1).join('-');
            uniqueQuestions.add(questionPrefix);
          }
        });
        resolve(uniqueQuestions.size);
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error(
      'IndexedDB getCompletedDiscussionQuestionsCount error:',
      error,
    );
    return 0;
  }
}

/**
 * Get total count of all discussion questions across all modules
 */
export function getTotalDiscussionQuestionsCount(
  modules: Array<{ module: { sections: Array<{ quiz?: unknown[] }> } }>,
): number {
  let total = 0;
  modules.forEach((moduleCategory) => {
    moduleCategory.module.sections.forEach((section) => {
      if (section.quiz) {
        total += section.quiz.length;
      }
    });
  });
  return total;
}

/**
 * Get total count of all multiple choice questions across all modules
 */
export function getTotalMultipleChoiceQuestionsCount(
  modules: Array<{
    module: { sections: Array<{ multipleChoice?: unknown[] }> };
  }>,
): number {
  let total = 0;
  modules.forEach((moduleCategory) => {
    moduleCategory.module.sections.forEach((section) => {
      if (section.multipleChoice) {
        total += section.multipleChoice.length;
      }
    });
  });
  return total;
}

/**
 * Get count of completed multiple choice questions across all modules
 */
export function getCompletedMultipleChoiceQuestionsCount(
  modules: Array<{
    id: string;
    module: { sections: Array<{ id?: string; multipleChoice?: unknown[] }> };
  }>,
): number {
  let completed = 0;
  modules.forEach((moduleCategory) => {
    moduleCategory.module.sections.forEach((section, sectionIndex) => {
      const sectionId = section.id || `section-${sectionIndex}`;
      if (section.multipleChoice && section.multipleChoice.length > 0) {
        const storageKey = `mc-quiz-${moduleCategory.id}-${sectionId}`;
        const stored = localStorage.getItem(storageKey);
        if (stored) {
          try {
            const completedQuestions = JSON.parse(stored);
            // Deduplicate in case of corrupted data
            const uniqueQuestions = [...new Set(completedQuestions)];

            // Fix corrupted data if duplicates found
            if (uniqueQuestions.length !== completedQuestions.length) {
              localStorage.setItem(storageKey, JSON.stringify(uniqueQuestions));
              console.warn(
                `Fixed duplicates in ${storageKey}: ${completedQuestions.length} → ${uniqueQuestions.length}`,
              );
            }

            completed += uniqueQuestions.length;
          } catch (e) {
            console.error('Failed to parse MC quiz progress:', e);
          }
        }
      }
    });
  });
  return completed;
}

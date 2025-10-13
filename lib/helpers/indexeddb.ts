/**
 * IndexedDB wrapper for persistent storage
 */

const DB_NAME = 'CodeBlanketDB';
const DB_VERSION = 1;
const STORE_NAME = 'progress';

interface ProgressData {
  id: string;
  data: unknown;
  timestamp: number;
}

let dbInstance: IDBDatabase | null = null;

/**
 * Initialize and open the IndexedDB database
 */
async function openDB(): Promise<IDBDatabase> {
  if (dbInstance) return dbInstance;

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      dbInstance = request.result;
      resolve(request.result);
    };

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
  });
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
      'module-',
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

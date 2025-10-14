/**
 * Export/Import functionality for user progress
 */

import {
  getAllData,
  importData,
  migrateFromLocalStorage,
  saveVideo,
} from './indexeddb';

export interface ExportData {
  version: string;
  exportDate: string;
  data: Record<string, unknown>;
  videos?: Array<{
    id: string;
    data: string; // base64 encoded video
    timestamp: number;
  }>;
}

/**
 * Get all data from localStorage that should be backed up
 */
function getLocalStorageData(): Record<string, unknown> {
  const data: Record<string, unknown> = {};
  const prefixes = [
    'codeblanket_completed_problems',
    'codeblanket_code_',
    'codeblanket_tests_', // Custom test cases
    'module-', // Module completion
    'mc-quiz-', // Multiple choice quiz progress
  ];

  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (!key) continue;

    // Skip migration and backup keys
    if (
      key === 'codeblanket_migration_complete' ||
      key === 'codeblanket-auto-backup'
    ) {
      continue;
    }

    // Check if key matches our prefixes
    const shouldInclude = prefixes.some((prefix) => key.startsWith(prefix));

    if (shouldInclude) {
      const value = localStorage.getItem(key);
      if (value) {
        try {
          data[key] = JSON.parse(value);
        } catch {
          data[key] = value;
        }
      }
    }
  }

  return data;
}

/**
 * Get all videos from IndexedDB video store
 */
async function getAllVideos(): Promise<
  Array<{ id: string; data: string; timestamp: number }>
> {
  try {
    const db = await openVideoStore();
    const transaction = db.transaction('videos', 'readonly');
    const store = transaction.objectStore('videos');

    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = async () => {
        const videos = request.result || [];
        const exportVideos: Array<{
          id: string;
          data: string;
          timestamp: number;
        }> = [];

        // Convert each video blob to base64
        for (const video of videos) {
          try {
            const base64 = await blobToBase64(video.video);
            exportVideos.push({
              id: video.id,
              data: base64,
              timestamp: video.timestamp,
            });
          } catch (error) {
            console.error(`Failed to convert video ${video.id}:`, error);
          }
        }

        resolve(exportVideos);
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('Failed to get videos:', error);
    return [];
  }
}

/**
 * Open IndexedDB database for video store
 */
async function openVideoStore(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('CodeBlanketDB', 2);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Convert Blob to base64 string
 */
function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      // Remove the data URL prefix (e.g., "data:video/webm;base64,")
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Export all progress data to a JSON file
 */
export async function exportProgress(): Promise<void> {
  try {
    // Get data from both IndexedDB and localStorage
    let data = await getAllData();

    // If IndexedDB is empty or has very little data, use localStorage as fallback
    const indexedDBKeys = Object.keys(data);
    if (indexedDBKeys.length < 2) {
      console.warn(
        'IndexedDB has minimal data, including localStorage data in export',
      );
      const localStorageData = getLocalStorageData();
      data = { ...localStorageData, ...data }; // Merge, preferring IndexedDB
    }

    // Get all videos
    const videos = await getAllVideos();

    const exportData: ExportData = {
      version: '1.0',
      exportDate: new Date().toISOString(),
      data,
      videos: videos.length > 0 ? videos : undefined,
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `codeblanket-progress-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    // Log export summary
    const videoCount = videos.length;
    const videoSize = videos.reduce((sum, v) => sum + v.data.length * 0.75, 0); // Approximate size in bytes
    if (videoCount > 0) {
      console.warn(
        `Exported ${Object.keys(data).length} data entries and ${videoCount} videos (${(videoSize / 1024 / 1024).toFixed(2)} MB)`,
      );
    }
  } catch (error) {
    console.error('Export error:', error);
    throw error;
  }
}

/**
 * Convert base64 string back to Blob
 */
function base64ToBlob(base64: string, mimeType = 'video/webm'): Blob {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mimeType });
}

/**
 * Import progress data from a JSON file
 */
export async function importProgress(file: File): Promise<void> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = async (e) => {
      try {
        const content = e.target?.result as string;
        const importedData: ExportData = JSON.parse(content);

        // Validate the data
        if (!importedData.version || !importedData.data) {
          throw new Error('Invalid export file format');
        }

        // Import to IndexedDB
        await importData(importedData.data);

        // Also restore to localStorage for immediate access
        Object.entries(importedData.data).forEach(([key, value]) => {
          try {
            // If value is already a string (like code), store it directly
            // Otherwise, stringify it (like arrays, objects)
            const valueToStore =
              typeof value === 'string' ? value : JSON.stringify(value);
            localStorage.setItem(key, valueToStore);
          } catch (error) {
            console.error(`Failed to restore ${key} to localStorage:`, error);
          }
        });

        // Import videos if present
        if (importedData.videos && importedData.videos.length > 0) {
          for (const video of importedData.videos) {
            try {
              const blob = base64ToBlob(video.data);
              await saveVideo(video.id, blob);
            } catch (error) {
              console.error(`Failed to import video ${video.id}:`, error);
            }
          }
        }

        // Trigger storage event to update UI
        window.dispatchEvent(new Event('storage'));

        resolve();
      } catch (error) {
        console.error('Import error:', error);
        reject(error);
      }
    };

    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

/**
 * Download a backup of current progress (auto-backup)
 */
export async function createAutoBackup(): Promise<void> {
  try {
    const data = await getAllData();
    const backupKey = 'codeblanket-auto-backup';
    const backup = {
      date: new Date().toISOString(),
      data,
    };

    // Store in localStorage as a last resort backup
    localStorage.setItem(backupKey, JSON.stringify(backup));
  } catch (error) {
    console.error('Auto-backup error:', error);
  }
}

/**
 * Force sync all localStorage data to IndexedDB
 * Useful for debugging or ensuring all data is backed up
 */
export async function forceSyncToIndexedDB(): Promise<void> {
  try {
    // Clear migration flag to force re-migration
    localStorage.removeItem('codeblanket_migration_complete');

    // Run migration
    await migrateFromLocalStorage();

    // Restore migration flag
    localStorage.setItem('codeblanket_migration_complete', 'true');
  } catch (error) {
    console.error('Force sync error:', error);
    throw error;
  }
}

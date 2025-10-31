/**
 * Export/Import functionality for user progress
 */

import {
  importData,
  migrateFromLocalStorage,
  saveVideo,
} from './storage-adapter';
import * as indexedDB from './indexeddb';

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
 * Validate and clean multiple choice quiz data
 */
function validateMCQuizData(key: string, data: string[]): string[] {
  // Extract section ID from key (e.g., "mc-quiz-python-fundamentals-variables-types" -> "variables-types")
  let sectionId = '';
  if (key.startsWith('mc-quiz-python-fundamentals-')) {
    sectionId = key.replace('mc-quiz-python-fundamentals-', '');
  } else if (key.startsWith('mc-quiz-python-intermediate-')) {
    sectionId = key.replace('mc-quiz-python-intermediate-', '');
  } else if (key.startsWith('mc-quiz-python-advanced-')) {
    sectionId = key.replace('mc-quiz-python-advanced-', '');
  } else if (key.startsWith('mc-quiz-python-oop-')) {
    sectionId = key.replace('mc-quiz-python-oop-', '');
  } else {
    // For other modules, can't validate, just deduplicate
    return [...new Set(data)];
  }

  // Define valid prefixes for each section
  const sectionPrefixes: Record<string, string[]> = {
    'variables-types': ['pf-variables-mc-'],
    'control-flow': ['pf-control-mc-'],
    'data-structures': ['pf-datastructures-mc-'],
    functions: ['pf-functions-mc-'],
    strings: ['pf-strings-mc-'],
    'none-handling': ['pf-none-mc-'],
    'modules-imports': ['mc'], // Generic IDs
    'list-comprehensions': ['mc'],
    'lambda-functions': ['mc'],
    'built-in-functions': ['mc'],
  };

  const validPrefixes = sectionPrefixes[sectionId];
  if (!validPrefixes) {
    // Unknown section, just deduplicate
    return [...new Set(data)];
  }

  // Filter to only valid questions for this section
  const validQuestions = data.filter((qId) =>
    validPrefixes.some((prefix) => qId.startsWith(prefix)),
  );

  // Deduplicate
  const uniqueQuestions = [...new Set(validQuestions)];

  if (uniqueQuestions.length !== data.length) {
    const removed = data.length - uniqueQuestions.length;
    const invalid = data.filter((qId) => !validQuestions.includes(qId));
    console.warn(
      `Export: Cleaned ${key}: ${data.length} → ${uniqueQuestions.length} (removed ${removed} invalid: ${invalid.join(', ')})`,
    );
  }

  return uniqueQuestions;
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
          const parsed = JSON.parse(value);

          // Special handling for mc-quiz data: validate and deduplicate
          if (key.startsWith('mc-quiz-') && Array.isArray(parsed)) {
            data[key] = validateMCQuizData(key, parsed);
          } else {
            data[key] = parsed;
          }
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
 * Always uses IndexedDB for export (includes both anonymous and authenticated users' local data)
 */
async function getAllVideos(): Promise<
  Array<{ id: string; data: string; timestamp: number }>
> {
  try {
    // Get all videos from IndexedDB
    const videos = await indexedDB.getVideosForQuestion(''); // Empty prefix = all videos
    const exportVideos: Array<{
      id: string;
      data: string;
      timestamp: number;
    }> = [];

    // Convert each video blob to base64
    for (const video of videos) {
      try {
        const base64 = await blobToBase64(video.blob);
        exportVideos.push({
          id: video.id,
          data: base64,
          timestamp: video.timestamp,
        });
      } catch (error) {
        console.error(`Failed to convert video ${video.id}:`, error);
      }
    }

    return exportVideos;
  } catch (error) {
    console.error('Failed to get videos:', error);
    return [];
  }
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
    // Get data from IndexedDB directly (works for both anonymous and authenticated)
    const indexedDBData = await indexedDB.getAllData();
    const localStorageData = getLocalStorageData();

    // Merge both sources, preferring IndexedDB for duplicates
    // This ensures we capture module completions and MC quiz progress even if not yet synced
    const data = { ...localStorageData, ...indexedDBData };

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
            let finalValue = value;

            // Special handling for mc-quiz data: validate and deduplicate on import
            if (key.startsWith('mc-quiz-') && Array.isArray(value)) {
              finalValue = validateMCQuizData(key, value);
            }

            // If value is already a string (like code), store it directly
            // Otherwise, stringify it (like arrays, objects)
            const valueToStore =
              typeof finalValue === 'string'
                ? finalValue
                : JSON.stringify(finalValue);
            localStorage.setItem(key, valueToStore);
          } catch (error) {
            console.error(`Failed to restore ${key} to localStorage:`, error);
          }
        });

        // Import videos if present
        if (importedData.videos && importedData.videos.length > 0) {
          console.warn(`Importing ${importedData.videos.length} videos...`);
          for (const video of importedData.videos) {
            try {
              const blob = base64ToBlob(video.data);
              await saveVideo(video.id, blob);
            } catch (error) {
              console.error(`Failed to import video ${video.id}:`, error);
            }
          }
          console.warn('Video import complete. Refresh the page to see them.');
        }

        // Don't auto-refresh - let user manually refresh after import completes
        // This ensures videos are fully imported before the UI tries to load them
        // window.dispatchEvent(new Event('storage'));

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
 * Uses IndexedDB directly to avoid API calls on page load
 */
export async function createAutoBackup(): Promise<void> {
  try {
    // Import directly from indexeddb to avoid API calls
    const indexedDB = await import('./indexeddb');
    const data = await indexedDB.getAllData();

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

/**
 * Storage Adapter - Dual storage strategy
 *
 * Switches between IndexedDB (anonymous users) and PostgreSQL (authenticated users)
 * All storage operations go through this adapter to ensure the right storage backend is used
 */

import * as indexedDB from './indexeddb';

/**
 * Check if user is authenticated by calling the API
 * This is client-side, so we need to check via API
 */
async function isUserAuthenticated(): Promise<boolean> {
  // Only run in browser
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

/**
 * Set a value in storage (IndexedDB or PostgreSQL)
 */
export async function setItem(key: string, value: unknown): Promise<void> {
  const isAuthenticated = await isUserAuthenticated();

  if (isAuthenticated) {
    // Use PostgreSQL via API
    try {
      const response = await fetch('/api/progress', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ key, value }),
      });

      if (!response.ok) {
        throw new Error('Failed to save to database');
      }
    } catch (error) {
      console.error(
        'Failed to save to PostgreSQL, falling back to IndexedDB:',
        error,
      );
      // Fallback to IndexedDB
      await indexedDB.setItem(key, value);
    }
  } else {
    // Use IndexedDB for anonymous users
    await indexedDB.setItem(key, value);
  }
}

/**
 * Get a value from storage (IndexedDB or PostgreSQL)
 */
export async function getItem<T>(key: string): Promise<T | null> {
  const isAuthenticated = await isUserAuthenticated();

  if (isAuthenticated) {
    // Use PostgreSQL via API
    try {
      const response = await fetch(
        `/api/progress?key=${encodeURIComponent(key)}`,
      );

      if (!response.ok) {
        throw new Error('Failed to fetch from database');
      }

      const data = await response.json();
      return data.value as T | null;
    } catch (error) {
      console.error(
        'Failed to fetch from PostgreSQL, falling back to IndexedDB:',
        error,
      );
      // Fallback to IndexedDB
      return await indexedDB.getItem<T>(key);
    }
  } else {
    // Use IndexedDB for anonymous users
    return await indexedDB.getItem<T>(key);
  }
}

/**
 * Remove a value from storage (IndexedDB or PostgreSQL)
 */
export async function removeItem(key: string): Promise<void> {
  const isAuthenticated = await isUserAuthenticated();

  if (isAuthenticated) {
    // Use PostgreSQL via API
    try {
      const response = await fetch(
        `/api/progress?key=${encodeURIComponent(key)}`,
        {
          method: 'DELETE',
        },
      );

      if (!response.ok) {
        throw new Error('Failed to delete from database');
      }
    } catch (error) {
      console.error(
        'Failed to delete from PostgreSQL, falling back to IndexedDB:',
        error,
      );
      // Fallback to IndexedDB
      await indexedDB.removeItem(key);
    }
  } else {
    // Use IndexedDB for anonymous users
    await indexedDB.removeItem(key);
  }
}

/**
 * Get all data from storage (IndexedDB or PostgreSQL)
 */
export async function getAllData(): Promise<Record<string, unknown>> {
  const isAuthenticated = await isUserAuthenticated();

  if (isAuthenticated) {
    // Use PostgreSQL via API
    try {
      const response = await fetch('/api/progress');

      if (!response.ok) {
        throw new Error('Failed to fetch all data from database');
      }

      const result = await response.json();
      return result.data as Record<string, unknown>;
    } catch (error) {
      console.error(
        'Failed to fetch all data from PostgreSQL, falling back to IndexedDB:',
        error,
      );
      // Fallback to IndexedDB
      return await indexedDB.getAllData();
    }
  } else {
    // Use IndexedDB for anonymous users
    return await indexedDB.getAllData();
  }
}

/**
 * Import data into storage (IndexedDB or PostgreSQL)
 */
export async function importData(data: Record<string, unknown>): Promise<void> {
  const isAuthenticated = await isUserAuthenticated();

  if (isAuthenticated) {
    // Use PostgreSQL via API
    try {
      const response = await fetch('/api/progress/import', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data }),
      });

      if (!response.ok) {
        throw new Error('Failed to import data to database');
      }
    } catch (error) {
      console.error(
        'Failed to import to PostgreSQL, falling back to IndexedDB:',
        error,
      );
      // Fallback to IndexedDB
      await indexedDB.importData(data);
    }
  } else {
    // Use IndexedDB for anonymous users
    await indexedDB.importData(data);
  }
}

/**
 * Save a video (IndexedDB or PostgreSQL)
 */
export async function saveVideo(id: string, video: Blob): Promise<void> {
  const isAuthenticated = await isUserAuthenticated();

  if (isAuthenticated) {
    // Use PostgreSQL via API
    try {
      const formData = new FormData();
      formData.append('videoId', id);
      formData.append('video', video);

      const response = await fetch('/api/videos', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to save video to database');
      }
    } catch (error) {
      console.error(
        'Failed to save video to PostgreSQL, falling back to IndexedDB:',
        error,
      );
      // Fallback to IndexedDB
      await indexedDB.saveVideo(id, video);
    }
  } else {
    // Use IndexedDB for anonymous users
    await indexedDB.saveVideo(id, video);
  }
}

/**
 * Get a video (IndexedDB or Vercel Blob via PostgreSQL)
 */
export async function getVideo(id: string): Promise<Blob | null> {
  const isAuthenticated = await isUserAuthenticated();

  if (isAuthenticated) {
    // Use Vercel Blob Storage via API
    try {
      // First, get the video metadata which includes the blob URL
      const response = await fetch('/api/videos');

      if (!response.ok) {
        throw new Error('Failed to fetch video metadata');
      }

      const data = await response.json();
      const video = data.videos.find(
        (v: { id: string; blobUrl: string }) => v.id === id,
      );

      if (!video) {
        return null;
      }

      // Fetch the actual video from Vercel Blob Storage
      const blobResponse = await fetch(video.blobUrl);

      if (!blobResponse.ok) {
        throw new Error('Failed to fetch video from blob storage');
      }

      return await blobResponse.blob();
    } catch (error) {
      console.error(
        'Failed to fetch video from Vercel Blob, falling back to IndexedDB:',
        error,
      );
      // Fallback to IndexedDB
      return await indexedDB.getVideo(id);
    }
  } else {
    // Use IndexedDB for anonymous users
    return await indexedDB.getVideo(id);
  }
}

/**
 * Migrate data from IndexedDB to PostgreSQL
 * Called when user signs in
 */
export async function migrateToPostgreSQL(): Promise<void> {
  try {
    console.log('Starting migration from IndexedDB to PostgreSQL...');

    // Get all data from IndexedDB
    const data = await indexedDB.getAllData();

    if (Object.keys(data).length === 0) {
      console.log('No data to migrate');
      return;
    }

    // Import to PostgreSQL
    const response = await fetch('/api/progress/import', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ data }),
    });

    if (!response.ok) {
      throw new Error('Failed to migrate data');
    }

    console.log(
      `Successfully migrated ${Object.keys(data).length} items to PostgreSQL`,
    );

    // Optionally: Clear IndexedDB after successful migration
    // await indexedDB.clearAllData();
  } catch (error) {
    console.error('Migration failed:', error);
    throw error;
  }
}

/**
 * Get video metadata only (without downloading the actual video)
 * Use this to show video indicators without consuming bandwidth
 */
export async function getVideoMetadataForQuestion(
  questionIdPrefix: string,
): Promise<
  Array<{ id: string; blobUrl: string; timestamp: number; size?: number }>
> {
  const isAuthenticated = await isUserAuthenticated();

  if (isAuthenticated) {
    // Use Vercel Blob Storage via API (metadata only, no bandwidth)
    try {
      const response = await fetch('/api/videos');

      if (!response.ok) {
        throw new Error('Failed to fetch videos metadata');
      }

      const result = await response.json();
      const videos = result.videos as Array<{
        id: string;
        blobUrl: string;
        mimeType: string;
        timestamp: string;
        size?: number;
      }>;

      // Filter by prefix and return metadata only
      const filtered = videos
        .filter((v) => v.id.startsWith(questionIdPrefix))
        .map((v) => ({
          id: v.id,
          blobUrl: v.blobUrl,
          timestamp: new Date(v.timestamp).getTime(),
          size: v.size,
        }))
        .sort((a, b) => a.timestamp - b.timestamp);

      return filtered;
    } catch (error) {
      console.error(
        'Failed to fetch videos metadata, falling back to IndexedDB:',
        error,
      );
      // Fallback to IndexedDB - get metadata from IndexedDB videos
      const videos = await indexedDB.getVideosForQuestion(questionIdPrefix);
      return videos.map((v) => ({
        id: v.id,
        blobUrl: '', // IndexedDB doesn't have blob URLs
        timestamp: v.timestamp,
      }));
    }
  } else {
    // Use IndexedDB for anonymous users
    const videos = await indexedDB.getVideosForQuestion(questionIdPrefix);
    return videos.map((v) => ({
      id: v.id,
      blobUrl: '', // IndexedDB doesn't have blob URLs
      timestamp: v.timestamp,
    }));
  }
}

/**
 * Get all videos for a specific question (by prefix)
 * WARNING: This downloads all video files - use getVideoMetadataForQuestion() instead
 * for displaying video lists to save bandwidth
 */
export async function getVideosForQuestion(
  questionIdPrefix: string,
): Promise<Array<{ id: string; blob: Blob; timestamp: number }>> {
  const isAuthenticated = await isUserAuthenticated();

  if (isAuthenticated) {
    // Use Vercel Blob Storage via API
    try {
      const response = await fetch('/api/videos');

      if (!response.ok) {
        throw new Error('Failed to fetch videos metadata');
      }

      const result = await response.json();
      const videos = result.videos as Array<{
        id: string;
        blobUrl: string;
        mimeType: string;
        timestamp: string;
      }>;

      // Filter by prefix and fetch each video from Vercel Blob
      const filtered = videos.filter((v) => v.id.startsWith(questionIdPrefix));
      const videosWithBlobs = await Promise.all(
        filtered.map(async (v) => {
          const blobResponse = await fetch(v.blobUrl);
          const blob = await blobResponse.blob();
          return {
            id: v.id,
            blob,
            timestamp: new Date(v.timestamp).getTime(),
          };
        }),
      );

      return videosWithBlobs.sort((a, b) => a.timestamp - b.timestamp);
    } catch (error) {
      console.error(
        'Failed to fetch videos from Vercel Blob, falling back to IndexedDB:',
        error,
      );
      // Fallback to IndexedDB
      return await indexedDB.getVideosForQuestion(questionIdPrefix);
    }
  } else {
    // Use IndexedDB for anonymous users
    return await indexedDB.getVideosForQuestion(questionIdPrefix);
  }
}

/**
 * Load a specific video on-demand from its blob URL
 * Use this when user clicks to view a video
 */
export async function loadVideoFromUrl(blobUrl: string): Promise<Blob | null> {
  if (!blobUrl) {
    return null;
  }

  try {
    const response = await fetch(blobUrl);
    if (!response.ok) {
      throw new Error('Failed to fetch video');
    }
    return await response.blob();
  } catch (error) {
    console.error('Failed to load video from URL:', error);
    return null;
  }
}

/**
 * Delete a video
 */
export async function deleteVideo(videoId: string): Promise<void> {
  const isAuthenticated = await isUserAuthenticated();

  if (isAuthenticated) {
    // Use PostgreSQL via API
    try {
      const response = await fetch(
        `/api/videos?videoId=${encodeURIComponent(videoId)}`,
        {
          method: 'DELETE',
        },
      );

      if (!response.ok) {
        throw new Error('Failed to delete video from database');
      }
    } catch (error) {
      console.error(
        'Failed to delete video from PostgreSQL, falling back to IndexedDB:',
        error,
      );
      // Fallback to IndexedDB
      await indexedDB.deleteVideo(videoId);
    }
  } else {
    // Use IndexedDB for anonymous users
    await indexedDB.deleteVideo(videoId);
  }
}

/**
 * Get count of completed discussion questions (questions with at least one video)
 */
export async function getCompletedDiscussionQuestionsCount(): Promise<number> {
  try {
    const isAuthenticated = await isUserAuthenticated();

    if (isAuthenticated) {
      // Use Vercel Blob Storage metadata via API
      try {
        const response = await fetch('/api/videos');

        if (!response.ok) {
          // API failed, fallback to IndexedDB
          console.warn(
            'Failed to fetch videos from API, using IndexedDB fallback',
          );
          return await indexedDB.getCompletedDiscussionQuestionsCount();
        }

        const result = await response.json();
        const videos = result.videos as Array<{ id: string }>;

        // Extract unique question prefixes (before the timestamp)
        const uniqueQuestions = new Set<string>();
        videos.forEach((item) => {
          // Video ID format: moduleSlug-sectionId-questionId-timestamp
          // We need to extract: moduleSlug-sectionId-questionId
          const parts = item.id.split('-');
          if (parts.length >= 4) {
            const questionPrefix = parts.slice(0, -1).join('-');
            uniqueQuestions.add(questionPrefix);
          }
        });

        return uniqueQuestions.size;
      } catch (error) {
        console.warn(
          'Error fetching videos from API, falling back to IndexedDB:',
          error,
        );
        // Fallback to IndexedDB
        return await indexedDB.getCompletedDiscussionQuestionsCount();
      }
    } else {
      // Use IndexedDB for anonymous users
      return await indexedDB.getCompletedDiscussionQuestionsCount();
    }
  } catch (error) {
    // Ultimate fallback
    console.error('Error in getCompletedDiscussionQuestionsCount:', error);
    return 0;
  }
}

/**
 * Re-export other IndexedDB functions that don't need adaptation
 */
export {
  migrateFromLocalStorage,
  getTotalMultipleChoiceQuestionsCount,
  getCompletedMultipleChoiceQuestionsCount,
  getTotalDiscussionQuestionsCount,
} from './indexeddb';

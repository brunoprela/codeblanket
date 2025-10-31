/**
 * Neon database client for server-side operations
 * Used for authenticated users to store their progress in PostgreSQL
 */

import { neon } from '@neondatabase/serverless';

if (!process.env.DATABASE_URL) {
  throw new Error('DATABASE_URL environment variable is not set');
}

export const sql = neon(process.env.DATABASE_URL);

/**
 * User progress operations
 */

export interface ProgressRecord {
  id: string;
  user_id: string;
  key: string;
  value: unknown;
  created_at: string;
  updated_at: string;
}

/**
 * Set a progress item for a user
 */
export async function setProgressItem(
  userId: string,
  key: string,
  value: unknown,
): Promise<void> {
  await sql`
    INSERT INTO user_progress (user_id, key, value)
    VALUES (${userId}, ${key}, ${JSON.stringify(value)})
    ON CONFLICT (user_id, key)
    DO UPDATE SET value = ${JSON.stringify(value)}, updated_at = NOW()
  `;
}

/**
 * Get a progress item for a user
 */
export async function getProgressItem(
  userId: string,
  key: string,
): Promise<unknown | null> {
  const result = await sql`
    SELECT value FROM user_progress
    WHERE user_id = ${userId} AND key = ${key}
  `;

  if (result.length === 0) {
    return null;
  }

  return result[0].value;
}

/**
 * Get all progress data for a user
 */
export async function getAllProgressData(
  userId: string,
): Promise<Record<string, unknown>> {
  const result = await sql`
    SELECT key, value FROM user_progress
    WHERE user_id = ${userId}
  `;

  const data: Record<string, unknown> = {};
  for (const row of result) {
    data[row.key as string] = row.value;
  }

  return data;
}

/**
 * Delete a progress item for a user
 */
export async function removeProgressItem(
  userId: string,
  key: string,
): Promise<void> {
  await sql`
    DELETE FROM user_progress
    WHERE user_id = ${userId} AND key = ${key}
  `;
}

/**
 * Import bulk progress data for a user (overwrites existing)
 */
export async function importProgressData(
  userId: string,
  data: Record<string, unknown>,
): Promise<void> {
  // Delete existing data
  await sql`DELETE FROM user_progress WHERE user_id = ${userId}`;

  // Insert new data
  const entries = Object.entries(data);
  if (entries.length === 0) return;

  // Batch insert for performance
  const values = entries.map(([key, value]) => ({
    userId,
    key,
    value: JSON.stringify(value),
  }));

  for (const { userId: uid, key, value } of values) {
    await sql`
      INSERT INTO user_progress (user_id, key, value)
      VALUES (${uid}, ${key}, ${value})
    `;
  }
}

/**
 * Video operations
 * Videos are stored in Vercel Blob Storage, metadata in PostgreSQL
 */

export interface VideoRecord {
  id: string;
  user_id: string;
  video_id: string;
  blob_url: string;
  blob_pathname: string;
  mime_type: string;
  file_size: number;
  created_at: string;
  updated_at: string;
}

export interface VideoMetadata {
  blobUrl: string;
  blobPathname: string;
  mimeType: string;
  fileSize: number;
}

/**
 * Save video metadata (after uploading to Vercel Blob)
 */
export async function saveVideoMetadata(
  userId: string,
  videoId: string,
  metadata: VideoMetadata,
): Promise<void> {
  await sql`
    INSERT INTO user_videos (user_id, video_id, blob_url, blob_pathname, mime_type, file_size)
    VALUES (${userId}, ${videoId}, ${metadata.blobUrl}, ${metadata.blobPathname}, ${metadata.mimeType}, ${metadata.fileSize})
    ON CONFLICT (user_id, video_id)
    DO UPDATE SET 
      blob_url = ${metadata.blobUrl}, 
      blob_pathname = ${metadata.blobPathname},
      mime_type = ${metadata.mimeType}, 
      file_size = ${metadata.fileSize},
      updated_at = NOW()
  `;
}

/**
 * Get video metadata for a user
 */
export async function getVideoMetadata(
  userId: string,
  videoId: string,
): Promise<VideoMetadata | null> {
  const result = await sql`
    SELECT blob_url, blob_pathname, mime_type, file_size FROM user_videos
    WHERE user_id = ${userId} AND video_id = ${videoId}
  `;

  if (result.length === 0) {
    return null;
  }

  return {
    blobUrl: result[0].blob_url as string,
    blobPathname: result[0].blob_pathname as string,
    mimeType: result[0].mime_type as string,
    fileSize: result[0].file_size as number,
  };
}

/**
 * Get all video metadata for a user
 */
export async function getAllVideoMetadata(userId: string): Promise<
  Array<{
    id: string;
    blobUrl: string;
    blobPathname: string;
    mimeType: string;
    timestamp: string;
  }>
> {
  const result = await sql`
    SELECT video_id, blob_url, blob_pathname, mime_type, created_at FROM user_videos
    WHERE user_id = ${userId}
  `;

  return result.map((row) => ({
    id: row.video_id as string,
    blobUrl: row.blob_url as string,
    blobPathname: row.blob_pathname as string,
    mimeType: row.mime_type as string,
    timestamp: row.created_at as string,
  }));
}

/**
 * Delete video metadata (call after deleting from Vercel Blob)
 */
export async function deleteVideoMetadata(
  userId: string,
  videoId: string,
): Promise<{ blobPathname: string } | null> {
  // Get the blob pathname before deleting (needed to delete from Vercel Blob)
  const result = await sql`
    SELECT blob_pathname FROM user_videos
    WHERE user_id = ${userId} AND video_id = ${videoId}
  `;

  if (result.length === 0) {
    return null;
  }

  const blobPathname = result[0].blob_pathname as string;

  // Delete the metadata
  await sql`
    DELETE FROM user_videos
    WHERE user_id = ${userId} AND video_id = ${videoId}
  `;

  return { blobPathname };
}

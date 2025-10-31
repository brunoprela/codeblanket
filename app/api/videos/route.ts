/**
 * API route for video operations
 * Handles GET (fetch), POST (upload), and DELETE operations
 * Videos are stored in Vercel Blob Storage, metadata in PostgreSQL
 */

import { NextRequest, NextResponse } from 'next/server';
import { put, del } from '@vercel/blob';
import { stackServerApp } from '@/lib/stack';
import {
  saveVideoMetadata,
  getVideoMetadata,
  getAllVideoMetadata,
  deleteVideoMetadata,
} from '@/lib/db/neon';

/**
 * GET /api/videos - Get all videos metadata or redirect to a specific video
 * Query params: videoId (optional)
 */
export async function GET(request: NextRequest) {
  try {
    console.log('[API /api/videos GET] Request received');
    const user = await stackServerApp.getUser();
    console.log(
      '[API /api/videos GET] User:',
      user ? `${user.id} (${user.primaryEmail})` : 'null',
    );

    if (!user) {
      console.log('[API /api/videos GET] Returning 401 - No user');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const searchParams = request.nextUrl.searchParams;
    const videoId = searchParams.get('videoId');

    if (videoId) {
      // Get specific video metadata
      const metadata = await getVideoMetadata(user.id, videoId);

      if (!metadata) {
        return NextResponse.json({ error: 'Video not found' }, { status: 404 });
      }

      // Redirect to the Vercel Blob URL
      // The blob URL is already signed and accessible
      return NextResponse.redirect(metadata.blobUrl);
    } else {
      // Get all videos metadata
      console.log(
        '[API /api/videos GET] Fetching all videos for user:',
        user.id,
      );
      const videos = await getAllVideoMetadata(user.id);
      console.log(
        '[API /api/videos GET] Success - found',
        videos.length,
        'videos',
      );
      const videosList = videos.map((v) => ({
        id: v.id,
        blobUrl: v.blobUrl,
        mimeType: v.mimeType,
        timestamp: v.timestamp,
      }));

      return NextResponse.json({ videos: videosList });
    }
  } catch (error) {
    console.error('[API /api/videos GET] Error:', error);
    return NextResponse.json(
      {
        error: 'Failed to fetch videos',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 },
    );
  }
}

/**
 * POST /api/videos - Upload a video to Vercel Blob Storage
 * Body: FormData with videoId and video file
 */
export async function POST(request: NextRequest) {
  const user = await stackServerApp.getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const formData = await request.formData();
    const videoId = formData.get('videoId') as string;
    const videoFile = formData.get('video') as File;

    if (!videoId || !videoFile) {
      return NextResponse.json(
        { error: 'Missing videoId or video file' },
        { status: 400 },
      );
    }

    // Upload to Vercel Blob Storage
    // Pathname format: videos/{userId}/{videoId}.webm
    const pathname = `videos/${user.id}/${videoId}.webm`;

    const blob = await put(pathname, videoFile, {
      access: 'public', // Make videos publicly accessible
      addRandomSuffix: false, // Use consistent pathname
    });

    // Save metadata to PostgreSQL
    await saveVideoMetadata(user.id, videoId, {
      blobUrl: blob.url,
      blobPathname: blob.pathname,
      mimeType: videoFile.type,
      fileSize: videoFile.size,
    });

    return NextResponse.json({
      success: true,
      blobUrl: blob.url,
    });
  } catch (error) {
    console.error('POST /api/videos error:', error);
    return NextResponse.json(
      { error: 'Failed to save video' },
      { status: 500 },
    );
  }
}

/**
 * DELETE /api/videos - Delete a video from Vercel Blob and database
 * Query params: videoId (required)
 */
export async function DELETE(request: NextRequest) {
  const user = await stackServerApp.getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const searchParams = request.nextUrl.searchParams;
    const videoId = searchParams.get('videoId');

    if (!videoId) {
      return NextResponse.json({ error: 'Missing videoId' }, { status: 400 });
    }

    // Get the blob pathname from database and delete the record
    const metadata = await deleteVideoMetadata(user.id, videoId);

    if (!metadata) {
      return NextResponse.json({ error: 'Video not found' }, { status: 404 });
    }

    // Delete from Vercel Blob Storage
    await del(metadata.blobPathname);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('DELETE /api/videos error:', error);
    return NextResponse.json(
      { error: 'Failed to delete video' },
      { status: 500 },
    );
  }
}

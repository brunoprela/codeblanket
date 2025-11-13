import { NextRequest, NextResponse } from 'next/server';
import { handleUpload, type HandleUploadBody } from '@vercel/blob/client';
import { stackServerApp } from '@/lib/stack';

const MAX_VIDEO_SIZE_BYTES = 200 * 1024 * 1024; // 200MB safety limit

export async function POST(request: NextRequest) {
  const user = await stackServerApp.getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = (await request.json()) as HandleUploadBody;

    const response = await handleUpload({
      request,
      body,
      onBeforeGenerateToken: async (pathname, clientPayload, multipart) => {
        let payload: Record<string, unknown> = {};
        try {
          payload = clientPayload ? JSON.parse(clientPayload) : {};
        } catch (error) {
          console.warn(
            '[API /api/videos/upload] Failed to parse clientPayload:',
            error,
          );
        }

        return {
          access: 'public' as const,
          addRandomSuffix: false,
          allowOverwrite: false,
          maximumSizeInBytes: MAX_VIDEO_SIZE_BYTES,
          allowedContentTypes: ['video/webm'],
          cacheControlMaxAge: 60 * 60 * 24 * 30, // 30 days
          tokenPayload: JSON.stringify({
            ...payload,
            userId: user.id,
            multipart,
            originalPathname: pathname,
          }),
        };
      },
      onUploadCompleted: async ({ blob, tokenPayload }) => {
        console.info('[API /api/videos/upload] Upload completed', {
          pathname: blob.pathname,
          tokenPayload,
        });
        // Metadata persistence is handled client-side after upload completes.
      },
    });

    return NextResponse.json(response);
  } catch (error) {
    console.error('[API /api/videos/upload] Error handling upload:', error);
    return NextResponse.json(
      { error: 'Failed to handle client upload' },
      { status: 400 },
    );
  }
}

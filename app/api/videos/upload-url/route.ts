import { NextRequest, NextResponse } from 'next/server';
import { createUploadURL } from '@vercel/blob';
import { stackServerApp } from '@/lib/stack';

export async function POST(request: NextRequest) {
  const user = await stackServerApp.getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const { videoId, contentType, fileSize } = (await request.json()) as {
      videoId?: string;
      contentType?: string;
      fileSize?: number;
    };

    if (!videoId || !contentType) {
      return NextResponse.json(
        { error: 'Missing videoId or contentType' },
        { status: 400 },
      );
    }

    const pathname = `videos/${user.id}/${videoId}.webm`;

    const { url, token } = await createUploadURL({
      access: 'public',
      addRandomSuffix: false,
      contentType,
      metadata: {
        userId: user.id,
        videoId,
        fileSize: fileSize ? String(fileSize) : undefined,
      },
      pathname,
    });

    return NextResponse.json({
      uploadUrl: url,
      uploadToken: token,
      blobPathname: pathname,
    });
  } catch (error) {
    console.error(
      '[API /api/videos/upload-url] Error creating upload URL:',
      error,
    );
    return NextResponse.json(
      { error: 'Failed to create upload URL' },
      { status: 500 },
    );
  }
}

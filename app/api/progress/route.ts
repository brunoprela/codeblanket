/**
 * API route for user progress operations
 * Handles GET (fetch), POST (set), and DELETE operations
 */

import { NextRequest, NextResponse } from 'next/server';
import { stackServerApp } from '@/lib/stack';
import {
  setProgressItem,
  getProgressItem,
  getAllProgressData,
  removeProgressItem,
} from '@/lib/db/neon';

/**
 * GET /api/progress - Get all progress data or a specific item
 * Query params: key (optional)
 */
export async function GET(request: NextRequest) {
  try {
    console.log('[API /api/progress GET] Request received');
    const user = await stackServerApp.getUser();
    console.log(
      '[API /api/progress GET] User:',
      user ? `${user.id} (${user.primaryEmail})` : 'null',
    );

    if (!user) {
      console.log('[API /api/progress GET] Returning 401 - No user');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const searchParams = request.nextUrl.searchParams;
    const key = searchParams.get('key');

    if (key) {
      // Get specific item
      console.log('[API /api/progress GET] Fetching key:', key);
      const value = await getProgressItem(user.id, key);
      return NextResponse.json({ key, value });
    } else {
      // Get all data
      console.log(
        '[API /api/progress GET] Fetching all data for user:',
        user.id,
      );
      const data = await getAllProgressData(user.id);
      console.log(
        '[API /api/progress GET] Success - found',
        Object.keys(data).length,
        'items',
      );
      return NextResponse.json({ data });
    }
  } catch (error) {
    console.error('[API /api/progress GET] Error:', error);
    return NextResponse.json(
      {
        error: 'Failed to fetch progress',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 },
    );
  }
}

/**
 * POST /api/progress - Set a progress item
 * Body: { key: string, value: any }
 */
export async function POST(request: NextRequest) {
  const user = await stackServerApp.getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = await request.json();
    const { key, value } = body;

    if (!key) {
      return NextResponse.json({ error: 'Missing key' }, { status: 400 });
    }

    await setProgressItem(user.id, key, value);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('POST /api/progress error:', error);
    return NextResponse.json(
      { error: 'Failed to save progress' },
      { status: 500 },
    );
  }
}

/**
 * DELETE /api/progress - Delete a progress item
 * Query params: key (required)
 */
export async function DELETE(request: NextRequest) {
  const user = await stackServerApp.getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const searchParams = request.nextUrl.searchParams;
    const key = searchParams.get('key');

    if (!key) {
      return NextResponse.json({ error: 'Missing key' }, { status: 400 });
    }

    await removeProgressItem(user.id, key);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('DELETE /api/progress error:', error);
    return NextResponse.json(
      { error: 'Failed to delete progress' },
      { status: 500 },
    );
  }
}

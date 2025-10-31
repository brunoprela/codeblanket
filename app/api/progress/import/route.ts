/**
 * API route for importing bulk progress data
 * Used when migrating from IndexedDB to PostgreSQL
 */

import { NextRequest, NextResponse } from 'next/server';
import { stackServerApp } from '@/lib/stack';
import { importProgressData } from '@/lib/db/neon';

/**
 * POST /api/progress/import - Import bulk progress data
 * Body: { data: Record<string, any> }
 */
export async function POST(request: NextRequest) {
  const user = await stackServerApp.getUser();

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = await request.json();
    const { data } = body;

    if (!data || typeof data !== 'object') {
      return NextResponse.json(
        { error: 'Invalid data format' },
        { status: 400 },
      );
    }

    await importProgressData(user.id, data);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('POST /api/progress/import error:', error);
    return NextResponse.json(
      { error: 'Failed to import progress' },
      { status: 500 },
    );
  }
}

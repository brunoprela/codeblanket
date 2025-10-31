/**
 * Database connection test endpoint
 * Tests raw connection to Neon without any auth
 */

import { NextResponse } from 'next/server';
import { sql } from '@/lib/db/neon';

export async function GET() {
  try {
    console.debug('[DB Test] Testing database connection...');
    console.debug('[DB Test] DATABASE_URL exists:', !!process.env.DATABASE_URL);

    // Test basic connection
    const result = await sql`SELECT 1 as test, NOW() as current_time`;

    console.debug('[DB Test] Connection successful!', result);

    return NextResponse.json({
      success: true,
      message: 'Database connection works!',
      result: result[0],
      databaseUrl: process.env.DATABASE_URL ? 'Set (hidden)' : 'Not set',
    });
  } catch (error) {
    console.error('[DB Test] Connection failed:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        stack: error instanceof Error ? error.stack : undefined,
        databaseUrl: process.env.DATABASE_URL ? 'Set (hidden)' : 'Not set',
      },
      { status: 500 },
    );
  }
}

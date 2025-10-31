/**
 * API route to check if user is authenticated
 * Used by the storage adapter to determine which backend to use
 */

import { NextResponse } from 'next/server';
import { stackServerApp } from '@/lib/stack';

export async function GET() {
  try {
    const user = await stackServerApp.getUser();
    
    console.log('[Auth Check] User:', user ? `Authenticated as ${user.id}` : 'Not authenticated');
    
    return NextResponse.json({
      authenticated: !!user,
      userId: user?.id || null,
    });
  } catch (error) {
    console.error('[Auth Check] Error:', error);
    return NextResponse.json({
      authenticated: false,
      userId: null,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}

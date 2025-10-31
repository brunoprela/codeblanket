/**
 * Efficient stats API - returns counts without downloading all data
 * Used for progress bars and completion indicators
 */

import { NextResponse } from 'next/server';
import { stackServerApp } from '@/lib/stack';
import { sql } from '@/lib/db/neon';

export async function GET() {
  try {
    const user = await stackServerApp.getUser();

    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    console.debug('[API Stats] Fetching stats for user:', user.id);

    // Efficient count queries - no data transfer
    const [progressCount, videoCount, progressKeys] = await Promise.all([
      // Count total progress items
      sql`SELECT COUNT(*) as count FROM user_progress WHERE user_id = ${user.id}`,

      // Count total videos
      sql`SELECT COUNT(*) as count FROM user_videos WHERE user_id = ${user.id}`,

      // Get just the keys (for completion checking) - much smaller than full data
      sql`SELECT key FROM user_progress WHERE user_id = ${user.id}`,
    ]);

    // Extract unique video question prefixes for discussion completion count
    const videoQuestions = await sql`
      SELECT video_id FROM user_videos WHERE user_id = ${user.id}
    `;

    const uniqueQuestions = new Set<string>();
    videoQuestions.forEach((row) => {
      const videoId = row.video_id as string;
      // Extract question prefix (remove timestamp)
      const parts = videoId.split('-');
      if (parts.length >= 4) {
        const questionPrefix = parts.slice(0, -1).join('-');
        uniqueQuestions.add(questionPrefix);
      }
    });

    // Extract completion stats from keys
    const keys = progressKeys.map((row) => row.key as string);
    const completedProblemsKey = keys.find(
      (k) => k === 'codeblanket_completed_problems',
    );
    const mcQuizKeys = keys.filter((k) => k.startsWith('mc-quiz-'));
    const moduleKeys = keys.filter(
      (k) => k.startsWith('module-') && k.endsWith('-completed'),
    );

    const stats = {
      totalProgressItems: Number(progressCount[0].count),
      totalVideos: Number(videoCount[0].count),
      completedDiscussionQuestions: uniqueQuestions.size,
      hasCompletedProblems: !!completedProblemsKey,
      multipleChoiceQuizCount: mcQuizKeys.length,
      moduleProgressCount: moduleKeys.length,
      keys: keys, // Return keys for checking specific completion
    };

    console.debug('[API Stats] Returning stats:', stats);

    return NextResponse.json(stats);
  } catch (error) {
    console.error('[API Stats] Error:', error);
    return NextResponse.json(
      {
        error: 'Failed to fetch stats',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 },
    );
  }
}

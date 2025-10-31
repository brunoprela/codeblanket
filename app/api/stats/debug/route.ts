/**
 * Debug endpoint to test stats without auth
 */

import { NextResponse } from 'next/server';
import { sql } from '@/lib/db/neon';

export async function GET() {
  try {
    // Test with your actual user ID
    const userId = '7e1a5eb9-423a-4a1c-883c-81533cc865d9';

    console.log('[Debug Stats] Testing for user:', userId);

    // Get progress data
    const progressData = await sql`
      SELECT key, value FROM user_progress WHERE user_id = ${userId}
    `;

    console.log('[Debug Stats] Found', progressData.length, 'progress items');

    // Get videos
    const videos = await sql`
      SELECT video_id FROM user_videos WHERE user_id = ${userId}
    `;

    console.log('[Debug Stats] Found', videos.length, 'videos');

    // Process data
    let totalMC = 0;
    let completedProblems = 0;
    const keys: string[] = [];

    progressData.forEach((row) => {
      const key = row.key as string;
      const value = row.value;
      keys.push(key);

      try {
        if (key.startsWith('mc-quiz-')) {
          const questions = JSON.parse(value as string);
          if (Array.isArray(questions)) {
            console.log(
              `[Debug Stats] MC key ${key}: ${questions.length} questions`,
            );
            totalMC += questions.length;
          }
        } else if (key === 'codeblanket_completed_problems') {
          const problems = JSON.parse(value as string);
          if (Array.isArray(problems)) {
            console.log(`[Debug Stats] Completed problems: ${problems.length}`);
            completedProblems = problems.length;
          }
        }
      } catch (error) {
        console.error(`[Debug Stats] Error parsing ${key}:`, error);
      }
    });

    // Process videos
    const uniqueQuestions = new Set<string>();
    videos.forEach((row) => {
      const videoId = row.video_id as string;
      const parts = videoId.split('-');
      if (parts.length >= 4) {
        const questionPrefix = parts.slice(0, -1).join('-');
        uniqueQuestions.add(questionPrefix);
      }
    });

    const result = {
      userId,
      progressItems: progressData.length,
      videoCount: videos.length,
      totalMC,
      completedProblems,
      uniqueDiscussions: uniqueQuestions.size,
      keys,
      videoIds: videos.map((v) => v.video_id),
      uniqueQuestions: Array.from(uniqueQuestions),
    };

    console.log('[Debug Stats] Result:', result);

    return NextResponse.json(result);
  } catch (error) {
    console.error('[Debug Stats] Error:', error);
    return NextResponse.json(
      {
        error: 'Failed to get debug stats',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 },
    );
  }
}

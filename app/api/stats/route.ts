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
    const moduleVideoMap: Record<string, Set<string>> = {};

    videoQuestions.forEach((row) => {
      const videoId = row.video_id as string;
      // Extract question prefix (everything before the last timestamp)
      // Format: moduleId-sectionId-questionId-timestamp
      // We want: moduleId-sectionId-questionId (everything except the timestamp)
      const parts = videoId.split('-');

      // Need at least 4 parts: module, section, question, timestamp
      if (parts.length >= 4) {
        // Take all parts except the last one (timestamp)
        // This gives us the unique question identifier (deduplicates multiple videos for same question)
        const questionPrefix = parts.slice(0, -1).join('-');
        uniqueQuestions.add(questionPrefix);

        // Extract module ID (first part)
        const moduleId = parts[0];
        if (!moduleVideoMap[moduleId]) {
          moduleVideoMap[moduleId] = new Set();
        }
        moduleVideoMap[moduleId].add(questionPrefix);
      }
    });

    console.debug(
      `[API Stats] Found ${videoQuestions.length} videos, ${uniqueQuestions.size} unique questions`,
    );

    // Convert module video map to counts
    const moduleVideoCounts: Record<string, number> = {};
    Object.keys(moduleVideoMap).forEach((moduleId) => {
      moduleVideoCounts[moduleId] = moduleVideoMap[moduleId].size;
    });

    // Extract completion stats from keys and values
    const keys = progressKeys.map((row) => row.key as string);
    const completedProblemsKey = keys.find(
      (k) => k === 'codeblanket_completed_problems',
    );
    const mcQuizKeys = keys.filter((k) => k.startsWith('mc-quiz-'));
    const moduleKeys = keys.filter(
      (k) => k.startsWith('module-') && k.endsWith('-completed'),
    );

    // Count actual multiple choice questions (not just sections)
    // Each mc-quiz key contains an array of completed question IDs
    let totalMCQuestions = 0;
    for (const mcKey of mcQuizKeys) {
      try {
        const mcData = await sql`
          SELECT value FROM user_progress 
          WHERE user_id = ${user.id} AND key = ${mcKey}
        `;
        if (mcData.length > 0 && mcData[0].value) {
          const completedQuestions = JSON.parse(mcData[0].value as string);
          if (Array.isArray(completedQuestions)) {
            totalMCQuestions += completedQuestions.length;
          }
        }
      } catch (error) {
        console.error(`Failed to count MC questions for key ${mcKey}:`, error);
      }
    }

    // Count completed problems
    let completedProblemsCount = 0;
    if (completedProblemsKey) {
      try {
        const problemsData = await sql`
          SELECT value FROM user_progress 
          WHERE user_id = ${user.id} AND key = ${completedProblemsKey}
        `;
        if (problemsData.length > 0 && problemsData[0].value) {
          const completedProblems = JSON.parse(problemsData[0].value as string);
          if (Array.isArray(completedProblems)) {
            completedProblemsCount = completedProblems.length;
          }
        }
      } catch (error) {
        console.error('Failed to count completed problems:', error);
      }
    }

    const stats = {
      totalProgressItems: Number(progressCount[0].count),
      totalVideos: Number(videoCount[0].count),
      completedDiscussionQuestions: uniqueQuestions.size,
      hasCompletedProblems: !!completedProblemsKey,
      completedProblemsCount: completedProblemsCount, // Actual count of completed problems
      multipleChoiceQuizCount: totalMCQuestions, // Actual count of completed MC questions
      moduleProgressCount: moduleKeys.length,
      moduleVideoCounts: moduleVideoCounts, // Module-specific video counts
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

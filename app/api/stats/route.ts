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

    // Efficient queries - get keys and small values in one go
    const [progressCount, videoCount, progressData] = await Promise.all([
      // Count total progress items
      sql`SELECT COUNT(*) as count FROM user_progress WHERE user_id = ${user.id}`,

      // Count total videos
      sql`SELECT COUNT(*) as count FROM user_videos WHERE user_id = ${user.id}`,

      // Get keys AND values (we need values to count questions, not just keys)
      sql`SELECT key, value FROM user_progress WHERE user_id = ${user.id}`,
    ]);

    // Extract unique video question prefixes for discussion completion count
    const videoQuestions = await sql`
      SELECT video_id FROM user_videos WHERE user_id = ${user.id}
    `;

    const uniqueQuestions = new Set<string>();
    const moduleVideoMap: Record<string, Set<string>> = {};

    console.debug(
      `[API Stats] Processing ${videoQuestions.length} video records...`,
    );

    videoQuestions.forEach((row) => {
      const videoId = row.video_id as string;
      console.debug(`[API Stats] Video ID: ${videoId}`);

      // Extract question prefix (everything before the last timestamp)
      // Format: moduleId-sectionId-questionId-timestamp
      // We want: moduleId-sectionId-questionId (everything except the timestamp)
      const parts = videoId.split('-');
      console.debug(`[API Stats]   Parts: [${parts.join(', ')}]`);

      // Need at least 4 parts: module, section, question, timestamp
      if (parts.length >= 4) {
        // Take all parts except the last one (timestamp)
        // This gives us the unique question identifier (deduplicates multiple videos for same question)
        const questionPrefix = parts.slice(0, -1).join('-');
        console.debug(
          `[API Stats]   Question prefix (deduplicated): ${questionPrefix}`,
        );

        uniqueQuestions.add(questionPrefix);

        // Extract module ID (first part)
        const moduleId = parts[0];
        if (!moduleVideoMap[moduleId]) {
          moduleVideoMap[moduleId] = new Set();
        }
        moduleVideoMap[moduleId].add(questionPrefix);
      } else {
        console.warn(
          `[API Stats]   WARNING: Unexpected format (less than 4 parts)`,
        );
      }
    });

    console.debug(
      `[API Stats] Deduplication complete: ${videoQuestions.length} videos → ${uniqueQuestions.size} unique questions`,
    );
    console.debug(`[API Stats] Unique questions:`, Array.from(uniqueQuestions));

    // Convert module video map to counts
    const moduleVideoCounts: Record<string, number> = {};
    Object.keys(moduleVideoMap).forEach((moduleId) => {
      moduleVideoCounts[moduleId] = moduleVideoMap[moduleId].size;
    });

    // Extract completion stats from keys and values (all in one pass)
    const keys = progressData.map((row) => row.key as string);

    // Build maps for module completions and MC questions in one pass
    const moduleCompletionMap: Record<string, number> = {};
    let totalMCQuestions = 0;
    let completedProblemsCount = 0;

    progressData.forEach((row) => {
      const key = row.key as string;
      const value = row.value;

      try {
        // IMPORTANT: Neon returns JSON columns as already-parsed objects, not strings
        // So we need to handle both cases: already parsed OR string
        const parseValue = (val: unknown) => {
          if (Array.isArray(val)) {
            return val; // Already parsed
          } else if (typeof val === 'string') {
            return JSON.parse(val); // Need to parse
          }
          return val;
        };

        if (key.startsWith('mc-quiz-')) {
          // Multiple choice: count questions in array
          const completedQuestions = parseValue(value);
          if (Array.isArray(completedQuestions)) {
            console.debug(
              `[API Stats] MC ${key}: ${completedQuestions.length} questions`,
            );
            totalMCQuestions += completedQuestions.length;
          }
        } else if (key.startsWith('module-') && key.endsWith('-completed')) {
          // Module completion: extract module ID and count sections
          const moduleId = key.replace('module-', '').replace('-completed', '');
          const completedSections = parseValue(value);
          if (Array.isArray(completedSections)) {
            console.debug(
              `[API Stats] Module ${moduleId}: ${completedSections.length} sections`,
            );
            moduleCompletionMap[moduleId] = completedSections.length;
          }
        } else if (key === 'codeblanket_completed_problems') {
          // Problem completion: count problems in array
          const completedProblems = parseValue(value);
          if (Array.isArray(completedProblems)) {
            console.debug(
              `[API Stats] Completed problems: ${completedProblems.length}`,
            );
            completedProblemsCount = completedProblems.length;
          }
        }
      } catch (error) {
        console.error(`Failed to parse value for key ${key}:`, error);
      }
    });

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
      completedProblemsCount: completedProblemsCount, // Actual count of completed problems
      multipleChoiceQuizCount: totalMCQuestions, // Actual count of completed MC questions
      moduleProgressCount: moduleKeys.length,
      moduleVideoCounts: moduleVideoCounts, // Module-specific video counts
      moduleCompletionMap: moduleCompletionMap, // Module-specific section completion counts
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

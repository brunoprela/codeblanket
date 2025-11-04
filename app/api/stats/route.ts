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

    // OPTIMIZED: Get data and count in JavaScript (2 queries instead of 4)
    const [progressData, videoQuestions] = await Promise.all([
      // Get all progress data (we'll count in JavaScript)
      sql`SELECT key, value FROM user_progress WHERE user_id = ${user.id}`,

      // Get all video IDs (we'll count in JavaScript)
      sql`SELECT video_id FROM user_videos WHERE user_id = ${user.id}`,
    ]);

    // Count in JavaScript (no extra queries needed)
    const progressCount = progressData.length;
    const videoCount = videoQuestions.length;

    console.debug(
      `[API Stats] Fetched ${progressCount} progress items and ${videoCount} videos in 2 queries`,
    );

    // Helper to extract module ID from a key
    // Some modules are 2 parts (python-fundamentals), others are 3+ (applied-ai-llm-fundamentals)
    const extractModuleId = (keyWithoutPrefix: string): string => {
      const parts = keyWithoutPrefix.split('-');

      // Try to match valid module ID patterns
      // Common patterns:
      // - applied-ai-* (3 parts): applied-ai-llm-fundamentals
      // - system-design-* (2-4 parts): system-design-fundamentals, system-design-core-building-blocks
      // - ml-* (2-5 parts): ml-supervised-learning, ml-ai-llm-applications-finance
      // - python-* (2 parts): python-fundamentals
      // - single word (1 part): dfs, bfs, etc.

      if (parts[0] === 'applied' && parts[1] === 'ai' && parts.length >= 3) {
        return parts.slice(0, 3).join('-'); // applied-ai-{something}
      } else if (
        parts[0] === 'system' &&
        parts[1] === 'design' &&
        parts.length >= 3
      ) {
        // For system-design, could be 3-5 parts
        // Try to find where the section name starts (usually after 2-4 parts)
        // Heuristic: if part contains common section words, stop before it
        for (let i = 2; i < Math.min(5, parts.length); i++) {
          const part = parts[i];
          // Common section indicators
          if (
            [
              'fundamentals',
              'core',
              'advanced',
              'database',
              'networking',
              'api',
              'tradeoffs',
              'authentication',
              'microservices',
              'message',
              'case',
              'trading',
            ].includes(part)
          ) {
            return parts.slice(0, i + 1).join('-');
          }
        }
        return parts.slice(0, 3).join('-'); // Default to 3 parts
      } else if (parts[0] === 'ml' && parts.length >= 3) {
        // For ML, could be 3-5 parts
        // Similar heuristic
        for (let i = 2; i < Math.min(5, parts.length); i++) {
          const part = parts[i];
          if (
            [
              'mathematical',
              'calculus',
              'linear',
              'probability',
              'statistics',
              'python',
              'eda',
              'supervised',
              'unsupervised',
              'deep',
              'advanced',
              'natural',
              'model',
              'system',
              'ai',
            ].includes(part)
          ) {
            return parts.slice(0, i + 1).join('-');
          }
        }
        return parts.slice(0, 3).join('-'); // Default to 3 parts
      } else if (
        parts.length >= 2 &&
        parts[0] !== 'applied' &&
        parts[0] !== 'system' &&
        parts[0] !== 'ml'
      ) {
        return parts.slice(0, 2).join('-'); // Standard 2-part module
      } else {
        return parts[0]; // Single-word module
      }
    };

    // Process video question prefixes for discussion completion count
    // (videoQuestions already fetched above)
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

        // Extract module ID using helper function (supports variable-length module IDs)
        // From ["python", "fundamentals", "variables", "types", "pf", "variables", "q", "2"]
        // or ["applied", "ai", "llm", "fundamentals", "section", "q", "1"]
        const moduleId = extractModuleId(videoId);
        console.debug(`[API Stats]   Module ID: ${moduleId}`);

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

    // Build maps for ALL module-specific progress in one pass
    const moduleCompletionMap: Record<string, number> = {}; // Section completions
    const moduleMCCounts: Record<string, number> = {}; // MC questions
    let totalMCQuestions = 0;
    let completedProblemsCount = 0;
    const completedProblemsList: string[] = [];

    // Helper to handle Neon's JSON columns (already parsed vs string)
    const parseValue = (val: unknown) => {
      if (Array.isArray(val)) {
        return val; // Already parsed
      } else if (typeof val === 'string') {
        return JSON.parse(val); // Need to parse
      }
      return val;
    };

    progressData.forEach((row) => {
      const key = row.key as string;
      const value = row.value;

      try {
        if (key.startsWith('mc-quiz-')) {
          // Multiple choice: count questions in array
          // Key format: mc-quiz-{moduleId}-{sectionId}
          const completedQuestions = parseValue(value);
          if (Array.isArray(completedQuestions)) {
            console.debug(
              `[API Stats] MC ${key}: ${completedQuestions.length} questions`,
            );
            totalMCQuestions += completedQuestions.length;

            // Extract module ID for module-specific counts
            const keyWithoutPrefix = key.replace('mc-quiz-', '');
            const moduleId = extractModuleId(keyWithoutPrefix);

            if (!moduleMCCounts[moduleId]) {
              moduleMCCounts[moduleId] = 0;
            }
            moduleMCCounts[moduleId] += completedQuestions.length;
            console.debug(
              `[API Stats] Key ${key} → Module ${moduleId} MC count: ${moduleMCCounts[moduleId]}`,
            );
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
            completedProblemsList.push(...completedProblems);
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
      totalProgressItems: progressCount,
      totalVideos: videoCount,
      completedDiscussionQuestions: uniqueQuestions.size,
      hasCompletedProblems: !!completedProblemsKey,
      completedProblemsCount: completedProblemsCount, // Actual count of completed problems
      completedProblemsList: completedProblemsList, // List of completed problem IDs
      multipleChoiceQuizCount: totalMCQuestions, // Actual count of completed MC questions
      moduleProgressCount: moduleKeys.length,
      moduleVideoCounts: moduleVideoCounts, // Module-specific video counts
      moduleMCCounts: moduleMCCounts, // Module-specific MC counts
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

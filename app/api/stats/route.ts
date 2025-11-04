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
    // Strategy: Try progressively longer prefixes and check against known module IDs
    const extractModuleId = (keyWithoutPrefix: string): string => {
      const parts = keyWithoutPrefix.split('-');

      // All known module ID patterns (from topics/index.ts)
      const knownModuleIds = new Set([
        // Python
        'python-fundamentals',
        'python-intermediate',
        'python-advanced',
        'python-oop',
        'sqlalchemy-database',
        'python-async',
        'python-celery',
        'python-django',
        'fastapi-production',
        'python-testing',
        // Algorithms
        'time-space-complexity',
        'arrays-hashing',
        'two-pointers',
        'sliding-window',
        'dfs',
        'bfs',
        'binary-search',
        'sorting',
        'recursion',
        'stack',
        'queue',
        'design-problems',
        'linked-list',
        'string-algorithms',
        'trees',
        'heap',
        'graphs',
        'backtracking',
        'dynamic-programming',
        'tries',
        'intervals',
        'greedy',
        'advanced-graphs',
        'segment-tree',
        'fenwick-tree',
        'bit-manipulation',
        'math-geometry',
        // System Design
        'system-design-fundamentals',
        'system-design-core-building-blocks',
        'system-design-database-design',
        'system-design-networking',
        'system-design-api-design',
        'system-design-tradeoffs',
        'system-design-authentication',
        'system-design-microservices',
        'observability-resilience',
        'system-design-advanced-algorithms',
        'distributed-system-patterns',
        'search-analytics-specialized-systems',
        'distributed-file-systems-databases',
        'system-design-message-queues',
        'real-world-architectures',
        'system-design-case-studies',
        'system-design-trading',
        // Machine Learning
        'ml-mathematical-foundations',
        'ml-calculus-fundamentals',
        'ml-linear-algebra-foundations',
        'ml-probability-theory',
        'ml-statistics-fundamentals',
        'ml-python-for-data-science',
        'ml-eda-feature-engineering',
        'ml-supervised-learning',
        'ml-unsupervised-learning',
        'ml-deep-learning-fundamentals',
        'ml-advanced-deep-learning',
        'ml-natural-language-processing',
        'ml-model-evaluation-optimization',
        'ml-system-design-production',
        'large-language-models',
        // Finance
        'quantitative-finance',
        'time-series-financial-ml',
        'ml-ai-llm-applications-finance',
        'quant-interview-prep',
        'finance-foundations',
        'professional-tools',
        'corporate-finance',
        'financial-markets-instruments',
        'financial-statements-analysis',
        'financial-modeling-valuation',
        'portfolio-theory',
        'options-trading-greeks',
        'time-series-analysis',
        'market-data-real-time-processing',
        'fixed-income-derivatives',
        'algorithmic-trading-strategies',
        'market-microstructure-order-flow',
        'building-trading-infrastructure',
        'backtesting-strategy-development',
        'risk-management-portfolio-systems',
        // Applied AI
        'applied-ai-llm-fundamentals',
        'applied-ai-prompt-engineering',
        'applied-ai-file-processing',
        'applied-ai-code-understanding',
        'applied-ai-code-generation',
        'applied-ai-tool-use',
        'applied-ai-multi-agent',
        'applied-ai-image-generation',
        'applied-ai-video-audio',
        'applied-ai-multi-modal',
        'applied-ai-rag-search',
        'applied-ai-production',
        'applied-ai-scaling',
        'applied-ai-safety',
        'applied-ai-complete-products',
        'applied-ai-evaluation-dataops-finetuning',
        // Others
        'linux-system-administration',
        'crypto-blockchain-fundamentals',
        'react-fundamentals',
        'product-management-fundamentals',
        'competitive-programming',
      ]);

      // Try longest match first (up to 6 parts, down to 1)
      for (let len = Math.min(6, parts.length); len >= 1; len--) {
        const candidate = parts.slice(0, len).join('-');
        if (knownModuleIds.has(candidate)) {
          console.debug(
            `[extractModuleId] Matched: ${candidate} from ${keyWithoutPrefix}`,
          );
          return candidate;
        }
      }

      // Fallback: return first 2 parts (most common pattern)
      const fallback = parts.slice(0, 2).join('-');
      console.debug(
        `[extractModuleId] Fallback: ${fallback} from ${keyWithoutPrefix}`,
      );
      return fallback;
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

            console.debug(`[API Stats] MC key: ${key}`);
            console.debug(
              `[API Stats]   Key without prefix: ${keyWithoutPrefix}`,
            );
            console.debug(`[API Stats]   Extracted module ID: ${moduleId}`);
            console.debug(
              `[API Stats]   Question count: ${completedQuestions.length}`,
            );

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

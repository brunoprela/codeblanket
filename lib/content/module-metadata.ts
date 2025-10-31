/**
 * Lightweight module metadata - only IDs, titles, and counts
 * Used for homepage rendering without loading full 50MB of content
 */

export interface ModuleMetadata {
  id: string;
  title: string;
  description: string;
  icon: string;
  sectionCount: number;
  mcCount: number;
  discussionCount: number;
  problemCount: number;
}

// Lightweight metadata for all modules (< 10KB vs 50MB!)
// This allows homepage to render without loading full content
export const moduleMetadataMap: Record<string, ModuleMetadata> = {
  'python-fundamentals': {
    id: 'python-fundamentals',
    title: 'Python Fundamentals',
    description:
      'Master the core concepts of Python programming, from basic syntax to essential data structures and control flow.',
    icon: '🐍',
    sectionCount: 10,
    mcCount: 50,
    discussionCount: 30,
    problemCount: 100,
  },
  'python-intermediate': {
    id: 'python-intermediate',
    title: 'Python Intermediate',
    description:
      'Build practical Python skills with file handling, error management, regular expressions, and more.',
    icon: '🔧',
    sectionCount: 10,
    mcCount: 50,
    discussionCount: 30,
    problemCount: 56,
  },
  'python-advanced': {
    id: 'python-advanced',
    title: 'Python Advanced',
    description:
      'Master advanced Python features including decorators, generators, context managers, and metaclasses.',
    icon: '🐍',
    sectionCount: 7,
    mcCount: 35,
    discussionCount: 21,
    problemCount: 50,
  },
  // TODO: Add metadata for all other modules
  // This is a starting point - you can generate this programmatically
};

/**
 * Get module metadata by ID
 */
export function getModuleMetadata(moduleId: string): ModuleMetadata | null {
  return moduleMetadataMap[moduleId] || null;
}

/**
 * Get all module IDs for a topic
 */
export function getModuleIdsForTopic(topicId: string): string[] {
  // Topic to module mapping
  const topicModules: Record<string, string[]> = {
    python: [
      'python-fundamentals',
      'python-intermediate',
      'python-advanced',
      'python-web-development',
      'python-data-science',
      'python-testing',
      'python-async',
      'python-performance',
      'python-packaging',
      'python-best-practices',
    ],
    // Add other topics...
  };

  return topicModules[topicId] || [];
}

import { Problem } from '@/lib/types';
import { advancedGraphsProblems } from './advanced-graphs';
import { arraysHashingProblems } from './arrays-hashing';
import { backtrackingProblems } from './backtracking';
import { bfsProblems } from './bfs';
import { binarySearchProblems } from './binary-search';
import { bitManipulationProblems } from './bit-manipulation';
import { dfsProblems } from './dfs';
import { dynamicProgrammingProblems } from './dynamic-programming';
import { fenwickTreeProblems } from './fenwick-tree';
import { graphsProblems } from './graphs';
import { greedyProblems } from './greedy';
import { heapProblems } from './heap';
import { intervalsProblems } from './intervals';
import { linkedListProblems } from './linked-list';
import { mathGeometryProblems } from './math-geometry';
import { segmentTreeProblems } from './segment-tree';
import { slidingWindowProblems } from './sliding-window';
import { stackProblems } from './stack';
import { treesProblems } from './trees';
import { triesProblems } from './tries';
import { twoPointersProblems } from './two-pointers';

export interface ProblemCategory {
  id: string;
  title: string;
  description: string;
  icon: string;
  problemCount: number;
  problems: Problem[];
}

export const problemCategories: ProblemCategory[] = [
  {
    id: 'advanced-graphs',
    title: 'Advanced Graphs',
    description:
      'Master advanced graph algorithms including shortest paths, minimum spanning trees, and network flow',
    icon: '🗺️',
    problemCount: advancedGraphsProblems.length,
    problems: advancedGraphsProblems,
  },
  {
    id: 'arrays-hashing',
    title: 'Arrays & Hashing',
    description:
      'Master array manipulation and hash table techniques for optimal performance',
    icon: '🔢',
    problemCount: arraysHashingProblems.length,
    problems: arraysHashingProblems,
  },
  {
    id: 'backtracking',
    title: 'Backtracking',
    description:
      'Master backtracking for exploring all possible solutions with pruning',
    icon: '🔙',
    problemCount: backtrackingProblems.length,
    problems: backtrackingProblems,
  },
  {
    id: 'bfs',
    title: 'Breadth-First Search (BFS)',
    description:
      'Master BFS for level-by-level traversal and finding shortest paths',
    icon: '📊',
    problemCount: bfsProblems.length,
    problems: bfsProblems,
  },
  {
    id: 'binary-search',
    title: 'Binary Search',
    description:
      'Master the art of dividing and conquering with logarithmic time complexity',
    icon: '🔍',
    problemCount: binarySearchProblems.length,
    problems: binarySearchProblems,
  },
  {
    id: 'bit-manipulation',
    title: 'Bit Manipulation',
    description:
      'Master bitwise operations and bit tricks for efficient problem solving',
    icon: '⚡',
    problemCount: bitManipulationProblems.length,
    problems: bitManipulationProblems,
  },
  {
    id: 'dfs',
    title: 'Depth-First Search (DFS)',
    description:
      'Master DFS for exploring trees and graphs by going deep before backtracking',
    icon: '🌊',
    problemCount: dfsProblems.length,
    problems: dfsProblems,
  },
  {
    id: 'dynamic-programming',
    title: 'Dynamic Programming',
    description:
      'Master the art of breaking problems into overlapping subproblems and building optimal solutions',
    icon: '🧩',
    problemCount: dynamicProgrammingProblems.length,
    problems: dynamicProgrammingProblems,
  },
  {
    id: 'fenwick-tree',
    title: 'Fenwick Tree',
    description:
      'Master Fenwick Trees for efficient prefix sum queries and updates',
    icon: '🎯',
    problemCount: fenwickTreeProblems.length,
    problems: fenwickTreeProblems,
  },
  {
    id: 'graphs',
    title: 'Graphs',
    description:
      'Master graph traversal, pathfinding, and connectivity problems for complex network structures',
    icon: '🕸️',
    problemCount: graphsProblems.length,
    problems: graphsProblems,
  },
  {
    id: 'greedy',
    title: 'Greedy',
    description:
      'Master greedy algorithms that make locally optimal choices to find global optima',
    icon: '🎯',
    problemCount: greedyProblems.length,
    problems: greedyProblems,
  },
  {
    id: 'heap',
    title: 'Heap / Priority Queue',
    description:
      'Master heaps and priority queues for efficient min/max operations',
    icon: '⛰️',
    problemCount: heapProblems.length,
    problems: heapProblems,
  },
  {
    id: 'intervals',
    title: 'Intervals',
    description:
      'Master interval manipulation including merging, overlapping, and intersection problems',
    icon: '↔️',
    problemCount: intervalsProblems.length,
    problems: intervalsProblems,
  },
  {
    id: 'linked-list',
    title: 'Linked List',
    description: 'Master linked list manipulation and pointer techniques',
    icon: '🔗',
    problemCount: linkedListProblems.length,
    problems: linkedListProblems,
  },
  {
    id: 'math-geometry',
    title: 'Math & Geometry',
    description:
      'Master mathematical algorithms and geometric computations for problem solving',
    icon: '📐',
    problemCount: mathGeometryProblems.length,
    problems: mathGeometryProblems,
  },
  {
    id: 'segment-tree',
    title: 'Segment Tree',
    description: 'Master segment trees for efficient range queries and updates',
    icon: '🌲',
    problemCount: segmentTreeProblems.length,
    problems: segmentTreeProblems,
  },
  {
    id: 'sliding-window',
    title: 'Sliding Window',
    description:
      'Master the sliding window technique for substring and subarray problems',
    icon: '🪟',
    problemCount: slidingWindowProblems.length,
    problems: slidingWindowProblems,
  },
  {
    id: 'stack',
    title: 'Stack',
    description:
      'Master the Last-In-First-Out (LIFO) data structure for parsing and backtracking',
    icon: '📚',
    problemCount: stackProblems.length,
    problems: stackProblems,
  },
  {
    id: 'trees',
    title: 'Trees',
    description:
      'Master tree structures, traversals, and recursive problem-solving',
    icon: '🌳',
    problemCount: treesProblems.length,
    problems: treesProblems,
  },
  {
    id: 'tries',
    title: 'Tries',
    description:
      'Master the prefix tree data structure for efficient string operations and searches',
    icon: '🌲',
    problemCount: triesProblems.length,
    problems: triesProblems,
  },
  {
    id: 'two-pointers',
    title: 'Two Pointers',
    description:
      'Learn to efficiently solve array problems with two-pointer technique',
    icon: '👉👈',
    problemCount: twoPointersProblems.length,
    problems: twoPointersProblems,
  },
];

export const allProblems: Problem[] = [
  ...advancedGraphsProblems,
  ...arraysHashingProblems,
  ...backtrackingProblems,
  ...bfsProblems,
  ...binarySearchProblems,
  ...bitManipulationProblems,
  ...dfsProblems,
  ...dynamicProgrammingProblems,
  ...fenwickTreeProblems,
  ...graphsProblems,
  ...greedyProblems,
  ...heapProblems,
  ...intervalsProblems,
  ...linkedListProblems,
  ...mathGeometryProblems,
  ...segmentTreeProblems,
  ...slidingWindowProblems,
  ...stackProblems,
  ...treesProblems,
  ...triesProblems,
  ...twoPointersProblems,
];

export function getProblemById(id: string): Problem | undefined {
  return allProblems.find((p) => p.id === id);
}

export function getCategoryById(id: string): ProblemCategory | undefined {
  return problemCategories.find((c) => c.id === id);
}

export function getProblemsByDifficulty(
  difficulty: 'Easy' | 'Medium' | 'Hard',
): Problem[] {
  return allProblems.filter((p) => p.difficulty === difficulty);
}

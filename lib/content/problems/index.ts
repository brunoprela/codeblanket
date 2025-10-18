/**
 * Problems Index
 * Central export for all coding problems and provides backward-compatible API
 */

import { Problem } from '../../types';

// Import all problem sets
import { pythonFundamentalsProblems } from './python-fundamentals';
import { pythonIntermediateProblems } from './python-intermediate';
import { pythonAdvancedProblems } from './python-advanced';
import { pythonOopProblems } from './python-oop';

import { advancedGraphsProblems } from './advanced-graphs';
import { arraysHashingProblems } from './arrays-hashing';
import { backtrackingProblems } from './backtracking';
import { bfsProblems } from './bfs';
import { binarySearchProblems } from './binary-search';
import { bitManipulationProblems } from './bit-manipulation';
import { designProblemsProblems } from './design-problems';
import { dfsProblems } from './dfs';
import { dynamicProgrammingProblems } from './dynamic-programming';
import { fenwickTreeProblems } from './fenwick-tree';
import { graphsProblems } from './graphs';
import { greedyProblems } from './greedy';
import { heapProblems } from './heap';
import { intervalsProblems } from './intervals';
import { linkedListProblems } from './linked-list';
import { mathGeometryProblems } from './math-geometry';
import { queueProblems } from './queue';
import { recursionProblems } from './recursion';
import { segmentTreeProblems } from './segment-tree';
import { slidingWindowProblems } from './sliding-window';
import { sortingProblems } from './sorting';
import { stackProblems } from './stack';
import { stringAlgorithmsProblems } from './string-algorithms';
import { timeSpaceComplexityProblems } from './time-space-complexity';
import { treesProblems } from './trees';
import { triesProblems } from './tries';
import { twoPointersProblems } from './two-pointers';

// Aggregate all problems
export const allProblems: Problem[] = [
  ...pythonFundamentalsProblems,
  ...pythonIntermediateProblems,
  ...pythonAdvancedProblems,
  ...pythonOopProblems,
  ...advancedGraphsProblems,
  ...arraysHashingProblems,
  ...backtrackingProblems,
  ...bfsProblems,
  ...binarySearchProblems,
  ...bitManipulationProblems,
  ...designProblemsProblems,
  ...dfsProblems,
  ...dynamicProgrammingProblems,
  ...fenwickTreeProblems,
  ...graphsProblems,
  ...greedyProblems,
  ...heapProblems,
  ...intervalsProblems,
  ...linkedListProblems,
  ...mathGeometryProblems,
  ...queueProblems,
  ...recursionProblems,
  ...segmentTreeProblems,
  ...slidingWindowProblems,
  ...sortingProblems,
  ...stackProblems,
  ...stringAlgorithmsProblems,
  ...timeSpaceComplexityProblems,
  ...treesProblems,
  ...triesProblems,
  ...twoPointersProblems,
];

// Build problemCategories for backward compatibility
export const problemCategories = [
  {
    id: 'python-fundamentals',
    title: 'Python Fundamentals',
    description: 'Master Python basics',
    icon: '🐍',
    problems: pythonFundamentalsProblems,
  },
  {
    id: 'python-intermediate',
    title: 'Python Intermediate',
    description: 'Advanced Python concepts',
    icon: '🔧',
    problems: pythonIntermediateProblems,
  },
  {
    id: 'python-advanced',
    title: 'Python Advanced',
    description: 'Expert-level Python',
    icon: '⚡',
    problems: pythonAdvancedProblems,
  },
  {
    id: 'python-oop',
    title: 'Python OOP',
    description: 'Object-oriented programming',
    icon: '🏗️',
    problems: pythonOopProblems,
  },
  {
    id: 'time-space-complexity',
    title: 'Time & Space Complexity',
    description: 'Algorithm analysis',
    icon: '⏱️',
    problems: timeSpaceComplexityProblems,
  },
  {
    id: 'arrays-hashing',
    title: 'Arrays & Hashing',
    description: 'Core data structures',
    icon: '📊',
    problems: arraysHashingProblems,
  },
  {
    id: 'two-pointers',
    title: 'Two Pointers',
    description: 'Efficient array traversal',
    icon: '👉',
    problems: twoPointersProblems,
  },
  {
    id: 'sliding-window',
    title: 'Sliding Window',
    description: 'Subarray patterns',
    icon: '🪟',
    problems: slidingWindowProblems,
  },
  {
    id: 'stack',
    title: 'Stack',
    description: 'LIFO data structure',
    icon: '📚',
    problems: stackProblems,
  },
  {
    id: 'queue',
    title: 'Queue',
    description: 'FIFO data structure',
    icon: '🎬',
    problems: queueProblems,
  },
  {
    id: 'linked-list',
    title: 'Linked List',
    description: 'Node-based structures',
    icon: '🔗',
    problems: linkedListProblems,
  },
  {
    id: 'binary-search',
    title: 'Binary Search',
    description: 'Efficient searching',
    icon: '🔍',
    problems: binarySearchProblems,
  },
  {
    id: 'trees',
    title: 'Trees',
    description: 'Hierarchical structures',
    icon: '🌳',
    problems: treesProblems,
  },
  {
    id: 'tries',
    title: 'Tries',
    description: 'Prefix trees',
    icon: '🌲',
    problems: triesProblems,
  },
  {
    id: 'heap',
    title: 'Heap / Priority Queue',
    description: 'Priority-based structures',
    icon: '⛰️',
    problems: heapProblems,
  },
  {
    id: 'graphs',
    title: 'Graphs',
    description: 'Network structures',
    icon: '🕸️',
    problems: graphsProblems,
  },
  {
    id: 'advanced-graphs',
    title: 'Advanced Graphs',
    description: 'Complex graph algorithms',
    icon: '🔷',
    problems: advancedGraphsProblems,
  },
  {
    id: 'dfs',
    title: 'Depth-First Search',
    description: 'Tree/graph traversal',
    icon: '🔽',
    problems: dfsProblems,
  },
  {
    id: 'bfs',
    title: 'Breadth-First Search',
    description: 'Level-order traversal',
    icon: '↔️',
    problems: bfsProblems,
  },
  {
    id: 'backtracking',
    title: 'Backtracking',
    description: 'Explore all possibilities',
    icon: '🔙',
    problems: backtrackingProblems,
  },
  {
    id: 'dynamic-programming',
    title: 'Dynamic Programming',
    description: 'Optimization problems',
    icon: '💎',
    problems: dynamicProgrammingProblems,
  },
  {
    id: 'greedy',
    title: 'Greedy Algorithms',
    description: 'Local optimal choices',
    icon: '🎯',
    problems: greedyProblems,
  },
  {
    id: 'intervals',
    title: 'Intervals',
    description: 'Range problems',
    icon: '📏',
    problems: intervalsProblems,
  },
  {
    id: 'bit-manipulation',
    title: 'Bit Manipulation',
    description: 'Bitwise operations',
    icon: '🔢',
    problems: bitManipulationProblems,
  },
  {
    id: 'math-geometry',
    title: 'Math & Geometry',
    description: 'Mathematical problems',
    icon: '📐',
    problems: mathGeometryProblems,
  },
  {
    id: 'string-algorithms',
    title: 'String Algorithms',
    description: 'String manipulation',
    icon: '🔤',
    problems: stringAlgorithmsProblems,
  },
  {
    id: 'design-problems',
    title: 'Design Problems',
    description: 'System design coding',
    icon: '🏗️',
    problems: designProblemsProblems,
  },
  {
    id: 'recursion',
    title: 'Recursion',
    description: 'Self-referencing functions',
    icon: '🔄',
    problems: recursionProblems,
  },
  {
    id: 'sorting',
    title: 'Sorting',
    description: 'Ordering algorithms',
    icon: '↕️',
    problems: sortingProblems,
  },
  {
    id: 'segment-tree',
    title: 'Segment Tree',
    description: 'Range query structure',
    icon: '🌿',
    problems: segmentTreeProblems,
  },
  {
    id: 'fenwick-tree',
    title: 'Fenwick Tree / BIT',
    description: 'Binary indexed tree',
    icon: '🎄',
    problems: fenwickTreeProblems,
  },
];

// Export helper to get a problem by ID
export const getProblemById = (id: string): Problem | undefined => {
  return allProblems.find((p) => p.id === id);
};

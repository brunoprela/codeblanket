/**
 * Task Scheduler
 * Problem ID: task-scheduler
 * Order: 6
 */

import { Problem } from '../../../types';

export const task_schedulerProblem: Problem = {
  id: 'task-scheduler',
  title: 'Task Scheduler',
  difficulty: 'Medium',
  topic: 'Heap / Priority Queue',
  description: `You are given an array of CPU \`tasks\`, each represented by letters A to Z, and a cooling time, \`n\`. Each cycle or interval allows the completion of one task. Tasks can be completed in any order, but there is a constraint: **identical** tasks must be separated by at least \`n\` intervals due to cooling time.

Return the **minimum number of intervals** required to complete all tasks.`,
  examples: [
    {
      input: 'tasks = ["A","A","A","B","B","B"], n = 2',
      output: '8',
      explanation:
        'A -> B -> idle -> A -> B -> idle -> A -> B. Only 8 units of time are needed.',
    },
    {
      input: 'tasks = ["A","C","A","B","D","B"], n = 1',
      output: '6',
      explanation: 'A -> B -> C -> D -> A -> B. Only 6 units needed.',
    },
  ],
  constraints: [
    '1 <= tasks.length <= 10^4',
    'tasks[i] is an uppercase English letter',
    '0 <= n <= 100',
  ],
  hints: [
    'Count task frequencies',
    'Use max heap to always pick most frequent task',
    'Track cooling time for each task',
  ],
  starterCode: `from typing import List
import heapq
from collections import Counter

def least_interval(tasks: List[str], n: int) -> int:
    """
    Find minimum intervals to complete all tasks.
    
    Args:
        tasks: Array of task letters
        n: Cooling time between same tasks
        
    Returns:
        Minimum number of intervals
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['A', 'A', 'A', 'B', 'B', 'B'], 2],
      expected: 8,
    },
    {
      input: [['A', 'C', 'A', 'B', 'D', 'B'], 1],
      expected: 6,
    },
    {
      input: [['A', 'A', 'A', 'B', 'B', 'B'], 0],
      expected: 6,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/task-scheduler/',
  youtubeUrl: 'https://www.youtube.com/watch?v=s8p8ukTyA2I',
};

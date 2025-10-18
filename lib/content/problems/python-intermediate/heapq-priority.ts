/**
 * Priority Queue with heapq
 * Problem ID: intermediate-heapq-priority
 * Order: 28
 */

import { Problem } from '../../../types';

export const intermediate_heapq_priorityProblem: Problem = {
  id: 'intermediate-heapq-priority',
  title: 'Priority Queue with heapq',
  difficulty: 'Medium',
  description: `Use heapq to implement a priority queue.

heapq functions:
- heappush: Add item
- heappop: Remove smallest
- heapify: Convert list to heap

**Use Case:** Task scheduling, Dijkstra's algorithm

This tests:
- heapq module
- Min-heap operations
- Priority queues`,
  examples: [
    {
      input: 'Push tasks with priorities',
      output: 'Pop in priority order',
    },
  ],
  constraints: ['Use heapq', 'Min-heap (smallest first)'],
  hints: [
    'import heapq',
    'Use tuples (priority, item)',
    'heappush and heappop',
  ],
  starterCode: `import heapq

def process_tasks(tasks):
    """
    Process tasks by priority (lowest number = highest priority).
    
    Args:
        tasks: List of (priority, task_name) tuples
        
    Returns:
        List of task names in priority order
        
    Examples:
        >>> tasks = [(3, 'low'), (1, 'high'), (2, 'medium')]
        >>> process_tasks(tasks)
        ['high', 'medium', 'low']
    """
    pass


# Test
print(process_tasks([(3, 'task3'), (1, 'task1'), (2, 'task2')]))
`,
  testCases: [
    {
      input: [
        [
          [3, 'task3'],
          [1, 'task1'],
          [2, 'task2'],
        ],
      ],
      expected: ['task1', 'task2', 'task3'],
    },
    {
      input: [
        [
          [5, 'e'],
          [2, 'b'],
          [4, 'd'],
          [1, 'a'],
          [3, 'c'],
        ],
      ],
      expected: ['a', 'b', 'c', 'd', 'e'],
    },
  ],
  solution: `import heapq

def process_tasks(tasks):
    heap = []
    for priority, task_name in tasks:
        heapq.heappush(heap, (priority, task_name))
    
    result = []
    while heap:
        priority, task_name = heapq.heappop(heap)
        result.append(task_name)
    
    return result


# Alternative: heapify existing list
def process_tasks_heapify(tasks):
    heapq.heapify(tasks)
    return [task_name for priority, task_name in sorted(tasks)]`,
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(n)',
  order: 28,
  topic: 'Python Intermediate',
};

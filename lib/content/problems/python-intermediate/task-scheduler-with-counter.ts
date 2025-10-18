/**
 * Task Scheduler
 * Problem ID: task-scheduler-with-counter
 * Order: 26
 */

import { Problem } from '../../../types';

export const task_scheduler_with_counterProblem: Problem = {
  id: 'task-scheduler-with-counter',
  title: 'Task Scheduler',
  difficulty: 'Medium',
  category: 'python-intermediate',
  description: `Given a characters array \`tasks\`, where each character represents a unique task and an integer \`n\` representing the cooldown period, return the minimum number of time units needed to complete all tasks.

The same task cannot be executed in two consecutive time units. During a cooldown period, you can either execute another task or stay idle.

Use \`Counter\` and smart scheduling to solve this.

**Example 1:**
\`\`\`
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B
\`\`\`

**Example 2:**
\`\`\`
Input: tasks = ["A","A","A","B","B","B"], n = 0
Output: 6
Explanation: No cooldown, so: A -> A -> A -> B -> B -> B
\`\`\`

**Example 3:**
\`\`\`
Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
Output: 16
Explanation: One optimal solution:
A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle -> idle -> A
\`\`\``,
  starterCode: `from collections import Counter

def least_interval(tasks, n):
    """
    Calculate minimum time units to complete all tasks with cooldown.
    
    Args:
        tasks: List of task characters
        n: Cooldown period
    
    Returns:
        Minimum time units needed
    """
    pass`,
  testCases: [
    {
      input: [['A', 'A', 'A', 'B', 'B', 'B'], 2],
      expected: 8,
    },
    {
      input: [['A', 'A', 'A', 'B', 'B', 'B'], 0],
      expected: 6,
    },
    {
      input: [['A', 'A', 'A', 'A', 'A', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], 2],
      expected: 16,
    },
  ],
  hints: [
    'Count task frequencies using Counter',
    'Most frequent task determines minimum time',
    'Calculate idle slots based on most frequent task',
    'Fill idle slots with other tasks',
  ],
  solution: `from collections import Counter

def least_interval(tasks, n):
    """
    Calculate minimum time units to complete all tasks with cooldown.
    
    Args:
        tasks: List of task characters
        n: Cooldown period
    
    Returns:
        Minimum time units needed
    """
    # Count task frequencies
    task_counts = Counter(tasks)
    
    # Find maximum frequency
    max_freq = max(task_counts.values())
    
    # Count how many tasks have maximum frequency
    max_freq_count = sum(1 for count in task_counts.values() if count == max_freq)
    
    # Calculate minimum intervals needed
    # For most frequent task: (max_freq - 1) complete cycles + final tasks
    # Each cycle has (n + 1) slots
    intervals = (max_freq - 1) * (n + 1) + max_freq_count
    
    # Result is max of calculated intervals or total tasks
    # (if n is small, no idle time needed)
    return max(intervals, len(tasks))


# Example walkthrough for ["A","A","A","B","B","B"], n=2:
# max_freq = 3 (both A and B appear 3 times)
# max_freq_count = 2
# intervals = (3-1) * (2+1) + 2 = 2*3 + 2 = 8
# Result: max(8, 6) = 8
#
# Schedule: A -> B -> idle -> A -> B -> idle -> A -> B`,
  timeComplexity: 'O(n) where n is number of tasks',
  spaceComplexity: 'O(1) - at most 26 unique tasks',
  order: 26,
  topic: 'Python Intermediate',
};

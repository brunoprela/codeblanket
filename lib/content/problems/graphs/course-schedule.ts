/**
 * Course Schedule
 * Problem ID: course-schedule
 * Order: 2
 */

import { Problem } from '../../../types';

export const course_scheduleProblem: Problem = {
  id: 'course-schedule',
  title: 'Course Schedule',
  difficulty: 'Medium',
  description: `There are a total of \`numCourses\` courses you have to take, labeled from \`0\` to \`numCourses - 1\`. You are given an array \`prerequisites\` where \`prerequisites[i] = [ai, bi]\` indicates that you **must** take course \`bi\` first if you want to take course \`ai\`.

Return \`true\` if you can finish all courses. Otherwise, return \`false\`.


**Approach:**
This is a **cycle detection** problem in a directed graph. If there is a cycle in the dependency graph, it is impossible to complete all courses. Use DFS with a recursion stack or BFS with topological sort (Kahn algorithm).

**Key Insight:**
- No cycle → Can complete all courses (DAG)
- Has cycle → Cannot complete all courses`,
  examples: [
    {
      input: 'numCourses = 2, prerequisites = [[1,0]]',
      output: 'true',
      explanation:
        'There are 2 courses. To take course 1 you should have finished course 0. So it is possible.',
    },
    {
      input: 'numCourses = 2, prerequisites = [[1,0],[0,1]]',
      output: 'false',
      explanation:
        'There are 2 courses. To take course 1 you need course 0, and to take course 0 you need course 1. Circular dependency!',
    },
  ],
  constraints: [
    '1 <= numCourses <= 2000',
    '0 <= prerequisites.length <= 5000',
    'prerequisites[i].length == 2',
    '0 <= ai, bi < numCourses',
    'All the pairs prerequisites[i] are unique',
  ],
  hints: [
    'Model this as a directed graph where edge a → b means "must take b before a"',
    'The problem becomes: does the graph have a cycle?',
    'DFS approach: use recursion stack to detect back edges (cycles)',
    "BFS approach: use topological sort (Kahn's) - if we process all nodes, no cycle",
    'Track three states: unvisited, visiting (in current path), visited',
  ],
  starterCode: `from typing import List

def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    Determine if it's possible to finish all courses.
    
    Args:
        num_courses: Total number of courses
        prerequisites: List of [course, prerequisite] pairs
        
    Returns:
        True if possible to complete all courses, False otherwise
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [2, [[1, 0]]],
      expected: true,
    },
    {
      input: [
        2,
        [
          [1, 0],
          [0, 1],
        ],
      ],
      expected: false,
    },
    {
      input: [
        3,
        [
          [1, 0],
          [2, 1],
        ],
      ],
      expected: true,
    },
    {
      input: [
        4,
        [
          [1, 0],
          [2, 1],
          [3, 2],
          [1, 3],
        ],
      ],
      expected: false,
    },
  ],
  solution: `from typing import List
from collections import defaultdict, deque


def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    DFS with cycle detection.
    Time: O(V + E), Space: O(V + E)
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    # Three states: 0 = unvisited, 1 = visiting, 2 = visited
    state = [0] * num_courses
    
    def has_cycle(course):
        if state[course] == 1:  # Currently visiting = cycle
            return True
        if state[course] == 2:  # Already visited
            return False
        
        # Mark as visiting
        state[course] = 1
        
        # Check all prerequisites
        for prereq in graph[course]:
            if has_cycle(prereq):
                return True
        
        # Mark as visited
        state[course] = 2
        return False
    
    # Check each course
    for course in range(num_courses):
        if has_cycle(course):
            return False
    
    return True


# Alternative: Topological Sort (Kahn's Algorithm)
def can_finish_bfs(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    BFS with topological sort.
    Time: O(V + E), Space: O(V + E)
    """
    # Build graph and calculate in-degrees
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)  # prereq → course
        in_degree[course] += 1
    
    # Start with courses having no prerequisites
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    processed = 0
    
    while queue:
        course = queue.popleft()
        processed += 1
        
        # Reduce in-degree of dependent courses
        for dependent in graph[course]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    
    # If we processed all courses, no cycle exists
    return processed == num_courses


# Alternative: DFS with visited set (simpler)
def can_finish_simple(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    Simple DFS approach.
    """
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    visited = set()
    rec_stack = set()
    
    def dfs(course):
        if course in rec_stack:
            return False  # Cycle detected
        if course in visited:
            return True  # Already checked
        
        rec_stack.add(course)
        
        for prereq in graph[course]:
            if not dfs(prereq):
                return False
        
        rec_stack.remove(course)
        visited.add(course)
        return True
    
    for course in range(num_courses):
        if course not in visited:
            if not dfs(course):
                return False
    
    return True`,
  timeComplexity: 'O(V + E) where V = courses, E = prerequisites',
  spaceComplexity: 'O(V + E)',

  leetcodeUrl: 'https://leetcode.com/problems/course-schedule/',
  youtubeUrl: 'https://www.youtube.com/watch?v=EgI5nU9etnU',
  order: 2,
  topic: 'Graphs',
};

/**
 * Course Schedule II
 * Problem ID: course-schedule-ii
 * Order: 9
 */

import { Problem } from '../../../types';

export const course_schedule_iiProblem: Problem = {
  id: 'course-schedule-ii',
  title: 'Course Schedule II',
  difficulty: 'Medium',
  topic: 'Graphs',
  description: `There are a total of \`numCourses\` courses you have to take, labeled from \`0\` to \`numCourses - 1\`. You are given an array \`prerequisites\` where \`prerequisites[i] = [ai, bi]\` indicates that you **must** take course \`bi\` first if you want to take course \`ai\`.

For example, the pair \`[0, 1]\`, indicates that to take course \`0\` you have to first take course \`1\`.

Return **the ordering of courses you should take to finish all courses**. If there are many valid answers, return **any** of them. If it is impossible to finish all courses, return **an empty array**.`,
  examples: [
    {
      input: 'numCourses = 2, prerequisites = [[1,0]]',
      output: '[0,1]',
      explanation:
        'There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].',
    },
    {
      input: 'numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]',
      output: '[0,2,1,3]',
      explanation:
        'There are a total of 4 courses. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after course 0. So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3].',
    },
    {
      input: 'numCourses = 1, prerequisites = []',
      output: '[0]',
      explanation: 'There is only one course, so the answer is [0].',
    },
  ],
  constraints: [
    '1 <= numCourses <= 2000',
    '0 <= prerequisites.length <= numCourses * (numCourses - 1)',
    'prerequisites[i].length == 2',
    '0 <= ai, bi < numCourses',
    'ai != bi',
    'All the pairs [ai, bi] are distinct',
  ],
  hints: [
    "Use topological sort - Kahn's algorithm with BFS",
    'Build adjacency list and track in-degrees',
    'Start with nodes that have in-degree 0',
    'If ordering includes all nodes, return it; otherwise return []',
  ],
  starterCode: `from typing import List
from collections import deque, defaultdict

def find_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Return the ordering of courses using topological sort.
    
    Args:
        num_courses: Total number of courses
        prerequisites: List of [course, prerequisite] pairs
        
    Returns:
        Valid course ordering, or [] if impossible
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [2, [[1, 0]]],
      expected: [0, 1],
    },
    {
      input: [
        4,
        [
          [1, 0],
          [2, 0],
          [3, 1],
          [3, 2],
        ],
      ],
      expected: [0, 2, 1, 3], // or [0, 1, 2, 3]
    },
    {
      input: [1, []],
      expected: [0],
    },
  ],
  timeComplexity: 'O(V + E)',
  spaceComplexity: 'O(V + E)',
  solution: `from typing import List
from collections import deque, defaultdict

def find_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Course Schedule II - Topological Sort using Kahn's Algorithm (BFS).
    
    Time: O(V + E) where V = numCourses, E = prerequisites
    Space: O(V + E) for adjacency list and in-degree array
    
    Key Insight:
    - This is topological sort - order nodes so all edges point forward
    - Use Kahn's algorithm: start with nodes that have no dependencies (in-degree 0)
    - Process each node, reduce in-degrees of neighbors
    - If we process all nodes, return ordering; otherwise cycle exists
    
    Difference from Course Schedule I:
    - Course Schedule I: just check if possible (return True/False)
    - Course Schedule II: return the actual ordering
    """
    # Build adjacency list and in-degree array
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)  # prereq -> course
        in_degree[course] += 1
    
    # Queue: all courses with no prerequisites (in-degree 0)
    queue = deque()
    for course in range(num_courses):
        if in_degree[course] == 0:
            queue.append(course)
    
    # Topological sort
    ordering = []
    
    while queue:
        # Take a course with no prerequisites
        course = queue.popleft()
        ordering.append(course)
        
        # Reduce in-degree for courses that depend on this one
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            
            # If all prerequisites satisfied, add to queue
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    # Check if we processed all courses
    if len(ordering) == num_courses:
        return ordering
    else:
        return []  # Cycle detected


# Alternative: DFS-based topological sort
def find_order_dfs(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    DFS-based topological sort.
    
    Time: O(V + E)
    Space: O(V + E)
    
    Strategy:
    - DFS from each unvisited node
    - Add to result in post-order (after visiting all descendants)
    - Reverse at end to get correct ordering
    - Detect cycles during DFS
    """
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # 0 = unvisited, 1 = visiting (in current path), 2 = visited
    state = [0] * num_courses
    ordering = []
    has_cycle = False
    
    def dfs(course):
        nonlocal has_cycle
        
        if state[course] == 1:
            # Cycle detected (found node in current path)
            has_cycle = True
            return
        
        if state[course] == 2:
            # Already processed
            return
        
        # Mark as visiting
        state[course] = 1
        
        # Visit all neighbors
        for next_course in graph[course]:
            dfs(next_course)
            if has_cycle:
                return
        
        # Mark as visited (done processing)
        state[course] = 2
        
        # Add to ordering in post-order
        ordering.append(course)
    
    # DFS from all nodes
    for course in range(num_courses):
        if state[course] == 0:
            dfs(course)
            if has_cycle:
                return []
    
    # Reverse to get correct topological order
    return ordering[::-1]


# Example walkthrough:
"""
Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]

Graph:
0 → 1
0 → 2
1 → 3
2 → 3

In-degrees:
0: 0 (no prerequisites)
1: 1 (needs 0)
2: 1 (needs 0)
3: 2 (needs 1 and 2)

Process:
1. Start with in-degree 0: queue = [0]
2. Process 0: ordering = [0]
   - Reduce in-degree of 1 and 2
   - in-degree[1] = 0, in-degree[2] = 0
   - queue = [1, 2]

3. Process 1: ordering = [0, 1]
   - Reduce in-degree of 3
   - in-degree[3] = 1
   - queue = [2]

4. Process 2: ordering = [0, 1, 2]
   - Reduce in-degree of 3
   - in-degree[3] = 0
   - queue = [3]

5. Process 3: ordering = [0, 1, 2, 3]
   - queue = []

Result: [0, 1, 2, 3] (or [0, 2, 1, 3] depending on queue order)
"""


# Comparison: BFS (Kahn's) vs DFS
"""
BFS (Kahn's Algorithm):
+ More intuitive for beginners
+ Natural for "level" processing
+ Easier to explain in interviews
- Requires in-degree array
- More space

DFS:
+ No need for in-degree tracking
+ More elegant code
+ Natural recursion
- Harder to explain cycle detection
- State management needed

Both have same complexity: O(V + E) time and space
"""`,
  leetcodeUrl: 'https://leetcode.com/problems/course-schedule-ii/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Akt3glAwyfY',
};

/**
 * Keys and Rooms
 * Problem ID: keys-and-rooms
 * Order: 8
 */

import { Problem } from '../../../types';

export const keys_and_roomsProblem: Problem = {
  id: 'keys-and-rooms',
  title: 'Keys and Rooms',
  difficulty: 'Medium',
  topic: 'Graphs',
  description: `There are \`n\` rooms labeled from \`0\` to \`n - 1\` and all the rooms are locked except for room \`0\`. Your goal is to visit all the rooms. However, you cannot enter a locked room without having its key.

When you visit a room, you may find a set of **distinct keys** in it. Each key has a number on it, denoting which room it unlocks, and you can take all of them with you to unlock the other rooms.

Given an array \`rooms\` where \`rooms[i]\` is the set of keys that you can obtain if you visited room \`i\`, return \`true\` if you can visit **all** the rooms, or \`false\` otherwise.`,
  examples: [
    {
      input: 'rooms = [[1],[2],[3],[]]',
      output: 'true',
      explanation:
        'Start in room 0, pick up key 1. Go to room 1, pick up key 2. Go to room 2, pick up key 3. Go to room 3. All rooms visited.',
    },
    {
      input: 'rooms = [[1,3],[3,0,1],[2],[0]]',
      output: 'false',
      explanation:
        'Cannot enter room 2 since the only key that unlocks it is in that room.',
    },
  ],
  constraints: [
    'n == rooms.length',
    '2 <= n <= 1000',
    '0 <= rooms[i].length <= 1000',
    '1 <= sum(rooms[i].length) <= 3000',
    '0 <= rooms[i][j] < n',
    'All the values of rooms[i] are unique',
  ],
  hints: [
    'Use DFS or BFS starting from room 0',
    'Track visited rooms',
    'Check if all rooms were visited',
  ],
  starterCode: `from typing import List

def can_visit_all_rooms(rooms: List[List[int]]) -> bool:
    """
    Check if all rooms can be visited.
    
    Args:
        rooms: List of keys in each room
        
    Returns:
        True if all rooms can be visited
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[[1], [2], [3], []]],
      expected: true,
    },
    {
      input: [[[1, 3], [3, 0, 1], [2], [0]]],
      expected: false,
    },
  ],
  timeComplexity: 'O(N + E)',
  spaceComplexity: 'O(N)',
  leetcodeUrl: 'https://leetcode.com/problems/keys-and-rooms/',
  youtubeUrl: 'https://www.youtube.com/watch?v=ye_7c2K0Ark',
};

/**
 * Quiz questions for Queue Interview Strategy section
 */

export const interviewstrategyQuiz = [
  {
    id: 'q1',
    question:
      'What are the key indicators that a problem requires BFS with a queue rather than DFS?',
    sampleAnswer:
      'Use BFS with queue when: (1) Problem asks for "shortest path" in unweighted graph - BFS guarantees finding shortest path first, (2) Need "level by level" or "layer by layer" processing, (3) Problem mentions "minimum depth/steps/moves", (4) Need to find nodes at distance k from source, (5) "Closest" or "nearest" in unweighted context. Use DFS when: detecting cycles, finding any path (not shortest), exploring all possibilities with backtracking, topological sort. Key difference: BFS explores breadth-first (all neighbors before going deeper), DFS explores depth-first (go as deep as possible). BFS with queue is also better when tree is very deep but we need shallow answer.',
    keyPoints: [
      'BFS: shortest path in unweighted graphs',
      'BFS: level-by-level processing',
      'BFS: minimum depth/steps',
      'DFS: any path, backtracking, cycles',
      'BFS better for shallow answers in deep trees',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk through the level-order traversal pattern. Why is capturing len (queue) crucial?',
    sampleAnswer:
      'Level-order processes tree level by level. At start of each iteration, len (queue) tells us how many nodes are in the current level. We loop exactly that many times to process the current level, adding children to queue for the next level. Without capturing len (queue), we would process children immediately instead of waiting for next level. Example: level 0 has root (1 node), len (queue)=1, process 1 node, add 2 children. Level 1: len (queue)=2, process 2 nodes, add 4 children. Level 2: len (queue)=4. The captured length separates levels. Alternative: use sentinel value (like None) to mark level boundaries, but len (queue) is cleaner.',
    keyPoints: [
      'len (queue) at start = nodes in current level',
      'Loop that many times to process just this level',
      'Children added for next level, not processed now',
      'Separates levels cleanly',
      'Without it, would process children immediately',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the common mistakes when implementing queue-based solutions in interviews?',
    sampleAnswer:
      'Common mistakes: (1) Using list.pop(0) instead of deque.popleft() - O(n) vs O(1), (2) Forgetting to mark nodes as visited in graph BFS - causes infinite loops, (3) Not capturing len (queue) before level loop - mixes levels, (4) Adding node to queue after checking visited instead of when marking visited - adds duplicates, (5) Using queue.pop() instead of popleft() - processes LIFO not FIFO, (6) Not checking empty queue before popleft() - IndexError, (7) Initializing queue incorrectly (forgetting starting nodes). Prevention: use templates, test with simple example, check visited logic, verify FIFO order.',
    keyPoints: [
      'Use deque.popleft(), not list.pop(0)',
      'Mark visited when adding to queue',
      'Capture len (queue) for level-order',
      'Use popleft() not pop()',
      'Check empty before popleft()',
    ],
  },
];

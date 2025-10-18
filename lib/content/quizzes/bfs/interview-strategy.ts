/**
 * Quiz questions for Interview Strategy section
 */

export const interviewstrategyQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize a BFS problem? What keywords and patterns signal BFS?',
    sampleAnswer:
      'Keywords: "shortest path", "minimum steps", "level", "nearest", "closest", "minimum distance", "level-order". Patterns: 1) Shortest path in unweighted graph. 2) Level-by-level traversal. 3) Minimum moves/steps. 4) Nearest neighbor. 5) Multi-source exploration. Signals: unweighted graph + shortest, tree + level-order, minimum steps in state space. For example, "shortest path maze" → BFS (each move costs 1). "Binary tree level order" → BFS natural fit. "Minimum moves to solve puzzle" → BFS on state graph. If "shortest" with weighted → Dijkstra not BFS. If exploring all paths → DFS. If shortest unweighted → always BFS.',
    keyPoints: [
      'Keywords: shortest, minimum, level, nearest',
      'Patterns: shortest path, level order, min steps',
      'Unweighted + shortest → BFS',
      'Weighted + shortest → Dijkstra',
      'All paths → DFS',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your BFS interview approach from recognition to implementation.',
    sampleAnswer:
      'First, recognize BFS from keywords (shortest, level, minimum steps). Second, identify if graph/tree/state-space (positions, configurations). Third, define state representation (node, position, configuration). Fourth, determine what to track (distance, parent, level). Fifth, initialize: queue with start, visited set, mark start visited. Sixth, BFS loop: dequeue, check if target, enqueue unvisited neighbors with updated distance. Seventh, test with examples and edges. Finally, analyze complexity O(V+E) or O(states). For example, "word ladder": recognize minimum steps (BFS), state is word, neighbors are one-letter changes, queue with start word, BFS until reaching end word, O(words × word_length).',
    keyPoints: [
      'Recognize: shortest, level, minimum',
      'Identify: graph/tree/state-space',
      'Define: state representation',
      'Track: distance, parent, level',
      'Initialize: queue, visited, mark start',
      'Test and analyze complexity',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the most common mistakes in BFS problems? How do you avoid them?',
    sampleAnswer:
      'First: using stack instead of queue (becomes DFS). Second: marking visited when dequeuing not enqueuing (duplicates in queue, higher space/time). Third: forgetting to mark start visited. Fourth: not handling disconnected graphs (only explore from one component). Fifth: not tracking level when problem needs it. Sixth: using BFS for weighted graphs (need Dijkstra). My strategy: 1) Always use queue (collections.deque in Python). 2) Mark visited when enqueuing neighbor. 3) Mark start before loop. 4) For components, loop through all unvisited nodes. 5) Track level: count nodes per level or store (node, level) in queue. 6) Unweighted only. Test: disconnected, single node, already visited start.',
    keyPoints: [
      'Mistakes: stack (DFS), visited timing, no start mark',
      'Mark visited when enqueuing not dequeuing',
      'Use queue (deque), not stack',
      'Disconnected: loop all unvisited',
      'Track level if needed',
      'Unweighted only (else Dijkstra)',
    ],
  },
];

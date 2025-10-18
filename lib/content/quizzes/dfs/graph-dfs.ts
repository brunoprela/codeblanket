/**
 * Quiz questions for DFS on Graphs section
 */

export const graphdfsQuiz = [
  {
    id: 'q1',
    question: 'Explain DFS on graphs. What is different from tree DFS?',
    sampleAnswer:
      'Graph DFS similar to tree but: 1) Need visited set (graphs have cycles). 2) Multiple edges to same node possible. 3) No clear parent-child (undirected or cycles). 4) Start node may not reach all (disconnected). Tree DFS: no visited needed (no cycles), parent known. Graph DFS: visited set prevents infinite loops, track visited in set or array. For example, graph 0→1, 1→2, 2→0 (cycle): without visited, 0→1→2→0→1... infinite. With visited: mark 0, visit 1, mark 1, visit 2, mark 2, try 0 (already visited, skip). Time O(V+E) visit each vertex once, each edge once. Space O(V) for visited. For disconnected graphs: run DFS from each unvisited node (connected components).',
    keyPoints: [
      'Graph needs visited set (cycles exist)',
      'Tree: no cycles, visited not needed',
      'Mark visited before recursing',
      'O(V+E) time, O(V) space',
      'Disconnected: run DFS from each unvisited',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe cycle detection using DFS. How do you differentiate back edge from tree edge?',
    sampleAnswer:
      'Cycle detection: use three states: unvisited, visiting (in current path), visited (finished). Cycle exists if we reach a "visiting" node (back edge to ancestor). Algorithm: mark node as visiting, recurse on neighbors, mark as visited when done. If neighbor is visiting → cycle. For example, 0→1→2→0: visit 0 (visiting), visit 1 (visiting), visit 2 (visiting), try 0 (visiting! cycle found). For DAG: never encounter visiting node. Directed graph: visiting state tracks current path. Undirected: simpler, just check if neighbor is visited and not parent. The key: visiting state means node is in current DFS path (ancestor), detecting back edge proves cycle.',
    keyPoints: [
      'Three states: unvisited, visiting, visited',
      'Cycle: reach node in "visiting" state',
      'Visiting = in current DFS path',
      'Back edge to ancestor = cycle',
      'Undirected: simpler, check visited except parent',
    ],
  },
  {
    id: 'q3',
    question: 'Explain connected components using DFS. How do you count them?',
    sampleAnswer:
      'Connected components: groups of nodes reachable from each other. Count: for each unvisited node, run DFS (marks entire component), increment counter. For example, graph with nodes 0,1,2,3,4,5 and edges 0-1, 2-3-4: two components. Algorithm: visited = set(), count = 0; for each node: if not visited, DFS(node), count++. First DFS from 0 marks 0,1. Second DFS from 2 marks 2,3,4. Result: 2 components. Each DFS explores one component completely. Time O(V+E) total (each node/edge visited once across all DFS calls). This works for undirected graphs. For directed: strongly connected components need different algorithm (Tarjan, Kosaraju).',
    keyPoints: [
      'Count: DFS from each unvisited node',
      'Each DFS marks one component',
      'Increment counter per DFS call',
      'O(V+E) total across all DFS',
      'Directed: use Tarjan/Kosaraju for SCC',
    ],
  },
];

/**
 * Quiz questions for Union-Find (Disjoint Set Union) section
 */

export const unionfindQuiz = [
  {
    id: 'q1',
    question:
      'Explain Union-Find data structure. How do find and union operations work?',
    sampleAnswer:
      'Union-Find (Disjoint Set Union) tracks connected components. Each element has parent pointer, root of tree represents component. Find(x): follow parent pointers until reaching root (element where parent[x] = x). Union(x, y): find roots of x and y, make one root point to other. Initially each element is own parent (separate components). For example, union(1,2) and union(3,4): creates two trees with roots 1 and 3. Then union(1,3) merges into one tree. Operations track which elements are connected without storing all edges. Used for: dynamic connectivity, Kruskal MST, network connectivity, image segmentation. Basic operations are O(n) worst case (linear tree), but optimizations make them nearly O(1).',
    keyPoints: [
      'Tracks connected components with parent pointers',
      'Find: follow parents to root',
      'Union: connect roots of two components',
      'Initially: each element separate',
      'Used: connectivity, Kruskal, segmentation',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe path compression and union by rank optimizations. Why do they make operations nearly constant time?',
    sampleAnswer:
      'Path compression: during find(x), make all nodes on path point directly to root. Flattens tree. Union by rank: track tree depth (rank), attach shorter tree under taller. Together achieve O(α(n)) per operation where α is inverse Ackermann - grows so slowly it is effectively O(1) (α(n) < 5 for any practical n). Without optimizations, trees can become linear chains O(n) tall. With optimizations, trees stay very flat. For example, path compression: find(5) in chain 1→2→3→4→5 flattens to all pointing to 1. Union by rank: prevents attaching big tree under small (which increases height). Mathematical proof shows amortized O(α(n)) - nearly constant for all practical purposes. This makes Union-Find incredibly efficient.',
    keyPoints: [
      'Path compression: flatten during find',
      'Union by rank: shorter under taller',
      'Achieves O(α(n)) ≈ O(1) practical',
      'Without: trees can be O(n) tall',
      'With: trees stay very flat (height < 5)',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through using Union-Find for Kruskal MST algorithm. Why is Union-Find perfect for this?',
    sampleAnswer:
      'Kruskal MST: sort edges by weight, iterate edges in order, add edge if it connects different components (no cycle). Union-Find checks connectivity efficiently. Algorithm: initialize Union-Find with n vertices (separate components), sort edges O(E log E), for each edge (u,v): if find(u) != find(v) (different components), add edge to MST and union(u,v). Why perfect? Need to: 1) check if edge creates cycle (different components?), 2) merge components. Union-Find does both in O(α(n)) ≈ O(1). Alternative: DFS to check cycle is O(V+E) per edge, total O(E(V+E)) - too slow. Union-Find makes Kruskal O(E log E) dominated by sorting. The "union" operation naturally represents merging components as we build MST.',
    keyPoints: [
      'Kruskal: sort edges, add if no cycle',
      'Union-Find checks: different components?',
      'Find: check connectivity, Union: merge components',
      'O(α(n)) per edge vs O(V+E) DFS',
      'Total: O(E log E) for sorting, Union-Find is fast',
    ],
  },
];

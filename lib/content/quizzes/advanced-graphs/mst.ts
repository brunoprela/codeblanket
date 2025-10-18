/**
 * Quiz questions for Minimum Spanning Tree (MST) section
 */

export const mstQuiz = [
  {
    id: 'q1',
    question: 'Compare Kruskal vs Prim for MST. When would you choose each?',
    sampleAnswer:
      'Kruskal: sort edges O(E log E), use Union-Find to add edges avoiding cycles. Total O(E log E). Prim: grow tree from vertex, use heap to pick minimum edge to unexplored vertex. Total O(E log V). Choose Kruskal when: sparse graph (E << V²), edges already sorted or can sort efficiently, easy to parallelize (edge-based), want simple implementation with Union-Find. Choose Prim when: dense graph (E ≈ V²), need to start from specific vertex, edges stored as adjacency list. For example, sparse graph 1000 vertices, 5000 edges: Kruskal O(5000 log 5000) ≈ 60K, Prim O(5000 log 1000) ≈ 50K - similar. Dense graph 1000 vertices, 500K edges: Kruskal O(500K log 500K) ≈ 9M, Prim O(500K log 1000) ≈ 5M - Prim better. In practice, both work well.',
    keyPoints: [
      'Kruskal: O(E log E) edge-based, sparse graphs',
      'Prim: O(E log V) vertex-based, dense graphs',
      'Kruskal: sort edges, union-find',
      'Prim: grow tree, heap for min edge',
      'Choice depends on graph density',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain why MST algorithms are greedy and why the greedy choice is optimal.',
    sampleAnswer:
      'MST greedy choice: always pick minimum weight edge that connects components (Kruskal) or minimum edge to unexplored vertex (Prim). Proof by cut property: for any cut of graph, minimum weight edge crossing cut is in some MST. Both algorithms respect this: Kruskal picks minimum globally avoiding cycles (implicit cut between connected components), Prim picks minimum locally (explicit cut between tree and non-tree vertices). The greedy choice is safe - adding minimum edge cannot be wrong because cut property guarantees it belongs to some MST. This is unlike general optimization where local optimum might not be global. MST has special structure making greedy optimal. Proof by contradiction: if greedy edge not in MST, can swap it for heavier edge, reducing total weight - contradiction.',
    keyPoints: [
      'Greedy: always pick minimum available edge',
      'Cut property: min edge crossing cut is in MST',
      'Kruskal respects cut between components',
      'Prim respects cut between tree and non-tree',
      'Proof: swapping greedy edge reduces weight',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe how MST is used for clustering. What is the connection?',
    sampleAnswer:
      'MST for clustering: build MST, remove k-1 heaviest edges to get k clusters. Intuition: MST connects all points with minimum total distance. Heavy edges represent large gaps between clusters. Removing heavy edges separates natural clusters. For example, customer segmentation: points are customers (features as coordinates), distance is dissimilarity. MST finds minimum connections. Remove 2 heaviest edges to get 3 clusters. Why works? MST captures global structure efficiently. Removing edge disconnects into components - natural clusters. This is single-linkage clustering. Alternative: use MST edge weights as threshold (edges > threshold separate clusters). MST gives hierarchical clustering by considering removal of edges in weight order. Simple, efficient O(E log E), but sensitive to outliers (single outlier edge can connect clusters).',
    keyPoints: [
      'Build MST, remove k-1 heaviest edges → k clusters',
      'Heavy edges = gaps between clusters',
      'MST captures global structure efficiently',
      'Single-linkage clustering',
      'Limitation: sensitive to outlier edges',
    ],
  },
];

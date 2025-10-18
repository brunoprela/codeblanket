/**
 * Algorithm Comparison Section
 */

export const comparisonSection = {
  id: 'comparison',
  title: 'Algorithm Comparison',
  content: `**Shortest Path Algorithm Selection:**

| Algorithm | Use Case | Time | Space | Negative Weights | Negative Cycles |
|-----------|----------|------|-------|------------------|-----------------|
| BFS | Unweighted | O(V+E) | O(V) | N/A | No |
| Dijkstra | Non-negative weights | O((V+E)logV) | O(V) | ❌ No | N/A |
| Bellman-Ford | Negative weights OK | O(VE) | O(V) | ✅ Yes | ✅ Detects |
| Floyd-Warshall | All pairs | O(V³) | O(V²) | ✅ Yes | ✅ Detects |

**Decision Tree:**

\`\`\`
Need shortest path?
│
├─ Unweighted graph?
│  └─ Use BFS (O(V+E))
│
├─ Single source?
│  │
│  ├─ Non-negative weights?
│  │  └─ Use Dijkstra (O((V+E)logV))
│  │
│  └─ Negative weights possible?
│     └─ Use Bellman-Ford (O(VE))
│
└─ All pairs?
   │
   ├─ Sparse graph, many queries?
   │  └─ Run Dijkstra V times
   │
   └─ Dense graph or small V?
      └─ Use Floyd-Warshall (O(V³))
\`\`\`

**Practical Guidelines:**

**Use Dijkstra when:**
- GPS/navigation (non-negative distances)
- Network routing (non-negative costs)
- Most real-world shortest path problems

**Use Bellman-Ford when:**
- Arbitrage detection (negative cycles)
- Some financial models
- Need to detect negative cycles

**Use Floyd-Warshall when:**
- Small graph (V ≤ 400)
- Need many shortest paths
- Transitive closure problems

**Use BFS when:**
- Unweighted graph (or all weights equal)
- Simplest and fastest for this case`,
};

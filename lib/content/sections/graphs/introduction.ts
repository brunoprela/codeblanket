/**
 * Introduction to Graphs Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'Introduction to Graphs',
  content: `A **graph** is a data structure consisting of **vertices (nodes)** connected by **edges**. Graphs model relationships and networks: social networks, maps, dependencies, etc.

**Graph Terminology:**

- **Vertex/Node**: A point in the graph
- **Edge**: Connection between two vertices
- **Directed Graph**: Edges have direction (A → B)
- **Undirected Graph**: Edges are bidirectional (A ↔ B)
- **Weighted Graph**: Edges have values/costs
- **Degree**: Number of edges connected to a vertex
- **Path**: Sequence of vertices connected by edges
- **Cycle**: Path that starts and ends at same vertex
- **Connected Graph**: Path exists between any two vertices
- **DAG**: Directed Acyclic Graph (no cycles)

**Graph Representations:**

**1. Adjacency List** (Most Common)
\`\`\`python
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}

# Visualized:
#   0---1
#   |   |
#   2---3
\`\`\`

**Pros**: Space efficient O(V + E), fast to iterate neighbors
**Cons**: Slow to check if edge exists

**2. Adjacency Matrix**
\`\`\`python
matrix = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]
# matrix[i][j] = 1 if edge exists from i to j
\`\`\`

**Pros**: O(1) edge lookup, simple
**Cons**: O(V²) space, inefficient for sparse graphs

**3. Edge List**
\`\`\`python
edges = [(0,1), (0,2), (1,3), (2,3)]
\`\`\`

**Pros**: Simple, good for algorithms processing all edges
**Cons**: Slow to find neighbors

**When to Use Graphs:**
- Social networks (friends, followers)
- Maps and navigation (cities, roads)
- Dependencies (tasks, packages)
- Networks (computers, websites)
- State machines and game trees`,
};

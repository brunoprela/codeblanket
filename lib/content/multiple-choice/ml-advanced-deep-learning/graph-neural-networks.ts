/**
 * Graph Neural Networks (GNNs) Multiple Choice Questions
 */

export const graphNeuralNetworksMultipleChoice = [
  {
    id: 'mc-1',
    question:
      'What is the key advantage of Graph Neural Networks over traditional neural networks like CNNs and RNNs?',
    options: [
      'GNNs are faster to train than CNNs and RNNs',
      'GNNs can handle arbitrary graph-structured data where relationships between entities are as important as the entities themselves',
      'GNNs require less training data than CNNs',
      'GNNs have fewer parameters than traditional networks',
    ],
    correctAnswer: 1,
    explanation:
      'GNNs are specifically designed for **graph-structured data** with arbitrary topology: **Traditional networks**: CNNs: Grid structure (images: neighbors in 2D grid), Fixed neighborhood (3×3, 5×5 kernels). RNNs: Sequential structure (text: left-to-right sequence), Fixed order. **Problem**: Many real-world data has ARBITRARY graph structure. Social networks: Variable number of friends per person, no grid/sequence. Molecules: Atoms with variable bonds, 3D structure. Citation networks: Papers cite variable numbers of other papers. **GNN solution**: Handles variable-size neighborhoods, No fixed structure required, Learns from BOTH node features AND graph topology. **Message passing**: Each node aggregates information from its specific neighbors (however many), Adapts to graph structure automatically. **Example**: Social network node with 3 friends vs. node with 30 friends - GNN handles both! **Not about**: Training speed (actually similar/slower), data requirements (similar), parameter count (similar). The fundamental advantage is flexibility to work with arbitrary graph structures where relationships matter.',
  },
  {
    id: 'mc-2',
    question:
      'In a Graph Convolutional Network (GCN), what does it mean when we say a 2-layer GCN captures 2-hop neighborhoods?',
    options: [
      'The GCN looks at the 2 nearest neighbors of each node',
      "Each node's final representation incorporates information from nodes up to 2 edges away",
      'The GCN has exactly 2 convolutional layers',
      'Each message passes through 2 nodes before aggregation',
    ],
    correctAnswer: 1,
    explanation:
      "k-layer GCN captures **k-hop neighborhood** - information propagates k edges away: **Message passing per layer**: Layer 1: Each node aggregates direct neighbors (1-hop). Node A receives features from nodes 1-edge away. Layer 2: Each node aggregates from layer-1 representations. Node A receives features that already contain 1-hop info. Now Node A has information from 2-hops away (friends of friends)! **Example graph**: A -- B -- C -- D (linear). **After 1 layer**: A knows about B (direct neighbor), B knows about A and C. **After 2 layers**: A knows about B AND C (2-hops), B knows about A, C, and D. **After 3 layers**: A knows about B, C, AND D (3-hops). **Mathematical**: h^(k)_A depends on {h^(k-1)_neighbor}, which depends on {h^(k-2)_neighbor_of_neighbor}, ..., recursively expanding k-hops! **Implications**: (1) **Receptive field**: k layers → k-hop information, (2) **Over-smoothing**: Too many layers (k=10) → all nodes have similar representations, (3) **Computational cost**: k layers → k matrix multiplications. **Not about**: Number of neighbors (varies by node), having 2 layers (that's related but not the definition), message routing (messages aggregate, not pass through). k-hop neighborhoods are fundamental to understanding GNN capacity and depth.",
  },
  {
    id: 'mc-3',
    question:
      'Why do we add self-loops (A + I) to the adjacency matrix in GCNs?',
    options: [
      'To make the matrix multiplication faster',
      'To allow each node to retain its own features when aggregating, preventing information loss',
      'To convert the graph from directed to undirected',
      'To increase the number of edges in the graph',
    ],
    correctAnswer: 1,
    explanation:
      "Self-loops ensure nodes **retain their own information** during aggregation: **Without self-loops** (only A): h'_i = AGG({h_j : j ∈ neighbors(i)}). Aggregates ONLY neighbors, NOT node i itself. Problem: Node i's original features get lost! Only receives neighbor information. **Example**: Node with features [1.0, 0.5, 2.0] and two neighbors [0.2, 0.8, 1.5], [0.9, 0.3, 1.0]. Without self-loop: h' = average([0.2, 0.8, 1.5], [0.9, 0.3, 1.0]) = [0.55, 0.55, 1.25]. Original [1.0, 0.5, 2.0] is LOST! **With self-loops** (A + I): h'_i = AGG({h_j : j ∈ neighbors(i) ∪ {i}}). Aggregates neighbors AND itself. Node i participates in its own aggregation! **Example (continued)**: With self-loop: h' = average([1.0, 0.5, 2.0], [0.2, 0.8, 1.5], [0.9, 0.3, 1.0]) = [0.70, 0.53, 1.50]. Original information preserved (weighted with neighbors). **Benefits**: (1) Prevents information loss, (2) Allows identity mapping (h' ≈ h if neighbors not informative), (3) Stabilizes training. **Not about**: Computational speed (negligible difference), directed/undirected (separate concern), edge count (implementation detail). Self-loops are ESSENTIAL for preserving node identity during message passing!",
  },
  {
    id: 'mc-4',
    question:
      'What is the main difference between Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT)?',
    options: [
      'GAT has more layers than GCN',
      'GAT learns attention weights to assign different importance to different neighbors, while GCN treats all neighbors equally (after normalization)',
      'GCN works on directed graphs while GAT works on undirected graphs',
      'GAT uses RNNs while GCN uses CNNs',
    ],
    correctAnswer: 1,
    explanation:
      "The key difference is **neighbor weighting**: **GCN**: Fixed weights based on degree: α_{ij} = 1/√(degree_i × degree_j). All neighbors weighted by graph structure (degree normalization). Weights determined by topology, not learned! **Example**: Node with 4 neighbors → each contributes 0.25 (equal after normalization). **GAT**: Learned attention weights: α_{ij} = softmax(attention_score(h_i, h_j)). Different neighbors get different weights based on features! Network learns which neighbors matter more. **Example**: Node with 4 neighbors → attention [0.5, 0.3, 0.15, 0.05] (learned during training). **Mathematical**: GCN: h'_i = Σ_j (1/√(d_i d_j)) × W h_j (fixed coefficients). GAT: h'_i = Σ_j α_{ij} × W h_j where α_{ij} = softmax(LeakyReLU(a^T [Wh_i || Wh_j])) (learned coefficients). **When GAT is better**: (1) **Heterogeneous graphs**: Different neighbor types/importance, (2) **Noisy edges**: Can downweight irrelevant connections, (3) **Interpretability**: Attention weights show which neighbors matter. **Trade-offs**: GAT: More parameters, slower, more expressive. GCN: Fewer parameters, faster, simpler. **Not about**: Network depth, graph types (both handle directed/undirected), or architectural paradigm (both message-passing, neither uses RNN/CNN directly). Attention vs. fixed weighting is the fundamental distinction!",
  },
  {
    id: 'mc-5',
    question:
      'For graph-level classification tasks (e.g., molecule property prediction), why is pooling necessary?',
    options: [
      'To reduce the computational cost of the forward pass',
      'To aggregate variable-size node representations into a fixed-size graph representation that can be fed to a classifier',
      'To prevent overfitting by reducing the number of features',
      'To make the graph easier to visualize',
    ],
    correctAnswer: 1,
    explanation:
      "Pooling solves the **variable-size input problem** for graph-level prediction: **Problem**: Graphs have variable number of nodes. Molecule A: 15 atoms (nodes), Molecule B: 32 atoms (nodes). After GCN: Molecule A → 15 node representations (15 × d), Molecule B → 32 node representations (32 × d). Classifier needs FIXED input size! Can't handle variable-size input. **Solution: Pooling**: Aggregate all node representations → single graph representation. Mean pooling: h_graph = (1/n) Σᵢ h_i (fixed size d regardless of n). Max pooling: h_graph = max(h_1, ..., h_n) (fixed size d). Sum pooling: h_graph = Σᵢ h_i (fixed size d). **Result**: Variable graphs (n nodes) → Fixed representation (d dimensions) → Classifier. **Example**: Molecule A (15 atoms, each with 64-d representation) → Pool → 64-d graph vector, Molecule B (32 atoms, each with 64-d representation) → Pool → 64-d graph vector, Classifier: 64-d input → 2 classes (toxic/non-toxic). **Types of pooling**: (1) **Global**: Aggregate all nodes (mean/max/sum), (2) **Hierarchical**: Coarsen graph iteratively (DiffPool), (3) **Attention-based**: Weighted sum with learned weights. **Not primarily about**: Computational cost (side benefit), overfitting (not main purpose), visualization (unrelated). Pooling is NECESSARY for graph-level tasks because classifier requires fixed-size input!",
  },
];

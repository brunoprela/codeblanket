/**
 * Graph Neural Networks (GNNs) Content
 */

export const graphNeuralNetworksSection = {
  id: 'graph-neural-networks',
  title: 'Graph Neural Networks (GNNs)',
  content: `
# Graph Neural Networks (GNNs)

## Introduction

**Graph Neural Networks (GNNs)** are a class of deep learning models designed to work with **graph-structured data** - data where relationships (edges) between entities (nodes) are as important as the entities themselves.

**Why graphs?**

Many real-world data is naturally graph-structured:
- **Social networks**: Users (nodes) connected by friendships (edges)
- **Molecules**: Atoms (nodes) connected by bonds (edges)
- **Citation networks**: Papers (nodes) linked by citations (edges)
- **Knowledge graphs**: Entities (nodes) linked by relationships (edges)
- **Transportation**: Locations (nodes) connected by roads/flights (edges)

**Limitation of traditional neural networks**:
- CNNs: Work on grid-structured data (images)
- RNNs: Work on sequential data (text, time series)
- Neither captures **arbitrary graph topology**

**GNN goal**: Learn representations of nodes, edges, or entire graphs that capture both **node features** and **graph structure**.

---

## Graph Basics

### Graph Definition

A graph G = (V, E) consists of:
- **V**: Set of nodes/vertices
- **E**: Set of edges connecting nodes

**Types**:
1. **Undirected**: Edges have no direction (Facebook friendship)
2. **Directed**: Edges have direction (Twitter follow)
3. **Weighted**: Edges have weights (distance between cities)
4. **Unweighted**: All edges equal importance

### Graph Representations

**Adjacency Matrix** (A):
- Size: n × n (n = number of nodes)
- A[i,j] = 1 if edge from node i to j, else 0
- For weighted graphs: A[i,j] = edge weight

Example (3 nodes, edges: 0→1, 1→2, 2→0):
\`\`\`
A = [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0]]
\`\`\`

**Node Features** (X):
- Size: n × d (d = feature dimension)
- X[i] = feature vector for node i

Example (3 nodes, 2 features each):
\`\`\`
X = [[0.5, 1.2],   # Node 0 features
     [0.8, 0.3],   # Node 1 features
     [1.1, 0.7]]   # Node 2 features
\`\`\`

---

## Core Idea: Message Passing

**Key insight**: A node's representation should depend on its neighbors!

**Message passing framework**:

1. **Initialize**: Each node starts with its features h^(0)_i = x_i
2. **Message aggregation**: Collect information from neighbors
3. **Update**: Combine aggregated messages with own features
4. **Repeat**: Multiple layers for multi-hop information propagation

**Mathematically** (layer ℓ):
\`\`\`
h^(ℓ+1)_i = UPDATE^(ℓ)(h^(ℓ)_i, AGGREGATE^(ℓ)({h^(ℓ)_j : j ∈ N(i)}))
                       ↑                             ↑
                   own features            neighbor features
\`\`\`

Where N(i) = neighbors of node i.

**After k layers**: Each node's representation incorporates information from k-hop neighborhood.

---

## Graph Convolutional Networks (GCN)

**GCN** (Kipf & Welling, 2017) is the most influential GNN architecture.

### Mathematical Formulation

**Layer-wise propagation rule**:
\`\`\`
H^(ℓ+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(ℓ) W^(ℓ))
\`\`\`

Where:
- H^(ℓ): Node representations at layer ℓ (n × d)
- Ã = A + I: Adjacency matrix + self-loops
- D̃: Degree matrix of Ã
- W^(ℓ): Learnable weight matrix (d × d')
- σ: Activation function (e.g., ReLU)

**Intuition**:
1. **Ã H^(ℓ)**: Aggregate neighbor features
2. **D̃^(-1/2) ... D̃^(-1/2)**: Normalize by degree (symmetric normalization)
3. **... W^(ℓ)**: Transform features (learnable)
4. **σ(...)**: Apply non-linearity

**Simplified version** (for understanding):
\`\`\`
h^(ℓ+1)_i = σ(W^(ℓ) Σ_{j∈N(i)∪{i}} h^(ℓ)_j / √(|N(i)| × |N(j)|))
\`\`\`

Each node averages its neighbors' features (including itself), transforms them, and applies activation.

### Implementation

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer (nn.Module):
    def __init__(self, in_features, out_features):
        """
        Single Graph Convolutional layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
        """
        super().__init__()
        self.linear = nn.Linear (in_features, out_features)
    
    def forward (self, X, A):
        """
        Args:
            X: Node features (n_nodes, in_features)
            A: Adjacency matrix (n_nodes, n_nodes)
        
        Returns:
            H: Updated node representations (n_nodes, out_features)
        """
        # Add self-loops
        A_hat = A + torch.eye(A.size(0), device=A.device)
        
        # Degree matrix
        D = torch.diag(A_hat.sum (dim=1))
        
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag()))
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        
        # Message passing: aggregate neighbors, then transform
        H = A_norm @ X  # Aggregate
        H = self.linear(H)  # Transform
        
        return H


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2):
        """
        Graph Convolutional Network with multiple layers.
        
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden dimension
            out_features: Output dimension
            num_layers: Number of GCN layers
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GCNLayer (in_features, hidden_features))
        
        # Hidden layers
        for _ in range (num_layers - 2):
            self.layers.append(GCNLayer (hidden_features, hidden_features))
        
        # Output layer
        self.layers.append(GCNLayer (hidden_features, out_features))
    
    def forward (self, X, A):
        """
        Args:
            X: Node features (n_nodes, in_features)
            A: Adjacency matrix (n_nodes, n_nodes)
        
        Returns:
            H: Final node representations (n_nodes, out_features)
        """
        H = X
        
        # Apply GCN layers with ReLU (except last)
        for i, layer in enumerate (self.layers):
            H = layer(H, A)
            if i < len (self.layers) - 1:
                H = F.relu(H)
                H = F.dropout(H, p=0.5, training=self.training)
        
        return H


# Example usage
n_nodes = 5  # Number of nodes in graph
in_features = 3  # Input feature dimension
hidden_features = 16
out_features = 2  # Output dimension (e.g., 2 classes)

# Create graph
X = torch.randn (n_nodes, in_features)  # Node features
A = torch.tensor([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0]
], dtype=torch.float32)  # Adjacency matrix

print(f"Number of nodes: {n_nodes}")
print(f"Number of edges: {int(A.sum() / 2)}")  # Divide by 2 for undirected

# Create model
model = GCN(in_features, hidden_features, out_features, num_layers=2)

# Forward pass
output = model(X, A)
print(f"\\nInput shape: {X.shape}")  # (5, 3)
print(f"Output shape: {output.shape}")  # (5, 2)
print(f"Output (node representations):")
print(output)

# For node classification
predictions = output.argmax (dim=1)
print(f"\\nPredicted classes: {predictions}")
\`\`\`

---

## Node Classification Task

**Problem**: Predict label for each node using graph structure and features.

**Example**: Classify papers in citation network by topic.

### Implementation

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Create synthetic graph data
n_nodes = 100
n_features = 10
n_classes = 3

# Random node features
X = torch.randn (n_nodes, n_features)

# Create adjacency matrix (random graph)
A = torch.rand (n_nodes, n_nodes)
A = (A > 0.9).float()  # Sparse graph
A = (A + A.T) / 2  # Make symmetric (undirected)
A.fill_diagonal_(0)  # Remove self-loops (added by GCN)

print(f"Graph: {n_nodes} nodes, {int(A.sum()/2)} edges")
print(f"Average degree: {A.sum()/n_nodes:.2f}")

# Create labels
labels = torch.randint(0, n_classes, (n_nodes,))

# Split nodes: train/val/test
n_train = 60
n_val = 20
n_test = 20

train_mask = torch.zeros (n_nodes, dtype=torch.bool)
val_mask = torch.zeros (n_nodes, dtype=torch.bool)
test_mask = torch.zeros (n_nodes, dtype=torch.bool)

train_mask[:n_train] = True
val_mask[n_train:n_train+n_val] = True
test_mask[n_train+n_val:] = True

# Create model
model = GCN(n_features, hidden_features=32, out_features=n_classes, num_layers=2)
optimizer = optim.Adam (model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 100

for epoch in range (num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(X, A)
    
    # Compute loss only on training nodes
    loss = criterion (output[train_mask], labels[train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Validation
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            output = model(X, A)
            
            # Train accuracy
            train_pred = output[train_mask].argmax (dim=1)
            train_acc = (train_pred == labels[train_mask]).float().mean()
            
            # Val accuracy
            val_pred = output[val_mask].argmax (dim=1)
            val_acc = (val_pred == labels[val_mask]).float().mean()
            
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

# Test evaluation
model.eval()
with torch.no_grad():
    output = model(X, A)
    test_pred = output[test_mask].argmax (dim=1)
    test_acc = (test_pred == labels[test_mask]).float().mean()
    print(f"\\nTest Accuracy: {test_acc:.4f}")
\`\`\`

---

## Graph Attention Networks (GAT)

**Problem with GCN**: All neighbors weighted equally.

**GAT idea**: Learn **attention weights** for neighbors - some neighbors more important than others!

### Architecture

**Attention mechanism**:
\`\`\`
α_{ij} = softmax_j (e_{ij})  (attention weight from j to i)

e_{ij} = LeakyReLU(a^T [W h_i || W h_j])  (attention coefficient)
\`\`\`

Where:
- || denotes concatenation
- a: Learnable attention vector
- W: Weight matrix

**Update rule**:
\`\`\`
h'_i = σ(Σ_{j∈N(i)} α_{ij} W h_j)
\`\`\`

Each node aggregates neighbors weighted by learned attention!

### Implementation

\`\`\`python
class GATLayer (nn.Module):
    def __init__(self, in_features, out_features, num_heads=1, concat=True):
        """
        Graph Attention layer with multi-head attention.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            num_heads: Number of attention heads
            concat: If True, concatenate heads; else average
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        # Multi-head attention
        self.W = nn.Parameter (torch.zeros (num_heads, in_features, out_features))
        self.a = nn.Parameter (torch.zeros (num_heads, 2 * out_features, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward (self, X, A):
        """
        Args:
            X: Node features (n_nodes, in_features)
            A: Adjacency matrix (n_nodes, n_nodes)
        
        Returns:
            H: Updated features (n_nodes, out_features * num_heads) or (n_nodes, out_features)
        """
        n_nodes = X.size(0)
        
        # Linear transformation for each head
        # (n_heads, n_nodes, out_features)
        Wh = torch.matmul(X, self.W)  # Broadcasting over heads
        
        # Compute attention coefficients
        # For each head and each pair of nodes
        attention_outputs = []
        
        for h in range (self.num_heads):
            Wh_h = Wh[h]  # (n_nodes, out_features)
            
            # Prepare for attention: [Wh_i || Wh_j] for all pairs
            # Wh_repeat_i: (n_nodes, n_nodes, out_features) - repeat along rows
            # Wh_repeat_j: (n_nodes, n_nodes, out_features) - repeat along cols
            Wh_repeat_i = Wh_h.unsqueeze(1).repeat(1, n_nodes, 1)
            Wh_repeat_j = Wh_h.unsqueeze(0).repeat (n_nodes, 1, 1)
            
            # Concatenate
            concat_features = torch.cat([Wh_repeat_i, Wh_repeat_j], dim=2)
            # Shape: (n_nodes, n_nodes, 2*out_features)
            
            # Compute attention scores
            e = self.leaky_relu (torch.matmul (concat_features, self.a[h]))
            # Shape: (n_nodes, n_nodes, 1)
            e = e.squeeze(2)  # (n_nodes, n_nodes)
            
            # Mask attention for non-neighbors
            e = torch.where(A > 0, e, torch.tensor (float('-inf'), device=e.device))
            
            # Softmax attention weights
            alpha = F.softmax (e, dim=1)  # (n_nodes, n_nodes)
            
            # Weighted aggregation
            h_prime = torch.matmul (alpha, Wh_h)  # (n_nodes, out_features)
            attention_outputs.append (h_prime)
        
        # Combine heads
        if self.concat:
            output = torch.cat (attention_outputs, dim=1)  # Concatenate heads
        else:
            output = torch.stack (attention_outputs, dim=0).mean (dim=0)  # Average heads
        
        return output


# Example usage
in_features, out_features, num_heads = 10, 8, 4
n_nodes = 5

X = torch.randn (n_nodes, in_features)
A = torch.tensor([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0]
], dtype=torch.float32)

gat_layer = GATLayer (in_features, out_features, num_heads=num_heads, concat=True)

output = gat_layer(X, A)
print(f"Input shape: {X.shape}")  # (5, 10)
print(f"Output shape: {output.shape}")  # (5, 8*4=32) with concat
\`\`\`

---

## Graph-Level Tasks

**Problem**: Predict properties of **entire graphs** (not individual nodes).

**Examples**:
- Molecule property prediction (toxicity, solubility)
- Graph classification (social network type)

**Solution**: **Graph pooling** - aggregate node representations into single graph representation.

### Pooling Methods

1. **Global mean pooling**: h_graph = mean (h_1, ..., h_n)
2. **Global max pooling**: h_graph = max (h_1, ..., h_n)
3. **Global sum pooling**: h_graph = sum (h_1, ..., h_n)
4. **Attention pooling**: Weighted sum with learned attention weights

### Implementation

\`\`\`python
class GraphClassifier (nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_classes):
        """
        Graph classification model: GCN + pooling + classifier
        """
        super().__init__()
        
        # GCN layers
        self.gcn1 = GCNLayer (in_features, hidden_features)
        self.gcn2 = GCNLayer (hidden_features, out_features)
        
        # Classifier
        self.classifier = nn.Linear (out_features, num_classes)
    
    def forward (self, X, A):
        """
        Args:
            X: Node features (n_nodes, in_features)
            A: Adjacency matrix (n_nodes, n_nodes)
        
        Returns:
            logits: Graph classification logits (num_classes,)
        """
        # GCN layers
        H = self.gcn1(X, A)
        H = F.relu(H)
        H = self.gcn2(H, A)
        H = F.relu(H)
        
        # Global pooling: aggregate all nodes
        h_graph = H.mean (dim=0)  # (out_features,)
        
        # Classify
        logits = self.classifier (h_graph)
        
        return logits


# Example: Classify multiple graphs (batch)
def classify_graph_batch (graphs, model):
    """
    Classify a batch of graphs.
    
    Args:
        graphs: List of (X, A) tuples
        model: Graph classifier
    
    Returns:
        predictions: Batch of predictions
    """
    predictions = []
    
    for X, A in graphs:
        logits = model(X, A)
        pred = logits.argmax().item()
        predictions.append (pred)
    
    return predictions
\`\`\`

---

## Applications

### 1. Social Network Analysis

**Task**: Predict user interests, detect communities, recommend connections.

**Graph**: Users as nodes, friendships as edges
**Features**: User profile, activity history
**GNN learns**: Homophily (friends have similar interests)

### 2. Molecule Property Prediction

**Task**: Predict chemical properties (toxicity, solubility, drug-likeness).

**Graph**: Atoms as nodes, bonds as edges
**Features**: Atom type, charge, bond type
**GNN learns**: Molecular structure → properties

### 3. Recommendation Systems

**Task**: Recommend items to users.

**Graph**: Users and items as nodes, interactions as edges (bipartite graph)
**Features**: User/item attributes
**GNN learns**: Complex user-item-user relationships

### 4. Traffic Prediction

**Task**: Predict traffic flow on road networks.

**Graph**: Intersections as nodes, roads as edges
**Features**: Historical traffic, time of day, weather
**GNN learns**: Spatial dependencies between locations

---

## Discussion Questions

1. **Why can't we just use standard CNNs or RNNs on graphs?**
   - Consider the differences in structure (grid vs. arbitrary topology)

2. **In GCNs, after k layers, each node's representation depends on its k-hop neighborhood. What are the implications for very deep GCNs?**
   - Think about over-smoothing and computational cost

3. **Graph Attention Networks learn different attention weights for different neighbors. When would this be more beneficial than GCN's equal weighting?**
   - Consider heterogeneous graphs and varying edge importance

4. **For graph-level classification, why is pooling necessary?**
   - Think about fixed vs. variable-size inputs

5. **How do GNNs handle graphs of different sizes (different numbers of nodes)?**
   - Consider the parameter sharing and pooling mechanisms

---

## Key Takeaways

- **GNNs** extend deep learning to graph-structured data by incorporating both node features and graph topology
- **Message passing**: Core mechanism where nodes aggregate information from neighbors iteratively
- **GCN**: Aggregates neighbors with symmetric normalization, simple and effective
- **Graph Attention Networks (GAT)**: Learn attention weights for neighbors, allowing different importance
- **Node classification**: Predict labels for nodes using local graph structure
- **Graph classification**: Predict properties of entire graphs using pooling
- **Multi-layer GNNs**: Capture k-hop neighborhoods with k layers
- **Applications**: Social networks, molecules, recommendation systems, knowledge graphs, traffic prediction
- **Challenges**: Over-smoothing with many layers, scalability to large graphs, heterogeneous graphs
- **Active research**: Graph transformers, geometric deep learning, heterogeneous GNNs

---

## Practical Tips

1. **Normalization**: Always normalize adjacency matrix (symmetric or row-wise)

2. **Self-loops**: Add self-loops (A + I) so nodes consider their own features

3. **Depth**: 2-3 layers often sufficient, more can cause over-smoothing

4. **Dropout**: Apply between layers to prevent overfitting

5. **Pooling**: For graph-level tasks, try mean/max/sum pooling

6. **Learning rate**: Lower than image tasks (1e-3 to 1e-2)

7. **Libraries**: Use PyTorch Geometric or DGL for efficient implementation

8. **Data augmentation**: Add/remove edges, mask features

---

## Further Reading

- ["Semi-Supervised Classification with Graph Convolutional Networks"](https://arxiv.org/abs/1609.02907) - Kipf & Welling, 2017 (GCN)
- ["Graph Attention Networks"](https://arxiv.org/abs/1710.10903) - Veličković et al., 2018 (GAT)
- ["How Powerful are Graph Neural Networks?"](https://arxiv.org/abs/1810.00826) - Xu et al., 2019 (GIN)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Popular GNN library

---

**Congratulations!** You've completed the Advanced Deep Learning Architectures module. You now understand CNNs, RNNs, Transformers, Transfer Learning, Autoencoders, GANs, and GNNs - the core architectures powering modern AI!

*Continue exploring the cutting edge of deep learning with our advanced modules on Reinforcement Learning, Computer Vision, and Natural Language Processing!*
`,
};

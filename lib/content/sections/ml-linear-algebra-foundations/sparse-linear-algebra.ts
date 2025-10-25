/**
 * Sparse Linear Algebra Section
 */

export const sparselinearalgebraSection = {
  id: 'sparse-linear-algebra',
  title: 'Sparse Linear Algebra',
  content: `
# Sparse Linear Algebra

## Introduction

**Sparse matrices** have mostly zero elements. Many real-world problems produce sparse matrices:
- Text data (word-document matrices: most documents don't contain most words)
- Graphs (adjacency matrices: most nodes don't connect to most others)
- Recommender systems (user-item matrices: users interact with tiny fraction of items)
- Scientific computing (finite element methods, PDEs)

**Why care?** Storing and computing with sparse matrices efficiently can save massive memory and computation.

\`\`\`python
import numpy as np
from scipy import sparse

print("=== Sparse vs Dense Matrices ===")

# Dense matrix (wasteful for sparse data)
dense = np.array([
    [0, 0, 3, 0, 0],
    [0, 0, 0, 0, 4],
    [0, 2, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 5, 0]
])

print("Dense matrix:")
print(dense)
print(f"Non-zero elements: {np.count_nonzero (dense)} / {dense.size}")
print(f"Sparsity: {1 - np.count_nonzero (dense)/dense.size:.1%}")
print(f"Memory: {dense.nbytes} bytes")
print()

# Sparse matrix (efficient)
sparse_csr = sparse.csr_matrix (dense)

print(f"Sparse (CSR) memory: {sparse_csr.data.nbytes + sparse_csr.indices.nbytes + sparse_csr.indptr.nbytes} bytes")
print(f"Compression ratio: {dense.nbytes / (sparse_csr.data.nbytes + sparse_csr.indices.nbytes + sparse_csr.indptr.nbytes):.1f}x")
\`\`\`

## Sparse Matrix Formats

### 1. COO (Coordinate Format)

Store (row, col, value) triplets for non-zero elements.

**Pros**: Simple, easy to construct
**Cons**: No efficient arithmetic, no random access

\`\`\`python
print("\\n=== COO Format ===")

# Create COO matrix
row = [0, 1, 2, 3, 4]
col = [2, 4, 1, 0, 3]
data = [3, 4, 2, 1, 5]

coo = sparse.coo_matrix((data, (row, col)), shape=(5, 5))

print("COO representation:")
print(f"Row indices: {coo.row}")
print(f"Col indices: {coo.col}")
print(f"Data: {coo.data}")
print()
print("As dense:")
print(coo.toarray())
\`\`\`

### 2. CSR (Compressed Sparse Row)

Store row pointers, column indices, and values.

**Pros**: Efficient row slicing, arithmetic, matrix-vector products
**Cons**: Column slicing slow, modifying structure expensive

\`\`\`python
print("\\n=== CSR Format ===")

csr = sparse.csr_matrix (dense)

print("CSR representation:")
print(f"Data: {csr.data}")        # Non-zero values
print(f"Indices: {csr.indices}")  # Column indices
print(f"Indptr: {csr.indptr}")    # Row pointers
print()

# Indptr interpretation: row i spans indices[indptr[i]:indptr[i+1]]
print("Row 0:")
start, end = csr.indptr[0], csr.indptr[1]
print(f"  Indices: {csr.indices[start:end]}")
print(f"  Data: {csr.data[start:end]}")
\`\`\`

### 3. CSC (Compressed Sparse Column)

Like CSR but column-oriented.

**Pros**: Efficient column slicing
**Cons**: Row slicing slow

\`\`\`python
print("\\n=== CSC Format ===")

csc = sparse.csc_matrix (dense)

print("CSC representation:")
print(f"Data: {csc.data}")
print(f"Indices: {csc.indices}")  # Row indices
print(f"Indptr: {csc.indptr}")    # Column pointers
\`\`\`

## Sparse Matrix Operations

### Matrix-Vector Product

\`\`\`python
print("\\n=== Sparse Matrix-Vector Product ===")

A_sparse = sparse.random(1000, 1000, density=0.01, format='csr')
x = np.random.randn(1000)

# Sparse product
y_sparse = A_sparse @ x

# Compare with dense (memory-intensive!)
# A_dense = A_sparse.toarray()
# y_dense = A_dense @ x

print(f"Matrix shape: {A_sparse.shape}")
print(f"Non-zeros: {A_sparse.nnz} / {A_sparse.shape[0] * A_sparse.shape[1]}")
print(f"Sparsity: {(1 - A_sparse.nnz / (A_sparse.shape[0] * A_sparse.shape[1])):.2%}")
print()
print("Sparse matvec is O(nnz), dense is O(n²)")
print(f"Speedup: ~{A_sparse.shape[0]**2 / A_sparse.nnz:.0f}x")
\`\`\`

### Matrix-Matrix Product

\`\`\`python
print("\\n=== Sparse Matrix-Matrix Product ===")

A = sparse.random(100, 100, density=0.1, format='csr')
B = sparse.random(100, 100, density=0.1, format='csr')

# Sparse product
C_sparse = A @ B

print(f"A non-zeros: {A.nnz}")
print(f"B non-zeros: {B.nnz}")
print(f"C non-zeros: {C_sparse.nnz}")
print()
print("Note: Product of sparse matrices may be denser")
\`\`\`

### Element-wise Operations

\`\`\`python
print("\\n=== Element-wise Operations ===")

A = sparse.random(5, 5, density=0.3, format='csr')
B = sparse.random(5, 5, density=0.3, format='csr')

# Element-wise multiplication (preserves sparsity)
C_mul = A.multiply(B)

# Addition (can increase non-zeros)
C_add = A + B

print(f"A nnz: {A.nnz}")
print(f"B nnz: {B.nnz}")
print(f"A * B (element-wise) nnz: {C_mul.nnz}")  # ≤ min(A.nnz, B.nnz)
print(f"A + B nnz: {C_add.nnz}")                  # ≤ A.nnz + B.nnz
\`\`\`

## Applications in Machine Learning

### 1. Text Data (TF-IDF)

\`\`\`python
print("\\n=== Application: TF-IDF (Text Data) ===")

from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
docs = [
    "machine learning is awesome",
    "deep learning is powerful",
    "linear algebra is fundamental",
    "machine learning uses linear algebra"
]

# Create TF-IDF matrix (sparse)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform (docs)

print(f"Shape: {X_tfidf.shape}")  # (4 docs, vocab_size)
print(f"Type: {type(X_tfidf)}")
print(f"Non-zeros: {X_tfidf.nnz} / {X_tfidf.shape[0] * X_tfidf.shape[1]}")
print(f"Sparsity: {(1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])):.1%}")
print()

print("Vocabulary:")
print(list (vectorizer.vocabulary_.keys()))
print()

# Cosine similarity (using sparse operations)
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(X_tfidf)
print("Document similarities:")
print(sim.round(2))
\`\`\`

### 2. Recommender Systems

\`\`\`python
print("\\n=== Application: Recommender Systems ===")

# User-item matrix (rows=users, cols=items)
# Most entries are 0 (users rate few items)
n_users, n_items = 1000, 5000
density = 0.001  # Each user rates 0.1% of items

ratings = sparse.random (n_users, n_items, density=density, format='csr')
ratings.data = np.random.randint(1, 6, size=ratings.nnz)  # Ratings 1-5

print(f"Users: {n_users}")
print(f"Items: {n_items}")
print(f"Total possible ratings: {n_users * n_items:,}")
print(f"Actual ratings: {ratings.nnz:,}")
print(f"Sparsity: {(1 - ratings.nnz / (n_users * n_items)):.2%}")
print()

# Collaborative filtering: item-item similarity
# Compute item vectors (each column of ratings)
item_similarity = cosine_similarity (ratings.T, dense_output=False)

print(f"Item similarity matrix shape: {item_similarity.shape}")
print(f"Item similarity nnz: {item_similarity.nnz}")
\`\`\`

### 3. Graph Data (Adjacency Matrix)

\`\`\`python
print("\\n=== Application: Graph Analysis ===")

# Create random graph (sparse adjacency matrix)
n_nodes = 100
n_edges = 300

# Random edges
edges = np.random.randint(0, n_nodes, size=(2, n_edges))
row, col = edges[0], edges[1]
data = np.ones (n_edges)

# Adjacency matrix
A_graph = sparse.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
A_graph = A_graph.tocsr()

print(f"Nodes: {n_nodes}")
print(f"Edges: {A_graph.nnz}")
print(f"Sparsity: {(1 - A_graph.nnz / (n_nodes**2)):.2%}")
print()

# Compute degree (sum of each row)
degrees = np.array(A_graph.sum (axis=1)).flatten()

print(f"Average degree: {degrees.mean():.2f}")
print(f"Max degree: {degrees.max()}")
\`\`\`

### 4. Sparse Neural Networks

\`\`\`python
print("\\n=== Application: Sparse Neural Networks ===")

# Sparse weight matrix (many zeros for regularization/compression)
input_dim = 1000
output_dim = 500
sparsity = 0.9  # 90% zeros

# Create sparse weights
W_sparse = sparse.random (input_dim, output_dim, density=1-sparsity, format='csr')

print(f"Weight matrix: {input_dim} → {output_dim}")
print(f"Dense parameters: {input_dim * output_dim:,}")
print(f"Sparse parameters: {W_sparse.nnz:,}")
print(f"Compression: {(input_dim * output_dim) / W_sparse.nnz:.1f}x")
print()

# Forward pass
x = np.random.randn (input_dim)
y = W_sparse.T @ x  # Sparse matvec

print(f"Input: {x.shape}")
print(f"Output: {y.shape}")
print("Sparse forward pass much faster and less memory!")
\`\`\`

## Iterative Solvers

For large sparse systems **Ax** = **b**, iterative methods are essential.

\`\`\`python
print("\\n=== Iterative Solvers for Sparse Systems ===")

from scipy.sparse.linalg import cg, spsolve

# Create sparse SPD system
n = 1000
A_sparse = sparse.random (n, n, density=0.01, format='csr')
A_sparse = A_sparse @ A_sparse.T + sparse.eye (n) * 0.1  # Make SPD

b = np.random.randn (n)

print(f"System size: {n}×{n}")
print(f"Non-zeros: {A_sparse.nnz}")
print()

# Conjugate Gradient (iterative)
x_cg, info = cg(A_sparse, b, tol=1e-6)

if info == 0:
    print("Conjugate Gradient converged")
    print(f"Solution norm: {np.linalg.norm (x_cg):.4f}")
    print(f"Residual: {np.linalg.norm(A_sparse @ x_cg - b):.2e}")

# Direct solve (LU factorization)
x_direct = spsolve(A_sparse, b)

print(f"\\nDirect solve solution norm: {np.linalg.norm (x_direct):.4f}")
print("Iterative solvers scale better for very large systems!")
\`\`\`

## Summary

**Sparse Matrices**: Store only non-zero elements
- **COO**: (row, col, value) triplets
- **CSR**: Compressed rows (efficient row ops)
- **CSC**: Compressed columns (efficient column ops)

**Why Sparse Matters**:
- **Memory**: O(nnz) vs O(n²) for dense
- **Speed**: Operations O(nnz) vs O(n²)
- **Scale**: Can handle millions of dimensions

**ML Applications**:
- **Text**: TF-IDF, word embeddings (most words absent in most docs)
- **Recommender systems**: User-item matrices (users interact with <0.1% of items)
- **Graphs**: Adjacency matrices (social networks, molecules)
- **Sparse neural networks**: Pruning, lottery ticket hypothesis

**Operations**:
- **Matvec**: O(nnz) vs O(n²)
- **Matmul**: Result may be denser
- **Iterative solvers**: Conjugate Gradient, GMRES for large systems

**Best Practices**:
- Use CSR for row operations, CSC for column
- Convert to sparse early (before operations)
- Leverage scipy.sparse for efficient implementations
- For very large sparse systems, use iterative solvers
- Be aware when operations densify (e.g., A⁻¹ usually dense even if A sparse)

Sparse linear algebra enables working with massive datasets that would be infeasible as dense matrices!
`,
};

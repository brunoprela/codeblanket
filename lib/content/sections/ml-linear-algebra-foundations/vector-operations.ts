/**
 * Vector Operations Section
 */

export const vectoroperationsSection = {
  id: 'vector-operations',
  title: 'Vector Operations',
  content: `
# Vector Operations

## Introduction

Beyond basic addition and scaling, vectors support several advanced operations that are fundamental to machine learning. The most important are the **dot product**, **cross product**, and various **norms**. These operations allow us to measure angles, distances, similarities, and projections—all crucial for ML algorithms.

## Dot Product (Inner Product)

The **dot product** (also called inner product or scalar product) is one of the most important operations in linear algebra and machine learning.

### Definition

For vectors **u** = [u₁, u₂, ..., uₙ] and **v** = [v₁, v₂, ..., vₙ]:

**u** · **v** = u₁v₁ + u₂v₂ + ... + uₙvₙ = Σᵢ uᵢvᵢ

The result is a **scalar** (single number), not a vector.

### Geometric Interpretation

**u** · **v** = ||**u**|| ||**v**|| cos(θ)

Where θ is the angle between the vectors.

**Implications**:
- If θ = 0° (parallel): **u** · **v** = ||**u**|| ||**v**|| (maximum positive)
- If θ = 90° (perpendicular/orthogonal): **u** · **v** = 0
- If θ = 180° (opposite directions): **u** · **v** = -||**u**|| ||**v**|| (maximum negative)

### Properties

1. **Commutative**: **u** · **v** = **v** · **u**
2. **Distributive**: **u** · (**v** + **w**) = **u** · **v** + **u** · **w**
3. **Associative with scalars**: (c**u**) · **v** = c(**u** · **v**)
4. **Positive definite**: **v** · **v** ≥ 0, equals 0 only if **v** = **0**

### Applications in ML

1. **Similarity measurement**: Large positive dot product → vectors point in similar directions
2. **Neural network computations**: Each neuron computes a dot product of inputs and weights
3. **Projections**: Project one vector onto another
4. **Cosine similarity**: cos(θ) = (**u** · **v**) / (||**u**|| ||**v**||)

## Vector Norms

A **norm** is a function that assigns a length or size to a vector. Different norms measure size differently.

### L2 Norm (Euclidean Norm)

The most common norm, representing geometric length:

||**v**||₂ = √(v₁² + v₂² + ... + vₙ²) = √(**v** · **v**)

This is what we typically call "magnitude."

### L1 Norm (Manhattan/Taxicab Norm)

Sum of absolute values:

||**v**||₁ = |v₁| + |v₂| + ... + |vₙ|

**Named "Manhattan"** because it measures distance as if you can only travel along grid lines (like city blocks).

**Use in ML**: 
- L1 regularization (Lasso) for feature selection
- More robust to outliers than L2

### L∞ Norm (Maximum/Chebyshev Norm)

The maximum absolute component:

||**v**||∞ = max(|v₁|, |v₂|, ..., |vₙ|)

### P-Norm (General Case)

||**v**||ₚ = (|v₁|ᵖ + |v₂|ᵖ + ... + |vₙ|ᵖ)^(1/p)

- p = 1: L1 norm
- p = 2: L2 norm
- p → ∞: L∞ norm

## Distance Metrics

Distance measures how far apart two vectors are. Different metrics are appropriate for different applications.

### Euclidean Distance (L2 Distance)

d(**u**, **v**) = ||**u** - **v**||₂ = √(Σᵢ (uᵢ - vᵢ)²)

Most intuitive, corresponds to straight-line distance.

### Manhattan Distance (L1 Distance)

d(**u**, **v**) = ||**u** - **v**||₁ = Σᵢ |uᵢ - vᵢ|

Sum of absolute differences.

### Cosine Distance

Measures angle between vectors (direction), ignoring magnitude:

cosine_similarity(**u**, **v**) = (**u** · **v**) / (||**u**|| ||**v**||)

cosine_distance = 1 - cosine_similarity

**Range**: 
- Similarity: -1 (opposite) to +1 (identical direction)
- 0 means orthogonal (no similarity)

**Use in ML**: Text similarity, recommendation systems, where magnitude doesn't matter.

## Cross Product (3D Only)

The **cross product** is only defined for 3D vectors and produces a vector perpendicular to both inputs.

### Definition

For **u** = [u₁, u₂, u₃] and **v** = [v₁, v₂, v₃]:

**u** × **v** = [u₂v₃ - u₃v₂, u₃v₁ - u₁v₃, u₁v₂ - u₂v₁]

### Properties

1. **Anti-commutative**: **u** × **v** = -(**v** × **u**)
2. **Not associative**: (**u** × **v**) × **w** ≠ **u** × (**v** × **w**)
3. **Magnitude**: ||**u** × **v**|| = ||**u**|| ||**v**|| sin(θ)
4. **Direction**: Perpendicular to both **u** and **v** (right-hand rule)

### Applications

- Computer graphics (computing normals to surfaces)
- Physics (torque, angular momentum)
- Less common in ML, but useful for 3D data

## Python Implementation

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Create sample vectors
u = np.array([3, 4])
v = np.array([1, 2])
w = np.array([2, -1])

print("=== Dot Product ===")
print(f"u = {u}")
print(f"v = {v}")
print()

# Dot product - multiple ways
dot_product_1 = np.dot(u, v)
dot_product_2 = u @ v  # Matrix multiplication operator
dot_product_3 = np.sum(u * v)  # Element-wise multiply then sum

print(f"u · v = {dot_product_1}")
print(f"u @ v = {dot_product_2}")
print(f"Sum(u * v) = {dot_product_3}")
print()

# Verify they're all the same
assert dot_product_1 == dot_product_2 == dot_product_3
\`\`\`

\`\`\`python
# Angle between vectors using dot product
def angle_between_vectors(u, v, degrees=True):
    """Calculate angle between two vectors"""
    cos_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    # Clip to handle numerical errors
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_rad = np.arccos(cos_angle)
    
    if degrees:
        return np.degrees(angle_rad)
    return angle_rad

angle = angle_between_vectors(u, v)
print(f"Angle between u and v: {angle:.2f}°")

# Orthogonal vectors have dot product = 0
u_orth = np.array([1, 0])
v_orth = np.array([0, 1])
print(f"\\nOrthogonal vectors: {u_orth} and {v_orth}")
print(f"Dot product: {np.dot(u_orth, v_orth)}")
print(f"Angle: {angle_between_vectors(u_orth, v_orth):.2f}°")
\`\`\`

\`\`\`python
print("\\n=== Vector Norms ===")

v = np.array([3, 4, 0])
print(f"v = {v}")
print()

# L2 norm (Euclidean)
l2_norm = np.linalg.norm(v)  # Default is L2
l2_norm_manual = np.sqrt(np.sum(v**2))
print(f"L2 norm: {l2_norm:.4f}")
print(f"L2 norm (manual): {l2_norm_manual:.4f}")
print()

# L1 norm (Manhattan)
l1_norm = np.linalg.norm(v, ord=1)
l1_norm_manual = np.sum(np.abs(v))
print(f"L1 norm: {l1_norm:.4f}")
print(f"L1 norm (manual): {l1_norm_manual:.4f}")
print()

# L-infinity norm (Maximum)
linf_norm = np.linalg.norm(v, ord=np.inf)
linf_norm_manual = np.max(np.abs(v))
print(f"L-inf norm: {linf_norm:.4f}")
print(f"L-inf norm (manual): {linf_norm_manual:.4f}")
print()

# P-norm for different values of p
for p in [0.5, 1, 2, 3, 10]:
    p_norm = np.linalg.norm(v, ord=p)
    print(f"L{p} norm: {p_norm:.4f}")
\`\`\`

\`\`\`python
# Visualizing different norms
def plot_unit_balls():
    """Plot unit circles for different norms"""
    theta = np.linspace(0, 2*np.pi, 1000)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # L1 norm (diamond)
    ax = axes[0]
    x_l1 = np.sign(np.cos(theta)) * np.abs(np.cos(theta))
    y_l1 = np.sign(np.sin(theta)) * (1 - np.abs(x_l1))
    ax.plot(x_l1, y_l1, 'b-', linewidth=2)
    ax.set_title('L1 Norm (||v||₁ = 1)', fontsize=14)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # L2 norm (circle)
    ax = axes[1]
    x_l2 = np.cos(theta)
    y_l2 = np.sin(theta)
    ax.plot(x_l2, y_l2, 'r-', linewidth=2)
    ax.set_title('L2 Norm (||v||₂ = 1)', fontsize=14)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # L-infinity norm (square)
    ax = axes[2]
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    ax.plot(square[:, 0], square[:, 1], 'g-', linewidth=2)
    ax.set_title('L∞ Norm (||v||∞ = 1)', fontsize=14)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.show()

plot_unit_balls()
\`\`\`

\`\`\`python
print("=== Distance Metrics ===")

point1 = np.array([1, 2])
point2 = np.array([4, 6])

print(f"Point 1: {point1}")
print(f"Point 2: {point2}")
print()

# Euclidean distance
euclidean_dist = np.linalg.norm(point1 - point2)
print(f"Euclidean distance: {euclidean_dist:.4f}")

# Manhattan distance
manhattan_dist = np.linalg.norm(point1 - point2, ord=1)
print(f"Manhattan distance: {manhattan_dist:.4f}")

# Cosine similarity and distance
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def cosine_distance(u, v):
    return 1 - cosine_similarity(u, v)

cos_sim = cosine_similarity(point1, point2)
cos_dist = cosine_distance(point1, point2)

print(f"Cosine similarity: {cos_sim:.4f}")
print(f"Cosine distance: {cos_dist:.4f}")
\`\`\`

\`\`\`python
print("\\n=== Cross Product (3D) ===")

u_3d = np.array([1, 0, 0])
v_3d = np.array([0, 1, 0])

# Cross product
cross = np.cross(u_3d, v_3d)
print(f"u = {u_3d}")
print(f"v = {v_3d}")
print(f"u × v = {cross}")
print()

# Verify it's perpendicular to both
print(f"(u × v) · u = {np.dot(cross, u_3d)}")
print(f"(u × v) · v = {np.dot(cross, v_3d)}")
print()

# Magnitude of cross product
cross_magnitude = np.linalg.norm(cross)
print(f"||u × v|| = {cross_magnitude:.4f}")

# Compare with ||u|| ||v|| sin(θ)
angle_rad = angle_between_vectors(u_3d, v_3d, degrees=False)
expected_magnitude = np.linalg.norm(u_3d) * np.linalg.norm(v_3d) * np.sin(angle_rad)
print(f"||u|| ||v|| sin(θ) = {expected_magnitude:.4f}")
\`\`\`

## Application: Cosine Similarity in Text Analysis

\`\`\`python
# Example: Document similarity using cosine similarity
# Documents represented as term frequency vectors

# Vocabulary: [python, java, machine, learning, web, development]
doc1 = np.array([3, 0, 5, 5, 0, 1])  # ML-focused document
doc2 = np.array([4, 0, 4, 6, 0, 0])  # Another ML document
doc3 = np.array([1, 4, 0, 0, 5, 5])  # Web development document

print("=== Document Similarity ===")
print(f"Doc1 (ML): {doc1}")
print(f"Doc2 (ML): {doc2}")
print(f"Doc3 (Web): {doc3}")
print()

# Compute pairwise cosine similarities
sim_12 = cosine_similarity(doc1, doc2)
sim_13 = cosine_similarity(doc1, doc3)
sim_23 = cosine_similarity(doc2, doc3)

print(f"Similarity(Doc1, Doc2): {sim_12:.4f}")
print(f"Similarity(Doc1, Doc3): {sim_13:.4f}")
print(f"Similarity(Doc2, Doc3): {sim_23:.4f}")
print()

print("Interpretation:")
print(f"Docs 1 and 2 are very similar (both ML): {sim_12:.4f}")
print(f"Docs 1 and 3 are dissimilar (ML vs Web): {sim_13:.4f}")
\`\`\`

## Application: K-Nearest Neighbors Distance

\`\`\`python
# Example: Finding nearest neighbors
from scipy.spatial.distance import cdist

# Training data points (2D for visualization)
X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 1],
    [6, 5],
    [7, 7],
    [8, 6]
])

# New point to classify
x_new = np.array([[5, 4]])

# Compute distances using different metrics
euclidean_distances = cdist(x_new, X_train, metric='euclidean')[0]
manhattan_distances = cdist(x_new, X_train, metric='cityblock')[0]
cosine_distances = cdist(x_new, X_train, metric='cosine')[0]

print("=== K-NN Distance Computation ===")
print(f"New point: {x_new[0]}")
print(f"\\nTraining points:\\n{X_train}")
print()

print("Euclidean distances:", euclidean_distances.round(2))
print("Manhattan distances:", manhattan_distances.round(2))
print("Cosine distances:", cosine_distances.round(2))
print()

# Find k=3 nearest neighbors (Euclidean)
k = 3
nearest_indices = np.argsort(euclidean_distances)[:k]
print(f"\\n{k} Nearest neighbors (indices): {nearest_indices}")
print(f"Nearest points:\\n{X_train[nearest_indices]}")
\`\`\`

\`\`\`python
# Visualize the K-NN example
plt.figure(figsize=(10, 8))

# Plot training points
plt.scatter(X_train[:, 0], X_train[:, 1], 
           c='blue', s=100, marker='o', label='Training points', alpha=0.6)

# Plot new point
plt.scatter(x_new[:, 0], x_new[:, 1], 
           c='red', s=200, marker='*', label='New point', zorder=5)

# Draw circles showing distances to nearest neighbors
for idx in nearest_indices:
    distance = euclidean_distances[idx]
    circle = plt.Circle(x_new[0], distance, fill=False, 
                       color='red', linestyle='--', alpha=0.3)
    plt.gca().add_patch(circle)
    
    # Draw line to nearest neighbor
    plt.plot([x_new[0, 0], X_train[idx, 0]], 
            [x_new[0, 1], X_train[idx, 1]], 
            'r--', alpha=0.5)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'K-NN: Finding {k} Nearest Neighbors')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
\`\`\`

## Application: Neural Network Forward Pass

\`\`\`python
# Simple neuron: computes weighted sum using dot product
def neuron_output(inputs, weights, bias):
    """
    Compute output of a single neuron
    output = activation(w · x + b)
    """
    # Dot product of weights and inputs
    z = np.dot(weights, inputs) + bias
    # ReLU activation
    output = np.maximum(0, z)
    return output

# Example
inputs = np.array([1.0, 2.0, 3.0])
weights = np.array([0.5, -0.3, 0.8])
bias = 0.1

output = neuron_output(inputs, weights, bias)
print("=== Neural Network Neuron ===")
print(f"Inputs: {inputs}")
print(f"Weights: {weights}")
print(f"Bias: {bias}")
print(f"Weighted sum: {np.dot(weights, inputs) + bias:.4f}")
print(f"Output (after ReLU): {output:.4f}")
\`\`\`

## Best Practices and Common Pitfalls

### 1. Numerical Stability

\`\`\`python
# Bad: Can have division by zero
def cosine_similarity_unsafe(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Good: Handle edge cases
def cosine_similarity_safe(u, v, epsilon=1e-8):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    if norm_u < epsilon or norm_v < epsilon:
        return 0.0  # Zero vectors
    
    return np.dot(u, v) / (norm_u * norm_v)

# Test with zero vector
zero_vec = np.array([0, 0, 0])
normal_vec = np.array([1, 2, 3])

print("Safe cosine similarity with zero vector:", 
      cosine_similarity_safe(zero_vec, normal_vec))
\`\`\`

### 2. Choose the Right Distance Metric

- **Euclidean**: When magnitude matters, continuous features
- **Manhattan**: More robust to outliers, grid-like spaces
- **Cosine**: When direction matters more than magnitude (text, embeddings)

### 3. Normalize Before Using Cosine Similarity

If using cosine similarity, sometimes normalize first for consistency:

\`\`\`python
# For very large or very small vectors, normalize first
v1_large = np.array([1000, 2000, 3000])
v2_large = np.array([1100, 1900, 3100])

# Normalize
v1_norm = v1_large / np.linalg.norm(v1_large)
v2_norm = v2_large / np.linalg.norm(v2_large)

# Now dot product equals cosine similarity
cos_sim_direct = cosine_similarity_safe(v1_large, v2_large)
cos_sim_normalized = np.dot(v1_norm, v2_norm)

print(f"Cosine similarity (direct): {cos_sim_direct:.6f}")
print(f"Dot product (normalized): {cos_sim_normalized:.6f}")
\`\`\`

## Summary

Vector operations form the computational backbone of machine learning:

1. **Dot Product**: Measures similarity, computes weighted sums (neurons), projects vectors
2. **Norms**: Measure vector magnitude (L1, L2, L∞) - used in regularization and distance
3. **Distance Metrics**: Euclidean, Manhattan, Cosine - fundamental to clustering, classification, retrieval
4. **Cross Product**: Perpendicular vectors (less common in ML, more in graphics)

**Key Applications in ML**:
- K-NN uses distance metrics to find similar data points
- Neural networks use dot products in every layer
- Text similarity uses cosine similarity
- Regularization uses L1 or L2 norms
- Gradient descent uses norm to measure gradient magnitude

**Performance Tip**: Always use vectorized NumPy operations. A single \`np.dot()\` is vastly faster than a Python loop.

Master these operations, and you understand how data flows through ML algorithms!
`,
};

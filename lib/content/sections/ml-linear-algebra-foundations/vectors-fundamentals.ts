/**
 * Vectors Fundamentals Section
 */

export const vectorsfundamentalsSection = {
  id: 'vectors-fundamentals',
  title: 'Vectors Fundamentals',
  content: `
# Vectors Fundamentals

## Introduction

Vectors are the foundational building blocks of linear algebra and machine learning. In ML, every data point, every feature, and every model parameter is represented as a vector. Understanding vectors deeply is essential for grasping how machine learning algorithms work under the hood.

A **vector** is an ordered collection of numbers. Geometrically, it represents a point in space or a directed arrow from the origin to that point.

### Why Vectors Matter in Machine Learning

- **Data Representation**: Each row in your dataset is a feature vector
- **Model Parameters**: Weights and biases are stored as vectors
- **Embeddings**: Words, images, and users are represented as dense vectors
- **Optimization**: Gradients are vectors pointing in the direction of steepest ascent
- **Predictions**: Model outputs are often vectors (e.g., probability distributions)

## Vector Notation and Representation

### Mathematical Notation

A vector is typically denoted with a lowercase bold letter or an arrow:
- **v** or **v⃗**

For a vector in n-dimensional space:

**v** = [v₁, v₂, v₃, ..., vₙ]

or in column form:

**v** = ⎡v₁⎤
      ⎢v₂⎥
      ⎢v₃⎥
      ⎢..⎥
      ⎣vₙ⎦

### Dimensions

- A vector in 2D space (ℝ²): **v** = [v₁, v₂]
- A vector in 3D space (ℝ³): **v** = [v₁, v₂, v₃]
- A vector in n-dimensional space (ℝⁿ): **v** = [v₁, v₂, ..., vₙ]

In machine learning, we often work with high-dimensional vectors (hundreds or thousands of dimensions).

### Row vs Column Vectors

- **Row vector**: [1, 2, 3] - written horizontally
- **Column vector**: Written vertically (more common in linear algebra)

In practice, NumPy arrays can be both, and context determines interpretation.

## Geometric Interpretation

### 2D Example

Consider **v** = [3, 2]:
- This represents a point at coordinates (3, 2)
- Or an arrow from origin (0, 0) to point (3, 2)
- Move 3 units right, 2 units up

### 3D Example

**v** = [2, 3, 1]:
- Point in 3D space at (2, 3, 1)
- Arrow from origin to this point
- x=2, y=3, z=1

### Higher Dimensions

While we can't visualize vectors in 100 dimensions, the mathematical operations remain the same. Each dimension represents a different feature or attribute.

**Example in ML**: A house with features [square_feet, bedrooms, bathrooms, age] is a 4D vector.

## Vector Addition and Scalar Multiplication

### Vector Addition

Adding two vectors means adding corresponding components:

**u** + **v** = [u₁ + v₁, u₂ + v₂, ..., uₙ + vₙ]

**Geometric interpretation**: Place the tail of **v** at the head of **u**. The result is the arrow from the tail of **u** to the head of **v**.

**Properties**:
- Commutative: **u** + **v** = **v** + **u**
- Associative: (**u** + **v**) + **w** = **u** + (**v** + **w**)
- Identity: **v** + **0** = **v**
- Inverse: **v** + (-**v**) = **0**

### Scalar Multiplication

Multiplying a vector by a scalar (number) scales each component:

c · **v** = [c·v₁, c·v₂, ..., c·vₙ]

**Geometric interpretation**: 
- If c > 1: stretches the vector
- If 0 < c < 1: shrinks the vector
- If c < 0: reverses direction and scales

**Properties**:
- Distributive: c(**u** + **v**) = c**u** + c**v**
- Distributive: (c + d)**v** = c**v** + d**v**
- Associative: (cd)**v** = c(d**v**)
- Identity: 1·**v** = **v**

## Unit Vectors and Normalization

### Unit Vector

A **unit vector** has magnitude (length) 1. It indicates direction without magnitude.

The magnitude of **v** is: ||**v**|| = √(v₁² + v₂² + ... + vₙ²)

### Normalization

To normalize a vector (convert it to a unit vector), divide by its magnitude:

**v̂** = **v** / ||**v**||

**Why normalize?**
- Focus on direction, ignore magnitude
- Many ML algorithms require normalized features
- Used in cosine similarity calculations
- Speeds up optimization in neural networks

### Standard Basis Vectors

In ℝ³, the standard basis vectors are:
- **e₁** = [1, 0, 0] - points along x-axis
- **e₂** = [0, 1, 0] - points along y-axis  
- **e₃** = [0, 0, 1] - points along z-axis

Any vector can be written as a linear combination of basis vectors.

## Python Implementation

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Creating vectors
v1 = np.array([3, 2])
v2 = np.array([1, 4])
v3 = np.array([2, 3, 1])  # 3D vector

print("2D Vector v1:", v1)
print("2D Vector v2:", v2)
print("3D Vector v3:", v3)
print()

# Vector dimensions
print("Dimension of v1:", v1.shape)
print("Number of components in v1:", len(v1))
print()

# Vector addition
v_sum = v1 + v2
print("v1 + v2 =", v_sum)
print()

# Scalar multiplication
v_scaled = 2 * v1
print("2 * v1 =", v_scaled)
print()

# Vector magnitude (L2 norm)
magnitude_v1 = np.linalg.norm(v1)
print(f"Magnitude of v1: {magnitude_v1:.4f}")
print()

# Manual calculation of magnitude
magnitude_manual = np.sqrt(np.sum(v1**2))
print(f"Magnitude (manual): {magnitude_manual:.4f}")
print()

# Normalization
v1_normalized = v1 / np.linalg.norm(v1)
print("Normalized v1:", v1_normalized)
print("Magnitude of normalized v1:", np.linalg.norm(v1_normalized))
print()

# Creating zero vector
zero_vector = np.zeros(5)
print("Zero vector (5D):", zero_vector)
print()

# Creating ones vector
ones_vector = np.ones(3)
print("Ones vector (3D):", ones_vector)
print()

# Standard basis vectors
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])
print("Standard basis vectors:")
print("e1:", e1)
print("e2:", e2)
print("e3:", e3)
\`\`\`

### Visualizing Vectors

\`\`\`python
# Visualizing 2D vectors
def plot_2d_vectors(*vectors, labels=None, colors=None):
    """Plot multiple 2D vectors"""
    plt.figure(figsize=(8, 8))
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, v in enumerate(vectors):
        label = labels[i] if labels else f'v{i+1}'
        color = colors[i % len(colors)]
        plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', 
                   scale=1, color=color, width=0.006, label=label)
        # Add text label at vector tip
        plt.text(v[0]*1.1, v[1]*1.1, label, fontsize=12, color=color)
    
    plt.xlim(-1, 6)
    plt.ylim(-1, 6)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('2D Vector Visualization')
    plt.axis('equal')
    plt.show()

# Plot vectors
v1 = np.array([3, 2])
v2 = np.array([1, 4])
v_sum = v1 + v2

plot_2d_vectors(v1, v2, v_sum, 
                labels=['v1', 'v2', 'v1+v2'],
                colors=['blue', 'red', 'green'])
\`\`\`

\`\`\`python
# Visualizing 3D vectors
def plot_3d_vectors(*vectors, labels=None):
    """Plot multiple 3D vectors"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, v in enumerate(vectors):
        label = labels[i] if labels else f'v{i+1}'
        color = colors[i % len(colors)]
        ax.quiver(0, 0, 0, v[0], v[1], v[2], 
                 color=color, arrow_length_ratio=0.1, linewidth=2, label=label)
    
    # Set labels and limits
    max_val = max([np.max(np.abs(v)) for v in vectors]) * 1.2
    ax.set_xlim([-1, max_val])
    ax.set_ylim([-1, max_val])
    ax.set_zlim([-1, max_val])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D Vector Visualization')
    plt.show()

# Plot 3D vectors
v3d_1 = np.array([2, 3, 1])
v3d_2 = np.array([1, 1, 4])
v3d_sum = v3d_1 + v3d_2

plot_3d_vectors(v3d_1, v3d_2, v3d_sum,
                labels=['v1', 'v2', 'v1+v2'])
\`\`\`

## Vectors in Machine Learning

### Example 1: Feature Vectors

\`\`\`python
# Representing data points as vectors
# House features: [square_feet, bedrooms, bathrooms, age_years]

house1 = np.array([2000, 3, 2, 10])
house2 = np.array([1500, 2, 1, 15])
house3 = np.array([2500, 4, 3, 5])

print("House 1:", house1)
print("House 2:", house2)
print("House 3:", house3)
print()

# Creating a dataset (matrix) from multiple vectors
dataset = np.array([house1, house2, house3])
print("Dataset shape:", dataset.shape)  # 3 samples, 4 features
print("Dataset:")
print(dataset)
\`\`\`

### Example 2: Normalizing Features

\`\`\`python
# Feature scaling is crucial for many ML algorithms
def normalize_features(X):
    """
    Normalize each feature to unit length
    This is L2 normalization (different from standardization)
    """
    # Normalize each row (sample) to unit length
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / norms
    return X_normalized

# Normalize our house data
dataset_normalized = normalize_features(dataset)
print("Normalized dataset:")
print(dataset_normalized)
print()

# Verify each row has unit length
print("Row norms:", np.linalg.norm(dataset_normalized, axis=1))
\`\`\`

### Example 3: Word Embeddings (Preview)

\`\`\`python
# In NLP, words are represented as dense vectors
# This is a simplified example - real embeddings have 100-300 dimensions

# Hypothetical word embeddings (3D for visualization)
word_vectors = {
    'king': np.array([0.5, 0.3, 0.8]),
    'queen': np.array([0.5, 0.7, 0.8]),
    'man': np.array([0.2, 0.3, 0.1]),
    'woman': np.array([0.2, 0.7, 0.1])
}

# Famous example: king - man + woman ≈ queen
result = word_vectors['king'] - word_vectors['man'] + word_vectors['woman']
print("king - man + woman =", result)
print("queen =", word_vectors['queen'])
print("Difference:", np.linalg.norm(result - word_vectors['queen']))
\`\`\`

## Common Operations and Properties

\`\`\`python
# Vector operations showcase
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print("=== Vector Operations ===")
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print()

# Addition
print(f"v1 + v2 = {v1 + v2}")

# Subtraction
print(f"v1 - v2 = {v1 - v2}")

# Scalar multiplication
print(f"3 * v1 = {3 * v1}")

# Element-wise multiplication (Hadamard product)
print(f"v1 ⊙ v2 (element-wise) = {v1 * v2}")

# Negative vector
print(f"-v1 = {-v1}")

# Linear combination: a*v1 + b*v2
a, b = 2, 3
linear_combo = a * v1 + b * v2
print(f"{a}*v1 + {b}*v2 = {linear_combo}")
\`\`\`

## Best Practices

1. **Use NumPy for vector operations**: Much faster than Python lists
2. **Be aware of dimensions**: Shape mismatches cause errors
3. **Normalize when appropriate**: Many algorithms benefit from normalized features
4. **Vectorize operations**: Avoid loops when working with vectors
5. **Use meaningful variable names**: v1, v2 are fine for examples, but use descriptive names in production

\`\`\`python
# Good practice: Vectorized operation
vectors = np.array([[1, 2], [3, 4], [5, 6]])
norms = np.linalg.norm(vectors, axis=1)  # Fast, vectorized

# Bad practice: Loop (slower)
norms_slow = []
for v in vectors:
    norms_slow.append(np.linalg.norm(v))
\`\`\`

## Connection to Machine Learning

### Gradient Vectors

In optimization, gradients are vectors indicating direction of steepest increase:

\`\`\`python
# Example: Gradient descent update
# θ_new = θ_old - learning_rate * gradient

theta = np.array([0.5, 1.2, -0.3])  # Model parameters
gradient = np.array([0.1, -0.05, 0.2])  # Computed gradient
learning_rate = 0.01

theta_new = theta - learning_rate * gradient
print("Old parameters:", theta)
print("Gradient:", gradient)
print("New parameters:", theta_new)
\`\`\`

### Distance Between Data Points

\`\`\`python
# Euclidean distance between two points (L2 norm of difference)
point1 = np.array([1, 2, 3])
point2 = np.array([4, 6, 8])

distance = np.linalg.norm(point1 - point2)
print(f"Distance between points: {distance:.4f}")

# This is used in k-NN, k-means, and many other algorithms
\`\`\`

## Summary

Vectors are the fundamental data structure in machine learning:
- They represent data points, features, model parameters, and gradients
- Vector operations (addition, scaling, normalization) are building blocks for complex algorithms
- NumPy provides efficient implementations of all vector operations
- Understanding vectors geometrically helps build intuition for higher-dimensional spaces
- Normalization and proper scaling are critical preprocessing steps

**Key Takeaway**: Master vectors, and you've mastered the language that all ML algorithms speak.
`,
};

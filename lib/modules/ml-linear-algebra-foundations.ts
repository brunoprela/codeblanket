import { Module } from '../types';

export const mlLinearAlgebraFoundationsModule: Module = {
  id: 'ml-linear-algebra-foundations',
  title: 'Linear Algebra Foundations',
  description:
    'Master vectors, matrices, tensors, and linear transformations - the language of machine learning and deep learning',
  icon: '🔷',
  sections: [
    {
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
      multipleChoice: [
        {
          id: 'vec-fund-q1',
          question: 'What is the magnitude of the vector v = [3, 4]?',
          options: ['7', '5', '12', '3.5'],
          correctAnswer: 1,
          explanation:
            'The magnitude is ||v|| = √(3² + 4²) = √(9 + 16) = √25 = 5. This is the Pythagorean theorem in 2D.',
        },
        {
          id: 'vec-fund-q2',
          question:
            'After normalizing a vector, what is guaranteed about its magnitude?',
          options: [
            'It equals zero',
            'It equals one',
            'It equals the original magnitude',
            'It is greater than one',
          ],
          correctAnswer: 1,
          explanation:
            'Normalization creates a unit vector with magnitude 1, preserving direction but setting length to 1.',
        },
        {
          id: 'vec-fund-q3',
          question:
            'In machine learning, what typically represents a single data point?',
          options: ['A scalar', 'A matrix', 'A vector', 'A tensor'],
          correctAnswer: 2,
          explanation:
            'Each data point is represented as a vector, where each component corresponds to a feature.',
        },
        {
          id: 'vec-fund-q4',
          question: 'If v = [2, 3] and w = [1, -1], what is 2v + w?',
          options: ['[3, 2]', '[5, 5]', '[5, 2]', '[4, 6]'],
          correctAnswer: 1,
          explanation:
            '2v + w = 2[2,3] + [1,-1] = [4,6] + [1,-1] = [5,5]. Multiply v by 2 first, then add w.',
        },
        {
          id: 'vec-fund-q5',
          question: 'Why is vectorization preferred over loops in NumPy?',
          options: [
            'It uses less memory',
            'It is significantly faster due to optimized C implementations',
            'It produces more accurate results',
            'It is required by Python syntax',
          ],
          correctAnswer: 1,
          explanation:
            'NumPy vectorized operations use optimized C and Fortran libraries (BLAS/LAPACK), making them much faster than Python loops.',
        },
      ],
      quiz: [
        {
          id: 'vec-fund-d1',
          question:
            'Explain why normalizing feature vectors is important in machine learning algorithms like k-NN and neural networks. What problems can occur if features are not normalized?',
          sampleAnswer:
            'Feature normalization is crucial because it ensures all features contribute equally to distance calculations and gradient updates. Without normalization, features with larger scales dominate. For example, in k-NN, if one feature is "house price in dollars" (range: 100,000-500,000) and another is "number of bedrooms" (range: 1-5), the distance calculation will be almost entirely determined by price, ignoring bedrooms. In neural networks, features with larger magnitudes can cause unstable gradients and slow convergence. Normalization (L2) or standardization (z-score) puts all features on comparable scales, allowing the model to learn the true importance of each feature rather than being biased by their original scales. This leads to faster training and better generalization.',
          keyPoints: [
            'Unnormalized features: larger-scale features dominate distance/gradient calculations',
            'Normalization ensures all features contribute equally',
            'Leads to faster training and better generalization in ML models',
          ],
        },
        {
          id: 'vec-fund-d2',
          question:
            'How do vectors enable us to work with high-dimensional data that we cannot visualize? Discuss the relationship between geometric intuition from 2D/3D and mathematical operations in higher dimensions.',
          sampleAnswer:
            'Vectors allow us to extend geometric intuition to arbitrarily high dimensions through algebraic operations. While we can only visualize up to 3D, the mathematical operations (addition, scaling, dot products, distances) work identically in any dimension. For example, the distance formula in 2D (Pythagorean theorem) extends naturally to n dimensions. This is powerful because real-world ML problems often involve hundreds or thousands of dimensions (features). A document might be represented as a 10,000-dimensional vector (one per word), yet we can still compute distances, similarities, and perform optimization. The key insight is that geometric concepts like "angle between vectors," "distance," and "projection" have precise algebraic definitions that work in any dimension, even when we cannot draw them. This mathematical abstraction is what makes modern ML possible.',
          keyPoints: [
            'Algebraic operations extend geometric intuition to high dimensions',
            'Distance, angle, and projection have precise definitions in n-dimensions',
            'Enables ML to work with thousands of features (document vectors, embeddings)',
          ],
        },
        {
          id: 'vec-fund-d3',
          question:
            'In the word embedding example (king - man + woman ≈ queen), what does this vector arithmetic represent conceptually? How do vectors capture semantic relationships between words?',
          sampleAnswer:
            'This vector arithmetic captures semantic relationships and analogies in language. When we compute "king - man + woman," we are performing conceptual reasoning: take the concept of "king," remove the "male" aspect, and add the "female" aspect, yielding "queen." This works because word embeddings (like Word2Vec or GloVe) are trained so that words appearing in similar contexts have similar vectors. During training, the model learns that "king" and "man" often appear in male contexts, while "queen" and "woman" appear in female contexts. The vector differences encode relationships: "king - man" captures "royalty minus maleness," and adding "woman" gives "royalty plus femaleness" = "queen." This demonstrates that vectors can encode not just individual meanings but also relationships and transformations between concepts. It is a remarkable example of how continuous representations (vectors) can capture discrete semantic knowledge, enabling machines to reason about language mathematically.',
          keyPoints: [
            'Vector arithmetic captures semantic analogies: king - man + woman ≈ queen',
            'Word embeddings learn from context: similar contexts → similar vectors',
            'Vectors encode relationships and transformations, not just meanings',
          ],
        },
      ],
    },

    {
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
      multipleChoice: [
        {
          id: 'vec-ops-q1',
          question:
            'What is the dot product of u = [1, 2, 3] and v = [4, 5, 6]?',
          options: ['18', '32', '21', '12'],
          correctAnswer: 1,
          explanation:
            'u · v = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32. Multiply corresponding components and sum.',
        },
        {
          id: 'vec-ops-q2',
          question:
            'If two vectors have a dot product of zero, what does this mean geometrically?',
          options: [
            'They are parallel',
            'They are orthogonal (perpendicular)',
            'They have the same magnitude',
            'They are opposite in direction',
          ],
          correctAnswer: 1,
          explanation:
            'A dot product of zero means cos(θ) = 0, so θ = 90°. The vectors are orthogonal/perpendicular.',
        },
        {
          id: 'vec-ops-q3',
          question:
            'Which distance metric is most appropriate for comparing text documents represented as term frequency vectors?',
          options: [
            'Euclidean distance',
            'Manhattan distance',
            'Cosine distance',
            'Chebyshev distance',
          ],
          correctAnswer: 2,
          explanation:
            "Cosine distance (or cosine similarity) is best for text because document length shouldn't matter—we care about the distribution of terms, not absolute frequencies. Two documents about the same topic should be similar regardless of one being longer.",
        },
        {
          id: 'vec-ops-q4',
          question: 'What is the L1 norm of vector v = [3, -4, 5]?',
          options: ['5', '7.07', '12', '50'],
          correctAnswer: 2,
          explanation:
            'L1 norm = |3| + |-4| + |5| = 3 + 4 + 5 = 12. Sum of absolute values of all components.',
        },
        {
          id: 'vec-ops-q5',
          question:
            'In a neural network, what operation does each neuron primarily perform with its inputs and weights?',
          options: [
            'Cross product',
            'Hadamard product',
            'Dot product',
            'Outer product',
          ],
          correctAnswer: 2,
          explanation:
            'Each neuron computes a dot product: z = w · x + b (weights dot inputs plus bias), then applies an activation function. This is the fundamental operation in neural networks.',
        },
      ],
      quiz: [
        {
          id: 'vec-ops-d1',
          question:
            'Compare and contrast Euclidean distance and cosine similarity. When would you choose one over the other in a machine learning application? Provide specific examples.',
          sampleAnswer:
            'Euclidean distance measures absolute spatial distance between vectors, while cosine similarity measures the angle between vectors, ignoring magnitude. Euclidean distance is sensitive to both direction and magnitude: vectors [1,1] and [10,10] are far apart (distance ≈ 12.7) even though they point in the same direction. Cosine similarity cares only about direction: these vectors have similarity = 1.0 (identical direction). Choose Euclidean when magnitude matters: measuring physical distances, detecting anomalies in sensor data, or comparing feature vectors where scale is meaningful. Choose cosine when direction matters more than magnitude: text similarity (a long document and short document can be similar if they discuss the same topics), recommendation systems (user preference patterns matter more than absolute ratings), and word embeddings (semantic similarity). In practice, for high-dimensional sparse data like text, cosine similarity often performs better because it is less affected by varying document lengths and focuses on content distribution.',
          keyPoints: [
            'Euclidean distance: sensitive to magnitude and direction (physical distance)',
            'Cosine similarity: only direction matters, ignores magnitude (text/embeddings)',
            'Choose Euclidean when scale matters, cosine when pattern/direction matters',
          ],
        },
        {
          id: 'vec-ops-d2',
          question:
            'Explain why L1 regularization (Lasso) tends to produce sparse models while L2 regularization (Ridge) does not. How do the different properties of L1 and L2 norms lead to this behavior?',
          sampleAnswer:
            'L1 and L2 regularization penalize model complexity differently due to the geometry of their norms. L2 regularization adds λ||w||₂² = λ(w₁² + w₂² + ... + wₙ²) to the loss, which creates a smooth, differentiable penalty. Gradients of L2 are proportional to the weights themselves, so weights shrink proportionally but rarely reach exactly zero. Geometrically, L2 creates a circular constraint region—the optimal solution typically lies where the elliptical error contours touch the circle, which is usually not at an axis (where weights are zero). L1 regularization adds λ||w||₁ = λ(|w₁| + |w₂| + ... + |wₙ|), creating a diamond-shaped constraint region with sharp corners at the axes. The penalty is constant regardless of weight magnitude (gradient is ±1), so small weights are penalized as much as large weights, driving them to exactly zero. The corners of the L1 diamond align with coordinate axes, so solutions often occur at corners where many weights are zero, producing sparsity. This makes L1 ideal for feature selection: it automatically identifies and zeros out irrelevant features, while L2 keeps all features but shrinks them.',
          keyPoints: [
            'L1 diamond geometry has corners on axes where weights = 0 (sparsity)',
            'L2 circular geometry rarely touches axes, shrinks weights proportionally',
            'L1 constant gradient drives small weights to zero; L2 gradient ∝ weight',
          ],
        },
        {
          id: 'vec-ops-d3',
          question:
            'The dot product can be computed as a sum of element-wise products or as ||u|| ||v|| cos(θ). Discuss how these two perspectives are used differently in machine learning, providing examples of each.',
          sampleAnswer:
            "These two equivalent definitions of the dot product serve different conceptual purposes in ML. The algebraic definition (u · v = Σ uᵢvᵢ) is used for **computation**: it is how we actually calculate dot products efficiently in code, and it is how we think about the mechanics of neural networks. Each neuron computes Σ wᵢxᵢ + b—a weighted sum of inputs. This perspective emphasizes the dot product as an aggregation operation, combining information from multiple sources with learned weights. It is also how we compute distances (||u - v||² = (u-v) · (u-v)) and norms (||v||² = v · v). The geometric definition (u · v = ||u|| ||v|| cos(θ)) is used for **interpretation** and **analysis**: it reveals that the dot product measures alignment or similarity between vectors. Cosine similarity for text comparison uses this explicitly. In neural networks, we can interpret what a neuron is computing: it is measuring how aligned the input is with the neuron's weight vector. Large positive dot product means strong alignment (similar direction), negative means opposite directions, zero means orthogonal (independent). This geometric view also explains why normalized vectors are important: after normalization, the dot product is purely the cosine of the angle, isolating directional similarity from magnitude effects. Both perspectives are essential—algebraic for implementation, geometric for understanding.",
          keyPoints: [
            'Algebraic (Σ uᵢvᵢ): used for computation in neural networks and algorithms',
            'Geometric (||u||||v||cos θ): used for interpretation of similarity/alignment',
            'Both perspectives essential: algebraic for implementation, geometric for understanding',
          ],
        },
      ],
    },

    {
      id: 'matrices-fundamentals',
      title: 'Matrices Fundamentals',
      content: `
# Matrices Fundamentals

## Introduction

A **matrix** is a rectangular array of numbers arranged in rows and columns. If vectors are the words of linear algebra, then matrices are the sentences—they enable us to express complex transformations, systems of equations, and entire datasets in a compact form.

In machine learning, matrices are everywhere:
- **Datasets**: Each row is a sample, each column is a feature
- **Weight matrices**: Connect layers in neural networks
- **Transformations**: Rotation, scaling, projection
- **Covariance**: Capture relationships between features

## Matrix Notation

### General Form

A matrix **A** with m rows and n columns is an **m × n matrix**:

**A** = ⎡a₁₁  a₁₂  ...  a₁ₙ⎤
      ⎢a₂₁  a₂₂  ...  a₂ₙ⎥
      ⎢ ...  ...  ...  ...⎥
      ⎣aₘ₁  aₘ₂  ...  aₘₙ⎦

- **m**: number of rows
- **n**: number of columns  
- **aᵢⱼ**: element in row i, column j

### Notation

- Bold capital letter: **A**, **B**, **X**
- Element: aᵢⱼ or Aᵢⱼ
- Size: **A** ∈ ℝᵐˣⁿ (m×n matrix of real numbers)

### Special Cases

- **Square matrix**: m = n (same number of rows and columns)
- **Row vector**: 1 × n matrix (single row)
- **Column vector**: m × 1 matrix (single column)
- **Scalar**: 1 × 1 matrix (single element)

## Matrix Indexing

In mathematics, indices typically start at 1:
- Rows: 1, 2, ..., m
- Columns: 1, 2, ..., n

In Python (NumPy), indices start at 0:
- Rows: 0, 1, ..., m-1
- Columns: 0, 1, ..., n-1

## Matrix Addition and Scalar Multiplication

### Matrix Addition

Two matrices can be added if they have the **same dimensions**:

**A** + **B** = [aᵢⱼ + bᵢⱼ]

Add corresponding elements.

**Properties**:
- Commutative: **A** + **B** = **B** + **A**
- Associative: (**A** + **B**) + **C** = **A** + (**B** + **C**)
- Identity: **A** + **0** = **A**
- Inverse: **A** + (-**A**) = **0**

### Scalar Multiplication

Multiply every element by a scalar c:

c**A** = [c·aᵢⱼ]

**Properties**:
- Distributive: c(**A** + **B**) = c**A** + c**B**
- Distributive: (c + d)**A** = c**A** + d**A**
- Associative: (cd)**A** = c(d**A**)
- Identity: 1**A** = **A**

## Matrix Multiplication

Matrix multiplication is more complex and incredibly important.

### Dimensions Must Be Compatible

To multiply **A** (m × n) by **B** (p × q):
- **Requirement**: n = p (columns of A = rows of B)
- **Result**: **C** = **AB** is m × q

The "inner dimensions" must match, and the result has the "outer dimensions."

### Definition

For **C** = **AB**, element cᵢⱼ is the dot product of row i of **A** and column j of **B**:

cᵢⱼ = Σₖ aᵢₖ bₖⱼ = aᵢ₁b₁ⱼ + aᵢ₂b₂ⱼ + ... + aᵢₙbₙⱼ

**Intuition**: Each element of the result is a dot product of a row from the first matrix and a column from the second matrix.

### Properties

1. **NOT commutative**: **AB** ≠ **BA** (in general)
2. **Associative**: (**AB**)**C** = **A**(**BC**)
3. **Distributive**: **A**(**B** + **C**) = **AB** + **AC**
4. **Identity**: **AI** = **IA** = **A**

### Why Matrix Multiplication Works This Way

This definition might seem arbitrary, but it corresponds to **composing linear transformations** and **representing systems of linear equations**. It is precisely the right operation for these purposes.

## Identity Matrix

The **identity matrix** **I** is a square matrix with 1s on the diagonal and 0s elsewhere:

**I** = ⎡1  0  0⎤
      ⎢0  1  0⎥
      ⎣0  0  1⎦

**Property**: **AI** = **IA** = **A** (for compatible dimensions)

The identity matrix is the multiplicative identity—like multiplying by 1 for scalars.

## Matrix Transpose

The **transpose** of **A** (denoted **Aᵀ**) flips rows and columns:

If **A** is m × n, then **Aᵀ** is n × m

(Aᵀ)ᵢⱼ = Aⱼᵢ

**Properties**:
- (**Aᵀ**)ᵀ = **A**
- (**A** + **B**)ᵀ = **Aᵀ** + **Bᵀ**
- (**AB**)ᵀ = **Bᵀ Aᵀ** (note the reversal!)
- (c**A**)ᵀ = c**Aᵀ**

## Python Implementation

\`\`\`python
import numpy as np

# Creating matrices
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])

print("Matrix A (2×3):")
print(A)
print(f"Shape: {A.shape}")
print()

print("Matrix B (3×2):")
print(B)
print(f"Shape: {B.shape}")
print()

# Accessing elements (0-indexed in Python)
print(f"Element A[0,0] (first row, first column): {A[0, 0]}")
print(f"Element A[1,2] (second row, third column): {A[1, 2]}")
print()

# Accessing rows and columns
print(f"First row of A: {A[0, :]}")
print(f"Second column of A: {A[:, 1]}")
print()

# Matrix dimensions
m, n = A.shape
print(f"A has {m} rows and {n} columns")
\`\`\`

\`\`\`python
# Matrix addition
C = np.array([[1, 2, 3], [4, 5, 6]])
D = np.array([[10, 20, 30], [40, 50, 60]])

print("=== Matrix Addition ===")
print("C:")
print(C)
print("\\nD:")
print(D)
print("\\nC + D:")
print(C + D)
print()

# Scalar multiplication
print("=== Scalar Multiplication ===")
print("C:")
print(C)
print("\\n3 * C:")
print(3 * C)
print()

# Element-wise multiplication (Hadamard product) - not standard matrix multiplication
print("=== Element-wise Multiplication ===")
print("C ⊙ D (element-wise):")
print(C * D)
print()
\`\`\`

\`\`\`python
# Matrix multiplication
print("=== Matrix Multiplication ===")
print("A (2×3):")
print(A)
print("\\nB (3×2):")
print(B)
print()

# Matrix multiplication: A @ B
C = A @ B  # Python 3.5+ operator
C_alt = np.dot(A, B)  # Alternative method
C_matmul = np.matmul(A, B)  # Another alternative

print("C = A @ B (2×2):")
print(C)
print()

# Verify all methods give same result
assert np.allclose(C, C_alt)
assert np.allclose(C, C_matmul)

# Show the computation for one element
print("Computing C[0,0]:")
print(f"C[0,0] = A[0,:] · B[:,0]")
print(f"       = {A[0,:]} · {B[:,0]}")
print(f"       = {A[0,0]}*{B[0,0]} + {A[0,1]}*{B[1,0]} + {A[0,2]}*{B[2,0]}")
print(f"       = {A[0,0]*B[0,0]} + {A[0,1]*B[1,0]} + {A[0,2]*B[2,0]}")
print(f"       = {C[0,0]}")
\`\`\`

\`\`\`python
# Matrix multiplication is NOT commutative
print("\\n=== Non-commutativity ===")
A_small = np.array([[1, 2], [3, 4]])
B_small = np.array([[5, 6], [7, 8]])

AB = A_small @ B_small
BA = B_small @ A_small

print("A:")
print(A_small)
print("\\nB:")
print(B_small)
print("\\nAB:")
print(AB)
print("\\nBA:")
print(BA)
print("\\nAB == BA:", np.allclose(AB, BA))
\`\`\`

\`\`\`python
# Identity matrix
print("\\n=== Identity Matrix ===")
I = np.eye(3)  # 3×3 identity matrix
print("Identity matrix I (3×3):")
print(I)
print()

# Matrix times identity
A_square = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("A:")
print(A_square)
print("\\nA @ I:")
print(A_square @ I)
print("\\nI @ A:")
print(I @ A_square)
print("\\nVerify A @ I == A:", np.allclose(A_square @ I, A_square))
\`\`\`

\`\`\`python
# Matrix transpose
print("\\n=== Matrix Transpose ===")
A_rect = np.array([[1, 2, 3], [4, 5, 6]])
print(f"A ({A_rect.shape[0]}×{A_rect.shape[1]}):")
print(A_rect)
print()

A_T = A_rect.T  # or np.transpose(A_rect)
print(f"Aᵀ ({A_T.shape[0]}×{A_T.shape[1]}):")
print(A_T)
print()

# Verify (Aᵀ)ᵀ = A
print("(Aᵀ)ᵀ == A:", np.allclose(A_T.T, A_rect))
print()

# Verify (AB)ᵀ = BᵀAᵀ
X = np.array([[1, 2], [3, 4], [5, 6]])
Y = np.array([[7, 8, 9], [10, 11, 12]])

XY = X @ Y
XY_T = XY.T
Y_T_X_T = Y.T @ X.T

print("Verify (XY)ᵀ = YᵀXᵀ:", np.allclose(XY_T, Y_T_X_T))
\`\`\`

## Matrices as Linear Transformations

A key insight: **matrix multiplication represents function composition**.

When we multiply a matrix **A** by a vector **x**, we transform **x**:

**y** = **Ax**

### Example: Rotation Matrix (2D)

Rotating a vector by angle θ:

**R**(θ) = ⎡cos(θ)  -sin(θ)⎤
          ⎣sin(θ)   cos(θ)⎦

\`\`\`python
# Rotation matrix example
def rotation_matrix_2d(theta):
    """Create 2D rotation matrix for angle theta (in radians)"""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

# Rotate vector [1, 0] by 90 degrees (π/2 radians)
v = np.array([1, 0])
theta = np.pi / 2  # 90 degrees

R = rotation_matrix_2d(theta)
v_rotated = R @ v

print("=== Rotation Example ===")
print(f"Original vector: {v}")
print(f"Rotation by {np.degrees(theta)}°")
print(f"Rotation matrix R:")
print(R)
print(f"Rotated vector: {v_rotated}")
print(f"Expected: [0, 1] (approximately)")
\`\`\`

\`\`\`python
# Visualize rotation
def plot_vector_transformation(v_original, v_transformed, title=""):
    """Plot original and transformed vectors"""
    plt.figure(figsize=(8, 8))
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    
    # Plot vectors
    plt.quiver(0, 0, v_original[0], v_original[1], 
              angles='xy', scale_units='xy', scale=1, 
              color='blue', width=0.008, label='Original')
    plt.quiver(0, 0, v_transformed[0], v_transformed[1], 
              angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.008, label='Transformed')
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(title)
    plt.axis('equal')
    plt.show()

plot_vector_transformation(v, v_rotated, "90° Rotation")
\`\`\`

### Example: Scaling Matrix

Scaling by factors sₓ and sᵧ:

**S** = ⎡sₓ  0 ⎤
      ⎣0   sᵧ⎦

\`\`\`python
# Scaling matrix
S = np.array([
    [2, 0],  # Scale x by 2
    [0, 3]   # Scale y by 3
])

v = np.array([1, 1])
v_scaled = S @ v

print("\\n=== Scaling Example ===")
print(f"Original vector: {v}")
print(f"Scaling matrix S:")
print(S)
print(f"Scaled vector: {v_scaled}")

plot_vector_transformation(v, v_scaled, "Scaling: 2x in X, 3x in Y")
\`\`\`

## Matrices in Machine Learning

### Dataset Representation

\`\`\`python
# Dataset as matrix: rows = samples, columns = features
# Example: Housing data
# Features: [square_feet, bedrooms, bathrooms, age_years]
# Price is separate (target variable)

X = np.array([
    [2000, 3, 2, 10],  # House 1
    [1500, 2, 1, 15],  # House 2
    [2500, 4, 3, 5],   # House 3
    [1800, 3, 2, 8],   # House 4
    [2200, 3, 2.5, 12] # House 5
])

y = np.array([300000, 220000, 380000, 290000, 320000])  # Prices

print("=== Dataset Matrix ===")
print(f"X shape: {X.shape} (5 samples, 4 features)")
print("X (feature matrix):")
print(X)
print(f"\\ny shape: {y.shape}")
print(f"y (target): {y}")
\`\`\`

### Linear Regression with Matrices

Linear regression can be expressed compactly using matrices:

**y** = **Xw** + **b**

Where:
- **X**: m × n matrix of features (m samples, n features)
- **w**: n × 1 vector of weights  
- **b**: scalar bias (or m × 1 vector)
- **y**: m × 1 vector of predictions

\`\`\`python
# Simple linear regression using matrix operations
# Predict house prices from features

# Initialize random weights
np.random.seed(42)
w = np.random.randn(4)  # 4 weights for 4 features
b = np.random.randn()   # 1 bias

print("\\n=== Linear Regression (Matrix Form) ===")
print(f"Weights w: {w}")
print(f"Bias b: {b:.4f}")
print()

# Make predictions: y_pred = Xw + b
y_pred = X @ w + b

print("Predictions y_pred = Xw + b:")
print(y_pred)
print()
print("Actual prices y:")
print(y)
print()

# Compute error (Mean Squared Error)
mse = np.mean((y_pred - y)**2)
print(f"MSE (with random weights): {mse:.2f}")
\`\`\`

### Neural Network Layer

A fully connected layer transforms input **x** to output **y**:

**y** = **Wx** + **b**

Where **W** is the weight matrix.

\`\`\`python
# Neural network layer as matrix multiplication
# Input layer: 4 neurons
# Hidden layer: 3 neurons

input_size = 4
hidden_size = 3

# Initialize weights and bias
np.random.seed(42)
W = np.random.randn(hidden_size, input_size) * 0.1
b = np.zeros(hidden_size)

print("\\n=== Neural Network Layer ===")
print(f"Weight matrix W ({hidden_size}×{input_size}):")
print(W)
print(f"\\nBias vector b ({hidden_size},):")
print(b)
print()

# Forward pass for one sample
x = np.array([1.0, 2.0, 3.0, 4.0])  # Input features
z = W @ x + b  # Linear transformation
a = np.maximum(0, z)  # ReLU activation

print(f"Input x: {x}")
print(f"Linear output z = Wx + b: {z}")
print(f"Activated output a = ReLU(z): {a}")
\`\`\`

### Batch Processing

Process multiple samples simultaneously:

\`\`\`python
# Batch processing: multiple samples at once
batch_size = 5
X_batch = X  # Use our housing data (5 samples, 4 features)

# Forward pass for entire batch
Z_batch = X_batch @ W.T + b  # Note: W.T to match dimensions
A_batch = np.maximum(0, Z_batch)

print("\\n=== Batch Processing ===")
print(f"Batch input X ({X_batch.shape}):")
print(X_batch)
print(f"\\nBatch output A ({A_batch.shape}):")
print(A_batch)
print()
print("Each row is the output for one sample")
\`\`\`

## Common Matrix Shapes in ML

| Matrix | Shape | Description |
|--------|-------|-------------|
| **X** | m × n | Dataset: m samples, n features |
| **y** | m × 1 | Target values for m samples |
| **W** | k × n | Weights: k outputs, n inputs |
| **b** | k × 1 | Biases for k outputs |
| **θ** | n × 1 | Model parameters |

## Best Practices

1. **Check dimensions before matrix multiplication**: Use \`.shape\` liberally
2. **Use @ operator for clarity**: \`A @ B\` is clearer than \`np.dot(A, B)\` for matrices
3. **Vectorize operations**: Process entire batches, not individual samples in loops
4. **Transpose when necessary**: Match dimensions for multiplication
5. **Initialize sensibly**: Random initialization for neural networks, not zeros

\`\`\`python
# Good practice: Check shapes
def matrix_multiply_safe(A, B):
    """Safely multiply matrices with dimension checking"""
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Cannot multiply {A.shape} × {B.shape}: "
                        f"incompatible dimensions")
    return A @ B

# Test
try:
    result = matrix_multiply_safe(A, B)
    print("Multiplication successful!")
except ValueError as e:
    print(f"Error: {e}")
\`\`\`

## Summary

Matrices are the workhorse of machine learning:

- **Compact representation**: Entire datasets, transformations, and models
- **Efficient computation**: Matrix operations are highly optimized (BLAS/LAPACK)
- **Batch processing**: Process many samples simultaneously
- **Linear transformations**: Rotation, scaling, projection, neural network layers

**Key Operations**:
- Addition: Element-wise, same dimensions required
- Scalar multiplication: Scale all elements
- Matrix multiplication: Compose transformations, compute outputs
- Transpose: Flip rows and columns

**Applications**:
- Datasets (rows = samples, columns = features)
- Linear regression (**y** = **Xw** + **b**)
- Neural networks (each layer is matrix multiplication + activation)
- Image processing (images are matrices)

Master matrices, and you can express complex ML operations concisely and compute them efficiently!
`,
      multipleChoice: [
        {
          id: 'mat-fund-q1',
          question:
            'If matrix A is 3×4 and matrix B is 4×2, what is the shape of AB?',
          options: ['3×2', '4×4', '3×4', 'Cannot multiply'],
          correctAnswer: 0,
          explanation:
            'For matrix multiplication, the inner dimensions must match (4=4) and the result has outer dimensions: 3×2.',
        },
        {
          id: 'mat-fund-q2',
          question:
            'What is the result of multiplying any matrix A by the identity matrix I (assuming compatible dimensions)?',
          options: [
            'The zero matrix',
            'The transpose of A',
            'The matrix A itself',
            'The inverse of A',
          ],
          correctAnswer: 2,
          explanation:
            'AI = IA = A. The identity matrix is the multiplicative identity - it leaves matrices unchanged, like multiplying by 1 for scalars.',
        },
        {
          id: 'mat-fund-q3',
          question:
            'If A is a 5×3 matrix, what is the shape of its transpose Aᵀ?',
          options: ['5×3', '3×5', '5×5', '3×3'],
          correctAnswer: 1,
          explanation:
            'Transpose flips rows and columns: a 5×3 matrix becomes a 3×5 matrix.',
        },
        {
          id: 'mat-fund-q4',
          question:
            'In a dataset matrix X with shape (100, 20), what do the dimensions represent?',
          options: [
            '20 samples with 100 features each',
            '100 samples with 20 features each',
            '100×20 = 2000 total data points',
            '20 classes and 100 possible predictions',
          ],
          correctAnswer: 1,
          explanation:
            'By convention, dataset matrices have rows as samples and columns as features: 100 samples, each with 20 features.',
        },
        {
          id: 'mat-fund-q5',
          question: 'Why is matrix multiplication NOT commutative (AB ≠ BA)?',
          options: [
            'It is a mathematical convention with no deeper reason',
            'Matrices represent ordered operations/transformations that compose in a specific order',
            'It would be too computationally expensive',
            'Only square matrices can be commutative',
          ],
          correctAnswer: 1,
          explanation:
            'Matrix multiplication represents function composition. Applying transformation A then B is different from applying B then A. For example, "rotate then scale" produces different results than "scale then rotate." The order matters because operations don\'t generally commute.',
        },
      ],
      quiz: [
        {
          id: 'mat-fund-d1',
          question:
            'Explain how matrix multiplication enables efficient batch processing in neural networks. Why is this crucial for modern deep learning?',
          sampleAnswer:
            'Matrix multiplication allows neural networks to process multiple samples simultaneously through a single operation. Instead of computing y = Wx + b for each sample individually in a loop (which would require n forward passes for n samples), we organize samples into a batch matrix X of shape (batch_size, input_dim) and compute Y = XW^T + b in one operation, producing outputs for all samples at once. This is crucial for several reasons: (1) Computational efficiency—GPUs are optimized for matrix operations and can parallelize thousands of arithmetic operations simultaneously. A single matrix multiplication is vastly faster than many sequential vector-matrix multiplications. (2) Memory efficiency—Modern deep learning frameworks can optimize memory access patterns when working with batches. (3) Statistical benefits—Using mini-batches provides better gradient estimates than single samples (less noisy than SGD) while being more computationally efficient than full-batch gradient descent. (4) Hardware utilization—GPUs have thousands of cores that sit idle when processing single samples; batching keeps them busy. Without efficient batching via matrix operations, training modern deep learning models would be prohibitively slow. A model that takes hours with batching might take weeks or months without it.',
          keyPoints: [
            'Batch matrix multiplication: Y = XW^T processes all samples simultaneously',
            'GPU optimization: matrix ops enable massive parallelization (100x+ speedup)',
            'Batch size trade-off: computational efficiency vs gradient update frequency',
          ],
        },
        {
          id: 'mat-fund-d2',
          question:
            'The transpose operation reverses multiplication order: (AB)ᵀ = BᵀAᵀ. Explain why this property is important in backpropagation for training neural networks.',
          sampleAnswer:
            'This transpose property is fundamental to backpropagation because gradients flow backward through the network in reverse order of the forward pass. In the forward pass, we compute Y = XW + b (input X times weights W). During backpropagation, we receive gradients dL/dY and need to compute gradients with respect to X and W. Using the chain rule: dL/dX = dL/dY × dY/dX. Since Y = XW, we have dY/dX = W^T (transpose of W). So dL/dX = (dL/dY)W^T. Similarly, dL/dW = X^T(dL/dY). Notice how the matrices appear in reversed, transposed form compared to the forward pass. This mirrors the mathematical property (AB)^T = B^T A^T. The transpose reversal is why backprop involves transposed weight matrices—we are literally reversing the flow of information. This property ensures that gradient dimensions match correctly: if forward pass goes from (batch, input_dim) → (batch, output_dim) via W of shape (input_dim, output_dim), then backward pass must go (batch, output_dim) → (batch, input_dim) via W^T of shape (output_dim, input_dim). Understanding this transpose property is key to implementing backpropagation correctly and debugging gradient computation errors.',
          keyPoints: [
            'Forward: Y = XW; Backward: dL/dX = (dL/dY)W^T (transposed weights)',
            'Transpose reversal mirrors (AB)^T = B^T A^T mathematical property',
            'Ensures gradient dimensions match: information flows backward through transposed weights',
          ],
        },
        {
          id: 'mat-fund-d3',
          question:
            'Compare representing a dataset as a list of vectors versus a single matrix. What are the trade-offs in terms of operations, memory, and coding practices in machine learning?',
          sampleAnswer:
            "Representing a dataset as a matrix (2D array) rather than a list of vectors (list of 1D arrays) offers significant advantages in ML, though with some trade-offs. Matrix representation enables: (1) Vectorized operations—computing statistics (mean, std) or transformations (scaling, PCA) on all features at once using optimized linear algebra libraries, orders of magnitude faster than looping through vectors. (2) Consistent dimensions—matrices enforce uniform feature counts across samples, catching data inconsistencies early. (3) Straightforward model application—neural networks expect matrix inputs for batch processing; constantly converting lists to matrices adds overhead. (4) Memory layout—contiguous memory storage enables CPU/GPU cache optimization and efficient data transfer. However, list-of-vectors can be advantageous when: (1) Samples have variable lengths (though padding/masking or ragged tensors often work), (2) Data arrives sequentially and doesn't fit in memory (though streaming matrix operations exist), (3) You need flexibility to modify individual samples independently. In practice, use matrices for structured, fixed-size data (tabular, images, most ML tasks). Use lists when dealing with truly variable-length sequences (before padding), or when prototyping with small datasets where performance doesn't matter. Modern ML libraries (NumPy, PyTorch, TensorFlow) are built around matrix operations, so embracing matrix representation aligns with the ecosystem and unlocks performance. The rule: default to matrices, use lists only when necessary.",
          keyPoints: [
            'Matrix: vectorized ops, GPU optimization, enforced dimensions (orders of magnitude faster)',
            'List-of-vectors: flexibility for variable lengths, sequential processing',
            'Default to matrices for ML (aligns with NumPy/PyTorch/TensorFlow ecosystem)',
          ],
        },
      ],
    },

    {
      id: 'matrix-operations',
      title: 'Matrix Operations',
      content: `
# Matrix Operations

## Introduction

Beyond basic addition and multiplication, matrices support advanced operations crucial for machine learning: matrix-vector multiplication, batch processing, broadcasting, and computing complex transformations. Understanding these operations deeply enables you to implement and optimize ML algorithms efficiently.

## Matrix-Vector Multiplication

One of the most fundamental operations: multiplying a matrix **A** (m × n) by a vector **x** (n × 1) produces a vector **y** (m × 1).

**y** = **Ax**

### Interpretation

Each element yᵢ is the dot product of row i of **A** with vector **x**:

yᵢ = **aᵢ** · **x** = Σⱼ aᵢⱼ xⱼ

This is exactly what happens in a neural network layer: the weight matrix multiplies the input vector.

### Geometric Interpretation

Matrix-vector multiplication is a **linear transformation**: it transforms vector **x** into vector **y** according to the transformation defined by **A**.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

print("=== Matrix-Vector Multiplication ===")

# Define matrix and vector
A = np.array([
    [2, 0],
    [0, 3]
])  # Scaling matrix

x = np.array([1, 1])

# Multiply
y = A @ x

print("Matrix A:")
print(A)
print(f"\\nVector x: {x}")
print(f"Result y = Ax: {y}")
print()

# Verify element-wise
print("Computing y manually:")
print(f"y[0] = A[0,:] · x = {A[0,:]} · {x} = {A[0,0]*x[0] + A[0,1]*x[1]}")
print(f"y[1] = A[1,:] · x = {A[1,:]} · {x} = {A[1,0]*x[0] + A[1,1]*x[1]}")
\`\`\`

\`\`\`python
# Visualize transformation
def visualize_matrix_vector_transform(A, x):
    """Visualize how matrix A transforms vector x"""
    y = A @ x
    
    plt.figure(figsize=(10, 5))
    
    # Plot original and transformed vector
    plt.subplot(1, 2, 1)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.008, label='x (original)')
    plt.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1,
              color='red', width=0.008, label='y = Ax (transformed)')
    plt.xlim(-0.5, 4)
    plt.ylim(-0.5, 4)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Matrix Transforms Vector')
    plt.axis('equal')
    
    # Show transformation of unit square
    plt.subplot(1, 2, 2)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    
    # Unit square corners
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    transformed_square = A @ square
    
    plt.plot(square[0, :], square[1, :], 'b-', linewidth=2, label='Original square')
    plt.plot(transformed_square[0, :], transformed_square[1, :], 'r-', 
            linewidth=2, label='Transformed square')
    
    max_val = max(np.max(transformed_square), 2)
    plt.xlim(-0.5, max_val)
    plt.ylim(-0.5, max_val)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('How A Transforms Space')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

visualize_matrix_vector_transform(A, x)
\`\`\`

## Matrix-Matrix Multiplication Revisited

Let's understand matrix multiplication from the perspective of **column combinations** and **row transformations**.

### Column Perspective

**AB** can be viewed as **A** transforming each column of **B**:

If **B** = [**b₁** **b₂** ... **bₙ**], then
**AB** = [**Ab₁** **Ab₂** ... **Abₙ**]

Each column of the result is **A** times the corresponding column of **B**.

\`\`\`python
print("\\n=== Column Perspective ===")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A:")
print(A)
print("\\nB:")
print(B)
print()

# Compute AB
AB = A @ B
print("AB:")
print(AB)
print()

# Verify column by column
print("Column perspective:")
print("First column of AB = A @ (first column of B)")
col1 = A @ B[:, 0]
print(f"  A @ {B[:, 0]} = {col1}")
print(f"  Matches AB[:, 0] = {AB[:, 0]}: {np.allclose(col1, AB[:, 0])}")
print()

print("Second column of AB = A @ (second column of B)")
col2 = A @ B[:, 1]
print(f"  A @ {B[:, 1]} = {col2}")
print(f"  Matches AB[:, 1] = {AB[:, 1]}: {np.allclose(col2, AB[:, 1])}")
\`\`\`

### Row Perspective

**AB** can also be viewed as each row of **A** forming a linear combination of rows of **B**.

\`\`\`python
print("\\n=== Row Perspective ===")
print("First row of AB = (first row of A) @ B")
row1 = A[0, :] @ B
print(f"  {A[0, :]} @ B = {row1}")
print(f"  Matches AB[0, :] = {AB[0, :]}: {np.allclose(row1, AB[0, :])}")
\`\`\`

## Batch Matrix-Vector Multiplication

In ML, we often process multiple vectors (a batch) simultaneously.

Given:
- **X**: (batch_size, n) - batch of input vectors
- **W**: (n, m) - weight matrix
- **Y** = **XW**: (batch_size, m) - batch of output vectors

Each row of **X** is multiplied by **W** to produce the corresponding row of **Y**.

\`\`\`python
print("\\n=== Batch Processing ===")

# Batch of 4 samples, each with 3 features
X = np.array([
    [1, 2, 3],  # Sample 1
    [4, 5, 6],  # Sample 2
    [7, 8, 9],  # Sample 3
    [2, 1, 0]   # Sample 4
])

# Weight matrix: 3 inputs → 2 outputs
W = np.array([
    [0.1, 0.4],
    [0.2, 0.5],
    [0.3, 0.6]
])

print(f"X shape: {X.shape} (4 samples, 3 features each)")
print(f"W shape: {W.shape} (3 inputs, 2 outputs)")
print()

# Batch multiplication
Y = X @ W

print(f"Y = X @ W, shape: {Y.shape} (4 samples, 2 outputs each)")
print("Y:")
print(Y)
print()

# Verify for first sample
print("Verification for first sample:")
y1 = X[0, :] @ W
print(f"X[0, :] @ W = {X[0, :]} @ W = {y1}")
print(f"Matches Y[0, :] = {Y[0, :]}: {np.allclose(y1, Y[0, :])}")
\`\`\`

## Matrix Powers

For square matrices, we can compute powers: **A²** = **AA**, **A³** = **AAA**, etc.

\`\`\`python
print("\\n=== Matrix Powers ===")

A = np.array([[1, 1], [0, 1]])
print("A:")
print(A)
print()

# A^2
A2 = A @ A
print("A² = A @ A:")
print(A2)
print()

# A^3
A3 = A2 @ A
print("A³:")
print(A3)
print()

# Using numpy's matrix_power
A4 = np.linalg.matrix_power(A, 4)
print("A⁴ (using np.linalg.matrix_power):")
print(A4)
\`\`\`

## Trace of a Matrix

The **trace** is the sum of diagonal elements:

tr(**A**) = Σᵢ aᵢᵢ = a₁₁ + a₂₂ + ... + aₙₙ

**Properties**:
- tr(**A** + **B**) = tr(**A**) + tr(**B**)
- tr(c**A**) = c·tr(**A**)
- tr(**AB**) = tr(**BA**) (even if **AB** ≠ **BA**!)
- tr(**A**ᵀ) = tr(**A**)

\`\`\`python
print("\\n=== Trace ===")

A = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])

trace_A = np.trace(A)
trace_manual = A[0,0] + A[1,1] + A[2,2]

print("A:")
print(A)
print(f"\\ntr(A) = {trace_A}")
print(f"tr(A) manual = {A[0,0]} + {A[1,1]} + {A[2,2]} = {trace_manual}")
print()

# Verify tr(AB) = tr(BA)
B = np.array([[1, 0, 2],
             [0, 1, 0],
             [2, 0, 1]])

AB = A @ B
BA = B @ A

print(f"tr(AB) = {np.trace(AB)}")
print(f"tr(BA) = {np.trace(BA)}")
print(f"Equal: {np.isclose(np.trace(AB), np.trace(BA))}")
\`\`\`

## Broadcasting in NumPy

**Broadcasting** allows operations on arrays of different shapes by automatically expanding dimensions.

### Rules

1. If arrays have different numbers of dimensions, pad the smaller one with 1s on the left
2. Arrays are compatible if dimensions are equal or one of them is 1
3. If compatible, broadcast the smaller dimension to match the larger

\`\`\`python
print("\\n=== Broadcasting ===")

# Example 1: Add vector to each row of matrix
X = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])

v = np.array([10, 20, 30])

print("X (3×3):")
print(X)
print(f"\\nv (3,): {v}")
print()

# Broadcasting: v is added to each row
X_plus_v = X + v
print("X + v (broadcasts v to each row):")
print(X_plus_v)
print()

# Example 2: Multiply matrix by column vector
col = np.array([[2], [3], [4]])  # 3×1
print(f"col shape {col.shape}:")
print(col)
print()

X_times_col = X * col  # Element-wise multiplication with broadcasting
print("X * col (broadcasts col to each column):")
print(X_times_col)
\`\`\`

\`\`\`python
# Common broadcasting patterns
print("\\n=== Common Broadcasting Patterns ===")

# 1. Add bias to each sample in batch
batch = np.random.randn(100, 5)  # 100 samples, 5 features
bias = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 5 biases

batch_with_bias = batch + bias  # Bias broadcast to all samples
print(f"1. Batch {batch.shape} + bias {bias.shape} → {batch_with_bias.shape}")

# 2. Normalize each feature (across samples)
mean = batch.mean(axis=0)  # Mean of each feature
std = batch.std(axis=0)  # Std of each feature
batch_normalized = (batch - mean) / std
print(f"2. Batch normalization: {batch.shape} → {batch_normalized.shape}")

# 3. Outer product without explicit loops
a = np.array([1, 2, 3])
b = np.array([4, 5])
outer = a[:, np.newaxis] @ b[np.newaxis, :]
print(f"\\n3. Outer product: a{a.shape} ⊗ b{b.shape} → {outer.shape}")
print("Outer product:")
print(outer)
\`\`\`

## Element-wise Operations

Operations that work element-by-element (require same shape or broadcasting).

\`\`\`python
print("\\n=== Element-wise Operations ===")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A:")
print(A)
print("\\nB:")
print(B)
print()

# Element-wise operations
print("A + B:")
print(A + B)

print("\\nA * B (Hadamard product):")
print(A * B)

print("\\nA / B:")
print(A / B)

print("\\nA ** 2 (element-wise square):")
print(A ** 2)

print("\\nnp.sqrt(A):")
print(np.sqrt(A))

print("\\nnp.exp(A):")
print(np.exp(A))
\`\`\`

## Matrix Operations in Neural Networks

Let's implement a complete forward pass through a neural network layer using pure matrix operations.

\`\`\`python
print("\\n=== Neural Network Forward Pass ===")

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def forward_pass(X, W1, b1, W2, b2):
    """
    Two-layer neural network forward pass
    
    X: (batch_size, input_dim)
    W1: (input_dim, hidden_dim)
    b1: (hidden_dim,)
    W2: (hidden_dim, output_dim)
    b2: (output_dim,)
    """
    # Layer 1
    Z1 = X @ W1 + b1  # Linear transformation
    A1 = relu(Z1)  # Activation
    
    # Layer 2
    Z2 = A1 @ W2 + b2  # Linear transformation
    A2 = Z2  # No activation (linear output)
    
    return Z1, A1, Z2, A2

# Initialize network
np.random.seed(42)
input_dim = 4
hidden_dim = 3
output_dim = 2
batch_size = 5

W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros(output_dim)

# Input data
X = np.random.randn(batch_size, input_dim)

print(f"Input X: {X.shape}")
print(f"W1: {W1.shape}, b1: {b1.shape}")
print(f"W2: {W2.shape}, b2: {b2.shape}")
print()

# Forward pass
Z1, A1, Z2, A2 = forward_pass(X, W1, b1, W2, b2)

print(f"After layer 1: Z1 {Z1.shape}, A1 {A1.shape}")
print(f"After layer 2: Z2 {Z2.shape}, A2 {A2.shape}")
print(f"\\nFinal output A2:\\n{A2}")
\`\`\`

## Common Mistakes and Best Practices

\`\`\`python
print("\\n=== Common Mistakes ===")

# Mistake 1: Shape mismatches
A = np.array([[1, 2, 3]])  # Shape (1, 3)
B = np.array([[4], [5], [6]])  # Shape (3, 1)

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")

try:
    result = A + B  # This will work due to broadcasting!
    print(f"A + B worked (broadcasting): shape {result.shape}")
    print(result)
except ValueError as e:
    print(f"Error: {e}")

print()

# Mistake 2: Confusing @ and *
X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6], [7, 8]])

print("X @ Y (matrix multiplication):")
print(X @ Y)

print("\\nX * Y (element-wise, Hadamard):")
print(X * Y)

# Mistake 3: Forgetting to reshape
v = np.array([1, 2, 3])  # Shape (3,)
print(f"\\nv shape: {v.shape}")

v_col = v[:, np.newaxis]  # Shape (3, 1)
v_row = v[np.newaxis, :]  # Shape (1, 3)

print(f"v as column: {v_col.shape}")
print(f"v as row: {v_row.shape}")

outer_product = v_col @ v_row
print(f"\\nOuter product v_col @ v_row: shape {outer_product.shape}")
print(outer_product)
\`\`\`

## Performance Tips

\`\`\`python
import time

print("\\n=== Performance Comparison ===")

# Setup
n = 1000
A = np.random.randn(n, n)
B = np.random.randn(n, n)

# Vectorized (fast)
start = time.time()
C_fast = A @ B
time_fast = time.time() - start

# Loop-based (slow) - don't do this!
start = time.time()
C_slow = np.zeros((n, n))
for i in range(min(10, n)):  # Only 10 rows to save time
    for j in range(n):
        C_slow[i, j] = np.dot(A[i, :], B[:, j])
time_slow = time.time() - start
time_slow_extrapolated = time_slow * (n / 10)

print(f"Vectorized @ operator: {time_fast:.4f} seconds")
print(f"Loop-based (10 rows): {time_slow:.4f} seconds")
print(f"Loop-based (extrapolated full): ~{time_slow_extrapolated:.2f} seconds")
print(f"Speedup: ~{time_slow_extrapolated/time_fast:.0f}x faster")
\`\`\`

## Summary

Matrix operations are the computational engine of machine learning:

1. **Matrix-vector multiplication**: Transforms vectors (neural network layers)
2. **Matrix-matrix multiplication**: Composes transformations, processes batches
3. **Batch processing**: Process multiple samples simultaneously for efficiency
4. **Broadcasting**: Automatic dimension expansion for convenient operations
5. **Element-wise operations**: Apply functions to each element
6. **Trace**: Sum of diagonal elements, useful in loss functions

**Best Practices**:
- Always check shapes: use \`.shape\` liberally
- Use @ for matrix multiplication, * for element-wise
- Vectorize operations: avoid Python loops
- Leverage broadcasting for concise code
- Use proper reshaping when needed

**Performance**:
- Vectorized operations are 10-1000x faster than loops
- GPUs accelerate matrix operations even more
- Modern ML frameworks (PyTorch, TensorFlow) optimize these operations

Master these operations, and you can implement any neural network architecture efficiently!
`,
      multipleChoice: [
        {
          id: 'mat-ops-q1',
          question:
            'In batch processing with matrix multiplication Y = XW, if X has shape (100, 20) and W has shape (20, 5), what is the shape of Y?',
          options: ['(100, 20)', '(20, 5)', '(100, 5)', '(5, 100)'],
          correctAnswer: 2,
          explanation:
            'Matrix multiplication: (m×n) @ (n×p) = (m×p). So (100×20) @ (20×5) = (100×5). Each of 100 samples transformed from 20 features to 5 outputs.',
        },
        {
          id: 'mat-ops-q2',
          question: 'What is the trace of a 3×3 identity matrix?',
          options: ['0', '1', '3', '9'],
          correctAnswer: 2,
          explanation:
            'The identity matrix has 1s on the diagonal and 0s elsewhere. trace(I₃) = 1 + 1 + 1 = 3.',
        },
        {
          id: 'mat-ops-q3',
          question:
            'When adding a vector of shape (5,) to a matrix of shape (10, 5) in NumPy, what happens?',
          options: [
            'Error: incompatible shapes',
            'The vector is added to each row of the matrix',
            'The vector is added to each column of the matrix',
            'Only the first row is modified',
          ],
          correctAnswer: 1,
          explanation:
            'NumPy broadcasting automatically adds the vector to each row. The vector is broadcast along axis 0 to match the matrix shape.',
        },
        {
          id: 'mat-ops-q4',
          question: 'What is the difference between A @ B and A * B in NumPy?',
          options: [
            'They are identical operations',
            '@ is matrix multiplication, * is element-wise multiplication',
            '@ only works for square matrices',
            '* is always faster',
          ],
          correctAnswer: 1,
          explanation:
            '@ performs matrix multiplication (dot product of rows and columns), while * performs element-wise Hadamard product (requires same shape or broadcasting).',
        },
        {
          id: 'mat-ops-q5',
          question:
            'Why is batch processing with matrices much faster than processing samples one at a time in a loop?',
          options: [
            'It uses less memory',
            'It produces more accurate results',
            'Optimized linear algebra libraries (BLAS) and hardware parallelization',
            'Python loops are forbidden in machine learning',
          ],
          correctAnswer: 2,
          explanation:
            'Matrix operations leverage highly optimized linear algebra libraries (BLAS/LAPACK) implemented in C/Fortran, and modern hardware (CPUs, GPUs) can parallelize thousands of operations simultaneously. A single matrix multiplication is orders of magnitude faster than equivalent Python loops.',
        },
      ],
      quiz: [
        {
          id: 'mat-ops-d1',
          question:
            'Explain how NumPy broadcasting works and why it is useful in machine learning. Provide examples of common broadcasting patterns in neural networks.',
          sampleAnswer:
            "Broadcasting is NumPy's mechanism for performing operations on arrays of different shapes by automatically expanding smaller dimensions. Rules: (1) If arrays differ in number of dimensions, prepend 1s to the smaller shape. (2) Arrays are compatible if each dimension is either equal or one is 1. (3) Arrays with dimension 1 are stretched to match the other dimension. This is invaluable in ML: (1) Adding bias: In a layer output = XW + b, if X@W has shape (batch, features) and b has shape (features,), broadcasting adds b to every sample automatically. Without broadcasting, we'd need explicit loops or tile b to (batch, features). (2) Batch normalization: (X - mean) / std where mean and std are per-feature statistics broadcasts automatically. (3) Attention mechanisms: When computing attention scores, we often need to broadcast queries/keys across different dimensions. (4) Loss computation: Comparing predictions (batch, classes) with one-hot labels involves broadcasting. Broadcasting makes code concise, readable, and fast (vectorized operations instead of loops). It eliminates the need for manually expanding arrays, reducing memory and computation. Understanding broadcasting is essential for implementing efficient ML code and debugging shape errors.",
          keyPoints: [
            'Broadcasting: automatic array expansion when dimensions are compatible (equal or 1)',
            'ML use cases: bias addition, batch norm, attention mechanisms (eliminates loops)',
            'Makes code concise, readable, fast; essential for efficient ML implementation',
          ],
        },
        {
          id: 'mat-ops-d2',
          question:
            'Compare the three perspectives of matrix multiplication: (1) element-wise computation, (2) column combination, (3) row transformation. When is each perspective most useful in understanding ML operations?',
          sampleAnswer:
            "Matrix multiplication C = AB can be understood three ways, each illuminating different aspects: (1) Element-wise: cᵢⱼ = Σₖ aᵢₖbₖⱼ. This is how we compute it manually and verify calculations. Useful for understanding computational complexity (O(n³) for n×n matrices) and debugging specific element calculations. (2) Column perspective: AB = [Ab₁ | Ab₂ | ... | Abₙ], each column of C is A transforming the corresponding column of B. This perspective is crucial for understanding how weight matrices transform input feature dimensions in neural networks. When we compute Y = XW, each output feature (column of Y) is a specific learned combination of input features. (3) Row perspective: Each row of C is a row of A combining rows of B. Useful for batch processing understanding: when X is (batch, features) and W is (features, outputs), each row of Y represents one sample's transformation. In backpropagation, the row perspective helps understand how gradients flow through layers. The column view is best for understanding feature transformations and weight matrix interpretation. The row view is best for understanding batch processing and sample-wise operations. The element view is best for mathematical verification and complexity analysis. Expert ML practitioners fluidly switch between perspectives depending on whether they are analyzing feature engineering, batch processing, or debugging.",
          keyPoints: [
            'Element-wise (Σₖ aᵢₖbₖⱼ): computational complexity, manual verification',
            'Column perspective: feature transformations, how weights combine input features',
            'Row perspective: batch processing, sample-wise transformations, gradient flow',
          ],
        },
        {
          id: 'mat-ops-d3',
          question:
            'The trace operation has the property tr(AB) = tr(BA) even though AB ≠ BA. Explain why this is true and discuss where this property is useful in machine learning and statistics.',
          sampleAnswer:
            'The trace property tr(AB) = tr(BA) is remarkable because matrix multiplication is not commutative. Proof sketch: tr(AB) = Σᵢ(AB)ᵢᵢ = Σᵢ Σₖ aᵢₖbₖᵢ. Similarly, tr(BA) = Σₖ(BA)ₖₖ = Σₖ Σᵢ bₖᵢaᵢₖ. These are the same sum with indices reversed. Geometrically, trace is invariant to cyclic permutations: tr(ABC) = tr(CAB) = tr(BCA). This property is crucial in several ML contexts: (1) Matrix derivatives: When deriving gradients, we often need to simplify expressions like tr(X^T A X). The cyclic property allows us to rearrange terms. (2) Frobenius norm: ||A||_F² = tr(A^T A) = tr(AA^T), giving equivalent expressions. (3) PCA and covariance: tr(Σ) gives total variance, and trace properties help simplify eigenvalue computations. (4) Fisher Information Matrix: In statistics, trace properties simplify expected value calculations. (5) Loss functions: Some regularization terms use trace, and the cyclic property helps derive gradients. (6) Quantum mechanics and physics: Trace is basis-independent, making it fundamental for observable quantities. The trace is one of the few matrix operations that treats AB and BA equivalently, making it especially useful when we need quantities invariant to certain transformations. In ML optimization, recognizing when to use trace properties can simplify complex derivative calculations significantly.',
          keyPoints: [
            'tr(AB) = tr(BA): cyclic permutation invariance despite AB ≠ BA',
            'Proof: both equal Σᵢ Σₖ aᵢₖbₖᵢ (same sum, reordered indices)',
            'ML uses: matrix derivatives, Frobenius norm, covariance, regularization terms',
          ],
        },
      ],
    },

    {
      id: 'special-matrices',
      title: 'Special Matrices',
      content: `
# Special Matrices

## Introduction

Certain types of matrices have special properties that make them particularly useful or efficient in machine learning. Understanding these matrices helps you recognize patterns, optimize computations, and gain insight into algorithm behavior.

## Diagonal Matrices

A **diagonal matrix** has non-zero elements only on the main diagonal (where row index = column index).

**D** = ⎡d₁  0   0   ...  0 ⎤
      ⎢0   d₂  0   ...  0 ⎥
      ⎢0   0   d₃  ...  0 ⎥
      ⎢...  ... ...  ... ...⎥
      ⎣0   0   0   ... dₙ⎦

**Properties**:
- Multiplication is very fast: O(n) instead of O(n³)
- **Dv** scales each component: (Dv)ᵢ = dᵢvᵢ
- Powers are trivial: **D**ᵏ has elements dᵢᵏ
- Determinant: det(**D**) = d₁d₂...dₙ (product of diagonal elements)
- Inverse: **D⁻¹** has elements 1/dᵢ (if all dᵢ ≠ 0)

\`\`\`python
import numpy as np

print("=== Diagonal Matrices ===")

# Create diagonal matrix
d = np.array([2, 3, 4])
D = np.diag(d)

print("Diagonal elements:", d)
print("\\nDiagonal matrix D:")
print(D)
print()

# Multiply by vector (fast scaling)
v = np.array([1, 1, 1])
Dv = D @ v

print(f"v: {v}")
print(f"Dv: {Dv}")
print("Effect: scales each component by diagonal element")
print()

# Matrix power (trivial for diagonal)
D2 = D @ D
D2_direct = np.diag(d**2)

print("D²:")
print(D2)
print("\\nD² (direct calculation):")
print(D2_direct)
print(f"Equal: {np.allclose(D2, D2_direct)}")
print()

# Inverse (if elements non-zero)
D_inv = np.linalg.inv(D)
D_inv_direct = np.diag(1/d)

print("D⁻¹:")
print(D_inv)
print("\\nD⁻¹ (direct 1/d):")
print(D_inv_direct)
\`\`\`

### Applications in ML

1. **Feature scaling**: Multiply by diagonal matrix to scale features
2. **Learning rate per parameter**: Diagonal preconditioning in optimization
3. **Covariance simplification**: Diagonal covariance assumes independent features

\`\`\`python
# Feature scaling with diagonal matrix
X = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])

# Scale factors for each feature
scale = np.array([0.1, 0.5, 1.0])
S = np.diag(scale)

X_scaled = X @ S  # Scale each column

print("\\n=== Feature Scaling ===")
print("Original X:")
print(X)
print("\\nScaling matrix S:")
print(S)
print("\\nScaled X:")
print(X_scaled)
\`\`\`

## Symmetric Matrices

A matrix **A** is **symmetric** if **A** = **Aᵀ** (equals its transpose).

aᵢⱼ = aⱼᵢ for all i, j

**Properties**:
- All eigenvalues are **real** numbers
- Eigenvectors corresponding to distinct eigenvalues are **orthogonal**
- Can be diagonalized with orthogonal eigenvectors
- Very important in optimization and statistics

\`\`\`python
print("\\n=== Symmetric Matrices ===")

# Create symmetric matrix
A_sym = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])

print("Symmetric matrix A:")
print(A_sym)
print()

# Verify symmetry
print("A transpose:")
print(A_sym.T)
print(f"\\nA == Aᵀ: {np.allclose(A_sym, A_sym.T)}")
print()

# Eigenvalues (will be real)
eigenvalues, eigenvectors = np.linalg.eig(A_sym)
print("Eigenvalues (all real):")
print(eigenvalues.real)
print(f"Imaginary parts (should be ~0): {eigenvalues.imag}")
\`\`\`

### Applications in ML

1. **Covariance matrices**: Always symmetric
2. **Hessian matrices**: Second derivatives in optimization
3. **Kernel matrices**: In SVM and kernel methods
4. **Graph Laplacians**: In graph neural networks

\`\`\`python
# Covariance matrix is always symmetric
print("\\n=== Covariance Matrix (Symmetric) ===")

# Generate random data
np.random.seed(42)
data = np.random.randn(100, 3)

# Compute covariance matrix
cov_matrix = np.cov(data.T)

print("Covariance matrix shape:", cov_matrix.shape)
print("Covariance matrix:")
print(cov_matrix.round(3))
print(f"\\nIs symmetric: {np.allclose(cov_matrix, cov_matrix.T)}")
\`\`\`

## Orthogonal Matrices

A square matrix **Q** is **orthogonal** if its columns are orthonormal (unit vectors that are mutually perpendicular).

**Q**ᵀ**Q** = **QQ**ᵀ = **I**

Equivalently: **Q**ᵀ = **Q⁻¹**

**Properties**:
- Preserves lengths: ||**Qv**|| = ||**v**||
- Preserves angles and dot products: (**Qu**) · (**Qv**) = **u** · **v**
- Determinant is ±1: det(**Q**) = ±1
- Inverse is just transpose (very cheap to compute!)

\`\`\`python
print("\\n=== Orthogonal Matrices ===")

# Example: Rotation matrix (orthogonal)
theta = np.pi / 4  # 45 degrees
Q = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

print("Rotation matrix Q (45°):")
print(Q)
print()

# Verify orthogonality: QᵀQ = I
QtQ = Q.T @ Q
print("QᵀQ:")
print(QtQ)
print(f"Is identity: {np.allclose(QtQ, np.eye(2))}")
print()

# Verify Q⁻¹ = Qᵀ
Q_inv = np.linalg.inv(Q)
print("Q⁻¹:")
print(Q_inv)
print("\\nQᵀ:")
print(Q.T)
print(f"Q⁻¹ == Qᵀ: {np.allclose(Q_inv, Q.T)}")
print()

# Verify length preservation
v = np.array([3, 4])
Qv = Q @ v

print(f"||v|| = {np.linalg.norm(v):.4f}")
print(f"||Qv|| = {np.linalg.norm(Qv):.4f}")
print("Length preserved!")
\`\`\`

### Applications in ML

1. **QR decomposition**: Orthogonal Q matrix
2. **SVD**: U and V matrices are orthogonal
3. **PCA**: Principal component directions are orthonormal
4. **Orthogonal initialization**: Some neural network weight initializations

## Triangular Matrices

**Upper triangular**: All elements below diagonal are zero
**Lower triangular**: All elements above diagonal are zero

\`\`\`python
print("\\n=== Triangular Matrices ===")

# Upper triangular
U = np.array([[1, 2, 3],
             [0, 4, 5],
             [0, 0, 6]])

# Lower triangular
L = np.array([[1, 0, 0],
             [2, 3, 0],
             [4, 5, 6]])

print("Upper triangular U:")
print(U)
print("\\nLower triangular L:")
print(L)
print()

# Determinant is product of diagonal elements
det_U = np.linalg.det(U)
det_U_diag = U[0,0] * U[1,1] * U[2,2]

print(f"det(U) = {det_U:.4f}")
print(f"Product of diagonal = {det_U_diag:.4f}")
\`\`\`

**Properties**:
- Determinant = product of diagonal elements
- Solving **Ux** = **b** is fast (back substitution)
- Solving **Lx** = **b** is fast (forward substitution)

### Applications in ML

1. **LU decomposition**: Factor **A** = **LU**
2. **Cholesky decomposition**: **A** = **LL**ᵀ for positive definite matrices
3. **Solving linear systems**: Much faster than generic methods

\`\`\`python
# Solving triangular system (fast!)
print("\\n=== Solving Triangular Systems ===")

b = np.array([14, 29, 66])

# Solve Ux = b using back substitution
x_U = np.linalg.solve(U, b)
print(f"Solving Ux = b:")
print(f"x = {x_U}")
print(f"Verification Ux = {U @ x_U}")

# Solve Lx = b using forward substitution  
x_L = np.linalg.solve(L, b)
print(f"\\nSolving Lx = b:")
print(f"x = {x_L}")
print(f"Verification Lx = {L @ x_L}")
\`\`\`

## Sparse Matrices

**Sparse matrices** have mostly zero elements. Instead of storing all elements, store only non-zero values and their positions.

\`\`\`python
from scipy import sparse

print("\\n=== Sparse Matrices ===")

# Create a large sparse matrix
n = 1000
# Most elements are zero
dense_matrix = np.eye(n)
dense_matrix[0, n-1] = 1
dense_matrix[n-1, 0] = 1

print(f"Dense matrix size: {dense_matrix.nbytes / 1024:.2f} KB")
print(f"Number of non-zeros: {np.count_nonzero(dense_matrix)}")
print()

# Convert to sparse (CSR format - Compressed Sparse Row)
sparse_matrix = sparse.csr_matrix(dense_matrix)

print(f"Sparse matrix size: {(sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes) / 1024:.2f} KB")
print(f"Space saving: {100 * (1 - sparse_matrix.data.nbytes / dense_matrix.nbytes):.1f}%")
print()

# Operations on sparse matrices
v = np.ones(n)
result_dense = dense_matrix @ v
result_sparse = sparse_matrix @ v

print(f"Results equal: {np.allclose(result_dense, result_sparse)}")
print(f"Result (first 5 elements): {result_sparse[:5]}")
\`\`\`

### Sparse Matrix Formats

\`\`\`python
print("\\n=== Sparse Matrix Formats ===")

# Small example for illustration
data_array = np.array([
    [0, 0, 3, 0],
    [1, 0, 0, 4],
    [0, 2, 0, 0]
])

print("Dense matrix:")
print(data_array)
print()

# COO (Coordinate) format - good for construction
coo = sparse.coo_matrix(data_array)
print("COO format:")
print(f"  Data: {coo.data}")
print(f"  Rows: {coo.row}")
print(f"  Cols: {coo.col}")
print()

# CSR (Compressed Sparse Row) - good for arithmetic and row slicing
csr = sparse.csr_matrix(data_array)
print("CSR format:")
print(f"  Data: {csr.data}")
print(f"  Indices: {csr.indices}")
print(f"  Indptr: {csr.indptr}")
print()

# CSC (Compressed Sparse Column) - good for column slicing
csc = sparse.csc_matrix(data_array)
print("CSC format:")
print(f"  Data: {csc.data}")
print(f"  Indices: {csc.indices}")
print(f"  Indptr: {csc.indptr}")
\`\`\`

### Applications in ML

1. **Text data**: Document-term matrices (mostly zeros)
2. **Recommender systems**: User-item interaction matrices
3. **Graph neural networks**: Adjacency matrices
4. **Feature engineering**: One-hot encoded categorical variables

\`\`\`python
# Example: Text document-term matrix
print("\\n=== Text Example (Sparse) ===")

from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "machine learning is great",
    "deep learning is powerful",
    "machine learning uses statistics",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print(f"Document-term matrix shape: {X.shape}")
print(f"Sparsity: {100 * (1 - X.nnz / (X.shape[0] * X.shape[1])):.1f}% zeros")
print(f"Type: {type(X)}")
print("\\nMatrix (dense view):")
print(X.toarray())
print("\\nVocabulary:")
print(vectorizer.get_feature_names_out())
\`\`\`

## Positive Definite Matrices

A symmetric matrix **A** is **positive definite** if:
- **x**ᵀ**Ax** > 0 for all non-zero vectors **x**
- All eigenvalues are positive
- **A** = **BB**ᵀ for some matrix **B** (Cholesky decomposition)

\`\`\`python
print("\\n=== Positive Definite Matrices ===")

# Create positive definite matrix
B = np.random.randn(3, 3)
A_pd = B.T @ B  # This construction guarantees positive definite

print("Positive definite matrix A:")
print(A_pd)
print()

# Check eigenvalues (all should be positive)
eigenvalues = np.linalg.eigvalsh(A_pd)
print(f"Eigenvalues: {eigenvalues}")
print(f"All positive: {np.all(eigenvalues > 0)}")
print()

# Test definition: xᵀAx > 0
x = np.random.randn(3)
xAx = x.T @ A_pd @ x
print(f"For random x: xᵀAx = {xAx:.4f} > 0: {xAx > 0}")
\`\`\`

### Applications in ML

1. **Covariance matrices**: Always positive semi-definite (or positive definite if full rank)
2. **Kernel matrices**: In kernel methods (SVM, Gaussian processes)
3. **Hessian matrices**: At local minima in convex optimization
4. **Guarantee unique solution**: For **Ax** = **b**

## Comparing Special Matrices

\`\`\`python
print("\\n=== Matrix Properties Comparison ===")

properties = {
    'Diagonal': {'Fast multiply': True, 'Fast inverse': True, 'Real eigenvalues': True},
    'Symmetric': {'Fast multiply': False, 'Fast inverse': False, 'Real eigenvalues': True},
    'Orthogonal': {'Fast multiply': False, 'Fast inverse': True, 'Real eigenvalues': False},
    'Triangular': {'Fast multiply': False, 'Fast inverse': True, 'Real eigenvalues': True},
    'Sparse': {'Fast multiply': True, 'Fast inverse': False, 'Real eigenvalues': False},
}

import pandas as pd
df = pd.DataFrame(properties).T
print(df)
\`\`\`

## Summary

Special matrices optimize computation and provide theoretical guarantees:

1. **Diagonal**: Fastest operations, feature scaling
2. **Symmetric**: Real eigenvalues, covariance matrices, optimization
3. **Orthogonal**: Preserve lengths and angles, rotations, SVD/QR
4. **Triangular**: Fast system solving, LU/Cholesky decompositions
5. **Sparse**: Memory efficient, text/graph data
6. **Positive Definite**: Convex optimization, covariance, kernels

**Key Insights**:
- Recognize matrix structure to choose optimal algorithms
- Special matrices often guarantee certain properties (e.g., real eigenvalues)
- Sparse matrices are essential for large-scale ML
- Many ML matrices have special structure (covariance is symmetric, adjacency is sparse)

**Performance**:
- Diagonal matrix multiply: O(n) vs O(n³) for general matrices
- Orthogonal matrix inverse: O(n) (just transpose) vs O(n³)
- Sparse operations: O(nnz) where nnz << n²

Recognizing and exploiting special matrix structure is a key skill for efficient ML implementations!
`,
      multipleChoice: [
        {
          id: 'special-mat-q1',
          question:
            'What is the computational advantage of multiplying a diagonal matrix D by a vector v compared to a general matrix?',
          options: [
            'No advantage, both are O(n²)',
            'Diagonal is O(n) instead of O(n²)',
            'Diagonal is O(log n) instead of O(n²)',
            'Diagonal uses less memory',
          ],
          correctAnswer: 1,
          explanation:
            'Diagonal matrix-vector multiplication is O(n) because you only multiply each component: (Dv)ᵢ = dᵢvᵢ. General matrix-vector multiplication is O(n²) (n elements per row, n rows).',
        },
        {
          id: 'special-mat-q2',
          question:
            'For an orthogonal matrix Q, what is the relationship between Q⁻¹ and Qᵀ?',
          options: [
            'Q⁻¹ = -Qᵀ',
            'Q⁻¹ = Qᵀ',
            'Q⁻¹ = 2Qᵀ',
            'No special relationship',
          ],
          correctAnswer: 1,
          explanation:
            'For orthogonal matrices, the inverse equals the transpose: Q⁻¹ = Qᵀ. This makes computing the inverse trivial (just transpose) and very fast.',
        },
        {
          id: 'special-mat-q3',
          question: 'Why are covariance matrices always symmetric?',
          options: [
            'They are not always symmetric',
            'Because covariance is commutative: Cov(X,Y) = Cov(Y,X)',
            'By mathematical convention only',
            'Because they are diagonal',
          ],
          correctAnswer: 1,
          explanation:
            'Covariance is symmetric: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] = E[(Y-μᵧ)(X-μₓ)] = Cov(Y,X). Therefore, the covariance matrix where element (i,j) is Cov(Xᵢ,Xⱼ) must be symmetric.',
        },
        {
          id: 'special-mat-q4',
          question:
            'When working with text data (document-term matrices), what matrix format is most memory efficient?',
          options: [
            'Dense matrix',
            'Diagonal matrix',
            'Sparse matrix',
            'Symmetric matrix',
          ],
          correctAnswer: 2,
          explanation:
            'Text data is naturally sparse—most documents only contain a small fraction of the total vocabulary. Sparse matrix formats (CSR/CSC) store only non-zero elements, saving massive amounts of memory.',
        },
        {
          id: 'special-mat-q5',
          question:
            'Which property guarantees that a symmetric matrix has all real eigenvalues?',
          options: [
            'The determinant is non-zero',
            'It is a fundamental theorem (spectral theorem)',
            'The matrix is invertible',
            'The diagonal elements are positive',
          ],
          correctAnswer: 1,
          explanation:
            'The spectral theorem states that real symmetric matrices have all real eigenvalues and orthogonal eigenvectors. This is a fundamental mathematical result, not dependent on other properties like invertibility.',
        },
      ],
      quiz: [
        {
          id: 'special-mat-d1',
          question:
            'Explain why orthogonal matrices preserve lengths and angles. How is this property useful in machine learning, particularly in dimensionality reduction and feature extraction?',
          sampleAnswer:
            'Orthogonal matrices preserve lengths because ||Qv||² = (Qv)ᵀ(Qv) = vᵀQᵀQv = vᵀIv = vᵀv = ||v||². They preserve angles because the dot product is preserved: (Qu)·(Qv) = (Qu)ᵀ(Qv) = uᵀQᵀQv = uᵀv = u·v, and angle is determined by dot product via cos(θ) = (u·v)/(||u||||v||). This property is crucial in ML because it means orthogonal transformations don\'t distort data—they only rotate or reflect it. In PCA, the principal components form an orthonormal basis, so projecting data onto these components preserves relative distances and angles, just in a lower-dimensional space. This ensures no artificial distortion is introduced by the dimensionality reduction. In SVD, the U and V matrices are orthogonal, guaranteeing that the decomposition preserves geometric relationships. In neural networks, some weight initialization schemes use orthogonal matrices to avoid vanishing/exploding gradients. Whitening transformations (decorrelating features) use orthogonal matrices to rotate data into independent components. The key insight: orthogonal transformations are "shape-preserving" transformations that change coordinates without changing the intrinsic geometry of the data, making them ideal for basis changes and feature transformations where we want to maintain data structure.',
          keyPoints: [
            'Orthogonal matrices (QᵀQ=I) preserve lengths and angles: no data distortion',
            'PCA uses orthonormal basis: dimensionality reduction without geometric distortion',
            'Shape-preserving: ideal for basis changes, feature transformations, whitening',
          ],
        },
        {
          id: 'special-mat-d2',
          question:
            'Compare dense versus sparse matrix storage and operations. When should you use each in machine learning, and what are the trade-offs?',
          sampleAnswer:
            "Dense matrices store all n² elements explicitly, while sparse matrices store only non-zero elements (typically many fewer). Trade-offs: Memory: Dense requires O(n²) space, sparse requires O(nnz) where nnz is the number of non-zeros. For a 10,000×10,000 matrix with 0.1% non-zeros, sparse uses 100MB vs dense's 800MB. Speed: For sparse matrices with sparsity s (fraction of zeros), operations are O(nnz) instead of O(n²). However, there's overhead in accessing non-contiguous memory. Operations: Dense supports all operations naturally. Sparse formats (CSR, CSC, COO) optimize different operations—CSR is fast for row operations, CSC for columns, COO for construction. Use sparse when: (1) Data is naturally sparse (>90% zeros): text (document-term matrices), graphs (adjacency matrices), recommender systems (user-item matrices), one-hot encoded features. (2) Scaling to large problems: a million-dimensional sparse vector fits in memory, dense doesn't. Use dense when: (1) Data is dense (<50% zeros): images, audio, embeddings. (2) Need fast, general operations without format constraints. (3) Using GPUs (dense operations more optimized). In practice: Start with dense for simplicity. Switch to sparse if memory/speed becomes an issue. Libraries like scikit-learn support both transparently. Know your data's sparsity: use df.values if dense, use sparse formats for text/graph. Modern deep learning mostly uses dense (images, embeddings) but transformers are exploring sparse attention. The key: sparse is essential for scaling to high-dimensional sparse data (text, graphs), but adds complexity—use only when necessary.",
          keyPoints: [
            'Dense: O(n²) storage, all operations; Sparse: O(nnz) storage, format-dependent ops',
            'Use sparse for >90% zeros (text, graphs, recommenders); dense for images/embeddings',
            'Trade-off: sparse saves memory/computation but adds complexity (CSR/CSC/COO formats)',
          ],
        },
        {
          id: 'special-mat-d3',
          question:
            'Positive definite matrices appear throughout machine learning (covariance matrices, kernels, Hessians at minima). Explain what positive definiteness means intuitively and why it provides useful guarantees in optimization and learning.',
          sampleAnswer:
            'Positive definiteness means xᵀAx > 0 for all non-zero vectors x. Intuitively, it means the quadratic form defined by A is always positive—imagine a bowl shape that curves upward in all directions, never downward or flat (except at the origin). Geometrically, A defines a distance metric that only equals zero at the origin. Mathematical equivalent conditions: all eigenvalues > 0, can be written as A = BᵀB. Why is this important in ML? (1) Covariance matrices: Var(aᵀX) = aᵀΣa where Σ is covariance. Variance must be non-negative, so Σ is positive semi-definite. If features are linearly independent, Σ is positive definite, meaning the distribution is non-degenerate (has spread in all principal directions). (2) Optimization: The Hessian at a local minimum must be positive definite—it curves upward in all directions. This guarantees the minimum is strict and unique (locally). In convex optimization, positive definite Hessian everywhere means globally convex. (3) Kernels: Valid kernel functions produce positive definite kernel matrices K, which guarantees a unique solution in kernel methods like SVMs and ensures the optimization problem is convex. (4) Numerical stability: Positive definite matrices are invertible (no zero eigenvalues) and well-conditioned (eigenvalues bounded away from zero), making numerical algorithms stable. (5) Cholesky decomposition: Only positive definite matrices have Cholesky decomposition A = LLᵀ, a fast and stable way to solve systems and generate multivariate Gaussian samples. In summary: positive definiteness provides convexity guarantees (unique solutions), stability guarantees (invertible, well-conditioned), and enables efficient algorithms (Cholesky). Recognizing when matrices are positive definite allows using specialized, faster, and more stable algorithms.',
          keyPoints: [
            'Positive definite: xᵀAx > 0 (bowl-shaped), all eigenvalues > 0, invertible',
            'Guarantees: covariance non-degenerate, Hessian at minima convex, kernels valid',
            'Enables: Cholesky decomposition, numerical stability, unique solutions',
          ],
        },
      ],
    },

    {
      id: 'matrix-inverse-determinants',
      title: 'Matrix Inverse & Determinants',
      content: `
# Matrix Inverse & Determinants

## Introduction

The **determinant** and **inverse** are fundamental concepts in linear algebra with deep geometric interpretations and critical applications in machine learning. Understanding them helps you solve systems of equations, analyze transformations, and diagnose numerical stability issues.

## Determinant

The **determinant** is a scalar value that encodes important properties of a square matrix.

### 2×2 Determinant

For a 2×2 matrix:

**A** = ⎡a  b⎤
      ⎣c  d⎦

det(**A**) = |**A**| = ad - bc

### 3×3 Determinant

For a 3×3 matrix, use cofactor expansion:

**A** = ⎡a  b  c⎤
      ⎢d  e  f⎥
      ⎣g  h  i⎦

det(**A**) = a(ei - fh) - b(di - fg) + c(dh - eg)

### General Case

For larger matrices, use recursive cofactor expansion or more efficient algorithms (LU decomposition).

\`\`\`python
import numpy as np

print("=== Determinants ===")

# 2x2 matrix
A_2x2 = np.array([[3, 2],
                  [1, 4]])

det_2x2 = np.linalg.det(A_2x2)
det_2x2_manual = A_2x2[0,0]*A_2x2[1,1] - A_2x2[0,1]*A_2x2[1,0]

print("2×2 Matrix A:")
print(A_2x2)
print(f"\\ndet(A) = {det_2x2:.4f}")
print(f"det(A) manual = {A_2x2[0,0]}*{A_2x2[1,1]} - {A_2x2[0,1]}*{A_2x2[1,0]} = {det_2x2_manual:.4f}")
print()

# 3x3 matrix
A_3x3 = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 10]])

det_3x3 = np.linalg.det(A_3x3)

print("3×3 Matrix A:")
print(A_3x3)
print(f"det(A) = {det_3x3:.4f}")
\`\`\`

### Properties of Determinants

\`\`\`python
print("\\n=== Determinant Properties ===")

A = np.array([[2, 1], [3, 4]])
B = np.array([[1, 2], [0, 3]])

print("Matrix A:")
print(A)
print(f"det(A) = {np.linalg.det(A):.4f}")
print()

print("Matrix B:")
print(B)
print(f"det(B) = {np.linalg.det(B):.4f}")
print()

# Property 1: det(AB) = det(A)det(B)
AB = A @ B
det_A = np.linalg.det(A)
det_B = np.linalg.det(B)
det_AB = np.linalg.det(AB)

print(f"det(A) * det(B) = {det_A:.4f} * {det_B:.4f} = {det_A * det_B:.4f}")
print(f"det(AB) = {det_AB:.4f}")
print(f"Equal: {np.isclose(det_A * det_B, det_AB)}")
print()

# Property 2: det(Aᵀ) = det(A)
det_A_T = np.linalg.det(A.T)
print(f"det(A) = {det_A:.4f}")
print(f"det(Aᵀ) = {det_A_T:.4f}")
print(f"Equal: {np.isclose(det_A, det_A_T)}")
print()

# Property 3: det(cA) = c^n det(A) for n×n matrix
c = 2
cA = c * A
det_cA = np.linalg.det(cA)
n = A.shape[0]
expected = c**n * det_A

print(f"For scalar c={c} and {n}×{n} matrix A:")
print(f"det(cA) = {det_cA:.4f}")
print(f"c^n * det(A) = {c}^{n} * {det_A:.4f} = {expected:.4f}")
print(f"Equal: {np.isclose(det_cA, expected)}")
\`\`\`

### Geometric Interpretation

The **absolute value** of the determinant represents:
- The **volume scaling factor** of the transformation
- In 2D: Area scaling factor
- In 3D: Volume scaling factor

**Sign** of determinant:
- Positive: Orientation preserved
- Negative: Orientation reversed (reflection)
- Zero: Collapses space (singular, not invertible)

\`\`\`python
import matplotlib.pyplot as plt

print("\\n=== Geometric Interpretation ===")

# Unit square
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

# Transformation matrix
A_transform = np.array([[2, 0.5],
                        [0.5, 1.5]])

transformed_square = A_transform @ square

# Determinant = area scaling factor
det_transform = np.linalg.det(A_transform)
original_area = 1.0  # Unit square
transformed_area = det_transform * original_area

print("Transformation matrix:")
print(A_transform)
print(f"\\ndet(A) = {det_transform:.4f}")
print(f"Original area: {original_area}")
print(f"Transformed area: {transformed_area:.4f}")
print(f"Area scaled by factor of {det_transform:.4f}")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(square[0, :], square[1, :], 'b-', linewidth=2)
plt.fill(square[0, :], square[1, :], alpha=0.3)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-0.5, 3)
plt.ylim(-0.5, 3)
plt.title(f'Original Square (Area = {original_area})')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.plot(transformed_square[0, :], transformed_square[1, :], 'r-', linewidth=2)
plt.fill(transformed_square[0, :], transformed_square[1, :], alpha=0.3, color='red')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-0.5, 3)
plt.ylim(-0.5, 3)
plt.title(f'Transformed (Area = {transformed_area:.2f}, det = {det_transform:.2f})')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
\`\`\`

### Singular Matrices (Determinant = 0)

When det(**A**) = 0, the matrix is **singular** (not invertible):
- Collapses space to lower dimension
- Rows/columns are linearly dependent
- No unique solutions to **Ax** = **b**

\`\`\`python
print("\\n=== Singular Matrix ===")

# Singular matrix (second row is 2× first row)
A_singular = np.array([[1, 2],
                       [2, 4]])

det_singular = np.linalg.det(A_singular)

print("Singular matrix A:")
print(A_singular)
print(f"det(A) = {det_singular:.10f}")
print(f"Is singular: {np.isclose(det_singular, 0)}")
print()

# Transform unit square (collapses to line)
transformed_singular = A_singular @ square

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(square[0, :], square[1, :], 'b-', linewidth=2)
plt.fill(square[0, :], square[1, :], alpha=0.3)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-0.5, 5)
plt.ylim(-0.5, 5)
plt.title('Original Square')

plt.subplot(1, 2, 2)
plt.plot(transformed_singular[0, :], transformed_singular[1, :], 'r-', linewidth=3)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-0.5, 5)
plt.ylim(-0.5, 5)
plt.title('Collapsed to Line (det = 0)')

plt.tight_layout()
plt.show()

print("Note: The square collapsed to a line!")
\`\`\`

## Matrix Inverse

The **inverse** of a matrix **A** is **A⁻¹** such that:

**AA⁻¹** = **A⁻¹A** = **I**

### When Does an Inverse Exist?

A matrix is **invertible** (non-singular) if and only if:
- det(**A**) ≠ 0
- Columns are linearly independent
- Rows are linearly independent
- **A** has full rank

### 2×2 Inverse Formula

For 2×2 matrix:

**A** = ⎡a  b⎤
      ⎣c  d⎦

**A⁻¹** = (1/det(A)) ⎡d  -b⎤
                     ⎣-c   a⎦

\`\`\`python
print("\\n=== Matrix Inverse ===")

A = np.array([[3, 2],
              [1, 4]])

# Compute inverse
A_inv = np.linalg.inv(A)

print("Matrix A:")
print(A)
print("\\nInverse A⁻¹:")
print(A_inv)
print()

# Verify AA⁻¹ = I
I_check = A @ A_inv
print("A @ A⁻¹:")
print(I_check)
print(f"Is identity: {np.allclose(I_check, np.eye(2))}")
print()

# Manual 2×2 inverse
det_A = np.linalg.det(A)
A_inv_manual = (1/det_A) * np.array([[A[1,1], -A[0,1]],
                                     [-A[1,0], A[0,0]]])

print("A⁻¹ (manual calculation):")
print(A_inv_manual)
print(f"Matches numpy: {np.allclose(A_inv, A_inv_manual)}")
\`\`\`

### Properties of Inverse

\`\`\`python
print("\\n=== Inverse Properties ===")

A = np.array([[2, 1], [1, 3]])
B = np.array([[1, 2], [0, 1]])

A_inv = np.linalg.inv(A)
B_inv = np.linalg.inv(B)

# Property 1: (A⁻¹)⁻¹ = A
A_inv_inv = np.linalg.inv(A_inv)
print("(A⁻¹)⁻¹ = A:")
print(f"Equal: {np.allclose(A_inv_inv, A)}")
print()

# Property 2: (AB)⁻¹ = B⁻¹A⁻¹ (order reverses!)
AB = A @ B
AB_inv = np.linalg.inv(AB)
B_inv_A_inv = B_inv @ A_inv

print("(AB)⁻¹ = B⁻¹A⁻¹ (note the reversal!):")
print(f"Equal: {np.allclose(AB_inv, B_inv_A_inv)}")
print()

# Property 3: (Aᵀ)⁻¹ = (A⁻¹)ᵀ
A_T_inv = np.linalg.inv(A.T)
A_inv_T = A_inv.T

print("(Aᵀ)⁻¹ = (A⁻¹)ᵀ:")
print(f"Equal: {np.allclose(A_T_inv, A_inv_T)}")
print()

# Property 4: det(A⁻¹) = 1/det(A)
det_A = np.linalg.det(A)
det_A_inv = np.linalg.det(A_inv)

print(f"det(A) = {det_A:.4f}")
print(f"det(A⁻¹) = {det_A_inv:.4f}")
print(f"1/det(A) = {1/det_A:.4f}")
print(f"Equal: {np.isclose(det_A_inv, 1/det_A)}")
\`\`\`

### Geometric Interpretation

The inverse transformation **undoes** the original transformation:
- **A** transforms **x** to **y** = **Ax**
- **A⁻¹** transforms **y** back to **x** = **A⁻¹y**

\`\`\`python
print("\\n=== Inverse as Undo Operation ===")

# Original vector
x = np.array([1, 2])

# Transform
A = np.array([[2, 1], [0, 3]])
y = A @ x

# Inverse transform (undo)
A_inv = np.linalg.inv(A)
x_recovered = A_inv @ y

print(f"Original x: {x}")
print(f"Transformed y = Ax: {y}")
print(f"Recovered x = A⁻¹y: {x_recovered}")
print(f"Recovered correctly: {np.allclose(x, x_recovered)}")
\`\`\`

## Applications in Machine Learning

### 1. Solving Linear Systems

Instead of solving **Ax** = **b** directly, use **x** = **A⁻¹b**:

\`\`\`python
print("\\n=== Solving Linear Systems ===")

# System: Ax = b
A = np.array([[3, 2],
              [1, 4]])
b = np.array([7, 13])

# Method 1: Using inverse (not recommended in practice)
A_inv = np.linalg.inv(A)
x_inv = A_inv @ b

# Method 2: Using solver (more stable)
x_solve = np.linalg.solve(A, b)

print("System: Ax = b")
print("A:")
print(A)
print(f"b: {b}")
print()

print(f"Solution (using inverse): x = {x_inv}")
print(f"Solution (using solver): x = {x_solve}")
print(f"Verification Ax = {A @ x_solve}")
print()

print("Note: np.linalg.solve() is more numerically stable!")
\`\`\`

### 2. Computing Covariance Matrix Inverse

In Gaussian distributions and many ML algorithms:

\`\`\`python
print("\\n=== Covariance Matrix Inverse ===")

# Generate data
np.random.seed(42)
data = np.random.randn(100, 3)

# Covariance matrix
cov = np.cov(data.T)
cov_inv = np.linalg.inv(cov)

print("Covariance matrix Σ:")
print(cov.round(3))
print("\\nInverse covariance Σ⁻¹ (precision matrix):")
print(cov_inv.round(3))
print()

# Used in Mahalanobis distance
x = np.array([1, 0, -1])
mean = np.zeros(3)

# Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
diff = x - mean
mahal_dist = np.sqrt(diff @ cov_inv @ diff)

print(f"Point: {x}")
print(f"Mahalanobis distance: {mahal_dist:.4f}")
print("(Used in anomaly detection, Gaussian processes)")
\`\`\`

### 3. Pseudo-inverse for Non-square Matrices

For non-square or singular matrices, use the **Moore-Penrose pseudo-inverse**:

\`\`\`python
print("\\n=== Pseudo-inverse ===")

# Non-square matrix (more rows than columns)
A_rect = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])

# Pseudo-inverse
A_pinv = np.linalg.pinv(A_rect)

print(f"A shape: {A_rect.shape}")
print("A:")
print(A_rect)
print(f"\\nA⁺ (pseudo-inverse) shape: {A_pinv.shape}")
print("A⁺:")
print(A_pinv)
print()

# Property: AA⁺A = A
AApinvA = A_rect @ A_pinv @ A_rect
print("AA⁺A = A:")
print(f"Equal: {np.allclose(AApinvA, A_rect)}")
print()

# Use in least squares
b = np.array([1, 2, 3])
x_ls = A_pinv @ b

print(f"\\nLeast squares solution x = A⁺b:")
print(f"x = {x_ls}")
print(f"Ax ≈ b: {A_rect @ x_ls}")
print(f"b: {b}")
print(f"Residual: {np.linalg.norm(A_rect @ x_ls - b):.6f}")
\`\`\`

## Numerical Considerations

### Condition Number

The **condition number** measures how sensitive a matrix is to numerical errors:

κ(**A**) = ||**A**|| · ||**A⁻¹**||

- κ = 1: Perfectly conditioned (e.g., identity matrix)
- κ large: Ill-conditioned (small changes → big errors)
- κ = ∞: Singular (not invertible)

\`\`\`python
print("\\n=== Condition Number ===")

# Well-conditioned matrix
A_good = np.eye(3)
cond_good = np.linalg.cond(A_good)

print("Well-conditioned (identity):")
print(f"κ(I) = {cond_good:.4f}")
print()

# Ill-conditioned matrix
A_bad = np.array([[1, 1],
                  [1, 1.0001]])
cond_bad = np.linalg.cond(A_bad)

print("Ill-conditioned:")
print(A_bad)
print(f"κ(A) = {cond_bad:.4f}")
print()

# Nearly singular
A_nearly_singular = np.array([[1, 1],
                              [1, 1.000001]])
cond_nearly_singular = np.linalg.cond(A_nearly_singular)

print("Nearly singular:")
print(f"κ(A) = {cond_nearly_singular:.4f}")
print("Very large condition number → numerical instability!")
\`\`\`

### Best Practices

\`\`\`python
print("\\n=== Best Practices ===")

# DON'T: Compute inverse explicitly
A = np.random.randn(100, 100)
b = np.random.randn(100)

# Slow and less stable
import time
start = time.time()
A_inv = np.linalg.inv(A)
x_bad = A_inv @ b
time_inv = time.time() - start

# DO: Use solver
start = time.time()
x_good = np.linalg.solve(A, b)
time_solve = time.time() - start

print(f"Using inverse: {time_inv:.6f}s")
print(f"Using solver: {time_solve:.6f}s")
print(f"Solver is {time_inv/time_solve:.2f}x faster")
print()
print(f"Solutions equal: {np.allclose(x_bad, x_good)}")
print()

print("Remember: Never compute A⁻¹ explicitly unless you need it!")
print("Use np.linalg.solve(A, b) instead of np.linalg.inv(A) @ b")
\`\`\`

## Summary

**Determinant**:
- Scalar that encodes matrix properties
- Geometric: Volume scaling factor
- det(**A**) = 0 ↔ singular (not invertible)
- det(**AB**) = det(**A**)·det(**B**)

**Inverse**:
- **AA⁻¹** = **I**
- Exists iff det(**A**) ≠ 0
- Geometric: Undoes transformation
- (**AB**)⁻¹ = **B⁻¹A⁻¹** (order reverses!)

**Applications in ML**:
- Solving linear systems
- Computing Mahalanobis distance
- Covariance inverse (precision matrix)
- Least squares (pseudo-inverse)

**Best Practices**:
- Don't compute inverse explicitly—use \`np.linalg.solve()\`
- Check condition number for numerical stability
- Use pseudo-inverse for non-square/singular matrices
- Be aware of computational cost: O(n³)

Understanding determinants and inverses is essential for linear algebra in ML!
`,
      multipleChoice: [
        {
          id: 'inv-det-q1',
          question: 'What does a determinant of zero indicate about a matrix?',
          options: [
            'The matrix is orthogonal',
            'The matrix is symmetric',
            'The matrix is singular (not invertible)',
            'The matrix is positive definite',
          ],
          correctAnswer: 2,
          explanation:
            'A determinant of zero means the matrix is singular—it collapses space to a lower dimension and is not invertible. The columns/rows are linearly dependent.',
        },
        {
          id: 'inv-det-q2',
          question: 'If det(A) = 3 and det(B) = 4, what is det(AB)?',
          options: ['7', '12', '1', '0.75'],
          correctAnswer: 1,
          explanation:
            'det(AB) = det(A) × det(B) = 3 × 4 = 12. This is a fundamental property of determinants.',
        },
        {
          id: 'inv-det-q3',
          question: 'What is the relationship between (AB)⁻¹, A⁻¹, and B⁻¹?',
          options: [
            '(AB)⁻¹ = A⁻¹B⁻¹',
            '(AB)⁻¹ = B⁻¹A⁻¹',
            '(AB)⁻¹ = A⁻¹ + B⁻¹',
            '(AB)⁻¹ = (A⁻¹)(B⁻¹)/2',
          ],
          correctAnswer: 1,
          explanation:
            '(AB)⁻¹ = B⁻¹A⁻¹. The order reverses! This is because (AB)(B⁻¹A⁻¹) = A(BB⁻¹)A⁻¹ = AIA⁻¹ = AA⁻¹ = I.',
        },
        {
          id: 'inv-det-q4',
          question:
            'Why should you use np.linalg.solve(A, b) instead of np.linalg.inv(A) @ b to solve Ax = b?',
          options: [
            'It produces different results',
            'It is more numerically stable and faster',
            "The inverse method doesn't work",
            'It uses less memory',
          ],
          correctAnswer: 1,
          explanation:
            'np.linalg.solve() is both faster and more numerically stable. It uses specialized algorithms (like LU decomposition) that avoid explicitly computing the inverse, which accumulates rounding errors and is computationally expensive.',
        },
        {
          id: 'inv-det-q5',
          question:
            'Geometrically, what does the absolute value of a determinant represent for a 2D transformation?',
          options: [
            'The angle of rotation',
            'The area scaling factor',
            'The direction of transformation',
            'The eigenvalues',
          ],
          correctAnswer: 1,
          explanation:
            "The absolute value |det(A)| is the area scaling factor—how much the transformation stretches or shrinks areas. In 3D, it's the volume scaling factor. The sign indicates whether orientation is preserved (+) or reversed (-).",
        },
      ],
      quiz: [
        {
          id: 'inv-det-d1',
          question:
            'Explain the geometric interpretation of a singular matrix (determinant = 0). What happens to space under this transformation, and why does this mean the matrix has no inverse?',
          sampleAnswer:
            "A singular matrix (det = 0) collapses space to a lower dimension. In 2D, it might collapse the plane to a line or point. In 3D, it might collapse 3D space to a plane, line, or point. Geometrically, imagine a transformation that squashes a square into a line—the area becomes zero, hence determinant = 0. This transformation loses information: multiple points map to the same output point (e.g., the entire square becomes a single line). An inverse would need to \"un-collapse\" this—to take a line back to a square—but this is impossible because we've lost the information about which point on the line came from which point in the square. Mathematically, the columns of a singular matrix are linearly dependent: one column is a combination of others. This means the transformation doesn't span the full output space—it only reaches a lower-dimensional subspace. Since the transformation isn't onto (doesn't cover all output space), it can't be inverted. In ML context: if your data matrix is singular, features are redundant (linearly dependent), and you can't solve certain systems uniquely. This is why we check rank and condition number—nearly singular matrices cause numerical instability even if not exactly singular.",
          keyPoints: [
            'det=0: space collapses to lower dimension, transformation loses information',
            'Columns linearly dependent: transformation not onto full output space',
            "ML impact: redundant features, can't solve systems uniquely, numerical instability",
          ],
        },
        {
          id: 'inv-det-d2',
          question:
            'The property (AB)⁻¹ = B⁻¹A⁻¹ shows that order reverses when inverting a product. Explain why this must be true using the concept of transformations "undoing" each other.',
          sampleAnswer:
            "Think of matrix multiplication as composing transformations applied right to left: AB means \"first apply B, then apply A.\" To undo this composition, we must undo the operations in reverse order: first undo A, then undo B. This is like putting on socks then shoes—to reverse it, you remove shoes first (undo A), then socks (undo B). Formally: suppose we want to verify (AB)(B⁻¹A⁻¹) = I. Expanding: (AB)(B⁻¹A⁻¹) = A(BB⁻¹)A⁻¹ = AIA⁻¹ = AA⁻¹ = I. The middle BB⁻¹ = I cancels out, leaving AA⁻¹ = I. If we tried (AB)(A⁻¹B⁻¹) instead, we'd get (AB)(A⁻¹B⁻¹) = A(BA⁻¹)B⁻¹, and BA⁻¹ ≠ I in general (matrices don't commute), so this doesn't work. In ML: this appears in backpropagation through composed layers. If forward pass is y = Layer2(Layer1(x)), the backward pass must go through Layer2's gradient first, then Layer1's—the reverse order. Understanding this reversal is crucial for implementing custom neural network layers and deriving gradients correctly. It also explains why (ABC)⁻¹ = C⁻¹B⁻¹A⁻¹—complete reversal for any number of matrices.",
          keyPoints: [
            '(AB)⁻¹ = B⁻¹A⁻¹: undo operations in reverse order (like socks then shoes)',
            'Proof: (AB)(B⁻¹A⁻¹) = A(BB⁻¹)A⁻¹ = AIA⁻¹ = AA⁻¹ = I',
            'ML: backpropagation through layers undoes forward pass in reverse order',
          ],
        },
        {
          id: 'inv-det-d3',
          question:
            'Discuss the condition number of a matrix and its importance in machine learning. What problems arise with ill-conditioned matrices, and how can you detect and mitigate them?',
          sampleAnswer:
            'The condition number κ(A) = ||A|| ||A⁻¹|| measures sensitivity to numerical errors. Small condition number (close to 1): well-conditioned, stable. Large condition number: ill-conditioned, small input changes or rounding errors cause large output changes. Why it matters: Computers use finite precision (typically 64-bit floats with ~15 significant digits). With κ(A) ≈ 10^k, you lose about k digits of precision. If κ ≈ 10^10, you lose 10 digits, leaving only ~5 accurate digits. This manifests as: (1) Solving Ax=b becomes inaccurate, (2) Gradient computations in optimization become unreliable, (3) Eigenvalue/SVD calculations may be wrong. Common causes in ML: (1) Features with vastly different scales (e.g., age in years vs income in dollars—scale difference of ~1000x), (2) Highly correlated features (multicollinearity), (3) Near-duplicate rows/columns in data matrix, (4) Using raw polynomials without orthogonalization. Detection: Check np.linalg.cond(A). Rule of thumb: κ > 10^10 is problematic, κ > 10^15 is critical. Mitigation: (1) Feature scaling/standardization: Scale all features to similar ranges (StandardScaler), (2) Regularization: Add λI to covariance matrix (Ridge regression), (3) PCA: Remove correlated dimensions, (4) Higher precision: Use float128 if needed, (5) Specialized algorithms: Use QR decomposition instead of normal equations for least squares. In deep learning: proper weight initialization and batch normalization help keep condition numbers reasonable throughout training. Understanding condition numbers helps diagnose why a model might be numerically unstable or producing nonsensical results despite correct code.',
          keyPoints: [
            'κ(A) = ||A|| ||A⁻¹||: measures numerical stability (κ ≈ 10^k loses k digits precision)',
            'Causes: vastly different feature scales, correlated features, multicollinearity',
            'Mitigation: feature scaling, regularization (Ridge), PCA, QR decomposition',
          ],
        },
      ],
    },

    {
      id: 'systems-linear-equations',
      title: 'Systems of Linear Equations',
      content: `
# Systems of Linear Equations

## Introduction

A system of linear equations is a collection of linear equations involving the same set of variables. Solving these systems is fundamental to many machine learning algorithms, from linear regression to training neural networks.

**General form**:
a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ = b₁
a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ = b₂
...
aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ = bₘ

**Matrix form**: **Ax** = **b**

Where:
- **A**: m × n coefficient matrix
- **x**: n × 1 solution vector  
- **b**: m × 1 constant vector

## Types of Systems

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

print("=== Types of Linear Systems ===")

# Case 1: Unique solution (m = n, det(A) ≠ 0)
print("\\n1. UNIQUE SOLUTION (Square, non-singular)")
A1 = np.array([[2, 1], [1, 3]])
b1 = np.array([5, 7])
x1 = np.linalg.solve(A1, b1)

print(f"A: \\n{A1}")
print(f"b: {b1}")
print(f"Solution x: {x1}")
print(f"Verification Ax = {A1 @ x1}")
print(f"det(A) = {np.linalg.det(A1):.4f} ≠ 0 → unique solution")

# Case 2: No solution (inconsistent)
print("\\n2. NO SOLUTION (Inconsistent)")
A2 = np.array([[1, 2], [2, 4]])  # Singular
b2 = np.array([3, 7])  # Not in column space
print(f"A: \\n{A2}")
print(f"b: {b2}")
print(f"det(A) = {np.linalg.det(A2):.10f} ≈ 0")
print("Rows are linearly dependent but b is not in column space")
print("→ No solution exists")

# Case 3: Infinite solutions
print("\\n3. INFINITE SOLUTIONS")
A3 = np.array([[1, 2], [2, 4]])
b3 = np.array([3, 6])  # In column space (b3 = 2*b3[0])
print(f"A: \\n{A3}")
print(f"b: {b3}")
print("b is in the column space of A")
print("→ Infinitely many solutions")

# Case 4: Overdetermined (m > n)
print("\\n4. OVERDETERMINED (More equations than unknowns)")
A4 = np.array([[1, 2], [3, 4], [5, 6]])
b4 = np.array([1, 2, 3])
print(f"A shape: {A4.shape} (3 equations, 2 unknowns)")
print("Usually no exact solution → use least squares")
x4_ls = np.linalg.lstsq(A4, b4, rcond=None)[0]
print(f"Least squares solution: {x4_ls}")
print(f"Residual: {np.linalg.norm(A4 @ x4_ls - b4):.6f}")

# Case 5: Underdetermined (m < n)
print("\\n5. UNDERDETERMINED (More unknowns than equations)")
A5 = np.array([[1, 2, 3], [4, 5, 6]])
b5 = np.array([1, 2])
print(f"A shape: {A5.shape} (2 equations, 3 unknowns)")
print("Infinitely many solutions → find minimum norm solution")
x5_min = np.linalg.lstsq(A5, b5, rcond=None)[0]
print(f"Minimum norm solution: {x5_min}")
\`\`\`

## Gaussian Elimination

The fundamental algorithm for solving systems.

**Steps**:
1. **Forward elimination**: Convert to upper triangular form
2. **Back substitution**: Solve from bottom to top

\`\`\`python
print("\\n=== Gaussian Elimination ===")

def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination
    Returns solution x
    """
    n = len(b)
    # Create augmented matrix [A|b]
    Ab = np.column_stack([A.astype(float), b.astype(float)])
    
    print("Augmented matrix [A|b]:")
    print(Ab)
    print()
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        Ab[[i, max_row]] = Ab[[max_row, i]]  # Swap rows
        
        # Eliminate below
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
        
        print(f"After elimination step {i+1}:")
        print(Ab)
        print()
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x

# Example
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

print("Solving system:")
print(f"A:\\n{A}")
print(f"b: {b}")
print()

x_ge = gaussian_elimination(A.copy(), b.copy())
print(f"\\nSolution: {x_ge}")
print(f"Verification Ax = {A @ x_ge}")
print(f"b = {b}")
print(f"Match: {np.allclose(A @ x_ge, b)}")
\`\`\`

## LU Decomposition

Factor **A** = **LU** where **L** is lower triangular and **U** is upper triangular.

**Advantages**:
- Solve multiple systems with same **A** efficiently
- Numerically stable with partial pivoting
- Foundation for many algorithms

\`\`\`python
from scipy.linalg import lu

print("\\n=== LU Decomposition ===")

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

# Compute LU decomposition with partial pivoting
P, L, U = lu(A)

print("Original matrix A:")
print(A)
print("\\nPermutation matrix P:")
print(P)
print("\\nLower triangular L:")
print(L)
print("\\nUpper triangular U:")
print(U)
print()

# Verify PA = LU
print("Verification PA = LU:")
print(f"PA:\\n{P @ A}")
print(f"LU:\\n{L @ U}")
print(f"Equal: {np.allclose(P @ A, L @ U)}")
print()

# Solving Ax = b using LU
b = np.array([4, 10, 24])

# Step 1: Solve Ly = Pb (forward substitution)
Pb = P @ b
y = np.linalg.solve(L, Pb)

# Step 2: Solve Ux = y (back substitution)
x = np.linalg.solve(U, y)

print(f"Solution: {x}")
print(f"Verification Ax = {A @ x}")
print(f"b = {b}")
\`\`\`

## Least Squares Solutions

For overdetermined systems (m > n), find **x** that minimizes ||**Ax** - **b**||².

**Normal equations**: **AᵀAx** = **Aᵀb**

**Solution**: **x** = (**AᵀA**)⁻¹**Aᵀb**

\`\`\`python
print("\\n=== Least Squares ===")

# Overdetermined system: fit line to points
# y = mx + c
# [x₁ 1] [m]   [y₁]
# [x₂ 1] [c] = [y₂]
# [x₃ 1]       [y₃]
# ...

# Generate noisy data
np.random.seed(42)
x_data = np.linspace(0, 10, 20)
y_true = 2 * x_data + 1
y_data = y_true + np.random.randn(20) * 2

# Set up overdetermined system
A = np.column_stack([x_data, np.ones_like(x_data)])
b = y_data

print(f"Data points: {len(x_data)}")
print(f"Parameters to fit: 2 (slope and intercept)")
print(f"System shape: {A.shape}")
print()

# Method 1: Normal equations (not recommended)
x_normal = np.linalg.solve(A.T @ A, A.T @ b)

# Method 2: np.linalg.lstsq (recommended)
x_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

# Method 3: Using pseudo-inverse
A_pinv = np.linalg.pinv(A)
x_pinv = A_pinv @ b

print("Solutions:")
print(f"Normal equations: {x_normal}")
print(f"np.linalg.lstsq: {x_lstsq}")
print(f"Pseudo-inverse: {x_pinv}")
print(f"True parameters: [2.0, 1.0]")
print()

print(f"Residual sum of squares: {residuals[0]:.4f}")
print(f"Residual norm: {np.linalg.norm(A @ x_lstsq - b):.4f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.6, label='Data points')
plt.plot(x_data, y_true, 'g--', label='True line: y = 2x + 1')
plt.plot(x_data, A @ x_lstsq, 'r-', linewidth=2, label=f'Fitted: y = {x_lstsq[0]:.2f}x + {x_lstsq[1]:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Least Squares Linear Regression')
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

## QR Decomposition for Least Squares

More numerically stable than normal equations.

**A** = **QR** where **Q** is orthogonal and **R** is upper triangular.

\`\`\`python
print("\\n=== QR Decomposition for Least Squares ===")

# Same data as before
Q, R = np.linalg.qr(A)

print("Q (orthogonal) shape:", Q.shape)
print("R (upper triangular) shape:", R.shape)
print()

# Verify QᵀQ = I
print("QᵀQ:")
print((Q.T @ Q).round(10))
print(f"Is identity: {np.allclose(Q.T @ Q, np.eye(Q.shape[1]))}")
print()

# Solve using QR: Ax = b → QRx = b → Rx = Qᵀb
Qtb = Q.T @ b
x_qr = np.linalg.solve(R, Qtb)

print(f"Solution using QR: {x_qr}")
print(f"Matches lstsq: {np.allclose(x_qr, x_lstsq)}")
print()

print("Advantages of QR over normal equations:")
print("1. Better numerical stability")
print("2. Avoids computing AᵀA (which squares condition number)")
print("3. Works well even when A is ill-conditioned")
\`\`\`

## Applications in Machine Learning

### 1. Linear Regression

\`\`\`python
print("\\n=== Linear Regression Application ===")

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate regression data
X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add intercept term
X_train_int = np.column_stack([X_train, np.ones(len(X_train))])
X_test_int = np.column_stack([X_test, np.ones(len(X_test))])

# Solve using least squares
weights = np.linalg.lstsq(X_train_int, y_train, rcond=None)[0]

# Predictions
y_pred = X_test_int @ weights

# Evaluate
mse = np.mean((y_pred - y_test)**2)
r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

print(f"Weights: {weights}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
\`\`\`

### 2. Polynomial Regression

\`\`\`python
print("\\n=== Polynomial Regression ===")

# Generate non-linear data
x = np.linspace(0, 1, 50)
y = np.sin(2 * np.pi * x) + np.random.randn(50) * 0.1

# Create polynomial features [x, x², x³]
degree = 3
A_poly = np.column_stack([x**i for i in range(degree + 1)])

# Fit
coeffs = np.linalg.lstsq(A_poly, y, rcond=None)[0]

# Predict
y_pred = A_poly @ coeffs

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, label='Data')
plt.plot(x, y_pred, 'r-', linewidth=2, label=f'Degree {degree} polynomial')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Regression using Least Squares')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Polynomial coefficients: {coeffs}")
\`\`\`

## Summary

**Key Methods**:
- **Gaussian Elimination**: Direct method for square systems
- **LU Decomposition**: Efficient for multiple systems with same **A**
- **Least Squares**: For overdetermined systems (more equations than unknowns)
- **QR Decomposition**: More stable than normal equations

**System Types**:
- **Square, non-singular** (m = n, det ≠ 0): Unique solution
- **Overdetermined** (m > n): Least squares solution
- **Underdetermined** (m < n): Minimum norm solution
- **Singular**: No solution or infinite solutions

**ML Applications**:
- Linear regression (least squares)
- Polynomial regression
- Ridge regression (regularized least squares)
- Neural network training (solving gradient equations)

**Best Practices**:
- Use \`np.linalg.solve()\` for square systems
- Use \`np.linalg.lstsq()\` for overdetermined systems
- Use QR decomposition for better numerical stability
- Check condition number for ill-conditioned systems
- Add regularization to prevent overfitting

Understanding how to solve linear systems is fundamental to implementing and debugging ML algorithms!
`,
      multipleChoice: [
        {
          id: 'sys-lin-q1',
          question:
            'What type of solution does an overdetermined system (more equations than unknowns) typically have?',
          options: [
            'Always a unique solution',
            'Always infinite solutions',
            'Usually no exact solution, use least squares',
            'Always no solution',
          ],
          correctAnswer: 2,
          explanation:
            'Overdetermined systems (m > n) usually have no exact solution because there are more constraints than degrees of freedom. We use least squares to find the best approximate solution that minimizes ||Ax - b||².',
        },
        {
          id: 'sys-lin-q2',
          question:
            'What is the advantage of LU decomposition over Gaussian elimination?',
          options: [
            'LU is always faster',
            'LU can solve multiple systems with the same A efficiently',
            'LU works for singular matrices',
            'LU is simpler to implement',
          ],
          correctAnswer: 1,
          explanation:
            'Once you compute A = LU, you can efficiently solve Ax = b for many different b vectors by just doing forward and back substitution (O(n²) each), without re-decomposing A (which is O(n³)).',
        },
        {
          id: 'sys-lin-q3',
          question:
            'Why is QR decomposition preferred over normal equations for least squares?',
          options: [
            'QR is always faster',
            'QR produces different results',
            'QR is more numerically stable, especially for ill-conditioned matrices',
            'QR works for non-square matrices only',
          ],
          correctAnswer: 2,
          explanation:
            'Normal equations require computing AᵀA, which squares the condition number, making the problem more ill-conditioned. QR decomposition avoids this and maintains numerical stability even when A is ill-conditioned.',
        },
        {
          id: 'sys-lin-q4',
          question:
            'In linear regression with n features and m samples, what shape is the design matrix X?',
          options: ['(n, m)', '(m, n)', '(n, n)', '(m, m)'],
          correctAnswer: 1,
          explanation:
            'The design matrix X has shape (m, n) where m is the number of samples (rows) and n is the number of features (columns). Each row represents one data point.',
        },
        {
          id: 'sys-lin-q5',
          question:
            'What does it mean when a system Ax = b has infinitely many solutions?',
          options: [
            'det(A) > 0',
            'A has full rank and b is arbitrary',
            'A is singular and b is in the column space of A',
            'The system is overdetermined',
          ],
          correctAnswer: 2,
          explanation:
            'Infinite solutions occur when A is singular (det = 0, not full rank) AND b is in the column space of A. This means the system is underdetermined—there are free variables that can take any value.',
        },
      ],
      quiz: [
        {
          id: 'sys-lin-d1',
          question:
            'Linear regression can be solved using normal equations (AᵀAx = Aᵀb) or QR decomposition. Compare these approaches in terms of computational cost, numerical stability, and when each should be used.',
          sampleAnswer:
            'Normal equations involve computing AᵀA and solving (AᵀA)x = Aᵀb. Computational cost: Computing AᵀA is O(mn²) for A with shape (m,n), then solving is O(n³), total O(mn² + n³). For m >> n, this is approximately O(mn²). QR decomposition computes A = QR in O(mn²), then solves Rx = Qᵀb in O(n²), total O(mn²). Both have similar asymptotic complexity. However, numerical stability differs dramatically. Normal equations square the condition number: κ(AᵀA) = κ(A)². If A has condition number 10⁴, AᵀA has condition number 10⁸, causing severe numerical errors. QR maintains the original condition number κ(R) ≈ κ(A), providing much better stability. Use normal equations when: (1) A is very well-conditioned (κ < 100), (2) AᵀA is already computed for other reasons, (3) Memory is severely constrained. Use QR when: (1) A is ill-conditioned, (2) Numerical accuracy is critical, (3) Standard case (recommended default). In practice: sklearn uses QR-based solvers by default. Normal equations are mainly of historical/educational interest. Modern recommendation: always use QR or SVD-based methods unless you have specific constraints.',
          keyPoints: [
            'Both O(mn²) complexity, but QR numerically superior',
            'Normal equations square condition number: κ(AᵀA) = κ(A)² (unstable)',
            'QR recommended default (sklearn uses it); normal equations only if well-conditioned',
          ],
        },
        {
          id: 'sys-lin-d2',
          question:
            'Explain why solving Ax = b by computing x = A⁻¹b is considered bad practice compared to using specialized solvers like np.linalg.solve(). What are the computational and numerical reasons?',
          sampleAnswer:
            "Computing x = A⁻¹b is bad practice for three main reasons: (1) Computational cost: Computing A⁻¹ explicitly costs O(n³). Then multiplying A⁻¹b costs O(n²), total O(n³). Using LU decomposition: PA = LU costs O(n³), then forward/back substitution costs O(n²), total O(n³). While asymptotically equal, LU has a smaller constant factor and avoids storing the full inverse matrix (n² space vs n space for the factorization). (2) Numerical stability: Matrix inversion accumulates rounding errors. The inverse A⁻¹ may have large entries even if A is well-behaved, amplifying errors. Specialized solvers use techniques like partial pivoting, scaling, and iterative refinement to minimize error propagation. For ill-conditioned systems, the difference can be several orders of magnitude in accuracy. (3) Flexibility: Computing A⁻¹ doesn't work for rectangular matrices or singular systems. LU/QR-based solvers can handle these cases (using least squares or finding minimum norm solutions). They also provide diagnostics like rank, condition number estimates, and residual norms. In practice: np.linalg.solve() uses LAPACK's optimized routines with partial pivoting and is typically 2-3x faster than computing the inverse, plus more accurate. The only time to compute A⁻¹ explicitly is when you actually need the inverse matrix itself (e.g., computing covariance inverse for Mahalanobis distance, or theoretical analysis). Even then, specialized methods often exist (e.g., Cholesky for positive definite matrices).",
          keyPoints: [
            'Computing A⁻¹ explicitly: same O(n³) cost, but worse numerical stability',
            'np.linalg.solve(): 2-3x faster, more accurate, better error handling',
            'Only compute A⁻¹ when you need the full inverse matrix (rare)',
          ],
        },
        {
          id: 'sys-lin-d3',
          question:
            'In machine learning, we often encounter regularized least squares (Ridge regression): minimize ||Ax - b||² + λ||x||². Explain how this modifies the normal equations and why regularization helps with ill-conditioned or underdetermined systems.',
          sampleAnswer:
            "Regularized least squares adds a penalty term λ||x||² to prevent overfitting. Taking the gradient and setting to zero: ∇(||Ax-b||² + λ||x||²) = 2Aᵀ(Ax-b) + 2λx = 0. This gives modified normal equations: (AᵀA + λI)x = Aᵀb. Compare to unregularized: AᵀAx = Aᵀb. The addition of λI (ridge term) has profound effects: (1) For ill-conditioned systems: AᵀA might have condition number κ = 10¹⁰. Adding λI increases all eigenvalues by λ, improving conditioning. If λ = 0.01 and smallest eigenvalue is 10⁻⁸, it becomes 10⁻² + 10⁻⁸ ≈ 10⁻², dramatically reducing κ. (2) For underdetermined systems (m < n): AᵀA is rank-deficient (not invertible). Adding λI makes (AᵀA + λI) full rank and invertible, providing a unique solution. (3) For nearly collinear features: High correlation creates near-zero eigenvalues in AᵀA. Regularization prevents coefficients from exploding to compensate for near-singularity. Geometric interpretation: λI shrinks all coefficients toward zero, preferring simpler models. This trades bias (slightly worse training fit) for variance (much better generalization). The optimal λ balances underfitting vs overfitting, typically chosen by cross-validation. Computational benefit: (AᵀA + λI) is better conditioned than AᵀA, so solving is more stable. For large λ, it's nearly diagonal, making solution very stable. In practice: Ridge regression is standard for high-dimensional problems (n large, possible multicollinearity). It's equivalent to imposing a Gaussian prior on weights in Bayesian framework: x ~ N(0, (1/λ)I). Modern variants include elastic net (L1 + L2 penalties) and adaptive regularization (different λ per feature).",
          keyPoints: [
            'Ridge modifies normal equations: (AᵀA + λI)x = Aᵀb (adds λI term)',
            'λI improves conditioning (increases eigenvalues), makes underdetermined solvable',
            'Trades bias for variance: shrinks coefficients, better generalization',
          ],
        },
      ],
    },

    {
      id: 'vector-spaces',
      title: 'Vector Spaces',
      content: `
# Vector Spaces

## Introduction

A **vector space** is a mathematical structure formed by a collection of vectors that can be added together and multiplied by scalars. Understanding vector spaces is essential for grasping the foundations of machine learning, where data lives in high-dimensional vector spaces.

## Vector Space Definition

A set **V** is a **vector space** over a field **F** (usually ℝ) if it satisfies these axioms for all **u**, **v**, **w** ∈ **V** and scalars a, b ∈ **F**:

### Closure Axioms
1. **Closure under addition**: **u** + **v** ∈ **V**
2. **Closure under scalar multiplication**: a**v** ∈ **V**

### Addition Axioms
3. **Commutative**: **u** + **v** = **v** + **u**
4. **Associative**: (**u** + **v**) + **w** = **u** + (**v** + **w**)
5. **Identity**: ∃ **0** ∈ **V** such that **v** + **0** = **v**
6. **Inverse**: ∃ -**v** ∈ **V** such that **v** + (-**v**) = **0**

### Scalar Multiplication Axioms
7. **Distributive** (vector): a(**u** + **v**) = a**u** + a**v**
8. **Distributive** (scalar): (a + b)**v** = a**v** + b**v**
9. **Associative**: (ab)**v** = a(b**v**)
10. **Identity**: 1**v** = **v**

\`\`\`python
import numpy as np

print("=== Vector Space Examples ===")

# Example 1: ℝⁿ (Euclidean space)
print("\\n1. ℝ² (2D Euclidean space)")
u = np.array([1, 2])
v = np.array([3, 4])
a = 2

print(f"u = {u}")
print(f"v = {v}")
print(f"u + v = {u + v} (still in ℝ²)")
print(f"{a}*u = {a*u} (still in ℝ²)")
print("✓ ℝ² is a vector space")

# Example 2: Space of 2×2 matrices
print("\\n2. Space of 2×2 matrices")
M1 = np.array([[1, 2], [3, 4]])
M2 = np.array([[5, 6], [7, 8]])
b = 3

print(f"M1 + M2 =\\n{M1 + M2}")
print(f"{b}*M1 =\\n{b*M1}")
print("✓ 2×2 matrices form a vector space")

# Example 3: Space of polynomials of degree ≤ 2
print("\\n3. Space of polynomials of degree ≤ 2")
# Represent polynomial a₀ + a₁x + a₂x² as [a₀, a₁, a₂]
p1 = np.array([1, 2, 3])  # 1 + 2x + 3x²
p2 = np.array([4, 5, 6])  # 4 + 5x + 6x²
c = 2

print(f"p1 + p2 = {p1 + p2} → {p1[0]+p2[0]} + {p1[1]+p2[1]}x + {p1[2]+p2[2]}x²")
print(f"{c}*p1 = {c*p1} → {c*p1[0]} + {c*p1[1]}x + {c*p1[2]}x²")
print("✓ Polynomials of degree ≤ 2 form a vector space")
\`\`\`

## Subspaces

A **subspace** is a subset of a vector space that is itself a vector space.

**Requirements** for **W** ⊆ **V** to be a subspace:
1. **0** ∈ **W** (contains zero vector)
2. Closed under addition: **u**, **v** ∈ **W** ⇒ **u** + **v** ∈ **W**
3. Closed under scalar multiplication: **v** ∈ **W**, a ∈ ℝ ⇒ a**v** ∈ **W**

\`\`\`python
print("\\n=== Subspaces ===")

# Example 1: Lines through origin in ℝ²
print("\\n1. Line through origin: y = 2x")
# Subspace: {[x, 2x] : x ∈ ℝ}

# Check properties
v1 = np.array([1, 2])
v2 = np.array([2, 4])
v_sum = v1 + v2
v_scaled = 3 * v1

print(f"v1 = {v1} is on line (y/x = 2)")
print(f"v2 = {v2} is on line (y/x = 2)")
print(f"v1 + v2 = {v_sum}, ratio = {v_sum[1]/v_sum[0]} ✓")
print(f"3*v1 = {v_scaled}, ratio = {v_scaled[1]/v_scaled[0]} ✓")
print("→ Line through origin is a subspace")

# Example 2: NOT a subspace (line not through origin)
print("\\n2. Line NOT through origin: y = 2x + 1")
w1 = np.array([0, 1])  # On line
w2 = np.array([1, 3])  # On line
w_sum = w1 + w2

print(f"w1 = {w1}, y = {w1[1]} = 2*{w1[0]} + 1 ✓")
print(f"w2 = {w2}, y = {w2[1]} = 2*{w2[0]} + 1 ✓")
print(f"w1 + w2 = {w_sum}, y = {w_sum[1]} ≠ 2*{w_sum[0]} + 1 ✗")
print("→ NOT a subspace (not closed under addition)")
\`\`\`

## Linear Independence

Vectors **v₁**, **v₂**, ..., **vₖ** are **linearly independent** if:

c₁**v₁** + c₂**v₂** + ... + cₖ**vₖ** = **0** ⇒ c₁ = c₂ = ... = cₖ = 0

**Otherwise**, they are **linearly dependent** (one can be written as a combination of the others).

\`\`\`python
print("\\n=== Linear Independence ===")

# Example 1: Linearly independent vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

# Stack as columns
A_indep = np.column_stack([v1, v2, v3])
rank_indep = np.linalg.matrix_rank(A_indep)

print("\\nLinearly independent vectors:")
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v3 = {v3}")
print(f"Rank = {rank_indep}, # vectors = {A_indep.shape[1]}")
print(f"Linearly independent: {rank_indep == A_indep.shape[1]}")

# Example 2: Linearly dependent vectors
w1 = np.array([1, 2, 3])
w2 = np.array([2, 4, 6])  # w2 = 2*w1
w3 = np.array([3, 6, 9])  # w3 = 3*w1

A_dep = np.column_stack([w1, w2, w3])
rank_dep = np.linalg.matrix_rank(A_dep)

print("\\nLinearly dependent vectors:")
print(f"w1 = {w1}")
print(f"w2 = {w2} = 2*w1")
print(f"w3 = {w3} = 3*w1")
print(f"Rank = {rank_dep}, # vectors = {A_dep.shape[1]}")
print(f"Linearly independent: {rank_dep == A_dep.shape[1]}")
\`\`\`

## Span

The **span** of vectors is the set of all linear combinations:

span(**v₁**, ..., **vₖ**) = {c₁**v₁** + ... + cₖ**vₖ** : c₁, ..., cₖ ∈ ℝ}

\`\`\`python
print("\\n=== Span ===")

# Span of standard basis in ℝ²
e1 = np.array([1, 0])
e2 = np.array([0, 1])

print(f"e1 = {e1}")
print(f"e2 = {e2}")
print()

# Any vector in ℝ² can be written as linear combination
target = np.array([3, 5])
c1, c2 = target[0], target[1]

combination = c1 * e1 + c2 * e2
print(f"Target: {target}")
print(f"As combination: {c1}*e1 + {c2}*e2 = {combination}")
print(f"Equal: {np.allclose(target, combination)}")
print("\\nspan(e1, e2) = ℝ²")

# Span of single vector (line)
v = np.array([1, 2])
print(f"\\nspan({v}) = all vectors of form c*{v}")
print(f"Examples: {0.5*v}, {v}, {2*v}, {-1*v}")
\`\`\`

## Basis and Dimension

A **basis** for vector space **V** is a set of vectors that:
1. **Spans** **V** (every vector can be written as their linear combination)
2. **Linearly independent**

The **dimension** of **V** is the number of vectors in a basis.

\`\`\`python
print("\\n=== Basis and Dimension ===")

# Standard basis for ℝ³
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

print("Standard basis for ℝ³:")
print(f"e1 = {e1}")
print(f"e2 = {e2}")
print(f"e3 = {e3}")
print("Dimension = 3")
print()

# Any vector can be expressed uniquely
v = np.array([5, 7, 9])
coords = v  # Coordinates in standard basis
reconstructed = coords[0]*e1 + coords[1]*e2 + coords[2]*e3

print(f"v = {v}")
print(f"Coordinates: [{coords[0]}, {coords[1]}, {coords[2]}]")
print(f"Reconstructed: {coords[0]}*e1 + {coords[1]}*e2 + {coords[2]}*e3 = {reconstructed}")

# Alternative basis for ℝ²
b1 = np.array([1, 1])
b2 = np.array([1, -1])

B = np.column_stack([b1, b2])
rank_B = np.linalg.matrix_rank(B)

print(f"\\nAlternative basis for ℝ²:")
print(f"b1 = {b1}")
print(f"b2 = {b2}")
print(f"Rank = {rank_B} (linearly independent)")
print(f"These also form a basis for ℝ²")
\`\`\`

## Column Space and Null Space

### Column Space

The **column space** of matrix **A** is span of its columns.

col(**A**) = {**Ax** : **x** ∈ ℝⁿ}

**Dimension**: rank(**A**)

### Null Space

The **null space** of **A** is the set of solutions to **Ax** = **0**.

null(**A**) = {**x** : **Ax** = **0**}

**Dimension**: n - rank(**A**) (nullity)

\`\`\`python
from scipy.linalg import null_space

print("\\n=== Column Space and Null Space ===")

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("Matrix A:")
print(A)
print()

# Column space
rank_A = np.linalg.matrix_rank(A)
print(f"Rank(A) = {rank_A}")
print(f"Column space dimension = {rank_A}")
print(f"Column space is spanned by {rank_A} independent columns")
print()

# Null space
null_A = null_space(A)
print(f"Null space dimension (nullity) = {null_A.shape[1]}")
print(f"n - rank = {A.shape[1]} - {rank_A} = {null_A.shape[1]} ✓")
print()

print("Null space basis:")
print(null_A)
print()

# Verify: A * null_vector ≈ 0
if null_A.shape[1] > 0:
    null_vector = null_A[:, 0]
    result = A @ null_vector
    print(f"A * null_vector = {result}")
    print(f"Approximately zero: {np.allclose(result, 0)}")
\`\`\`

## Applications in Machine Learning

### 1. Feature Space

\`\`\`python
print("\\n=== ML Application: Feature Space ===")

# Dataset as vectors in feature space
# Each sample is a vector in ℝⁿ where n = number of features

# Example: House data
# Features: [size, bedrooms, bathrooms, age]
houses = np.array([
    [2000, 3, 2, 10],
    [1500, 2, 1, 15],
    [2500, 4, 3, 5],
    [1800, 3, 2, 8]
])

print(f"Number of samples: {houses.shape[0]}")
print(f"Feature space dimension: {houses.shape[1]}")
print(f"Each house is a point in ℝ⁴")
print()

# The set of all possible house vectors forms a vector space
# We can add houses, scale them, etc. (though may not have physical meaning)
\`\`\`

### 2. Checking Linear Independence of Features

\`\`\`python
print("\\n=== Checking Feature Independence ===")

# Sometimes features are linearly dependent (redundant)
X = np.array([
    [1, 2, 3],    # Feature 3 = Feature 1 + Feature 2
    [2, 3, 5],
    [3, 4, 7],
    [4, 5, 9]
])

print("Feature matrix X:")
print(X)
print()

rank_X = np.linalg.matrix_rank(X)
print(f"Number of features: {X.shape[1]}")
print(f"Rank: {rank_X}")
print(f"Linearly independent: {rank_X == X.shape[1]}")
print()

if rank_X < X.shape[1]:
    print(f"Only {rank_X} truly independent features!")
    print("Consider removing redundant features or using PCA")
\`\`\`

### 3. Range and Null Space in Regression

\`\`\`python
print("\\n=== Regression: Range and Null Space ===")

# In linear regression: y = Xw + b
# Column space of X determines which target vectors are achievable

X_reg = np.array([[1, 2], [2, 4], [3, 6]])  # Rank 1 (columns dependent)
print("Design matrix X:")
print(X_reg)
print()

rank_reg = np.linalg.matrix_rank(X_reg)
print(f"Rank: {rank_reg}")
print()

# This means only certain y vectors can be perfectly fit
# If y is not in column space, we can only approximate (least squares)

y_in_col = np.array([2, 4, 6])  # = 2 * first column
y_not_in_col = np.array([1, 2, 4])  # Not in column space

print(f"y1 = {y_in_col} is in column space")
print(f"y2 = {y_not_in_col} is NOT in column space")
print("→ For y2, least squares finds best approximation")
\`\`\`

## Summary

**Vector Space**: Set closed under addition and scalar multiplication with 10 axioms

**Subspace**: Subset that is itself a vector space
- Contains zero vector
- Closed under addition and scalar multiplication

**Linear Independence**: No vector is a combination of others
- Check: rank = number of vectors

**Span**: All linear combinations of vectors

**Basis**: Linearly independent spanning set
- Every vector has unique coordinates

**Dimension**: Number of vectors in basis

**Column Space**: Span of columns, dimension = rank

**Null Space**: Solutions to Ax = 0, dimension = n - rank

**ML Connections**:
- Data lives in feature space (vector space)
- Linear independence → non-redundant features
- Dimension = intrinsic dimensionality of data
- Column space = outputs achievable by linear model
- Null space = unidentifiable parameter directions

Understanding vector spaces provides the mathematical foundation for dimensionality reduction, feature engineering, and many ML algorithms!
`,
      multipleChoice: [
        {
          id: 'vec-space-q1',
          question: 'Which of the following is NOT a vector space?',
          options: [
            'ℝⁿ (n-dimensional Euclidean space)',
            'The set of all 2×2 matrices',
            'The set of all points on a line NOT passing through the origin',
            'The set of all polynomials of degree ≤ 3',
          ],
          correctAnswer: 2,
          explanation:
            "A line not passing through the origin is not a vector space because it doesn't contain the zero vector and is not closed under addition. For example, adding two points on the line y = x + 1 gives a point not on the line.",
        },
        {
          id: 'vec-space-q2',
          question:
            'What is the dimension of the vector space of all 3×3 matrices?',
          options: ['3', '6', '9', '27'],
          correctAnswer: 2,
          explanation:
            'A 3×3 matrix has 9 entries, each of which can vary independently. The standard basis consists of 9 matrices with a single 1 and all other entries 0. Therefore, the dimension is 9.',
        },
        {
          id: 'vec-space-q3',
          question:
            'If vectors v₁, v₂, v₃ are linearly dependent, what can you conclude?',
          options: [
            'All three vectors are zero',
            'At least one vector can be written as a combination of the others',
            'The vectors are all parallel',
            'The vectors form a basis',
          ],
          correctAnswer: 1,
          explanation:
            'Linear dependence means there exist non-zero coefficients c₁, c₂, c₃ (not all zero) such that c₁v₁ + c₂v₂ + c₃v₃ = 0. This is equivalent to saying at least one vector is a linear combination of the others.',
        },
        {
          id: 'vec-space-q4',
          question:
            'For a matrix A with shape (m, n) and rank r, what is the dimension of its null space?',
          options: ['r', 'm - r', 'n - r', 'mn - r'],
          correctAnswer: 2,
          explanation:
            'By the rank-nullity theorem: dim(null space) = n - rank(A) = n - r, where n is the number of columns.',
        },
        {
          id: 'vec-space-q5',
          question:
            'In machine learning, if your feature matrix X has fewer linearly independent columns than total columns, what does this indicate?',
          options: [
            'You have too few samples',
            'Some features are redundant (linearly dependent)',
            'The model will always overfit',
            'You need more complex algorithms',
          ],
          correctAnswer: 1,
          explanation:
            "If rank(X) < number of columns, some columns are linear combinations of others, meaning you have redundant features that don't add new information. This can cause numerical instability and should be addressed via feature selection or PCA.",
        },
      ],
      quiz: [
        {
          id: 'vec-space-d1',
          question:
            'Explain why a line through the origin is a subspace of ℝ², but a line not through the origin is not. Use both algebraic (axioms) and geometric arguments.',
          sampleAnswer:
            'A line through the origin in ℝ² has the form {t·v : t ∈ ℝ} for some direction vector v. Algebraically, this satisfies subspace requirements: (1) Contains zero: t=0 gives 0·v = 0 ✓ (2) Closed under addition: t₁v + t₂v = (t₁+t₂)v, still on the line ✓ (3) Closed under scalar multiplication: c(tv) = (ct)v, still on the line ✓ Geometrically: adding two vectors on the line through origin gives another vector on the same line (parallelogram law), and scaling a vector keeps it on the same line. A line NOT through origin, say y = 2x + 1, fails multiple requirements: (1) No zero vector: (0,0) is not on the line since 0 ≠ 2·0 + 1 ✗ (2) Not closed under addition: points (0,1) and (1,3) are both on the line, but their sum (1,4) is not (4 ≠ 2·1 + 1) ✗ Geometrically: adding two position vectors on a shifted line produces a vector that "jumps" away from the line. The geometric intuition is that subspaces must include the origin (natural reference point for vector addition) and must contain all scaled versions and sums of their vectors. A shifted line is missing the origin, so vector operations escape the line. In ML context: centering data (subtracting mean) transforms our data cloud to pass through the origin, converting it into a proper subspace where linear operations behave nicely.',
          keyPoints: [
            'Subspace requirements: contains zero, closed under addition and scalar multiplication',
            'Line through origin: satisfies all axioms; shifted line: fails (no zero, not closed)',
            'ML: centering data (subtract mean) makes it pass through origin (proper subspace)',
          ],
        },
        {
          id: 'vec-space-d2',
          question:
            'The rank-nullity theorem states that for an m×n matrix A: rank(A) + dim(null(A)) = n. Explain this theorem intuitively and discuss its significance in understanding linear transformations and solving Ax = b.',
          sampleAnswer:
            'The rank-nullity theorem reveals a fundamental trade-off: the n input dimensions are partitioned into two complementary spaces. Rank(A) = dimension of column space = number of independent output dimensions that A can produce. Dim(null(A)) = nullity = number of independent input directions that get mapped to zero. Together they must sum to n (total input dimensions). Intuition: imagine A as a transformation. The null space consists of inputs that A "destroys" (maps to zero). The rank counts how many independent directions survive the transformation. Every input dimension either contributes to output (counted in rank) or gets destroyed (counted in nullity). For solving Ax = b: (1) If b is in column space (possible with rank dimensions), solutions exist. (2) If nullity > 0, there are multiple solutions—the null space provides "free parameters" that can be added without changing Ax. The solution set is x_particular + null(A), an affine subspace of dimension = nullity. (3) If nullity = 0 (full column rank), solutions are unique when they exist. Example: A is 3×5 with rank 3. Then nullity = 5-3 = 2. This means: (a) A can produce any vector in a 3D subspace of ℝ³ (column space). (b) For any b in that subspace, there are infinitely many solutions forming a 2D affine subspace (particular solution + 2D null space). In ML: underdetermined systems (more unknowns than equations) always have nullity > 0, giving infinitely many solutions. We typically choose the minimum norm solution (closest to origin). Overdetermined systems typically have nullity = 0, and we use least squares. Understanding the rank-nullity theorem helps diagnose whether a system has no solution, unique solution, or infinite solutions, and explains why regularization (adding constraints) is needed for underdetermined problems.',
          keyPoints: [
            'rank(A) + nullity(A) = n: input dimensions partition into output and destroyed',
            'Nullity > 0: infinite solutions (null space = free parameters); nullity = 0: unique',
            'ML: underdetermined has nullity > 0, use minimum norm solution or regularization',
          ],
        },
        {
          id: 'vec-space-d3',
          question:
            'In machine learning, feature matrices with linearly dependent columns can cause problems. Explain what linear dependence means geometrically, why it causes issues computationally and statistically, and how techniques like PCA address this.',
          sampleAnswer:
            "Linear dependence means some features are redundant—they can be written as combinations of other features, adding no new information. Geometrically: if features f₁, f₂, f₃ are linearly dependent, they don't span a 3D space but only a 2D plane (or even 1D line). Data points lie in a lower-dimensional subspace than the ambient feature space suggests. For example, if f₃ = 2f₁ + f₂ always, the data lies on a 2D plane in 3D space. Computational issues: (1) Near-singular matrices: XᵀX becomes nearly singular (determinant ≈ 0), making (XᵀX)⁻¹ numerically unstable or undefined. Small errors get amplified. (2) Non-unique solutions: In regression, w = (XᵀX)⁻¹Xᵀy fails if XᵀX is singular. Multiple weight combinations produce identical predictions—the problem is underdetermined. (3) Inflated coefficients: With multicollinearity (high but not perfect correlation), coefficients become unstable and interpretation breaks down. Small data changes cause wild coefficient swings. Statistical issues: (1) Variance inflation: Standard errors of coefficients explode, making hypothesis testing unreliable. (2) Loss of interpretability: Can't isolate individual feature effects when they're entangled. (3) Overfitting: Unnecessary parameters waste degrees of freedom. PCA addresses this by: (1) Finding orthogonal (linearly independent) principal components, removing redundancy. (2) Ordered by variance: first PC captures most variation, last PCs capture noise/redundancy. (3) Dimensionality reduction: keeping top k PCs gives k truly independent features spanning the same subspace as original data. This regularizes the problem, improving numerical stability and generalization. Alternative solutions: (1) Feature selection: manually remove redundant features. (2) Ridge regression: add λI to XᵀX, making it invertible even if singular. (3) Feature engineering: create genuinely independent features. Understanding linear independence helps diagnose multicollinearity (check rank or condition number), explains why regularization helps, and motivates dimensionality reduction techniques.",
          keyPoints: [
            'Linear dependence: redundant features, data in lower-dimensional subspace',
            'Issues: singular XᵀX, non-unique solutions, inflated coefficients, poor generalization',
            'Solutions: PCA (orthogonal components), Ridge (regularize), feature selection',
          ],
        },
      ],
    },

    {
      id: 'eigenvalues-eigenvectors',
      title: 'Eigenvalues & Eigenvectors',
      content: `
# Eigenvalues & Eigenvectors

## Introduction

**Eigenvalues** and **eigenvectors** are among the most important concepts in linear algebra and machine learning. They reveal the fundamental structure of linear transformations and appear throughout ML: in PCA, covariance matrices, graph analysis, Markov chains, and neural network analysis.

## Definition

For a square matrix **A** (n × n), a non-zero vector **v** is an **eigenvector** if:

**Av** = λ**v**

Where λ is a scalar called the **eigenvalue**.

**Geometric interpretation**: Matrix **A** stretches (or shrinks) **v** by factor λ, without changing direction.

**Key insight**: Most vectors change direction when multiplied by **A**, but eigenvectors only get scaled!

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

print("=== Eigenvector Intuition ===")

# Simple 2×2 matrix
A = np.array([[3, 1],
              [0, 2]])

print("Matrix A:")
print(A)
print()

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("\\nEigenvectors (as columns):")
print(eigenvectors)
print()

# Verify: Av = λv for each eigenvector
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_i = eigenvalues[i]
    
    Av = A @ v
    lambda_v = lambda_i * v
    
    print(f"\\nEigenvector {i+1}: v = {v}")
    print(f"Eigenvalue {i+1}: λ = {lambda_i}")
    print(f"Av = {Av}")
    print(f"λv = {lambda_v}")
    print(f"Equal: {np.allclose(Av, lambda_v)}")
\`\`\`

## Computing Eigenvalues: Characteristic Equation

**Av** = λ**v**
**Av** - λ**v** = **0**
(**A** - λ**I**)**v** = **0**

For non-trivial solution (**v** ≠ **0**), (**A** - λ**I**) must be singular:

**det(A - λI) = 0**

This is the **characteristic equation**. Solving gives eigenvalues.

\`\`\`python
print("\\n=== Computing Eigenvalues Manually (2×2 case) ===")

A = np.array([[4, 2],
              [1, 3]])

print("Matrix A:")
print(A)
print()

# Characteristic equation: det(A - λI) = 0
# For 2×2: det([[4-λ, 2], [1, 3-λ]]) = 0
# (4-λ)(3-λ) - (2)(1) = 0
# 12 - 4λ - 3λ + λ² - 2 = 0
# λ² - 7λ + 10 = 0
# (λ - 5)(λ - 2) = 0
# λ = 5 or λ = 2

print("Characteristic equation: λ² - 7λ + 10 = 0")
print("Solutions: λ₁ = 5, λ₂ = 2")
print()

# Verify with NumPy
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"NumPy eigenvalues: {eigenvalues}")
print(f"Match: {np.allclose(sorted(eigenvalues), [2, 5])}")
\`\`\`

## Properties of Eigenvalues and Eigenvectors

### Property 1: Trace and Determinant

**Trace(A)** = sum of eigenvalues

**Det(A)** = product of eigenvalues

\`\`\`python
print("\\n=== Trace and Determinant ===")

A = np.array([[6, 2],
              [2, 3]])

eigenvalues = np.linalg.eigvals(A)

trace_A = np.trace(A)
sum_eigenvalues = np.sum(eigenvalues)

det_A = np.linalg.det(A)
prod_eigenvalues = np.prod(eigenvalues)

print("Matrix A:")
print(A)
print()
print(f"Eigenvalues: {eigenvalues}")
print()
print(f"Trace(A) = {trace_A}")
print(f"Sum of eigenvalues = {sum_eigenvalues}")
print(f"Equal: {np.allclose(trace_A, sum_eigenvalues)}")
print()
print(f"Det(A) = {det_A}")
print(f"Product of eigenvalues = {prod_eigenvalues}")
print(f"Equal: {np.allclose(det_A, prod_eigenvalues)}")
\`\`\`

### Property 2: Linear Independence

Eigenvectors corresponding to **distinct** eigenvalues are **linearly independent**.

\`\`\`python
print("\\n=== Linear Independence of Eigenvectors ===")

A = np.array([[5, 2],
              [2, 5]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print()
print(f"Eigenvalues: {eigenvalues}")
print(f"Distinct: {len(set(eigenvalues)) == len(eigenvalues)}")
print()

# Check linear independence via rank
rank = np.linalg.matrix_rank(eigenvectors)
print("Eigenvector matrix (columns are eigenvectors):")
print(eigenvectors)
print()
print(f"Rank: {rank}")
print(f"Number of eigenvectors: {eigenvectors.shape[1]}")
print(f"Linearly independent: {rank == eigenvectors.shape[1]}")
\`\`\`

### Property 3: Eigenvalues of Special Matrices

\`\`\`python
print("\\n=== Eigenvalues of Special Matrices ===")

# 1. Diagonal matrix: eigenvalues = diagonal entries
D = np.array([[3, 0, 0],
              [0, 7, 0],
              [0, 0, 5]])

eigenvalues_D = np.linalg.eigvals(D)
print("Diagonal matrix D:")
print(D)
print(f"Eigenvalues: {sorted(eigenvalues_D)}")
print(f"Diagonal entries: {[D[i,i] for i in range(3)]}")
print()

# 2. Symmetric matrix: all eigenvalues are real
S = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

eigenvalues_S = np.linalg.eigvals(S)
print("Symmetric matrix S:")
print(S)
print(f"Eigenvalues: {eigenvalues_S}")
print(f"All real: {np.all(np.isreal(eigenvalues_S))}")
print()

# 3. Orthogonal matrix: |λ| = 1
# Rotation matrix (90 degrees)
theta = np.pi / 2
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

eigenvalues_Q = np.linalg.eigvals(Q)
print("Rotation matrix Q (90°):")
print(Q)
print(f"Eigenvalues: {eigenvalues_Q}")
print(f"Magnitudes: {np.abs(eigenvalues_Q)}")
print(f"All have |λ| = 1: {np.allclose(np.abs(eigenvalues_Q), 1)}")
\`\`\`

## Eigendecomposition (Diagonalization)

If **A** has n linearly independent eigenvectors, it can be **diagonalized**:

**A** = **PDP⁻¹**

Where:
- **P**: eigenvectors as columns
- **D**: diagonal matrix of eigenvalues
- **P⁻¹**: inverse of **P**

**Consequence**: **Aⁿ** = **PDⁿP⁻¹** (very efficient to compute)

\`\`\`python
print("\\n=== Eigendecomposition ===")

A = np.array([[5, 2],
              [2, 5]])

# Compute eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

print("Matrix A:")
print(A)
print()

print("P (eigenvectors as columns):")
print(P)
print()

print("D (diagonal of eigenvalues):")
print(D)
print()

# Reconstruct: A = PDP⁻¹
A_reconstructed = P @ D @ P_inv

print("A reconstructed (PDP⁻¹):")
print(A_reconstructed)
print()
print(f"Equal to original: {np.allclose(A, A_reconstructed)}")
print()

# Application: compute A^10 efficiently
A_10_direct = np.linalg.matrix_power(A, 10)
D_10 = np.diag(eigenvalues ** 10)
A_10_decomp = P @ D_10 @ P_inv

print("A^10 (via eigendecomposition):")
print(A_10_decomp)
print()
print(f"Equal to direct computation: {np.allclose(A_10_direct, A_10_decomp)}")
\`\`\`

## Symmetric Matrices and Orthogonal Eigenvectors

**Spectral Theorem**: Every **symmetric** matrix can be diagonalized by an **orthogonal** matrix:

**A** = **QΛQᵀ**

Where:
- **Q**: orthogonal matrix (columns are orthonormal eigenvectors)
- **Λ**: diagonal matrix of real eigenvalues
- **QᵀQ** = **QQᵀ** = **I**

\`\`\`python
print("\\n=== Spectral Theorem (Symmetric Matrices) ===")

A = np.array([[6, 2, 1],
              [2, 5, 2],
              [1, 2, 4]])

print("Symmetric matrix A:")
print(A)
print(f"Is symmetric: {np.allclose(A, A.T)}")
print()

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

Q = eigenvectors
Lambda = np.diag(eigenvalues)

print("Q (eigenvectors as columns):")
print(Q)
print()

# Verify Q is orthogonal: QᵀQ = I
Q_T_Q = Q.T @ Q
print("QᵀQ:")
print(Q_T_Q)
print(f"Is identity: {np.allclose(Q_T_Q, np.eye(3))}")
print()

# Verify: A = QΛQᵀ
A_reconstructed = Q @ Lambda @ Q.T
print("A reconstructed (QΛQᵀ):")
print(A_reconstructed)
print(f"Equal to original: {np.allclose(A, A_reconstructed)}")
print()

# Eigenvalues are real
print(f"Eigenvalues: {eigenvalues}")
print(f"All real: {np.all(np.isreal(eigenvalues))}")
\`\`\`

## Applications in Machine Learning

### 1. Covariance Matrix Analysis

\`\`\`python
print("\\n=== Application: Covariance Matrix ===")

# Generate sample data
np.random.seed(42)
n_samples = 100

# Correlated 2D data
X = np.random.randn(n_samples, 2)
X[:, 1] = X[:, 0] + 0.5 * X[:, 1]  # Add correlation

# Covariance matrix
cov = np.cov(X.T)

print("Covariance matrix:")
print(cov)
print()

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Eigenvalues (variances along principal axes):")
print(eigenvalues)
print()

print("Eigenvectors (principal directions):")
print(eigenvectors)
print()

# Eigenvalues represent variance in principal directions
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance

print("Explained variance ratio:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"  PC{i+1}: {ratio:.2%}")
\`\`\`

### 2. PageRank (Power Method)

\`\`\`python
print("\\n=== Application: PageRank (Simplified) ===")

# Transition matrix for web graph
# 4 pages: A, B, C, D
# Links: A→B, A→C, B→C, C→A, D→C

# Stochastic matrix (columns sum to 1)
P = np.array([
    [0,   0,   0.5, 0  ],   # A
    [0.5, 0,   0,   0  ],   # B
    [0.5, 1,   0,   1  ],   # C
    [0,   0,   0.5, 0  ]    # D
])

print("Transition matrix P:")
print(P)
print()

# Find dominant eigenvector (eigenvalue = 1)
eigenvalues, eigenvectors = np.linalg.eig(P)

# Find index of eigenvalue closest to 1
idx = np.argmin(np.abs(eigenvalues - 1))
dominant_eigenvalue = eigenvalues[idx]
page_rank = np.real(eigenvectors[:, idx])

# Normalize to probability distribution
page_rank = page_rank / np.sum(page_rank)

print(f"Dominant eigenvalue: {dominant_eigenvalue}")
print()
print("PageRank scores:")
pages = ['A', 'B', 'C', 'D']
for page, score in zip(pages, page_rank):
    print(f"  Page {page}: {score:.3f}")
\`\`\`

### 3. Principal Component Analysis (Preview)

\`\`\`python
print("\\n=== Application: PCA Preview ===")

# Generate 2D data with correlation
np.random.seed(42)
n = 50
theta = np.pi / 4
X_orig = np.random.randn(n, 2) @ np.diag([3, 1])

# Rotation
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
X = (R @ X_orig.T).T

# Center data
X_centered = X - X.mean(axis=0)

# Covariance matrix
cov = np.cov(X_centered.T)

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov)

# Sort by eigenvalue
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues (principal component variances):")
print(eigenvalues)
print()

# Project onto first principal component
PC1 = eigenvectors[:, 0]
X_projected = X_centered @ PC1.reshape(-1, 1)

print(f"First principal component direction: {PC1}")
print(f"Variance explained: {eigenvalues[0] / np.sum(eigenvalues):.2%}")
print()
print("PCA reduces dimensionality while preserving maximum variance!")
\`\`\`

### 4. Matrix Powers and Markov Chains

\`\`\`python
print("\\n=== Application: Markov Chain Convergence ===")

# Transition matrix for weather model
# States: Sunny, Rainy
P = np.array([[0.7, 0.4],
              [0.3, 0.6]])

print("Transition matrix P:")
print(P)
print()

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(P)

print(f"Eigenvalues: {eigenvalues}")
print()

# Steady-state distribution (eigenvector for λ=1)
idx = np.argmin(np.abs(eigenvalues - 1))
steady_state = np.real(eigenvectors[:, idx])
steady_state = steady_state / np.sum(steady_state)

print(f"Steady-state distribution: {steady_state}")
print("Long-run probabilities: {:.1%} Sunny, {:.1%} Rainy".format(
    steady_state[0], steady_state[1]))
print()

# Verify by computing P^100
P_100 = np.linalg.matrix_power(P, 100)
print("P^100 (each column converges to steady state):")
print(P_100)
\`\`\`

## Summary

**Eigenvalues & Eigenvectors**: **Av** = λ**v**
- Special vectors that only get scaled, not rotated

**Finding Eigenvalues**: Solve **det(A - λI) = 0** (characteristic equation)

**Properties**:
- **Trace(A)** = Σλᵢ
- **Det(A)** = Πλᵢ  
- Eigenvectors for distinct eigenvalues are linearly independent

**Eigendecomposition**: **A** = **PDP⁻¹**
- Efficient for computing **Aⁿ**

**Symmetric Matrices**: **A** = **QΛQᵀ** (Spectral Theorem)
- Real eigenvalues
- Orthonormal eigenvectors

**ML Applications**:
- **PCA**: Principal components = eigenvectors of covariance matrix
- **Covariance analysis**: Eigenvalues = variance in principal directions
- **PageRank**: Dominant eigenvector of transition matrix
- **Markov chains**: Steady state = eigenvector for λ = 1
- **Spectral clustering**: Eigenvectors of graph Laplacian
- **Neural network analysis**: Eigenvalues of Hessian → optimization landscape

**Why important in ML**:
1. **Dimensionality reduction**: PCA, Kernel PCA
2. **Data understanding**: Principal directions of variation
3. **Optimization**: Second-order methods (Newton)
4. **Stability analysis**: Eigenvalues of Jacobian/Hessian
5. **Graph algorithms**: Spectral methods
6. **Iterative algorithms**: Convergence rates depend on eigenvalues

Understanding eigenvalues and eigenvectors is essential for modern machine learning!
`,
      multipleChoice: [
        {
          id: 'eigen-q1',
          question:
            'If v is an eigenvector of matrix A with eigenvalue λ = 3, what is A(2v)?',
          options: ['2v', '3v', '6v', '9v'],
          correctAnswer: 2,
          explanation:
            'Since Av = λv = 3v, we have A(2v) = 2(Av) = 2(3v) = 6v by linearity. Eigenvectors remain eigenvectors when scaled, and the transformation scales linearly.',
        },
        {
          id: 'eigen-q2',
          question:
            'For a 3×3 matrix with eigenvalues λ₁ = 2, λ₂ = 3, λ₃ = 5, what is det(A)?',
          options: ['10', '15', '30', '235'],
          correctAnswer: 2,
          explanation:
            'The determinant equals the product of eigenvalues: det(A) = λ₁ × λ₂ × λ₃ = 2 × 3 × 5 = 30.',
        },
        {
          id: 'eigen-q3',
          question:
            'Which statement is TRUE about eigenvalues of a symmetric matrix?',
          options: [
            'They are always positive',
            'They are always real',
            'They are always distinct',
            'They sum to zero',
          ],
          correctAnswer: 1,
          explanation:
            'By the Spectral Theorem, symmetric matrices always have real eigenvalues. They can be positive, negative, or zero; they can be repeated; and their sum equals the trace (not necessarily zero).',
        },
        {
          id: 'eigen-q4',
          question: 'In PCA, the first principal component corresponds to:',
          options: [
            'The eigenvector with the smallest eigenvalue',
            'The eigenvector with the largest eigenvalue',
            'Any orthogonal direction',
            'The mean of the data',
          ],
          correctAnswer: 1,
          explanation:
            'The first principal component is the eigenvector of the covariance matrix with the largest eigenvalue. This direction captures the maximum variance in the data.',
        },
        {
          id: 'eigen-q5',
          question:
            'If matrix A has eigenvalues 0.9, 0.5, and 0.1, what happens to the iterative process xₖ₊₁ = Axₖ as k → ∞?',
          options: [
            'Diverges to infinity',
            'Converges to zero',
            'Oscillates indefinitely',
            'Converges to a non-zero value',
          ],
          correctAnswer: 1,
          explanation:
            'Since all eigenvalues have magnitude less than 1 (|λ| < 1), repeated multiplication by A shrinks vectors. Thus Aᵏx → 0 as k → ∞. If any |λ| > 1, it would diverge; if max|λ| = 1, it might converge to non-zero or oscillate.',
        },
      ],
      quiz: [
        {
          id: 'eigen-d1',
          question:
            'Explain geometrically what eigenvalues and eigenvectors represent for a 2×2 matrix. Why do eigenvectors only get scaled and not rotated? Provide an example with a diagonal matrix and a general matrix.',
          sampleAnswer:
            'Eigenvalues and eigenvectors reveal the "invariant directions" of a linear transformation. When matrix A transforms space, most vectors change both direction and magnitude. But eigenvectors are special: they only get scaled (stretched or compressed) by factor λ (eigenvalue), maintaining their direction (or reversing if λ < 0). Geometric intuition: imagine A as stretching/rotating space. Eigenvectors point along axes where only stretching occurs, no rotation. Example 1 (Diagonal matrix): A = [[3, 0], [0, 2]] represents stretching by 3× horizontally and 2× vertically. Eigenvectors are [1,0] and [0,1] (coordinate axes) with eigenvalues 3 and 2. A already shows its action clearly—it scales each axis independently. Example 2 (General matrix): A = [[2, 1], [1, 2]] is not aligned with coordinate axes. Computing eigenvalues: det(A - λI) = (2-λ)² - 1 = λ² - 4λ + 3 = (λ-3)(λ-1) = 0, giving λ₁=3, λ₂=1. For λ=3: (A-3I)v = 0 → [[-1,1],[1,-1]]v = 0 → v₁ = [1,1] (direction along y=x). For λ=1: (A-I)v = 0 → [[1,1],[1,1]]v = 0 → v₂ = [1,-1] (direction along y=-x). This matrix stretches 3× along the [1,1] diagonal and 1× (no change) along the [1,-1] diagonal. Why no rotation for eigenvectors? Rotation would change direction, but Av = λv means A maps v to a scalar multiple of itself, which is parallel to v. Any rotation would violate this. Physically: eigenvectors represent "principal axes" of the transformation. In ML: for covariance matrix, eigenvectors are principal components—directions of maximum variance. Data naturally spreads along these directions, and we can understand data structure by examining these invariant directions.',
          keyPoints: [
            'Eigenvectors: invariant directions (only scaled, not rotated) Av = λv',
            'Eigenvalues: scaling factor along each eigenvector direction',
            'ML: covariance eigenvectors = principal components (max variance directions)',
          ],
        },
        {
          id: 'eigen-d2',
          question:
            'The Spectral Theorem states that symmetric matrices can be written as A = QΛQᵀ where Q is orthogonal. Explain why this decomposition is so powerful, particularly for computing matrix functions (like A^n, exp(A), sqrt(A)), and why symmetric matrices are ubiquitous in machine learning.',
          sampleAnswer:
            'The Spectral Theorem (A = QΛQᵀ for symmetric A) is extraordinarily powerful because: (1) Computational efficiency: To compute A^n = (QΛQᵀ)^n = QΛ^nQᵀ. Since Λ is diagonal, Λ^n just raises each diagonal entry to power n, taking O(n) time vs O(n³) for matrix multiplication. Similarly, exp(A) = Q·exp(Λ)·Qᵀ where exp(Λ) = diag(e^λ₁, e^λ₂, ...), and sqrt(A) = Q·sqrt(Λ)·Qᵀ = Q·diag(√λ₁, √λ₂, ...)·Qᵀ. This makes complex matrix functions tractable. (2) Interpretability: Q provides orthonormal basis of eigenvectors—the "natural coordinates" for the transformation. Λ shows pure scaling along each axis. This diagonal form reveals the transformation\'s essential behavior. (3) Stability: Orthogonal matrices preserve lengths and angles (QᵀQ=I), making computations numerically stable. No amplification of errors. (4) Real eigenvalues: Symmetric matrices have real λ (no complex numbers), simplifying analysis and ensuring positive definiteness when all λ>0. Why symmetric matrices dominate ML: (a) Covariance matrices: Cov(X) = E[(X-μ)(X-μ)ᵀ] is symmetric. Eigenvectors = principal components (PCA), eigenvalues = variance along each component. This is foundational for dimensionality reduction. (b) Kernel matrices: K(i,j) = k(xᵢ, xⱼ) is symmetric (kernel functions are symmetric). Used in SVMs, Gaussian processes, kernel PCA. (c) Graph Laplacians: L = D - A (degree - adjacency) is symmetric for undirected graphs. Spectral clustering uses eigenvectors of L. (d) Hessian matrices: Second derivatives H(i,j) = ∂²f/∂xᵢ∂xⱼ are symmetric (by Schwarz\'s theorem). Used in optimization (Newton method) to understand loss surface curvature. (e) Gram matrices: XᵀX is always symmetric. Appears in normal equations (linear regression), momentum in neural networks. Example: To find optimal learning rate in gradient descent, analyze eigenvalues of Hessian. Largest eigenvalue determines maximum stable learning rate. Without Spectral Theorem, this analysis would be intractable. The ubiquity of symmetric matrices in ML stems from: natural occurrence (covariance, similarity), mathematical properties (real eigenvalues, orthogonal eigenvectors), and computational advantages (efficient diagonalization). The Spectral Theorem turns abstract transformations into understandable, computable operations.',
          keyPoints: [
            'A = QΛQᵀ (symmetric): efficient matrix functions A^n = QΛ^nQᵀ (O(n) vs O(n³))',
            'Real eigenvalues, orthogonal eigenvectors: numerical stability, interpretability',
            'ML ubiquity: covariance, kernels, graph Laplacians, Hessians, Gram matrices',
          ],
        },
        {
          id: 'eigen-d3',
          question:
            'In PCA, why do we use eigenvectors of the covariance matrix as principal components? Explain the connection between eigenvalues and explained variance, and discuss how to choose the number of components to retain.',
          sampleAnswer:
            'PCA finds directions of maximum variance in data. These directions are precisely the eigenvectors of the covariance matrix, and their importance is measured by eigenvalues. Mathematical justification: Given centered data X (n samples, d features), we want to find unit vector v₁ that maximizes variance of projected data. Variance of X projected onto v₁ is Var(Xv₁) = v₁ᵀ·Cov(X)·v₁ = v₁ᵀCv₁ where C is the covariance matrix. Using Lagrange multipliers to maximize v₁ᵀCv₁ subject to ||v₁||=1, we get: Cv₁ = λv₁. This is exactly the eigenvector equation! The maximum variance equals the largest eigenvalue λ₁, achieved when v₁ is the corresponding eigenvector. For subsequent components: the second PC maximizes variance among directions orthogonal to v₁, which is the eigenvector for second-largest eigenvalue, etc. Why eigenvalues = variance: For principal component vᵢ, the variance of projected data is λᵢ. Total variance = Tr(C) = Σλᵢ. Thus, explained variance ratio = λᵢ / Σλⱼ. This quantifies how much information each PC captures. Example: If λ = [50, 30, 15, 5], total variance = 100. PC1 explains 50%, PC2 explains 30%, PC1+PC2 explain 80%. Choosing number of components: (1) Explained variance threshold: Keep components until cumulative explained variance ≥ target (e.g., 95%). If cumulative is [0.5, 0.8, 0.95, 1.0], keep 3 components. (2) Elbow method: Plot eigenvalues (scree plot). Look for "elbow" where curve flattens—subsequent components add little information. (3) Kaiser criterion: Keep components with λᵢ > 1 (for standardized data, this means PC captures more variance than a single feature). (4) Cross-validation: Choose number that optimizes downstream task performance. (5) Interpretability: Sometimes fewer components are preferred for visualization (2-3) even if explained variance is lower. Trade-offs: More components = more information retained but higher dimensionality. Fewer components = more compression but information loss. Practical considerations: (a) Curse of dimensionality: In high dimensions (d >> n), many eigenvalues are near-zero noise. Aggressive reduction often helps. (b) Computational cost: Full eigendecomposition is O(d³). For very high d, use truncated SVD to compute only top k components efficiently. (c) Non-linear structure: PCA finds linear directions. If data has non-linear structure (manifold), kernel PCA or autoencoders might be better. Why PCA works: Data often has redundancy (correlated features). PCA decorrelates features (PCs are orthogonal) and orders them by importance (eigenvalues). This separates signal (large λ) from noise (small λ), enabling effective dimensionality reduction. In summary: eigenvectors = directions of variance, eigenvalues = amount of variance. PCA exploits eigen-structure of covariance to find optimal low-rank approximation of data.',
          keyPoints: [
            'PCA: max variance direction solves Cv₁ = λv₁ (eigen problem)',
            'Eigenvalue λᵢ = variance explained by PCᵢ; explained variance ratio = λᵢ/Σλⱼ',
            'Choose components: 95% variance threshold, scree plot elbow, cross-validation',
          ],
        },
      ],
    },

    {
      id: 'matrix-decompositions',
      title: 'Matrix Decompositions',
      content: `
# Matrix Decompositions

## Introduction

**Matrix decompositions** (or factorizations) express a matrix as a product of simpler matrices. These are fundamental to numerical linear algebra, enabling efficient computation and revealing matrix structure.

**Why decompose**?
1. **Numerical stability**: Solve systems more accurately
2. **Efficiency**: Faster computation for repeated operations
3. **Insight**: Reveal geometric/algebraic structure
4. **Applications**: Least squares, dimensionality reduction, data compression

We'll cover: **LU**, **QR**, **Cholesky**, and **SVD** decompositions.

## LU Decomposition

**LU decomposition** factors **A** into:

**A** = **LU**

Where:
- **L**: Lower triangular matrix (1s on diagonal)
- **U**: Upper triangular matrix

**Use case**: Solving **Ax** = **b** for multiple **b** efficiently.

\`\`\`python
import numpy as np
from scipy.linalg import lu

print("=== LU Decomposition ===")

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

print("Matrix A:")
print(A)
print()

# Compute LU decomposition
P, L, U = lu(A)

print("L (lower triangular):")
print(L)
print()

print("U (upper triangular):")
print(U)
print()

print("P (permutation matrix):")
print(P)
print()

# Verify: PA = LU
PA = P @ A
LU_prod = L @ U

print("PA:")
print(PA)
print()

print("LU:")
print(LU_prod)
print()

print(f"PA = LU: {np.allclose(PA, LU_prod)}")
\`\`\`

### Solving Systems with LU

**Ax** = **b** becomes **LUx** = **b**

**Two steps**:
1. **Forward substitution**: Solve **Ly** = **b** for **y**
2. **Back substitution**: Solve **Ux** = **y** for **x**

\`\`\`python
print("\\n=== Solving Ax = b with LU ===")

b1 = np.array([2, 4, 12])
b2 = np.array([1, 5, 15])

print(f"b1 = {b1}")
print(f"b2 = {b2}")
print()

# Solve for b1
from scipy.linalg import solve_triangular

# Step 1: Solve Ly = Pb
Pb1 = P @ b1
y1 = solve_triangular(L, Pb1, lower=True)

# Step 2: Solve Ux = y
x1 = solve_triangular(U, y1, lower=False)

print(f"Solution for b1: x1 = {x1}")
print(f"Verify Ax1 = b1: {np.allclose(A @ x1, b1)}")
print()

# Solve for b2 (reusing L and U!)
Pb2 = P @ b2
y2 = solve_triangular(L, Pb2, lower=True)
x2 = solve_triangular(U, y2, lower=False)

print(f"Solution for b2: x2 = {x2}")
print(f"Verify Ax2 = b2: {np.allclose(A @ x2, b2)}")
print()

print("✓ Reusing LU decomposition is much faster for multiple systems!")
\`\`\`

## QR Decomposition

**QR decomposition** factors **A** into:

**A** = **QR**

Where:
- **Q**: Orthogonal matrix (**QᵀQ** = **I**)
- **R**: Upper triangular matrix

**Use case**: Numerically stable solution of least squares problems.

\`\`\`python
print("\\n=== QR Decomposition ===")

A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

print("Matrix A:")
print(A)
print()

# Compute QR decomposition
Q, R = np.linalg.qr(A)

print("Q (orthogonal matrix):")
print(Q)
print()

print("R (upper triangular):")
print(R)
print()

# Verify: A = QR
QR_prod = Q @ R

print("QR:")
print(QR_prod)
print()

print(f"A = QR: {np.allclose(A, QR_prod)}")
print()

# Verify Q is orthogonal
Q_T_Q = Q.T @ Q

print("QᵀQ:")
print(Q_T_Q)
print(f"Is identity: {np.allclose(Q_T_Q, np.eye(3))}")
\`\`\`

### Solving Least Squares with QR

For overdetermined system **Ax** = **b** (more equations than unknowns):

**QRx** = **b**
**Rx** = **Qᵀb** (multiply both sides by **Qᵀ**)

Solve upper triangular system **Rx** = **Qᵀb** by back substitution.

\`\`\`python
print("\\n=== Least Squares with QR ===")

# Overdetermined system (4 equations, 2 unknowns)
A_over = np.array([[1, 1],
                   [1, 2],
                   [1, 3],
                   [1, 4]], dtype=float)

b_over = np.array([6, 5, 7, 10], dtype=float)

print("Overdetermined system:")
print(f"A shape: {A_over.shape}")
print(f"b shape: {b_over.shape}")
print()

# QR decomposition
Q_over, R_over = np.linalg.qr(A_over)

print(f"Q shape: {Q_over.shape}")
print(f"R shape: {R_over.shape}")
print()

# Solve Rx = Qᵀb
Q_T_b = Q_over.T @ b_over
x_qr = solve_triangular(R_over[:2, :], Q_T_b[:2], lower=False)

print(f"Least squares solution: x = {x_qr}")
print()

# Compare with np.linalg.lstsq
x_lstsq, _, _, _ = np.linalg.lstsq(A_over, b_over, rcond=None)

print(f"np.linalg.lstsq solution: {x_lstsq}")
print(f"Equal: {np.allclose(x_qr, x_lstsq)}")
print()

# Compute residual
residual = b_over - A_over @ x_qr
print(f"Residual norm: {np.linalg.norm(residual):.6f}")
\`\`\`

## Cholesky Decomposition

For **symmetric positive definite** matrix **A**:

**A** = **LLᵀ**

Where **L** is lower triangular with positive diagonal entries.

**Properties**:
- More efficient than LU (half the operations)
- Only works for positive definite matrices
- Commonly used for covariance matrices

\`\`\`python
print("\\n=== Cholesky Decomposition ===")

# Create a positive definite matrix
A_spd = np.array([[4, 2, 2],
                  [2, 5, 3],
                  [2, 3, 6]], dtype=float)

print("Symmetric positive definite matrix A:")
print(A_spd)
print()

# Check positive definiteness
eigenvalues = np.linalg.eigvals(A_spd)
print(f"Eigenvalues: {eigenvalues}")
print(f"All positive: {np.all(eigenvalues > 0)}")
print()

# Compute Cholesky decomposition
L_chol = np.linalg.cholesky(A_spd)

print("L (lower triangular):")
print(L_chol)
print()

# Verify: A = LLᵀ
L_L_T = L_chol @ L_chol.T

print("LLᵀ:")
print(L_L_T)
print()

print(f"A = LLᵀ: {np.allclose(A_spd, L_L_T)}")
\`\`\`

### Solving Systems with Cholesky

\`\`\`python
print("\\n=== Solving Ax = b with Cholesky ===")

b_chol = np.array([12, 15, 17])

print(f"b = {b_chol}")
print()

# Step 1: Solve Ly = b (forward substitution)
y_chol = solve_triangular(L_chol, b_chol, lower=True)

# Step 2: Solve Lᵀx = y (back substitution)
x_chol = solve_triangular(L_chol.T, y_chol, lower=False)

print(f"Solution: x = {x_chol}")
print(f"Verify Ax = b: {np.allclose(A_spd @ x_chol, b_chol)}")
print()

print("Cholesky is ~2x faster than LU for positive definite matrices!")
\`\`\`

## Singular Value Decomposition (SVD)

**SVD** is the most powerful decomposition. For any **A** (m × n):

**A** = **UΣVᵀ**

Where:
- **U**: m × m orthogonal matrix (left singular vectors)
- **Σ**: m × n diagonal matrix (singular values σ₁ ≥ σ₂ ≥ ... ≥ 0)
- **V**: n × n orthogonal matrix (right singular vectors)

**Key properties**:
- Works for ANY matrix (rectangular, singular, etc.)
- Reveals rank, null space, column space
- Optimal low-rank approximation
- Foundation of PCA

\`\`\`python
print("\\n=== Singular Value Decomposition (SVD) ===")

A_svd = np.array([[3, 1, 1],
                  [2, 1, 0],
                  [2, 0, 1],
                  [3, 1, 0]], dtype=float)

print("Matrix A (4×3):")
print(A_svd)
print()

# Compute SVD
U, S, Vt = np.linalg.svd(A_svd, full_matrices=True)

print(f"U shape: {U.shape}")
print(f"S (singular values): {S}")
print(f"Vt shape: {Vt.shape}")
print()

# Reconstruct Sigma as matrix
Sigma = np.zeros((A_svd.shape[0], A_svd.shape[1]))
Sigma[:S.shape[0], :S.shape[0]] = np.diag(S)

print("Σ (as matrix):")
print(Sigma)
print()

# Verify: A = UΣVᵀ
A_reconstructed = U @ Sigma @ Vt

print("Reconstructed A:")
print(A_reconstructed)
print()

print(f"A = UΣVᵀ: {np.allclose(A_svd, A_reconstructed)}")
print()

# Verify orthogonality
print(f"UᵀU = I: {np.allclose(U.T @ U, np.eye(U.shape[0]))}")
print(f"VVᵀ = I: {np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]))}")
\`\`\`

### Low-Rank Approximation with SVD

Keep only top **k** singular values for best rank-k approximation.

\`\`\`python
print("\\n=== Low-Rank Approximation ===")

# Create a larger matrix
np.random.seed(42)
A_large = np.random.randn(5, 5)

U_l, S_l, Vt_l = np.linalg.svd(A_large)

print("Singular values:")
print(S_l)
print()

# Rank-2 approximation
k = 2
Sigma_k = np.zeros((k, k))
Sigma_k[:k, :k] = np.diag(S_l[:k])

A_rank2 = U_l[:, :k] @ Sigma_k @ Vt_l[:k, :]

print(f"Original rank: {np.linalg.matrix_rank(A_large)}")
print(f"Approximation rank: {np.linalg.matrix_rank(A_rank2)}")
print()

# Error
error = np.linalg.norm(A_large - A_rank2, 'fro')
print(f"Frobenius norm error: {error:.6f}")
print(f"Error is σ₃² + σ₄² + σ₅²")
print(f"Computed: {np.sqrt(np.sum(S_l[k:]**2)):.6f}")
\`\`\`

### SVD for Pseudoinverse

For non-square or singular matrices, SVD computes Moore-Penrose pseudoinverse:

**A⁺** = **VΣ⁺Uᵀ**

Where **Σ⁺** has 1/σᵢ on diagonal (0 if σᵢ = 0).

\`\`\`python
print("\\n=== Pseudoinverse with SVD ===")

A_pseudo = np.array([[1, 2],
                     [3, 4],
                     [5, 6]])

print("Matrix A (3×2):")
print(A_pseudo)
print()

# SVD
U_p, S_p, Vt_p = np.linalg.svd(A_pseudo, full_matrices=False)

# Pseudoinverse: Σ⁺ = diag(1/σ₁, 1/σ₂)
Sigma_plus = np.diag(1 / S_p)

A_plus = Vt_p.T @ Sigma_plus @ U_p.T

print("A⁺ (pseudoinverse):")
print(A_plus)
print()

# Compare with np.linalg.pinv
A_plus_np = np.linalg.pinv(A_pseudo)

print("np.linalg.pinv:")
print(A_plus_np)
print()

print(f"Equal: {np.allclose(A_plus, A_plus_np)}")
\`\`\`

## Comparison of Decompositions

\`\`\`python
print("\\n=== Comparison of Decompositions ===")

comparison = """
| Decomposition | Form      | Requirements           | Use Case                        |
|---------------|-----------|------------------------|---------------------------------|
| LU            | A = LU    | Square, invertible*    | Solving Ax=b (multiple b)       |
| QR            | A = QR    | Any shape              | Least squares (stable)          |
| Cholesky      | A = LLᵀ   | Symmetric pos. def.    | Covariance matrices, fast solve |
| Eigendecomp   | A = PDP⁻¹ | Square, full rank*     | PCA, matrix powers              |
| SVD           | A = UΣVᵀ  | Any shape              | Dimensionality reduction, PCA   |

*With pivoting, LU works for all square matrices
*Not all matrices are diagonalizable

**Computational Cost** (n×n matrix):
- LU: O(n³) decomposition, O(n²) per solve
- QR: O(mn²) for m×n matrix
- Cholesky: O(n³/3) (half of LU)
- Eigen: O(n³)
- SVD: O(mn²) for m×n matrix (most expensive)

**Numerical Stability** (best to worst):
1. SVD (most stable)
2. QR
3. Cholesky (for SPD matrices)
4. LU with pivoting
5. LU without pivoting (least stable)
"""

print(comparison)
\`\`\`

## Applications in Machine Learning

### 1. Linear Regression via QR

\`\`\`python
print("\\n=== Application: Linear Regression ===")

from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

print(f"X shape: {X_reg.shape}")
print(f"y shape: {y_reg.shape}")
print()

# Add intercept column
X_reg_intercept = np.column_stack([np.ones(X_reg.shape[0]), X_reg])

# Solve using QR
Q_reg, R_reg = np.linalg.qr(X_reg_intercept)
coeffs_qr = solve_triangular(R_reg, Q_reg.T @ y_reg, lower=False)

print(f"Coefficients: {coeffs_qr}")
print()

# Compare with normal equations
coeffs_normal = np.linalg.inv(X_reg_intercept.T @ X_reg_intercept) @ X_reg_intercept.T @ y_reg

print(f"Normal equations: {coeffs_normal}")
print(f"QR more stable than normal equations: {np.allclose(coeffs_qr, coeffs_normal)}")
\`\`\`

### 2. Data Compression via SVD

\`\`\`python
print("\\n=== Application: Image Compression (Simulated) ===")

# Simulate grayscale image as matrix
np.random.seed(42)
image = np.random.rand(50, 50)

print(f"Original image shape: {image.shape}")
print(f"Original size: {image.size} values")
print()

# SVD
U_img, S_img, Vt_img = np.linalg.svd(image, full_matrices=False)

# Compress: keep only top k singular values
k_vals = [5, 10, 20]

for k in k_vals:
    compressed = U_img[:, :k] @ np.diag(S_img[:k]) @ Vt_img[:k, :]
    error = np.linalg.norm(image - compressed, 'fro') / np.linalg.norm(image, 'fro')
    
    # Storage: U[:, :k] + S[:k] + Vt[:k, :]
    storage = 50*k + k + k*50
    compression_ratio = image.size / storage
    
    print(f"Rank-{k} approximation:")
    print(f"  Relative error: {error:.4f}")
    print(f"  Storage: {storage} values")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print()
\`\`\`

## Summary

**Matrix Decompositions** factor matrices into simpler forms:

**LU Decomposition**: **A** = **LU**
- Efficient for solving multiple systems
- Forward + back substitution

**QR Decomposition**: **A** = **QR**
- Numerically stable
- Best for least squares problems

**Cholesky Decomposition**: **A** = **LLᵀ**
- For symmetric positive definite matrices
- Fastest option (2× LU)
- Used for covariance matrices

**SVD**: **A** = **UΣVᵀ**
- Works for ANY matrix
- Optimal low-rank approximation
- Foundation of PCA, dimensionality reduction

**Choosing decomposition**:
- **Multiple solves, same A**: LU or Cholesky
- **Least squares**: QR or SVD
- **Positive definite**: Cholesky (fastest)
- **Dimensionality reduction**: SVD
- **Numerical stability critical**: QR or SVD

**ML Applications**:
- Linear regression: QR decomposition
- PCA: SVD of centered data matrix (or eigendecomposition of covariance)
- Recommender systems: SVD for matrix completion
- Image compression: Truncated SVD
- Numerical stability: QR/SVD over normal equations

Understanding these decompositions is essential for implementing robust ML algorithms!
`,
      multipleChoice: [
        {
          id: 'decomp-q1',
          question:
            'Which decomposition is specifically designed for symmetric positive definite matrices and is computationally more efficient than LU?',
          options: [
            'QR decomposition',
            'Cholesky decomposition',
            'SVD',
            'Eigendecomposition',
          ],
          correctAnswer: 1,
          explanation:
            'Cholesky decomposition (A = LLᵀ) is specifically for symmetric positive definite matrices and requires about half the operations of LU decomposition, making it the most efficient choice for covariance matrices.',
        },
        {
          id: 'decomp-q2',
          question:
            'For solving an overdetermined least squares problem Ax = b (more equations than unknowns), which decomposition is most numerically stable?',
          options: [
            'LU decomposition',
            'Normal equations (XᵀX)⁻¹Xᵀy',
            'QR decomposition',
            'Cholesky decomposition',
          ],
          correctAnswer: 2,
          explanation:
            'QR decomposition is most numerically stable for least squares. Normal equations can be ill-conditioned (squaring condition number), while QR maintains stability through orthogonal transformations.',
        },
        {
          id: 'decomp-q3',
          question:
            'In SVD (A = UΣVᵀ), what do the singular values in Σ represent?',
          options: [
            'The eigenvalues of A',
            'The square roots of eigenvalues of AᵀA',
            'The rank of A',
            'The trace of A',
          ],
          correctAnswer: 1,
          explanation:
            'Singular values are the square roots of eigenvalues of AᵀA (or AAᵀ). They measure the "strength" of each principal direction and determine the optimal low-rank approximation.',
        },
        {
          id: 'decomp-q4',
          question:
            'Why is LU decomposition efficient for solving Ax = b with multiple different b vectors?',
          options: [
            'LU is faster than other methods',
            'Once A is decomposed into LU, each solve only requires forward and back substitution',
            'LU automatically handles singular matrices',
            'LU requires less memory',
          ],
          correctAnswer: 1,
          explanation:
            'LU decomposition requires O(n³) operations, but once computed, each solve (Ly = b, then Ux = y) takes only O(n²). For k different b vectors, total cost is O(n³ + kn²) vs O(kn³) for k separate solves.',
        },
        {
          id: 'decomp-q5',
          question:
            'For a rank-k approximation of matrix A using SVD, what is the Frobenius norm error?',
          options: [
            'σₖ (the k-th singular value)',
            '√(σₖ₊₁² + σₖ₊₂² + ... + σₙ²) (root sum of squares of discarded singular values)',
            'σ₁ - σₖ',
            '(σ₁ + σ₂ + ... + σₖ) / n',
          ],
          correctAnswer: 1,
          explanation:
            'By the Eckart-Young theorem, the optimal rank-k approximation has Frobenius error equal to √(σₖ₊₁² + ... + σₙ²), the root sum of squares of discarded singular values. This is the best possible error for any rank-k approximation.',
        },
      ],
      quiz: [
        {
          id: 'decomp-d1',
          question:
            'Explain why QR decomposition is preferred over normal equations (XᵀX)⁻¹Xᵀy for solving least squares problems. Discuss the numerical stability issues with normal equations and how QR avoids them.',
          sampleAnswer:
            "Normal equations and QR both solve least squares, but QR is much more numerically stable. The problem with normal equations: To solve Ax = b (overdetermined, A is m×n with m > n), normal equations give x = (AᵀA)⁻¹Aᵀb. Computing AᵀA squares the condition number: κ(AᵀA) = κ(A)². If A is already ill-conditioned (κ(A) = 10⁶), then κ(AᵀA) = 10¹². This means: (1) Small errors in data get amplified by factor 10¹², making solution unreliable. (2) AᵀA may become numerically singular even if A has full rank. (3) Loss of precision: forming AᵀA loses roughly half the significant digits. Example: If A has condition number 10⁸ and we work with 16 decimal digits, AᵀA has condition number 10¹⁶, leaving essentially no accurate digits! QR decomposition avoids this: A = QR where Q is orthogonal and R is upper triangular. To solve Ax = b: QRx = b → Rx = Qᵀb (multiply by Qᵀ). Since Q is orthogonal, Qᵀ doesn't amplify errors (κ(Q) = 1). We solve Rx = Qᵀb by back substitution. Condition number: κ(R) = κ(A), not κ(A)². Benefits: (1) Maintains original conditioning of A. (2) No squaring of errors. (3) Orthogonal Q preserves vector lengths and angles. (4) Direct computation without forming AᵀA. Practical impact: For moderately ill-conditioned problems (κ(A) = 10⁶), normal equations may fail completely while QR gives accurate solution. Example scenario: Linear regression with highly correlated features. If features have correlation 0.99999, normal equations become unstable, but QR handles it gracefully. Implementation note: Modern libraries (scikit-learn, np.linalg.lstsq) use QR or SVD internally, never normal equations in production code. Normal equations are only safe for well-conditioned problems (κ(A) < 10⁴) and are sometimes used for speed when stability isn't critical. In summary: QR costs slightly more (O(mn²) vs O(n³) for forming AᵀA + O(n³) for inversion), but this is negligible compared to the massive gain in numerical stability. Always prefer QR for least squares in practice.",
          keyPoints: [
            'Normal equations square condition number: κ(AᵀA) = κ(A)² (catastrophic instability)',
            "QR maintains κ(R) = κ(A): orthogonal Q doesn't amplify errors (κ(Q) = 1)",
            'Modern libraries (sklearn) always use QR/SVD for least squares, never normal equations',
          ],
        },
        {
          id: 'decomp-d2',
          question:
            "SVD provides the optimal low-rank approximation (Eckart-Young theorem). Explain how truncated SVD works, why it's optimal, and discuss its applications in dimensionality reduction, data compression, and recommender systems.",
          sampleAnswer:
            'Truncated SVD keeps only the k largest singular values, providing the best possible rank-k approximation of a matrix. Mechanics: Full SVD gives A = UΣVᵀ where Σ = diag(σ₁, σ₂, ..., σₙ) with σ₁ ≥ σ₂ ≥ ... ≥ σₙ ≥ 0. Truncated SVD keeps only first k components: A_k = Σᵢ₌₁ᵏ σᵢ uᵢvᵢᵀ = U_k Σ_k V_k^T, where U_k = first k columns of U, Σ_k = top-left k×k block of Σ, V_k^T = first k rows of Vᵀ. Optimality (Eckart-Young theorem): A_k minimizes ||A - B||_F over all rank-k matrices B. The error is ||A - A_k||_F = √(σₖ₊₁² + ... + σₙ²). This means no other rank-k matrix can approximate A better (in Frobenius or spectral norm). Intuition: Singular values measure "importance" of each direction. Large σᵢ = important structure, small σᵢ = noise. By keeping top k, we capture the k most important patterns while discarding noise. Applications: (1) Dimensionality Reduction (PCA): Data matrix X (n samples, d features). SVD: X = UΣVᵀ. Principal components = columns of V. Reduced data: X_reduced = XVₖ = UₖΣₖ (n×k). This projects data onto k principal axes capturing most variance. Example: MNIST digits (28×28 = 784 pixels). Top 50 singular values capture >90% of variance, reducing from 784 to 50 dimensions. (2) Data Compression: Images, video, text corpora. Grayscale image = m×n matrix. Store U_k (m×k) + Σ_k (k values) + V_k^T (k×n) = k(m+n+1) values instead of mn. Compression ratio = mn / k(m+n+1). Example: 1000×1000 image, rank-50 approximation: 10⁶ → 100,050 (10× compression) with minimal perceptible loss. (3) Recommender Systems: User-item matrix R (m users, n items), very sparse. SVD factors R ≈ UₖΣₖVₖᵀ. User i = row i of UₖΣₖ (low-dimensional user representation). Item j = column j of VₖᵀΣₖ (low-dimensional item representation). Predicted rating: rᵢⱼ = (UₖΣₖ)ᵢ · (VₖᵀΣₖ)ⱼ. This fills in missing entries (collaborative filtering). Netflix Prize used SVD extensively for recommendations. (4) Latent Semantic Analysis: Document-term matrix A (m documents, n terms). SVD reveals latent topics (k "concepts"). Documents/terms embedded in k-dimensional semantic space. Query matching in reduced space captures semantic similarity (synonyms, related concepts). Advantages: (1) Optimal approximation (provably best). (2) Automatic noise reduction (small singular values often = noise). (3) Reveals hidden structure (latent factors). (4) Dimensionality reduction with minimal information loss. Practical considerations: (1) Computation: Full SVD is O(mn²) for m×n matrix. For very large matrices, use randomized SVD or iterative methods (Lanczos) to compute only top k singular values/vectors. (2) Sparsity: SVD destroys sparsity (U, V are dense). For sparse data, consider sparse decompositions or matrix factorization methods (NMF, sparse coding). (3) Interpretability: PCA/SVD components are linear combinations, sometimes hard to interpret. Compare with sparse methods (e.g., sparse PCA) for more interpretable factors. In summary: Truncated SVD is the gold standard for low-rank approximation, combining optimality with computational efficiency and broad applicability across ML domains.',
          keyPoints: [
            'Eckart-Young: A_k = top k singular values is optimal rank-k approximation',
            'Error: ||A - A_k||_F = √(σₖ₊₁² + ... + σₙ²) (root sum of squares of discarded)',
            'Applications: PCA, image compression, recommender systems (Netflix), LSA',
          ],
        },
        {
          id: 'decomp-d3',
          question:
            'Compare and contrast the use cases, computational costs, and numerical stability of LU, QR, and Cholesky decompositions. When would you choose each one for solving linear systems or least squares problems in machine learning?',
          sampleAnswer:
            'LU, QR, and Cholesky are workhorses of numerical linear algebra, each with specific strengths. LU Decomposition (A = LU): Use case: Solving Ax = b for square, invertible A (with pivoting, works for all non-singular square matrices). Ideal when solving multiple systems with same A but different b vectors. Cost: O(n³) for decomposition, O(n²) per solve. For k systems: O(n³ + kn²) total. Stability: Moderate. LU without pivoting can be unstable. LU with partial pivoting is generally stable but can still fail for badly scaled matrices. Form AᵀA loses conditioning. ML usage: Less common in modern ML (QR preferred), but appears in specialized solvers and when repeated solves are needed. QR Decomposition (A = QR): Use case: Solving least squares (Ax = b, overdetermined). Works for any shape, most numerically stable option. Cost: O(mn²) for m×n matrix. For least squares, also O(mn²). Slightly more expensive than LU for square matrices, but stability makes it worthwhile. Stability: Excellent. Orthogonal transformations (Q) don\'t amplify errors. Condition number κ(R) = κ(A), not squared. "Gold standard" for numerical stability in least squares. ML usage: Linear regression, ridge regression, any least squares problem. Scikit-learn uses QR internally. Preferred over normal equations in production code. Cholesky Decomposition (A = LLᵀ): Use case: Symmetric positive definite matrices only (covariance matrices, kernel matrices, Hessians with positive curvature). Cost: O(n³/3), roughly half of LU. Fastest option for SPD matrices. Stability: Good for well-conditioned SPD matrices. Can fail if matrix is nearly singular or not quite positive definite due to numerical errors. ML usage: Covariance matrices in Gaussian processes, kernel ridge regression, optimization (Newton method with SPD Hessian), multivariate Gaussian sampling (L @ randn()). Decision Guide: (1) Square system, multiple b vectors, same A: Use LU (reuse factorization). (2) Least squares (overdetermined): Use QR (stability). If very large and sparse, consider iterative methods (CG, LSQR). (3) Symmetric positive definite (covariance, kernel): Use Cholesky (speed + natural for SPD). Check positive definiteness first. (4) Numerical stability critical: QR or SVD. Never use normal equations for ill-conditioned problems. (5) Rank-deficient or nearly singular: SVD with pseudoinverse. (6) Very large, sparse: Iterative methods (Conjugate Gradient for SPD, GMRES for general). Concrete ML Examples: Linear Regression: X is m×n, y is m×1, solve Xw = y. QR decomposition: X = QR, solve Rw = Qᵀy. Don\'t form XᵀX! Ridge Regression: Minimize ||Xw - y||² + λ||w||². Form (XᵀX + λI)w = Xᵀy (now SPD), solve with Cholesky. Or augment system and use QR on [X; √λI]. Gaussian Process: Predictive mean requires solving (K + σ²I)α = y where K is kernel matrix (SPD). Use Cholesky. Logistic Regression (Newton): Hessian H = XᵀWX (SPD for logistic). Solve Hδ = -∇L with Cholesky. Principal Component Analysis: Covariance C = XᵀX/n (SPD). Eigendecomposition or SVD of X. Numerical Stability Ranking: 1. SVD (most stable, most expensive) 2. QR 3. Cholesky (for SPD) 4. LU with partial pivoting 5. LU without pivoting 6. Normal equations (least stable) In modern ML: QR and Cholesky dominate for dense systems, iterative methods for large sparse systems, SVD for dimensionality reduction. LU is less common but still useful for specialized applications. Key lesson: Never sacrifice numerical stability for marginal speed gains. A fast but inaccurate solution is worthless. Choose decomposition based on matrix structure (SPD? square? rectangular?) and stability requirements.',
          keyPoints: [
            'LU: square systems, multiple solves; QR: least squares (most stable); Cholesky: SPD (fastest)',
            'Costs: Cholesky O(n³/3), LU O(n³), QR O(mn²); Stability: SVD > QR > Cholesky > LU',
            'ML: QR for linear regression, Cholesky for Gaussian processes/kernels, SVD for PCA',
          ],
        },
      ],
    },

    {
      id: 'principal-component-analysis',
      title: 'Principal Component Analysis (PCA)',
      content: `
# Principal Component Analysis (PCA)

## Introduction

**Principal Component Analysis (PCA)** is one of the most important dimensionality reduction techniques in machine learning. It uses linear algebra to find the directions of maximum variance in high-dimensional data, enabling:
- Data compression
- Visualization
- Noise reduction
- Feature extraction
- Speeding up learning algorithms

**Key Idea**: Transform correlated features into uncorrelated principal components, ordered by importance.

## The PCA Problem

Given data **X** (n samples × d features), find **k** orthogonal directions that capture maximum variance.

**Mathematically**:
1. Center the data: **X** = **X** - mean(**X**)
2. Find covariance matrix: **C** = **XᵀX** / n
3. Compute eigenvectors and eigenvalues of **C**
4. Principal components = eigenvectors sorted by eigenvalue (descending)

**Result**: Project data onto top k principal components, reducing dimensions from d to k.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

print("=== PCA: Basic Example ===")

# Load iris dataset
iris = load_iris()
X = iris.data  # 150 samples, 4 features
y = iris.target

print(f"Original data shape: {X.shape}")
print(f"Features: {iris.feature_names}")
print()

# Step 1: Standardize data (mean=0, std=1 for each feature)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Step 1: Standardized data")
print(f"Mean: {X_scaled.mean(axis=0)}")
print(f"Std: {X_scaled.std(axis=0)}")
print()

# Step 2: Compute covariance matrix
cov_matrix = np.cov(X_scaled.T)

print("Step 2: Covariance matrix:")
print(cov_matrix)
print()

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Step 3: Eigenvalues (variance explained by each PC):")
print(eigenvalues)
print()

print("Eigenvectors (principal components):")
print(eigenvectors)
print()

# Step 4: Explained variance ratio
total_variance = eigenvalues.sum()
explained_variance_ratio = eigenvalues / total_variance

print("Step 4: Explained variance ratio:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

cumulative_variance = np.cumsum(explained_variance_ratio)
print("\\nCumulative explained variance:")
for i, cum_var in enumerate(cumulative_variance):
    print(f"  First {i+1} PCs: {cum_var:.4f} ({cum_var*100:.2f}%)")
print()

# Step 5: Project onto first 2 principal components
X_pca = X_scaled @ eigenvectors[:, :2]

print(f"Step 5: Projected data shape: {X_pca.shape}")
print(f"Reduced from {X.shape[1]} to {X_pca.shape[1]} dimensions")
\`\`\`

## PCA via SVD

SVD provides a more numerically stable and efficient way to compute PCA.

For centered data **X** (n × d):

**X** = **UΣVᵀ**

**Principal components**: Columns of **V**
**Transformed data**: **XVₖ** = **UₖΣₖ**
**Variance**: σᵢ² / n = eigenvalue λᵢ

**Advantages**:
- No need to form covariance matrix (saves memory, more stable)
- More efficient for tall matrices (n >> d)
- Directly gives transformed data

\`\`\`python
print("\\n=== PCA via SVD ===")

# Center data
X_centered = X_scaled - X_scaled.mean(axis=0)

# SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

print(f"U shape: {U.shape}")
print(f"S (singular values): {S}")
print(f"Vt shape: {Vt.shape}")
print()

# Principal components = rows of Vt (or columns of V)
V = Vt.T
print("Principal components (columns of V):")
print(V)
print()

# Eigenvalues from singular values
eigenvalues_svd = (S ** 2) / (len(X_centered) - 1)

print("Eigenvalues from SVD:")
print(eigenvalues_svd)
print()

print("Match eigenvalues from covariance:")
print(f"Equal: {np.allclose(eigenvalues_svd, eigenvalues)}")
print()

# Transformed data
X_pca_svd = X_centered @ V[:, :2]

print(f"Transformed data via SVD: {X_pca_svd.shape}")
print(f"Equal to previous method: {np.allclose(X_pca, X_pca_svd)}")
\`\`\`

## Choosing Number of Components

### Method 1: Explained Variance Threshold

Keep components until cumulative explained variance ≥ threshold (e.g., 95%).

\`\`\`python
print("\\n=== Method 1: Explained Variance Threshold ===")

threshold = 0.95
n_components = np.argmax(cumulative_variance >= threshold) + 1

print(f"Threshold: {threshold*100:.0f}%")
print(f"Components needed: {n_components}")
print(f"Actual variance explained: {cumulative_variance[n_components-1]:.4f}")
\`\`\`

### Method 2: Scree Plot (Elbow Method)

Plot eigenvalues and look for "elbow" where curve flattens.

\`\`\`python
print("\\n=== Method 2: Scree Plot ===")

# Would visualize in practice
print("Eigenvalues:")
for i, val in enumerate(eigenvalues):
    print(f"  PC{i+1}: {val:.4f}")

print("\\n→ Look for elbow in plot where additional components add little variance")
\`\`\`

### Method 3: Kaiser Criterion

Keep components with eigenvalue > 1 (for standardized data).

\`\`\`python
print("\\n=== Method 3: Kaiser Criterion ===")

n_components_kaiser = np.sum(eigenvalues > 1)

print(f"Components with eigenvalue > 1: {n_components_kaiser}")
print("Eigenvalues > 1:")
for i, val in enumerate(eigenvalues):
    if val > 1:
        print(f"  PC{i+1}: {val:.4f}")
\`\`\`

## Reconstructing Data from PCA

Project to k dimensions, then back to original space (lossy reconstruction).

\`\`\`python
print("\\n=== Data Reconstruction ===")

k = 2  # Use only first 2 components

# Project to k dimensions
X_reduced = X_centered @ V[:, :k]

# Reconstruct (back to d dimensions)
X_reconstructed = X_reduced @ V[:, :k].T

print(f"Original shape: {X_centered.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Reconstructed shape: {X_reconstructed.shape}")
print()

# Reconstruction error
reconstruction_error = np.linalg.norm(X_centered - X_reconstructed, 'fro')**2 / X_centered.shape[0]

print(f"Reconstruction error (MSE): {reconstruction_error:.6f}")
print()

# This equals sum of discarded eigenvalues
expected_error = np.sum(eigenvalues[k:])
print(f"Expected error (sum of discarded eigenvalues): {expected_error:.6f}")
print(f"Match: {np.allclose(reconstruction_error, expected_error)}")
\`\`\`

## PCA with Scikit-learn

\`\`\`python
print("\\n=== PCA with Scikit-learn ===")

from sklearn.decomposition import PCA

# Create PCA object
pca = PCA(n_components=2)

# Fit and transform
X_pca_sklearn = pca.fit_transform(X_scaled)

print(f"Transformed data shape: {X_pca_sklearn.shape}")
print()

print("Principal components:")
print(pca.components_)
print()

print("Explained variance:")
print(pca.explained_variance_)
print()

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
print()

print(f"Cumulative variance: {pca.explained_variance_ratio_.sum():.4f}")
print()

# Inverse transform (reconstruction)
X_reconstructed_sklearn = pca.inverse_transform(X_pca_sklearn)

print(f"Reconstructed shape: {X_reconstructed_sklearn.shape}")
\`\`\`

## Applications in Machine Learning

### 1. Visualization

\`\`\`python
print("\\n=== Application: Visualization ===")

# Reduce high-dimensional data to 2D for plotting
pca_viz = PCA(n_components=2)
X_2d = pca_viz.fit_transform(X_scaled)

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced to: {X_2d.shape[1]} (for visualization)")
print(f"Variance preserved: {pca_viz.explained_variance_ratio_.sum():.2%}")
print()

# In practice, would create scatter plot with colors by class
print("Can now visualize 4D iris data in 2D scatter plot!")
\`\`\`

### 2. Speeding Up Learning

\`\`\`python
print("\\n=== Application: Speeding Up Learning ===")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Train on original data
start = time.time()
clf_original = LogisticRegression(max_iter=1000, random_state=42)
clf_original.fit(X_train, y_train)
time_original = time.time() - start

y_pred_original = clf_original.predict(X_test)
acc_original = accuracy_score(y_test, y_pred_original)

print(f"Original data ({X.shape[1]} features):")
print(f"  Training time: {time_original:.4f}s")
print(f"  Accuracy: {acc_original:.4f}")
print()

# Train on PCA-reduced data
pca_fast = PCA(n_components=2)
X_train_pca = pca_fast.fit_transform(X_train)
X_test_pca = pca_fast.transform(X_test)

start = time.time()
clf_pca = LogisticRegression(max_iter=1000, random_state=42)
clf_pca.fit(X_train_pca, y_train)
time_pca = time.time() - start

y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print(f"PCA data ({X_train_pca.shape[1]} features):")
print(f"  Training time: {time_pca:.4f}s")
print(f"  Accuracy: {acc_pca:.4f}")
print(f"  Speedup: {time_original/time_pca:.2f}x")
print(f"  Accuracy loss: {acc_original - acc_pca:.4f}")
\`\`\`

### 3. Noise Reduction

\`\`\`python
print("\\n=== Application: Noise Reduction ===")

# Add noise to data
np.random.seed(42)
X_noisy = X_scaled + np.random.normal(0, 0.5, X_scaled.shape)

print(f"Added Gaussian noise (std=0.5)")
print()

# Denoise using PCA (keep only top components)
pca_denoise = PCA(n_components=2)  # Keep top 2 (most signal)
X_denoised = pca_denoise.inverse_transform(
    pca_denoise.fit_transform(X_noisy)
)

# Compare errors
error_noisy = np.linalg.norm(X_scaled - X_noisy, 'fro') / np.sqrt(X_scaled.size)
error_denoised = np.linalg.norm(X_scaled - X_denoised, 'fro') / np.sqrt(X_scaled.size)

print(f"RMSE (noisy vs original): {error_noisy:.4f}")
print(f"RMSE (denoised vs original): {error_denoised:.4f}")
print(f"Noise reduction: {(1 - error_denoised/error_noisy)*100:.1f}%")
\`\`\`

### 4. Feature Extraction

\`\`\`python
print("\\n=== Application: Feature Extraction ===")

# Use PCA components as new features
pca_features = PCA(n_components=3)
X_new_features = pca_features.fit_transform(X_scaled)

print(f"Original features ({X.shape[1]}): {iris.feature_names}")
print()

print(f"New features ({X_new_features.shape[1]}): PC1, PC2, PC3")
print(f"These are uncorrelated and capture {pca_features.explained_variance_ratio_.sum():.1%} of variance")
print()

# Interpretation: what does PC1 represent?
print("PC1 loadings (contribution of each original feature):")
for i, feature in enumerate(iris.feature_names):
    print(f"  {feature}: {pca_features.components_[0, i]:.4f}")
\`\`\`

## Limitations and Considerations

### 1. Linearity

PCA finds linear combinations. For nonlinear structure, consider **Kernel PCA** or **t-SNE**.

\`\`\`python
print("\\n=== Limitation: Linearity ===")

# Generate nonlinear data (circle)
theta = np.linspace(0, 2*np.pi, 100)
X_circle = np.column_stack([np.cos(theta), np.sin(theta)])
X_circle += np.random.normal(0, 0.05, X_circle.shape)

# PCA fails to capture circular structure
pca_circle = PCA(n_components=1)
X_circle_pca = pca_circle.fit_transform(X_circle)

print("Circular data (inherently 1D structure)")
print(f"PCA variance explained with 1 component: {pca_circle.explained_variance_ratio_[0]:.2%}")
print("→ PCA requires 2 components for 1D circular manifold (inefficient)")
print("→ Use Kernel PCA or manifold learning for nonlinear structure")
\`\`\`

### 2. Scaling Sensitivity

PCA is sensitive to feature scales. Always standardize!

\`\`\`python
print("\\n=== Limitation: Scaling Sensitivity ===")

# Create data with different scales
X_unscaled = np.column_stack([
    np.random.randn(50) * 1,    # Feature 1: std = 1
    np.random.randn(50) * 100   # Feature 2: std = 100
])

# PCA without scaling
pca_unscaled = PCA(n_components=2)
pca_unscaled.fit(X_unscaled)

print("Unscaled data:")
print(f"  Feature 1 std: 1")
print(f"  Feature 2 std: 100")
print(f"  PC1 variance ratio: {pca_unscaled.explained_variance_ratio_[0]:.4f}")
print("  → PC1 dominated by high-variance feature!")
print()

# PCA with scaling
X_scaled_demo = StandardScaler().fit_transform(X_unscaled)
pca_scaled_demo = PCA(n_components=2)
pca_scaled_demo.fit(X_scaled_demo)

print("Scaled data:")
print(f"  PC1 variance ratio: {pca_scaled_demo.explained_variance_ratio_[0]:.4f}")
print("  → More balanced!")
\`\`\`

### 3. Interpretability

Principal components are linear combinations, sometimes hard to interpret.

\`\`\`python
print("\\n=== Limitation: Interpretability ===")

print("PC1 = 0.52*sepal_length + 0.37*sepal_width + ...")
print("→ Not always clear what this represents conceptually")
print("→ Trade-off: mathematical optimality vs human interpretability")
\`\`\`

## PCA vs Other Methods

\`\`\`python
print("\\n=== PCA vs Other Dimensionality Reduction Methods ===")

comparison = """
| Method     | Linear | Preserves | Use Case                          |
|------------|--------|-----------|-----------------------------------|
| PCA        | Yes    | Variance  | General purpose, fast, interpretable |
| t-SNE      | No     | Local structure | Visualization, nonlinear manifolds |
| UMAP       | No     | Global + local | Modern alternative to t-SNE |
| LDA        | Yes    | Class separation | Supervised dimensionality reduction |
| Kernel PCA | No     | Variance (in kernel space) | Nonlinear patterns |
| Autoencoders | No   | Reconstruction | Deep learning, very high dimensions |

**When to use PCA**:
- Data is (approximately) linear
- Want to preserve global variance
- Need fast, deterministic method
- Want interpretable components
- First step before trying complex methods

**When NOT to use PCA**:
- Data lies on nonlinear manifold (circle, Swiss roll, etc.)
- Need to preserve local neighborhood structure
- Supervised task (use LDA instead)
- Have very high dimensions (consider random projections, sparse PCA)
"""

print(comparison)
\`\`\`

## Summary

**PCA**: Projects data onto orthogonal directions of maximum variance.

**Mathematical Foundation**:
- Eigendecomposition of covariance matrix: **C** = **VΛVᵀ**
- Or SVD of data matrix: **X** = **UΣVᵀ**
- Principal components = eigenvectors of **C** = columns of **V**

**Key Steps**:
1. Center (and standardize) data
2. Compute covariance or use SVD
3. Find eigenvectors/eigenvalues (or singular vectors/values)
4. Sort by eigenvalue (descending)
5. Project onto top k components

**Choosing k**:
- Explained variance threshold (e.g., 95%)
- Scree plot (elbow method)
- Kaiser criterion (λ > 1)
- Cross-validation

**Applications**:
- **Visualization**: Reduce to 2D/3D
- **Speedup**: Fewer features → faster training
- **Noise reduction**: Keep signal, discard noise
- **Feature extraction**: Uncorrelated features
- **Data compression**: Approximate with fewer dimensions

**Limitations**:
- Linear only (use Kernel PCA for nonlinear)
- Sensitive to scaling (always standardize!)
- Components may be hard to interpret
- Assumes variance = importance (not always true)

**Best Practices**:
- Always standardize features first
- Check cumulative explained variance
- Validate on downstream task performance
- Consider nonlinear methods if PCA performs poorly

PCA is the workhorse of dimensionality reduction—fast, interpretable, and effective for many real-world problems!
`,
      multipleChoice: [
        {
          id: 'pca-q1',
          question:
            'What do the principal components in PCA represent geometrically?',
          options: [
            'The mean of the data',
            'Orthogonal directions of maximum variance',
            'The features with highest correlation',
            'Random projections of the data',
          ],
          correctAnswer: 1,
          explanation:
            'Principal components are orthogonal (uncorrelated) directions in feature space ordered by the amount of variance they capture. PC1 points in the direction of maximum variance, PC2 in the direction of maximum remaining variance orthogonal to PC1, etc.',
        },
        {
          id: 'pca-q2',
          question:
            'Why is standardization (scaling features to mean=0, std=1) important before applying PCA?',
          options: [
            'It makes computation faster',
            'PCA is sensitive to feature scales and will be dominated by high-variance features',
            'It is required for SVD to work',
            'It improves interpretability',
          ],
          correctAnswer: 1,
          explanation:
            'PCA finds directions of maximum variance. If features have different scales (e.g., one in meters, another in kilometers), the high-scale feature will dominate the first principal component regardless of its importance. Standardization ensures all features are treated equally.',
        },
        {
          id: 'pca-q3',
          question:
            'If the first 3 principal components explain 95% of the variance, what happens to the remaining 5% when you project data onto these 3 components?',
          options: [
            'It is stored separately',
            'It is lost (cannot be recovered)',
            'It is distributed among the 3 components',
            'It becomes noise',
          ],
          correctAnswer: 1,
          explanation:
            'The remaining 5% of variance is discarded when projecting onto only 3 components. This is lossy compression - you can approximate the original data from the 3 components, but cannot perfectly reconstruct it. The reconstruction error equals the sum of discarded eigenvalues.',
        },
        {
          id: 'pca-q4',
          question:
            'Why might PCA via SVD be preferred over eigendecomposition of the covariance matrix?',
          options: [
            'SVD is always faster',
            'SVD works for non-square matrices',
            "SVD is more numerically stable and doesn't require forming the covariance matrix",
            'SVD gives different results',
          ],
          correctAnswer: 2,
          explanation:
            'SVD is more numerically stable because it avoids forming XᵀX, which squares the condition number and can lead to numerical errors. For tall matrices (many samples, fewer features), SVD is also more efficient. Both methods give the same principal components.',
        },
        {
          id: 'pca-q5',
          question:
            'PCA is limited to linear dimensionality reduction. Which statement best describes when to use alternative methods?',
          options: [
            'Always use PCA first regardless of data structure',
            'Use Kernel PCA or manifold learning (t-SNE, UMAP) when data lies on a nonlinear manifold',
            'Never use PCA for high-dimensional data',
            'PCA only works for 2D data',
          ],
          correctAnswer: 1,
          explanation:
            'PCA finds linear combinations of features. For data with nonlinear structure (e.g., Swiss roll, circles), PCA is inefficient or ineffective. Kernel PCA applies PCA in a high-dimensional feature space (nonlinear), while t-SNE and UMAP preserve local neighborhood structure for visualization.',
        },
      ],
      quiz: [
        {
          id: 'pca-d1',
          question:
            'Derive why the principal components of PCA are the eigenvectors of the covariance matrix. Start from the optimization problem: find unit vector v₁ that maximizes variance of projected data Xv₁.',
          sampleAnswer:
            'PCA seeks directions of maximum variance. Mathematical formulation: Given centered data X (n samples, d features), find unit vector v₁ that maximizes variance of projected data. Variance of projection: Var(Xv₁) = (1/n)·||Xv₁||² = (1/n)·(Xv₁)ᵀ(Xv₁) = (1/n)·v₁ᵀXᵀXv₁ = v₁ᵀCv₁, where C = (1/n)XᵀX is the covariance matrix. Optimization problem: maximize v₁ᵀCv₁ subject to ||v₁||² = v₁ᵀv₁ = 1 (unit vector constraint). Lagrangian: L(v₁, λ) = v₁ᵀCv₁ - λ(v₁ᵀv₁ - 1). Taking gradient with respect to v₁: ∇L = 2Cv₁ - 2λv₁ = 0. This gives: Cv₁ = λv₁. This is the eigenvector equation! v₁ is an eigenvector of C with eigenvalue λ. To maximize variance, substitute back: v₁ᵀCv₁ = v₁ᵀ(λv₁) = λ(v₁ᵀv₁) = λ. So variance equals the eigenvalue λ. To maximize variance, choose v₁ to be the eigenvector corresponding to the largest eigenvalue. For subsequent components: Second PC v₂ maximizes variance subject to: (a) ||v₂|| = 1, (b) v₂ ⊥ v₁ (orthogonal to first PC). This gives v₂ = eigenvector for second-largest eigenvalue. By induction, all PCs are eigenvectors of C ordered by eigenvalue. Geometric intuition: Covariance matrix C describes how features co-vary. Eigenvectors are the "natural axes" of the data distribution. Eigenvalues measure spread along each axis. PCA rotates coordinate system to align with these natural axes, ordering them by importance (variance). Alternative derivation via SVD: X = UΣVᵀ (SVD of data matrix). Then C = (1/n)XᵀX = (1/n)VΣᵀUᵀUΣVᵀ = (1/n)VΣ²Vᵀ (since UᵀU = I). This is eigendecomposition of C with eigenvalues λᵢ = σᵢ²/n and eigenvectors = columns of V. Why this is profound: PCA is not just a heuristic—it\'s the mathematically optimal linear dimensionality reduction that preserves maximum variance. The connection to eigenanalysis makes it tractable and gives deep geometric insight.',
          keyPoints: [
            'Optimization: max v₁ᵀCv₁ subject to ||v₁||=1 → Lagrangian → Cv₁ = λv₁ (eigenvector equation)',
            'Variance = eigenvalue λ; max variance → largest eigenvalue eigenvector',
            'PCA = mathematically optimal linear dimensionality reduction (preserves max variance)',
          ],
        },
        {
          id: 'pca-d2',
          question:
            'Compare PCA computed via eigendecomposition of the covariance matrix versus SVD of the data matrix. Discuss computational complexity, numerical stability, and when each approach is preferred.',
          sampleAnswer:
            "PCA can be computed in two ways: eigendecomposition of covariance matrix or SVD of data matrix. Both give identical principal components, but differ in efficiency and stability. Method 1 - Eigendecomposition of Covariance Matrix: Steps: (1) Center data X (n×d). (2) Compute covariance C = XᵀX/n (d×d). (3) Eigendecomposition: C = VΛVᵀ. (4) Principal components = columns of V, variances = diagonal of Λ. Complexity: O(nd²) to form C, O(d³) for eigendecomposition. Total: O(nd² + d³). For wide matrices (n < d), this is O(nd²). For tall matrices (n >> d), this is O(nd²). Memory: Store C (d×d), efficient for moderate d. Method 2 - SVD of Data Matrix: Steps: (1) Center data X (n×d). (2) SVD: X = UΣVᵀ. (3) Principal components = columns of V (or rows of Vᵀ). (4) Variances = σᵢ²/(n-1). Complexity: O(min(nd², n²d)) for full SVD. For n >> d: O(nd²). For d >> n: O(n²d). Thin SVD (only first min(n,d) components): O(ndk) where k = rank. Memory: Never form XᵀX, work directly with X. Comparison: Numerical Stability: SVD is significantly more stable. Reason: forming XᵀX squares condition number. If κ(X) = 10⁶, then κ(XᵀX) = 10¹². This causes loss of precision. SVD avoids this by working directly with X. For ill-conditioned data, eigendecomposition can produce inaccurate eigenvalues/eigenvectors, while SVD remains accurate. Recommendation: Always use SVD for PCA in practice. Computational Efficiency: Tall matrices (n >> d, e.g., 10,000 samples, 50 features): Eigen: O(nd²) = O(10,000·50²) = O(25M). SVD: O(nd²) = O(10,000·50²) = O(25M). Both are O(nd²), similar speed. Thin SVD can be faster. Wide matrices (d >> n, e.g., 100 samples, 10,000 features like gene expression): Eigen: O(nd²) = O(100·10,000²) = O(10B). SVD: O(n²d) = O(100²·10,000) = O(100M). SVD is much faster (100× speedup)! Use randomized SVD for even better performance. Memory: Eigen requires storing d×d covariance matrix. For d = 10,000, C has 100M entries (800MB for float64). SVD works directly with X, no covariance matrix needed. For large d, SVD is essential. When to use each: Use SVD (almost always): - Default choice for numerical stability. - Large d (wide matrices). - Limited memory. - Industry/production code. Use Eigendecomposition: - Educational purposes (more direct connection to theory). - When covariance matrix is already computed (e.g., from streaming data). - Very small d where stability isn't critical. Practical note: sklearn.decomposition.PCA uses randomized SVD by default for n_components < min(n, d), which is even faster for large matrices while maintaining accuracy. This allows PCA on millions of features. In summary: SVD is the gold standard for computing PCA—more stable, often faster, and scales to larger dimensions. Always prefer SVD in practice.",
          keyPoints: [
            'Eigen: forms XᵀX (squares κ, unstable); SVD: works on X directly (stable)',
            'Complexity: both O(nd²) for n>>d; SVD faster for d>>n (100× speedup possible)',
            'Always use SVD in practice: sklearn uses randomized SVD (stable, fast, scales)',
          ],
        },
        {
          id: 'pca-d3',
          question:
            'PCA assumes that directions of maximum variance correspond to the most important structure in data. Discuss scenarios where this assumption breaks down and alternative dimensionality reduction methods would be more appropriate.',
          sampleAnswer:
            'PCA\'s core assumption is that variance = importance. While often reasonable, this fails in several important scenarios: Scenario 1: Signal in low-variance directions. Example: Classifying digits. Suppose one pixel has high variance because it randomly flickers (noise), while another pixel with low variance contains the crucial edge of a digit (signal). PCA would emphasize the noisy pixel and discard the informative one! Alternative: Linear Discriminant Analysis (LDA) finds directions that maximize class separation rather than variance. For supervised tasks, LDA often outperforms PCA. Scenario 2: Nonlinear manifolds. Example: Swiss roll dataset. Data lives on a 2D manifold (intrinsic dimension = 2) embedded in 3D space. PCA requires all 3 dimensions because the manifold isn\'t aligned with any linear subspace. PCA "unfolds" by finding linear directions, which is inefficient. Alternative: Manifold learning methods (Isomap, Locally Linear Embedding, t-SNE, UMAP) preserve local neighborhood structure and can capture the 2D manifold with 2 components. These methods use geodesic distances or local neighborhoods rather than global variance. Scenario 3: Multimodal distributions. Example: Data with two clusters separated along axis 1, but high spread within each cluster along axis 2. PCA\'s first PC might point along axis 2 (high variance), missing the cluster separation (axis 1, lower variance but more meaningful). Alternative: Cluster-specific PCA, mixture models, or supervised methods like LDA. Scenario 4: Adversarial noise. Example: Image data with imperceptible adversarial perturbations. These perturbations might have low variance but are carefully crafted to fool classifiers. PCA focusing on high variance would keep the natural image variation and discard the adversarial signal. Alternative: Robust PCA separates low-rank structure from sparse outliers. Used in video surveillance to separate background (low-rank) from moving objects (sparse). Scenario 5: Temporal or spatial structure. Example: Time series where important patterns occur at specific frequencies or spatial correlations matter. PCA treats all timepoints/locations independently, ignoring temporal/spatial structure. Alternative: - Fourier/wavelet analysis for frequency-domain patterns. - Autoregressive models for temporal dependencies. - Convolutional autoencoders for spatial structure. Scenario 6: Sparse or interpretable structure. Example: Gene expression data where biologists want to identify specific genes (sparse loadings), not linear combinations of all genes (dense PCA loadings). Alternative: Sparse PCA constrains loadings to have many zeros, giving interpretable components. Non-negative Matrix Factorization (NMF) gives additive, parts-based representations. Scenario 7: Extreme high dimensions with limited samples (p >> n). Example: Genomics (20,000 genes, 100 patients). Covariance matrix is rank-deficient and many eigenvalues are effectively noise. PCA overfits to noise rather than signal. Alternative: - Regularized PCA (ridge PCA, sparse PCA). - Random projections (Johnson-Lindenstrauss lemma). - Domain-specific feature selection. Scenario 8: Non-Gaussian distributions. PCA is optimal for Gaussian data. For heavy-tailed or multi-modal distributions, variance may not capture important structure. Alternative: Independent Component Analysis (ICA) for non-Gaussian sources. Kernel PCA with appropriate kernel for specific distributions. When PCA works well: - Data is approximately Gaussian or unimodal. - Variance correlates with signal (not noise). - Linear structure dominates. - Unsupervised setting (no labels). - Need fast, deterministic method. - Want global, coarse-grained compression. When to try alternatives: - Supervised task: Try LDA first. - Nonlinear patterns: Kernel PCA, t-SNE, UMAP, autoencoders. - Sparse/interpretable: Sparse PCA, NMF. - Robust to outliers: Robust PCA. - Time/spatial structure: Domain-specific methods. - Extreme high-dim: Regularization, random projections. Practical workflow: (1) Start with PCA (fast baseline). (2) Check if downstream task benefits. (3) If poor performance, investigate: Is data nonlinear? Use manifold learning. Is variance uninformative? Try LDA or ICA. Need interpretability? Use sparse methods. (4) Visualize principal components and loadings for diagnostics. In summary: PCA is a powerful and widely applicable method, but not universal. Understanding its assumptions helps recognize when alternatives are needed. The "right" dimensionality reduction depends on the data structure, task, and goals (compression, visualization, classification, interpretability).',
          keyPoints: [
            'PCA fails when: signal in low-variance, nonlinear manifolds, multimodal data',
            'Alternatives: LDA (supervised), t-SNE/UMAP (nonlinear), Sparse PCA (interpretable)',
            'Choose based on: supervised vs unsupervised, linear vs nonlinear, interpretability needs',
          ],
        },
      ],
    },

    {
      id: 'linear-transformations',
      title: 'Linear Transformations',
      content: `
# Linear Transformations

## Introduction

A **linear transformation** is a mapping **T**: ℝⁿ → ℝᵐ that preserves vector addition and scalar multiplication. Every linear transformation can be represented as matrix multiplication, making them fundamental to understanding neural networks, computer graphics, and data transformations.

**Definition**: **T** is linear if for all vectors **u**, **v** and scalar c:
1. **T**(**u** + **v**) = **T**(**u**) + **T**(**v**) (additivity)
2. **T**(c**u**) = c**T**(**u**) (homogeneity)

**Matrix Representation**: **T**(**x**) = **Ax**

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

print("=== Linear Transformation Basics ===")

# Example: 2D rotation by 45 degrees
theta = np.pi / 4
A = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

print("Rotation matrix (45°):")
print(A)
print()

# Test vectors
u = np.array([1, 0])
v = np.array([0, 1])

# Transform
T_u = A @ u
T_v = A @ v

print(f"u = {u} → T(u) = {T_u}")
print(f"v = {v} → T(v) = {T_v}")
print()

# Verify linearity
u_plus_v = u + v
T_u_plus_v_direct = A @ u_plus_v
T_u_plus_T_v = T_u + T_v

print("Linearity check:")
print(f"T(u + v) = {T_u_plus_v_direct}")
print(f"T(u) + T(v) = {T_u_plus_T_v}")
print(f"Equal: {np.allclose(T_u_plus_v_direct, T_u_plus_T_v)}")
\`\`\`

## Common 2D Transformations

### 1. Rotation

Rotate by angle θ counterclockwise:

**R**(θ) = [[cos θ, -sin θ],
         [sin θ,  cos θ]]

\`\`\`python
print("\\n=== 2D Rotation ===")

def rotation_matrix_2d(theta):
    """Create 2D rotation matrix."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

# Rotate vector [1, 0] by 90 degrees
R_90 = rotation_matrix_2d(np.pi / 2)
v = np.array([1, 0])
v_rotated = R_90 @ v

print(f"Rotate {v} by 90°:")
print(f"Result: {v_rotated}")
print(f"Expected: [0, 1]")
\`\`\`

### 2. Scaling

Scale by factors sₓ and sᵧ:

**S** = [[sₓ, 0],
      [0, sᵧ]]

\`\`\`python
print("\\n=== 2D Scaling ===")

S = np.array([[2, 0],
              [0, 3]])

v = np.array([1, 1])
v_scaled = S @ v

print("Scaling matrix:")
print(S)
print()
print(f"Transform {v} → {v_scaled}")
print("(2× horizontally, 3× vertically)")
\`\`\`

### 3. Reflection

Reflect across x-axis: **F**ₓ = [[1, 0], [0, -1]]
Reflect across y-axis: **F**ᵧ = [[-1, 0], [0, 1]]
Reflect across y=x: **F**_diagonal = [[0, 1], [1, 0]]

\`\`\`python
print("\\n=== 2D Reflection ===")

# Reflect across x-axis
F_x = np.array([[1, 0],
                [0, -1]])

v = np.array([2, 3])
v_reflected = F_x @ v

print(f"Reflect {v} across x-axis:")
print(f"Result: {v_reflected}")
\`\`\`

### 4. Shear

Horizontal shear: **H** = [[1, k], [0, 1]]
Vertical shear: **V** = [[1, 0], [k, 1]]

\`\`\`python
print("\\n=== 2D Shear ===")

# Horizontal shear
H = np.array([[1, 0.5],
              [0, 1]])

v = np.array([1, 2])
v_sheared = H @ v

print("Horizontal shear matrix:")
print(H)
print()
print(f"Transform {v} → {v_sheared}")
\`\`\`

### 5. Projection

Project onto x-axis: **P**ₓ = [[1, 0], [0, 0]]
Project onto line through origin with direction **u**: **P** = **uuᵀ**/(||**u**||²)

\`\`\`python
print("\\n=== 2D Projection ===")

# Project onto x-axis
P_x = np.array([[1, 0],
                [0, 0]])

v = np.array([3, 4])
v_projected = P_x @ v

print(f"Project {v} onto x-axis:")
print(f"Result: {v_projected}")
print()

# Project onto arbitrary line (direction [1, 1])
u = np.array([1, 1])
u_normalized = u / np.linalg.norm(u)
P_line = np.outer(u_normalized, u_normalized)

v_proj_line = P_line @ v

print(f"Project {v} onto line y=x:")
print(f"Result: {v_proj_line}")
\`\`\`

## Composition of Transformations

Applying transformations sequentially: **T₂**(**T₁**(**x**)) = **A₂A₁x**

**Order matters!** Matrix multiplication is not commutative.

\`\`\`python
print("\\n=== Composition of Transformations ===")

# Rotation by 45° then scaling by 2
R = rotation_matrix_2d(np.pi / 4)
S = np.array([[2, 0],
              [0, 2]])

# Composition 1: Scale then Rotate
T1 = R @ S

# Composition 2: Rotate then Scale
T2 = S @ R

print("R (Rotation 45°):")
print(R)
print()

print("S (Scale 2×):")
print(S)
print()

print("T1 = R @ S (scale then rotate):")
print(T1)
print()

print("T2 = S @ R (rotate then scale):")
print(T2)
print()

v = np.array([1, 0])

print(f"Apply T1 to {v}: {T1 @ v}")
print(f"Apply T2 to {v}: {T2 @ v}")
print("Results differ! Order matters.")
\`\`\`

## Properties of Linear Transformations

### 1. Range and Null Space

**Range (Image)**: Set of all possible outputs
**Null space (Kernel)**: Set of vectors mapped to zero

\`\`\`python
print("\\n=== Range and Null Space ===")

# Projection matrix (onto x-axis)
A = np.array([[1, 0],
              [0, 0]])

print("Projection matrix A (onto x-axis):")
print(A)
print()

# Range: all vectors of form [x, 0]
print("Range: All vectors [x, 0] (x-axis)")
print()

# Null space: all vectors of form [0, y]
print("Null space: All vectors [0, y] (y-axis)")
print()

# Test
v_in_null = np.array([0, 5])
result = A @ v_in_null
print(f"A @ {v_in_null} = {result} (in null space)")
\`\`\`

### 2. Invertibility

A transformation is invertible if there exists **T⁻¹** such that **T⁻¹**(**T**(**x**)) = **x**.

**Matrix condition**: **A** is invertible ⟺ det(**A**) ≠ 0

\`\`\`python
print("\\n=== Invertibility ===")

# Invertible: Rotation
R = rotation_matrix_2d(np.pi / 6)
R_inv = np.linalg.inv(R)

print("Rotation matrix R (30°):")
print(R)
print()

print("Inverse R⁻¹:")
print(R_inv)
print()

v = np.array([1, 2])
v_transformed = R @ v
v_recovered = R_inv @ v_transformed

print(f"v = {v}")
print(f"R(v) = {v_transformed}")
print(f"R⁻¹(R(v)) = {v_recovered}")
print(f"Recovered original: {np.allclose(v, v_recovered)}")
print()

# Non-invertible: Projection
P = np.array([[1, 0],
              [0, 0]])

print(f"Projection matrix det(P) = {np.linalg.det(P)}")
print("Not invertible (information loss)")
\`\`\`

## Linear Transformations in Higher Dimensions

### 3D Rotations

\`\`\`python
print("\\n=== 3D Rotations ===")

def rotation_matrix_3d_z(theta):
    """Rotate around z-axis."""
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rotation_matrix_3d_x(theta):
    """Rotate around x-axis."""
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

# Rotate around z-axis
R_z = rotation_matrix_3d_z(np.pi / 4)
v = np.array([1, 0, 0])
v_rotated = R_z @ v

print("Rotate [1,0,0] around z-axis by 45°:")
print(f"Result: {v_rotated}")
\`\`\`

## Applications in Machine Learning

### 1. Neural Network Layers

Each layer is a linear transformation followed by non-linearity:

**h** = σ(**Wx** + **b**)

\`\`\`python
print("\\n=== Application: Neural Network Layer ===")

# Input dimension: 3, Output dimension: 2
W = np.array([[0.5, 0.3, -0.2],
              [-0.1, 0.4, 0.6]])
b = np.array([0.1, -0.05])

x = np.array([1.0, 2.0, 3.0])

# Linear transformation
z = W @ x + b

# Non-linearity (ReLU)
h = np.maximum(0, z)

print(f"Input x: {x}")
print(f"Weights W:\\n{W}")
print(f"Bias b: {b}")
print()
print(f"Linear: z = Wx + b = {z}")
print(f"After ReLU: h = max(0, z) = {h}")
\`\`\`

### 2. Data Augmentation (Images)

\`\`\`python
print("\\n=== Application: Data Augmentation ===")

# Simulate small image patch as vector
image_patch = np.array([[100, 150],
                        [120, 140]])

print("Original image patch:")
print(image_patch)
print()

# Flatten to vector
image_vec = image_patch.flatten()

# Apply transformation (e.g., rotation)
# For images, we'd use homogeneous coordinates
# Here's simplified: horizontal flip
F_horiz = np.array([[-1, 0],
                     [0, 1]])

# This would be applied in image space
print("Transformations for data augmentation:")
print("- Rotation")
print("- Scaling")
print("- Flipping")
print("- Shearing")
print("→ All are linear transformations!")
\`\`\`

### 3. Change of Basis

Represent data in different coordinate system.

\`\`\`python
print("\\n=== Application: Change of Basis ===")

# Standard basis
e1 = np.array([1, 0])
e2 = np.array([0, 1])

# New basis (PCA principal components)
v1 = np.array([1, 1]) / np.sqrt(2)
v2 = np.array([1, -1]) / np.sqrt(2)

# Change of basis matrix (columns are new basis vectors)
P = np.column_stack([v1, v2])

print("Standard basis: e1=[1,0], e2=[0,1]")
print(f"New basis: v1={v1}, v2={v2}")
print()

# Vector in standard coordinates
x_standard = np.array([3, 1])

# Convert to new coordinates
x_new = P.T @ x_standard

print(f"Vector in standard basis: {x_standard}")
print(f"Vector in new basis: {x_new}")
print()

# Convert back
x_recovered = P @ x_new
print(f"Converted back: {x_recovered}")
print(f"Match: {np.allclose(x_standard, x_recovered)}")
\`\`\`

### 4. Dimensionality Reduction

Project high-dimensional data to lower dimensions.

\`\`\`python
print("\\n=== Application: Dimensionality Reduction ===")

# 3D data
X_3d = np.random.randn(100, 3)

# Project onto first 2 principal components
# (In practice, use PCA from sklearn)
U, S, Vt = np.linalg.svd(X_3d - X_3d.mean(axis=0), full_matrices=False)
V = Vt.T

# Projection matrix (to 2D)
P_2d = V[:, :2]

# Project
X_2d = X_3d @ P_2d

print(f"Original shape: {X_3d.shape}")
print(f"Projection matrix shape: {P_2d.shape}")
print(f"Projected shape: {X_2d.shape}")
print()
print("Linear transformation reduces dimensions while preserving structure!")
\`\`\`

## Geometric Interpretation

Linear transformations can:
1. **Rotate**: Change direction without changing length (orthogonal matrices)
2. **Scale**: Change length without changing direction (diagonal matrices)
3. **Shear**: Skew space while keeping some lines fixed
4. **Project**: Collapse to lower dimension (singular matrices)

\`\`\`python
print("\\n=== Geometric Effects ===")

transformations = {
    "Rotation (Orthogonal)": rotation_matrix_2d(np.pi/4),
    "Scaling (Diagonal)": np.array([[2, 0], [0, 3]]),
    "Shear": np.array([[1, 0.5], [0, 1]]),
    "Projection (Singular)": np.array([[1, 0], [0, 0]])
}

v = np.array([1, 1])

for name, T in transformations.items():
    v_transformed = T @ v
    det_T = np.linalg.det(T)
    
    print(f"\\n{name}:")
    print(f"  Matrix: {T.tolist()}")
    print(f"  {v} → {v_transformed}")
    print(f"  Determinant: {det_T:.2f}")
    print(f"  Area scaling factor: |det| = {abs(det_T):.2f}")
\`\`\`

## Summary

**Linear Transformation**: **T**(**x**) = **Ax**
- Preserves addition and scalar multiplication
- Completely determined by matrix **A**

**Common Transformations**:
- **Rotation**: Preserves length and angles
- **Scaling**: Stretches/compresses along axes
- **Reflection**: Mirrors across line/plane
- **Shear**: Skews space
- **Projection**: Reduces dimension

**Key Properties**:
- **Range**: dim(range) = rank(**A**)
- **Null space**: dim(null) = n - rank(**A**)
- **Invertibility**: det(**A**) ≠ 0
- **Composition**: **T₂** ∘ **T₁** = **A₂A₁**

**ML Applications**:
- **Neural networks**: Each layer is a linear transformation
- **Data augmentation**: Rotate, scale, flip images
- **Dimensionality reduction**: Project to lower dimensions (PCA)
- **Change of basis**: Represent data in new coordinates
- **Computer graphics**: Transforming 3D models

**Geometric Insight**:
- Determinant = volume scaling factor
- Eigenvectors = invariant directions
- Rank = dimension of output space
- Singular values = scaling factors along principal axes

Understanding linear transformations provides intuition for:
- How neural networks process data
- Why certain operations preserve or lose information
- How to design effective data augmentation
- The geometry underlying dimensionality reduction

Linear transformations are the foundation of linear algebra in machine learning!
`,
      multipleChoice: [
        {
          id: 'linear-trans-q1',
          question: 'Which property defines a linear transformation T?',
          options: [
            'T(x + y) = T(x) + T(y) only',
            'T(cx) = cT(x) only',
            'Both T(x + y) = T(x) + T(y) and T(cx) = cT(x)',
            'T(0) = 0 only',
          ],
          correctAnswer: 2,
          explanation:
            'A linear transformation must satisfy both additivity T(x + y) = T(x) + T(y) and homogeneity T(cx) = cT(x). Together, these ensure linearity. Note that T(0) = 0 follows from these properties but is not sufficient by itself.',
        },
        {
          id: 'linear-trans-q2',
          question:
            'For composition of transformations T₂(T₁(x)) = A₂A₁x, why does order matter?',
          options: [
            'Matrix multiplication is commutative',
            'Matrix multiplication is not commutative (generally A₂A₁ ≠ A₁A₂)',
            'The transformations are always the same',
            'Order only matters for non-square matrices',
          ],
          correctAnswer: 1,
          explanation:
            'Matrix multiplication is generally not commutative: A₂A₁ ≠ A₁A₂. For example, rotating then scaling gives a different result than scaling then rotating. The order of operations matters for transformations.',
        },
        {
          id: 'linear-trans-q3',
          question:
            'What does the determinant of a transformation matrix tell you geometrically?',
          options: [
            'The rotation angle',
            'The scaling factor along each axis',
            'The volume/area scaling factor of the transformation',
            'The rank of the matrix',
          ],
          correctAnswer: 2,
          explanation:
            'The determinant represents how much the transformation scales volumes (or areas in 2D). |det(A)| is the volume scaling factor. If det(A) = 0, the transformation collapses space to a lower dimension. If det(A) < 0, orientation is reversed.',
        },
        {
          id: 'linear-trans-q4',
          question:
            'In a neural network, each layer performs h = σ(Wx + b). Which part is the linear transformation?',
          options: [
            'Only σ (activation function)',
            'Only Wx',
            'Wx + b',
            'The entire expression σ(Wx + b)',
          ],
          correctAnswer: 2,
          explanation:
            'The linear transformation is Wx + b (affine transformation, technically). The activation function σ provides non-linearity. Without the non-linear activation, stacking multiple layers would collapse to a single linear transformation.',
        },
        {
          id: 'linear-trans-q5',
          question:
            'A projection matrix P projects vectors onto a subspace. What happens when you apply P twice?',
          options: [
            'P² scales the projection by 2',
            'P² = P (idempotent: applying twice same as once)',
            'P² = I (returns to original)',
            'P² = 0 (maps everything to zero)',
          ],
          correctAnswer: 1,
          explanation:
            "Projection matrices are idempotent: P² = P. Once a vector is projected onto a subspace, projecting again doesn't change it (it's already in the subspace). Mathematically, if Px is the projection, then P(Px) = Px.",
        },
      ],
      quiz: [
        {
          id: 'linear-trans-d1',
          question:
            'Explain why every linear transformation T: ℝⁿ → ℝᵐ can be represented as matrix multiplication T(x) = Ax. How do you construct the matrix A from the transformation T?',
          sampleAnswer:
            'Every linear transformation from ℝⁿ to ℝᵐ corresponds to a unique m×n matrix. This is a fundamental theorem in linear algebra. Proof and construction: Let T: ℝⁿ → ℝᵐ be a linear transformation. Let {e₁, e₂, ..., eₙ} be the standard basis of ℝⁿ (eᵢ has 1 in position i, 0 elsewhere). Any vector x ∈ ℝⁿ can be written as: x = x₁e₁ + x₂e₂ + ... + xₙeₙ. By linearity: T(x) = T(x₁e₁ + x₂e₂ + ... + xₙeₙ) = x₁T(e₁) + x₂T(e₂) + ... + xₙT(eₙ). Let aᵢ = T(eᵢ) be the image of the i-th basis vector (an m-dimensional vector). Then: T(x) = x₁a₁ + x₂a₂ + ... + xₙaₙ = [a₁ a₂ ... aₙ] [x₁, x₂, ..., xₙ]ᵀ = Ax. Construction: The matrix A has columns {a₁, a₂, ..., aₙ} where aᵢ = T(eᵢ). Column i of A tells us where the i-th basis vector goes under T. Example: T: ℝ² → ℝ² is rotation by 90°. Apply T to basis vectors: T(e₁) = T([1,0]) = [0,1] (horizontal vector rotates to vertical). T(e₂) = T([0,1]) = [-1,0] (vertical vector rotates to negative horizontal). Matrix: A = [[0, -1], [1, 0]]. Verify: A[x, y]ᵀ = [0, -1; 1, 0][x; y] = [-y; x], which is indeed (x,y) rotated 90° counterclockwise. Why this matters: (1) Linearity reduces infinite possibilities to finite representation (n² entries for n×n matrix). (2) Computing T(x) reduces to matrix-vector multiplication (efficient algorithms). (3) Composition of transformations = matrix multiplication. (4) Properties of T (invertibility, rank, null space) can be studied via linear algebra. Converse: Given any m×n matrix A, T(x) = Ax defines a linear transformation. The correspondence is bijective: every linear transformation ↔ unique matrix. This unification is powerful: abstract transformations become concrete matrices, enabling computation and analysis. In ML: Neural network layers (Wx + b), PCA projections, data preprocessing—all are linear transformations represented as matrices.',
          keyPoints: [
            'Matrix A construction: columns are T(eᵢ) where eᵢ are standard basis vectors',
            'Every linear T: ℝⁿ → ℝᵐ ↔ unique m×n matrix A (bijective correspondence)',
            'ML: neural layers (Wx+b), PCA projections are all matrix representations of linear T',
          ],
        },
        {
          id: 'linear-trans-d2',
          question:
            "Discuss the geometric meaning of the determinant of a transformation matrix. How does the sign and magnitude of det(A) relate to the transformation's effect on space?",
          sampleAnswer:
            "The determinant captures how a linear transformation scales volumes and whether it preserves orientation. Geometric interpretation: Magnitude: |det(A)| = volume scaling factor. If we transform a unit cube (or unit square in 2D), |det(A)| is the volume (or area) of the resulting parallelepiped (parallelogram). Example: A = [[2, 0], [0, 3]] (scaling by 2× horizontally, 3× vertically). det(A) = 6. A unit square (area 1) becomes a rectangle with area 6. Sign: det(A) > 0: Orientation preserved (right-handed basis stays right-handed). det(A) < 0: Orientation reversed (right-handed becomes left-handed, like mirror reflection). det(A) = 0: Space collapses to lower dimension (volume becomes 0). Examples: (1) Rotation matrix R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]. det(R) = cos²θ + sin²θ = 1. Rotations preserve volume and orientation (rigid motion). (2) Reflection across x-axis: F = [[1, 0], [0, -1]]. det(F) = -1. Preserves area (|det| = 1) but reverses orientation (det < 0). (3) Projection onto x-axis: P = [[1, 0], [0, 0]]. det(P) = 0. Collapses 2D to 1D line (area becomes 0). Information loss: determinant measures loss. Properties: (1) det(AB) = det(A)·det(B). Volume scales multiplicatively under composition. (2) det(A⁻¹) = 1/det(A). Inverse transformation scales by reciprocal. (3) det(Aᵀ) = det(A). Transpose doesn't change volume scaling. (4) det(cA) = cⁿ·det(A) for n×n matrix. Scaling all dimensions by c scales volume by cⁿ. Why determinant relates to invertibility: If det(A) = 0, transformation collapses space (loses dimension). Information is lost—cannot uniquely invert. Example: projection P maps infinite vectors to same output. If det(A) ≠ 0, transformation is bijective (one-to-one and onto). Can invert: A⁻¹ exists. In ML applications: (1) Checking invertibility: For autoencoders, encoder-decoder should be invertible (no information loss). Check if det ≈ 0. (2) Numerical stability: Near-zero determinant (det ≈ 10⁻¹⁰) indicates ill-conditioned matrix. Small input perturbations cause large output changes. Regularization or SVD helps. (3) Jacobian determinant: In normalizing flows (generative models), det(J) is the volume change for probability density transformation. (4) Data augmentation: Determinant tells if transformation preserves, expands, or contracts regions of feature space. In summary: determinant is not just an algebraic formula—it's the fundamental geometric quantity measuring how transformations warp space. Positive = preserve orientation, negative = flip, zero = collapse. Magnitude = volume scaling.",
          keyPoints: [
            'det(A) magnitude: volume scaling factor (|det|=1 preserves, >1 expands, <1 contracts)',
            'det(A) sign: positive preserves orientation, negative reverses (mirror), zero collapses',
            'ML: det≈0 → ill-conditioned (numerical instability); Jacobian det in normalizing flows',
          ],
        },
        {
          id: 'linear-trans-d3',
          question:
            'In neural networks, each layer computes h = σ(Wx + b) where σ is a non-linear activation. Explain why the non-linearity is essential: what would happen if we stacked multiple linear transformations without activation functions?',
          sampleAnswer:
            'Non-linear activation functions are crucial for neural networks\' expressive power. Without them, deep networks collapse to shallow linear models. The problem with stacking linear transformations: Consider a 2-layer network without activation: h₁ = W₁x + b₁ (first layer). h₂ = W₂h₁ + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂) = Wx + b. Where W = W₂W₁ and b = W₂b₁ + b₂. Result: Equivalent to a single linear transformation! Generalizing: Stack L layers, each computing hₗ = Wₗhₗ₋₁ + bₗ. Final output: h_L = W_combined·x + b_combined, where W_combined = W_L·W_{L-1}·...·W₁. Composition of linear transformations is linear. No matter how many layers, a purely linear network can only learn linear functions. Why this is limiting: Linear models can only separate data with linear boundaries (hyperplanes). Many real-world problems require non-linear decision boundaries: - XOR problem: Cannot be solved by any linear classifier. Points (0,0) and (1,1) in one class, (0,1) and (1,0) in another. Linear boundary cannot separate them. - Image classification: Distinguishing cats from dogs requires highly non-linear feature combinations. - Natural language: Semantic relationships are fundamentally non-linear. Example (XOR): Input: x ∈ {(0,0), (0,1), (1,0), (1,1)}. Target: y = x₁ XOR x₂ = {0, 1, 1, 0}. Any linear model: f(x) = w₁x₁ + w₂x₂ + b. Cannot fit this data—no choice of w₁, w₂, b works. With non-linearity (2-layer network): h = σ(W₁x + b₁). y = W₂h + b₂. With ReLU activation, this can solve XOR. Hidden units learn features that linearly separate in transformed space. Universal Approximation Theorem: A neural network with even a single hidden layer and non-linear activation can approximate any continuous function (given enough neurons). Key requirement: non-linearity. Role of different activations: (1) Sigmoid σ(z) = 1/(1+e⁻ᶻ): Smooth, bounded [0,1]. Historical favorite. Issues: vanishing gradients. (2) Tanh: Bounded [-1,1], zero-centered. Better than sigmoid. Still vanishing gradients. (3) ReLU σ(z) = max(0, z): Dominant in modern networks. Advantages: - No vanishing gradients for positive values. - Sparse activation (many neurons output 0). - Efficient computation. - Biologically inspired. (4) Leaky ReLU, ELU, GELU: Variants addressing "dying ReLU" problem. Why depth helps (with non-linearity): Each layer can learn increasingly abstract features. Early layers: edges, textures (simple non-linear combinations). Middle layers: parts, shapes (compositions of early features). Deep layers: objects, concepts (high-level semantic features). This hierarchical feature learning is only possible with non-linearity at each layer. Concrete example (vision): Layer 1: Learns edge detectors (Gabor-like filters). Layer 2: Combines edges into corners, curves. Layer 3: Combines corners into parts (wheel, face). Layer 4: Combines parts into objects (car, person). Without activation, all layers collapse to one linear transformation—no hierarchy! Practical implications: (1) Always use non-linear activations between layers (except final regression output). (2) ReLU is default choice; try others if it fails. (3) Batch normalization can reduce dependence on activation choice. (4) For purely linear relationships (rare), linear regression suffices—no need for deep network. In summary: Composition of linear functions is linear. Non-linear activations break this constraint, enabling neural networks to learn arbitrarily complex functions. Without activation, a 100-layer network has no more expressive power than logistic regression. This is why σ is as important as W!',
          keyPoints: [
            'Composition: T₂(T₁(x)) = A₂A₁x (matrix multiplication); order matters (non-commutative)',
            'Invertible ⟺ det(A)≠0 ⟺ full rank ⟺ bijection (lossless transformation)',
            'ML: backprop through layers reverses composition; autoencoders need invertibility',
          ],
        },
      ],
    },

    {
      id: 'tensor-operations',
      title: 'Tensor Operations in Deep Learning',
      content: `
# Tensor Operations in Deep Learning

## Introduction

**Tensors** are multi-dimensional arrays that generalize scalars (0D), vectors (1D), and matrices (2D) to higher dimensions. They are the fundamental data structure in deep learning frameworks like PyTorch and TensorFlow.

**Dimensionality**:
- **Scalar**: 0D tensor (single number)
- **Vector**: 1D tensor (array of numbers)
- **Matrix**: 2D tensor (table of numbers)
- **3D Tensor**: Batch of matrices or RGB image (width × height × channels)
- **4D Tensor**: Batch of images (batch × channels × height × width)
- **Higher**: Video, sequences, etc.

\`\`\`python
import numpy as np

print("=== Tensor Basics ===")

# Scalars (0D)
scalar = 42
print(f"Scalar (0D): {scalar}")
print(f"Shape: {np.array(scalar).shape}")
print()

# Vectors (1D)
vector = np.array([1, 2, 3, 4])
print(f"Vector (1D): {vector}")
print(f"Shape: {vector.shape}")
print()

# Matrices (2D)
matrix = np.array([[1, 2], [3, 4], [5, 6]])
print(f"Matrix (2D):\\n{matrix}")
print(f"Shape: {matrix.shape}")
print()

# 3D Tensor
tensor_3d = np.random.randn(2, 3, 4)  # 2 matrices, each 3×4
print(f"3D Tensor shape: {tensor_3d.shape}")
print(f"Interpretation: 2 samples, each 3×4")
print()

# 4D Tensor (typical for images)
tensor_4d = np.random.randn(8, 3, 32, 32)  # batch, channels, height, width
print(f"4D Tensor (images) shape: {tensor_4d.shape}")
print(f"Interpretation: batch of 8 images, 3 channels (RGB), 32×32 pixels")
\`\`\`

## Basic Tensor Operations

### Element-wise Operations

\`\`\`python
print("\\n=== Element-wise Operations ===")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A:")
print(A)
print("\\nB:")
print(B)
print()

# Addition
print("A + B (element-wise):")
print(A + B)
print()

# Multiplication
print("A * B (element-wise, Hadamard product):")
print(A * B)
print()

# Exponentiation
print("A ** 2 (element-wise):")
print(A ** 2)
\`\`\`

### Reduction Operations

\`\`\`python
print("\\n=== Reduction Operations ===")

X = np.array([[1, 2, 3],
              [4, 5, 6]])

print("X:")
print(X)
print()

# Sum along different axes
print(f"Sum all: {np.sum(X)}")
print(f"Sum axis=0 (columns): {np.sum(X, axis=0)}")
print(f"Sum axis=1 (rows): {np.sum(X, axis=1)}")
print()

# Mean
print(f"Mean all: {np.mean(X)}")
print(f"Mean axis=0: {np.mean(X, axis=0)}")
print()

# Max/Min
print(f"Max: {np.max(X)}")
print(f"Argmax (index of max): {np.argmax(X)}")
\`\`\`

## Broadcasting

**Broadcasting** allows operations on arrays of different shapes by automatically expanding dimensions.

**Rules**:
1. Align shapes from right
2. Dimensions must be compatible (equal or one is 1)
3. Broadcast smaller dimension to match larger

\`\`\`python
print("\\n=== Broadcasting ===")

# Example 1: Vector + Scalar
v = np.array([1, 2, 3])
s = 10

result1 = v + s  # s broadcast to [10, 10, 10]
print(f"{v} + {s} = {result1}")
print()

# Example 2: Matrix + Vector (row)
M = np.array([[1, 2, 3],
              [4, 5, 6]])
v_row = np.array([10, 20, 30])

result2 = M + v_row  # v_row broadcast to each row
print("Matrix + row vector:")
print(f"M:\\n{M}")
print(f"v: {v_row}")
print(f"M + v:\\n{result2}")
print()

# Example 3: Matrix + Vector (column)
v_col = np.array([[100], [200]])  # Shape (2, 1)

result3 = M + v_col  # v_col broadcast to each column
print("Matrix + column vector:")
print(f"v_col:\\n{v_col}")
print(f"M + v_col:\\n{result3}")
\`\`\`

## Tensor Reshaping

\`\`\`python
print("\\n=== Tensor Reshaping ===")

X = np.arange(12)  # [0, 1, 2, ..., 11]
print(f"Original (1D): {X}")
print(f"Shape: {X.shape}")
print()

# Reshape to 2D
X_2d = X.reshape(3, 4)
print(f"Reshaped to (3, 4):\\n{X_2d}")
print()

# Reshape to 3D
X_3d = X.reshape(2, 2, 3)
print(f"Reshaped to (2, 2, 3):\\n{X_3d}")
print()

# Flatten
X_flat = X_3d.flatten()
print(f"Flattened: {X_flat}")
print()

# Transpose (swap axes)
X_T = X_2d.T
print(f"Transpose of (3, 4):\\n{X_T}")
print(f"Shape: {X_T.shape}")
\`\`\`

## Tensor Contraction (Einstein Summation)

Einstein summation (\`einsum\`) is a powerful notation for tensor operations.

\`\`\`python
print("\\n=== Einstein Summation ===")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication: C[i,j] = Σₖ A[i,k] * B[k,j]
C_matmul = np.einsum('ik,kj->ij', A, B)
print("Matrix multiplication via einsum:")
print(f"A @ B =\\n{C_matmul}")
print(f"Verify: {np.allclose(C_matmul, A @ B)}")
print()

# Trace: sum of diagonal
trace = np.einsum('ii->', A)
print(f"Trace of A: {trace}")
print(f"Verify: {np.trace(A)}")
print()

# Outer product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5])
outer = np.einsum('i,j->ij', v1, v2)
print(f"Outer product:\\n{outer}")
print(f"Verify: {np.allclose(outer, np.outer(v1, v2))}")
\`\`\`

## Batched Operations

Deep learning processes multiple samples simultaneously (batching).

\`\`\`python
print("\\n=== Batched Operations ===")

# Batch of samples
batch_size = 4
input_dim = 3
output_dim = 2

# Weight matrix
W = np.random.randn(input_dim, output_dim)
b = np.random.randn(output_dim)

# Batch of inputs
X_batch = np.random.randn(batch_size, input_dim)

print(f"Batch size: {batch_size}")
print(f"Input dim: {input_dim}")
print(f"Output dim: {output_dim}")
print()

# Batched matrix multiplication
Z = X_batch @ W + b  # Broadcasting bias

print(f"X_batch shape: {X_batch.shape}")
print(f"W shape: {W.shape}")
print(f"Z shape: {Z.shape}")
print()
print("Each row of Z is the transformation of corresponding row in X_batch")
\`\`\`

## Convolutional Operations

Convolution is a key operation in computer vision.

\`\`\`python
print("\\n=== Convolution (Simplified) ===")

from scipy.signal import correlate2d

# 5×5 image
image = np.array([
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 2, 3, 4, 5]
])

# 3×3 edge detection kernel
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Convolution (technically cross-correlation)
output = correlate2d(image, kernel, mode='valid')

print(f"Image shape: {image.shape}")
print(f"Kernel shape: {kernel.shape}")
print(f"Output shape: {output.shape}")
print()
print("Output (edge detected):")
print(output)
\`\`\`

## Applications in Deep Learning

### 1. Matrix Multiplication (Dense Layers)

\`\`\`python
print("\\n=== Application: Dense Layer ===")

batch_size = 32
input_features = 784  # e.g., 28×28 image flattened
hidden_units = 128

# Simulate layer
X = np.random.randn(batch_size, input_features)
W = np.random.randn(input_features, hidden_units) * 0.01
b = np.zeros(hidden_units)

# Forward pass
Z = X @ W + b
A = np.maximum(0, Z)  # ReLU

print(f"Input: {X.shape}")
print(f"Weights: {W.shape}")
print(f"Output (before activation): {Z.shape}")
print(f"Output (after ReLU): {A.shape}")
\`\`\`

### 2. Batch Normalization

\`\`\`python
print("\\n=== Application: Batch Normalization ===")

# Batch of activations
X_bn = np.random.randn(32, 10)  # 32 samples, 10 features

# Compute mean and std across batch
mean = np.mean(X_bn, axis=0, keepdims=True)
std = np.std(X_bn, axis=0, keepdims=True)

# Normalize
X_normalized = (X_bn - mean) / (std + 1e-8)

# Scale and shift (learnable parameters)
gamma = np.ones((1, 10))
beta = np.zeros((1, 10))
X_bn_out = gamma * X_normalized + beta

print(f"Input shape: {X_bn.shape}")
print(f"Mean shape: {mean.shape}")
print(f"Output shape: {X_bn_out.shape}")
print()
print(f"Output mean: {np.mean(X_bn_out, axis=0)}")  # ≈ 0
print(f"Output std: {np.std(X_bn_out, axis=0)}")    # ≈ 1
\`\`\`

### 3. Attention Mechanism

\`\`\`python
print("\\n=== Application: Attention Mechanism (Simplified) ===")

seq_len = 5
d_model = 8

# Query, Key, Value matrices
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

# Attention scores: QK^T / sqrt(d_model)
scores = (Q @ K.T) / np.sqrt(d_model)

# Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scores)

# Weighted sum of values
output = attention_weights @ V

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print(f"Output shape: {output.shape}")
print()
print("Attention weights (row i shows attention from token i to all tokens):")
print(attention_weights.round(2))
\`\`\`

## Memory Layout and Performance

\`\`\`python
print("\\n=== Memory Layout ===")

# Row-major (C-style) vs Column-major (Fortran-style)
A_row_major = np.array([[1, 2, 3], [4, 5, 6]], order='C')
A_col_major = np.array([[1, 2, 3], [4, 5, 6]], order='F')

print("Row-major (C-style): consecutive elements in same row are contiguous")
print("Column-major (F-style): consecutive elements in same column are contiguous")
print()

# Check strides (bytes to skip for each dimension)
print(f"Row-major strides: {A_row_major.strides}")
print(f"Column-major strides: {A_col_major.strides}")
print()

print("→ Access patterns matter for performance!")
print("→ Iterate along contiguous dimension for cache efficiency")
\`\`\`

## Summary

**Tensors**: Multi-dimensional arrays (generalize vectors/matrices)
- 0D: Scalar
- 1D: Vector
- 2D: Matrix
- 3D+: Higher-order tensors

**Key Operations**:
- **Element-wise**: +, *, exp, etc. (Hadamard)
- **Reduction**: sum, mean, max along axes
- **Broadcasting**: Automatic shape expansion
- **Reshaping**: Change dimensions without copying data
- **Contraction**: Einstein summation (einsum)
- **Batching**: Process multiple samples simultaneously

**Deep Learning Applications**:
- **Dense layers**: Batched matrix multiplication
- **Convolution**: Local connectivity for images
- **Batch normalization**: Normalize across batch
- **Attention**: Weighted aggregation of sequences
- **Memory layout**: Row vs column major affects performance

**Why Tensors in ML**:
1. **Batching**: Process multiple samples in parallel (GPU efficiency)
2. **High-dimensional data**: Images (4D), videos (5D), sequences
3. **Unified operations**: Same code for scalars, vectors, matrices, tensors
4. **Automatic differentiation**: Deep learning frameworks compute gradients
5. **Hardware acceleration**: GPUs/TPUs optimized for tensor operations

**Best Practices**:
- Use batching for parallel processing
- Leverage broadcasting to avoid loops
- Be mindful of shapes (debug tool: print tensor.shape)
- Use contiguous memory layouts when possible
- Prefer built-in operations over manual loops (vectorization)

Understanding tensor operations is essential for implementing and optimizing deep learning models!
`,
      multipleChoice: [
        {
          id: 'tensor-q1',
          question:
            'What is the shape of a batch of 16 RGB images, each 64×64 pixels, in the standard (batch, channels, height, width) format?',
          options: [
            '(16, 64, 64, 3)',
            '(16, 3, 64, 64)',
            '(64, 64, 3, 16)',
            '(3, 16, 64, 64)',
          ],
          correctAnswer: 1,
          explanation:
            'In PyTorch/TensorFlow, images are typically stored as (batch, channels, height, width). For 16 RGB images (3 channels) of size 64×64: (16, 3, 64, 64). Some frameworks like TensorFlow can use (batch, height, width, channels).',
        },
        {
          id: 'tensor-q2',
          question:
            'Broadcasting allows operations on tensors of different shapes. What is the result shape of A (3, 1) + B (1, 4)?',
          options: ['(3, 4)', '(3, 1)', '(1, 4)', 'Error: incompatible shapes'],
          correctAnswer: 0,
          explanation:
            'Broadcasting expands dimensions of size 1. A (3, 1) broadcasts column-wise to (3, 4), B (1, 4) broadcasts row-wise to (3, 4). Result shape: (3, 4).',
        },
        {
          id: 'tensor-q3',
          question:
            'In a dense neural network layer with input shape (batch_size, input_dim) and weight matrix (input_dim, output_dim), what is the output shape?',
          options: [
            '(input_dim, output_dim)',
            '(batch_size, input_dim)',
            '(batch_size, output_dim)',
            '(output_dim, batch_size)',
          ],
          correctAnswer: 2,
          explanation:
            'Matrix multiplication (batch_size, input_dim) @ (input_dim, output_dim) = (batch_size, output_dim). Each sample (row) is transformed from input_dim to output_dim features.',
        },
        {
          id: 'tensor-q4',
          question: 'What does np.einsum("ij,jk->ik", A, B) compute?',
          options: [
            'Element-wise multiplication',
            'Matrix multiplication A @ B',
            'Outer product',
            'Transpose',
          ],
          correctAnswer: 1,
          explanation:
            'The einsum notation "ij,jk->ik" means: sum over repeated index j, resulting in i×k matrix. This is exactly matrix multiplication: C[i,k] = Σⱼ A[i,j] * B[j,k].',
        },
        {
          id: 'tensor-q5',
          question: 'Why is batching important in deep learning?',
          options: [
            'It makes code simpler',
            'It enables parallel processing on GPUs and reduces per-sample overhead',
            'It always improves model accuracy',
            'It reduces memory usage',
          ],
          correctAnswer: 1,
          explanation:
            "Batching processes multiple samples simultaneously, enabling GPU parallelism (thousands of cores) and amortizing overhead across samples. However, it increases memory usage (trade-off) and doesn't directly affect accuracy (training dynamics may change).",
        },
      ],
      quiz: [
        {
          id: 'tensor-d1',
          question:
            'Explain broadcasting in NumPy/PyTorch. What are the rules for broadcasting, and why is it useful for deep learning? Provide examples showing both valid and invalid broadcasting scenarios.',
          sampleAnswer:
            "Broadcasting is a mechanism that allows tensor operations on arrays of different shapes without explicit replication, saving memory and computation. Rules: (1) Align shapes from the right (trailing dimensions). (2) Dimensions are compatible if equal or one is 1. (3) Missing dimensions are treated as 1. (4) Broadcast smaller tensor by repeating along dimensions of size 1. Examples: Valid broadcasting: (1) Scalar + Matrix: (1,) + (3, 4) → (3, 4). Scalar broadcasts to every element. (2) Vector + Matrix (row): (4,) + (3, 4) → (3, 4). Vector (4,) becomes (1, 4), broadcasts row-wise. (3) Column + Matrix: (3, 1) + (3, 4) → (3, 4). Column broadcasts across columns. (4) 3D + 2D: (2, 3, 4) + (3, 4) → (2, 3, 4). 2D (3, 4) broadcasts across first dimension. (5) Different missing dims: (5, 1, 7) + (1, 6, 1) → (5, 6, 7). Both broadcast along their size-1 dimensions. Invalid broadcasting: (1) (3,) + (4,): Shapes don't align. 3 ≠ 4 and neither is 1. (2) (3, 4) + (3, 5): Last dimensions 4 ≠ 5. (3) (2, 3) + (3, 2): Shapes differ in incompatible ways. Why useful in deep learning: (1) Adding bias: X @ W + b where X is (batch, features), b is (features,). Broadcasting adds bias to each sample without loops. (2) Batch normalization: Normalize features across batch, then scale/shift with learnable (1, features) parameters. (3) Attention masks: (batch, seq_len, seq_len) attention scores + (1, 1, seq_len) mask. (4) Memory efficiency: Don't need to explicitly replicate data. A (1000, 1) array broadcasted to (1000, 1000) uses 1000× less memory than materializing full array. (5) Performance: Vectorized operations faster than Python loops. Internals: Broadcasting doesn't copy data—it adjusts strides. For array A with shape (3, 1), stride (8, 8) becomes (3, 4) with stride (8, 0). Stride 0 means \"repeat same element.\" Common pitfalls: (1) Unexpected broadcasting: A (3,) + B (3, 1) → (3, 3), not (3,). Solution: be explicit about shapes (reshape). (2) Memory explosion: Broadcasting large tensors can consume lots of memory if result is materialized. (3) Debugging: Print shapes religiously! Most bugs are shape mismatches. Best practices: (1) Use keepdims=True in reductions to preserve dimensions: np.mean(X, axis=1, keepdims=True) gives (n, 1) not (n,). (2) Be explicit with unsqueeze/expand_dims when needed. (3) Understand your framework: PyTorch and NumPy follow same rules, but TensorFlow has subtle differences. In summary: Broadcasting is a powerful abstraction that makes tensor code concise and efficient. Master the rules and common patterns—they're fundamental to writing effective deep learning code.",
          keyPoints: [
            'Broadcasting rules: align shapes from right, dimensions compatible if equal or 1',
            'Enables (batch, features) + (features,) without loops; fails if incompatible',
            'ML: bias addition, batch norm (X-mean)/std, attention mechanisms use broadcasting',
          ],
        },
        {
          id: 'tensor-d2',
          question:
            'Compare memory layout (row-major vs column-major) and its impact on performance. Why does access pattern matter, and how should you structure loops and operations for optimal cache utilization?',
          sampleAnswer:
            "Memory layout determines how multi-dimensional arrays are stored in linear (1D) memory. This profoundly affects performance due to cache locality. Row-major (C-style): Store rows contiguously. For 2D array A[i,j], element A[i, j] is at offset i*n_cols + j. Consecutive elements in same row are adjacent in memory. Column-major (Fortran-style): Store columns contiguously. Element A[i, j] is at offset i + j*n_rows. Consecutive elements in same column are adjacent. NumPy default: Row-major. MATLAB/Fortran: Column-major. Why it matters: Modern CPUs have hierarchical cache (L1, L2, L3). Accessing contiguous memory is ~100× faster than random access (cache lines fetch ~64 bytes). Cache-friendly access: If data is contiguous, CPU prefetches nearby elements. Cache-hostile access: Jumping around memory causes cache misses, stalling computation. Example (row-major matrix): Good: Iterate rows in inner loop: for i: for j: A[i, j]. Elements A[i, 0], A[i, 1], ... are contiguous. Bad: Iterate columns in inner loop: for j: for i: A[i, j]. Elements A[0, j], A[1, j], ... are strided by n_cols. For 1000×1000 matrix, bad pattern is 10-100× slower! Matrix multiplication performance: Consider C = A @ B. Naive triple loop: for i: for j: for k: C[i,j] += A[i,k] * B[k,j]. Access patterns: A[i,k]: row-wise (good). B[k,j]: column-wise (bad if row-major). C[i,j]: scattered writes (also bad). Optimized: Transpose B first, or use blocked algorithms (BLAS libraries do this). Modern BLAS (like OpenBLAS, MKL) achieve >90% peak hardware performance via careful cache optimization. Deep learning implications: (1) Batch dimension first: Store data as (batch, features). When processing batch, consecutive samples are contiguous—good for cache. (2) Vectorization: Use built-in operations (numpy, PyTorch) that exploit SIMD and cache. Hand-written loops in Python are 100-1000× slower. (3) Memory-bound vs compute-bound: Small models are often memory-bound (waiting for data). Large models (GPUs) are compute-bound (waiting for FLOPS). Cache matters more for memory-bound. (4) Convolutional layers: Implement as im2col (image-to-column) + matrix multiply. Transforms 2D convolution into cache-friendly matmul. Practical tips: (1) Prefer np.sum(axis=1) over manual loops. NumPy/PyTorch iterate optimally. (2) Use contiguous() in PyTorch if tensor is strided: x = x.transpose(0, 1).contiguous(). (3) Profile code: Use tools to find cache misses (e.g., perf on Linux). (4) For large matrices: Use libraries (BLAS, cuBLAS). They've invested decades in optimization. (5) On GPUs: Memory coalescing similar concept. Threads in same warp should access contiguous memory. Example benchmark (1000×1000 matrix): Python loops (bad): 1000ms. NumPy (good): 10ms (100× faster). Optimized BLAS (best): 1ms (1000× faster than naive). Takeaway: Cache matters. Structure data access contiguously. Use vetorized libraries. Don't write raw loops in Python.",
          keyPoints: [
            'Batched ops: process multiple samples simultaneously (matrix ops, not loops)',
            'GPU efficiency: 1000s of cores parallelize; batch size 32-256 typical (powers of 2)',
            'Trade-off: larger batch → better GPU utilization but more memory, fewer updates',
          ],
        },
        {
          id: 'tensor-d3',
          question:
            'Explain the attention mechanism in Transformers using tensor operations. Describe the shapes at each step and the role of Q, K, V matrices. Why is attention computed as softmax(QKᵀ/√d)V?',
          sampleAnswer:
            'Attention is the core of Transformers, enabling models to focus on relevant parts of input. Mathematically, it\'s elegant tensor operations. Setup: Input sequence: X (seq_len, d_model). E.g., (10, 512) for 10 tokens, each 512-dim embedding. Learn three projections: Query Q = XWq, Key K = XWk, Value V = XWv. Where Wq, Wk, Wv are (d_model, d_k) matrices (d_k often = d_model). Shapes: Q: (seq_len, d_k). K: (seq_len, d_k). V: (seq_len, d_v), often d_v = d_k. Step 1: Compute attention scores. Scores = Q @ Kᵀ. Shape: (seq_len, d_k) @ (d_k, seq_len) = (seq_len, seq_len). Interpretation: Scores[i, j] = similarity between query i and key j (dot product). High score = query i attends strongly to position j. Step 2: Scale. Scores_scaled = Scores / √d_k. Why? Dot products grow with dimension d_k. For large d_k, scores can be huge, making softmax saturate (gradients vanish). Dividing by √d_k keeps variance ~1, stabilizing training. Step 3: Softmax. Attention_weights = softmax(Scores_scaled, dim=-1). Shape: (seq_len, seq_len). Each row is a probability distribution (sums to 1). Row i gives attention distribution: how much query i attends to each position. Step 4: Weighted sum of values. Output = Attention_weights @ V. Shape: (seq_len, seq_len) @ (seq_len, d_v) = (seq_len, d_v). Interpretation: Output[i] = weighted combination of all values, with weights from attention_weights[i]. Position i "looks at" all positions, weighted by relevance. Why this works: (1) Q @ Kᵀ measures compatibility (dot product = cosine similarity if normalized). (2) Softmax converts scores to probabilities (non-negative, sum to 1). (3) Weighted sum aggregates information from relevant positions. Multi-head attention: Instead of single attention, use h heads in parallel. Split d_model into h heads: each head has dimension d_k = d_model / h. Concatenate outputs from all heads, then project. Allows model to attend to different aspects (e.g., syntax, semantics) simultaneously. Masked attention (for autoregressive models): Add mask to prevent position i from attending to future positions j > i. Set Scores[i, j > i] = -∞ before softmax, so they get 0 weight. Efficiency considerations: (1) Scores matrix (seq_len, seq_len) is O(seq_len²) memory and compute. Problem for long sequences (e.g., 10k tokens → 100M matrix). (2) Solutions: Sparse attention (attend only to subset), linear attention (approximate with low-rank), FlashAttention (optimize memory access). Example (simplified):seq_len = 4, d_k = 8. Q = [[q1], [q2], [q3], [q4]], each qᵢ is 8-dim. K = [[k1], [k2], [k3], [k4]]. Scores = [[q1·k1, q1·k2, q1·k3, q1·k4], [q2·k1, q2·k2, q2·k3, q2·k4], ..]. Each row: how much token i attends to all tokens. After softmax, row 1 might be [0.1, 0.6, 0.2, 0.1]: token 1 mostly attends to token 2. Output[1] = 0.1*v1 + 0.6*v2 + 0.2*v3 + 0.1*v4. Why attention revolutionized NLP: (1) Captures long-range dependencies (RNNs struggled). (2) Parallelizable (unlike sequential RNNs). (3) Interpretable (attention weights show what model attends to). In code (PyTorch-style): scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k). attn_weights = F.softmax(scores, dim=-1). output = attn_weights @ V. Three lines, yet transforms NLP! Understanding tensor shapes and operations is key to implementing and debugging Transformers.',
          keyPoints: [
            'Deep learning ops: matrix mult (layers), batch norm (normalization), attention (QK^T)',
            'einsum: flexible notation for tensor contractions (matrix mult, trace, etc.)',
            'Memory layout: row-major (C) vs column-major (Fortran); contiguous for efficiency',
          ],
        },
      ],
    },

    {
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
print(f"Non-zero elements: {np.count_nonzero(dense)} / {dense.size}")
print(f"Sparsity: {1 - np.count_nonzero(dense)/dense.size:.1%}")
print(f"Memory: {dense.nbytes} bytes")
print()

# Sparse matrix (efficient)
sparse_csr = sparse.csr_matrix(dense)

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

csr = sparse.csr_matrix(dense)

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

csc = sparse.csc_matrix(dense)

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
X_tfidf = vectorizer.fit_transform(docs)

print(f"Shape: {X_tfidf.shape}")  # (4 docs, vocab_size)
print(f"Type: {type(X_tfidf)}")
print(f"Non-zeros: {X_tfidf.nnz} / {X_tfidf.shape[0] * X_tfidf.shape[1]}")
print(f"Sparsity: {(1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])):.1%}")
print()

print("Vocabulary:")
print(list(vectorizer.vocabulary_.keys()))
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

ratings = sparse.random(n_users, n_items, density=density, format='csr')
ratings.data = np.random.randint(1, 6, size=ratings.nnz)  # Ratings 1-5

print(f"Users: {n_users}")
print(f"Items: {n_items}")
print(f"Total possible ratings: {n_users * n_items:,}")
print(f"Actual ratings: {ratings.nnz:,}")
print(f"Sparsity: {(1 - ratings.nnz / (n_users * n_items)):.2%}")
print()

# Collaborative filtering: item-item similarity
# Compute item vectors (each column of ratings)
item_similarity = cosine_similarity(ratings.T, dense_output=False)

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
data = np.ones(n_edges)

# Adjacency matrix
A_graph = sparse.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
A_graph = A_graph.tocsr()

print(f"Nodes: {n_nodes}")
print(f"Edges: {A_graph.nnz}")
print(f"Sparsity: {(1 - A_graph.nnz / (n_nodes**2)):.2%}")
print()

# Compute degree (sum of each row)
degrees = np.array(A_graph.sum(axis=1)).flatten()

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
W_sparse = sparse.random(input_dim, output_dim, density=1-sparsity, format='csr')

print(f"Weight matrix: {input_dim} → {output_dim}")
print(f"Dense parameters: {input_dim * output_dim:,}")
print(f"Sparse parameters: {W_sparse.nnz:,}")
print(f"Compression: {(input_dim * output_dim) / W_sparse.nnz:.1f}x")
print()

# Forward pass
x = np.random.randn(input_dim)
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
A_sparse = sparse.random(n, n, density=0.01, format='csr')
A_sparse = A_sparse @ A_sparse.T + sparse.eye(n) * 0.1  # Make SPD

b = np.random.randn(n)

print(f"System size: {n}×{n}")
print(f"Non-zeros: {A_sparse.nnz}")
print()

# Conjugate Gradient (iterative)
x_cg, info = cg(A_sparse, b, tol=1e-6)

if info == 0:
    print("Conjugate Gradient converged")
    print(f"Solution norm: {np.linalg.norm(x_cg):.4f}")
    print(f"Residual: {np.linalg.norm(A_sparse @ x_cg - b):.2e}")

# Direct solve (LU factorization)
x_direct = spsolve(A_sparse, b)

print(f"\\nDirect solve solution norm: {np.linalg.norm(x_direct):.4f}")
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
      multipleChoice: [
        {
          id: 'sparse-q1',
          question:
            'What is the primary advantage of sparse matrices over dense matrices?',
          options: [
            'They are always faster',
            'They store only non-zero elements, saving memory and computation',
            'They are easier to implement',
            'They give more accurate results',
          ],
          correctAnswer: 1,
          explanation:
            'Sparse matrices store only non-zero elements (and their positions), using O(nnz) memory vs O(n²) for dense. Operations are also O(nnz) vs O(n²), making them much faster for highly sparse data. Accuracy and ease of implementation are not primary advantages.',
        },
        {
          id: 'sparse-q2',
          question:
            'Which sparse format is most efficient for row-wise operations (e.g., accessing rows, matrix-vector products)?',
          options: [
            'COO (Coordinate)',
            'CSR (Compressed Sparse Row)',
            'CSC (Compressed Sparse Column)',
            'Dense',
          ],
          correctAnswer: 1,
          explanation:
            'CSR (Compressed Sparse Row) is optimized for row operations. It stores row pointers, making row access O(nnz_row) and matrix-vector products efficient. CSC is for columns, COO is simple but not optimized for arithmetic.',
        },
        {
          id: 'sparse-q3',
          question:
            'In a recommender system with 100,000 users and 50,000 items, why is the user-item rating matrix typically very sparse?',
          options: [
            'Items are usually identical',
            'Users typically rate only a tiny fraction of all items',
            'The matrix is stored inefficiently',
            'Ratings are always zero',
          ],
          correctAnswer: 1,
          explanation:
            'Users interact with (rate/buy/view) only a small fraction of items—often <0.1%. For 100k users × 50k items = 5 billion possible ratings, actual ratings might be ~10 million (99.8% sparse). This is why sparse formats are essential for recommender systems.',
        },
        {
          id: 'sparse-q4',
          question:
            'What happens to sparsity when you multiply two sparse matrices A and B?',
          options: [
            'The result is always as sparse as the sparser of A and B',
            'The result can be significantly denser than both A and B',
            'The result is always dense',
            'Sparsity is exactly preserved',
          ],
          correctAnswer: 1,
          explanation:
            'Matrix multiplication can increase density. If A has non-zero at (i,k) and B at (k,j), result has non-zero at (i,j). With many such "paths," C = AB can be much denser than A or B individually. This is called "fill-in."',
        },
        {
          id: 'sparse-q5',
          question:
            'For solving a very large sparse linear system Ax = b, which approach is typically preferred?',
          options: [
            'LU decomposition (direct)',
            'Matrix inversion A⁻¹b',
            'Iterative methods (Conjugate Gradient, GMRES)',
            'Normal equations (AᵀA)⁻¹Aᵀb',
          ],
          correctAnswer: 2,
          explanation:
            'For large sparse systems, iterative methods (CG for SPD, GMRES for general) are preferred. Direct methods (LU) suffer from "fill-in" (L and U can be much denser than A). Matrix inversion is never recommended (numerical instability, fill-in). Normal equations square the condition number.',
        },
      ],
      quiz: [
        {
          id: 'sparse-d1',
          question:
            'Explain the three main sparse matrix formats (COO, CSR, CSC). When would you choose each one, and what are the trade-offs in terms of memory, construction time, and operation efficiency?',
          sampleAnswer:
            'Sparse matrix formats trade-off between construction simplicity, memory overhead, and operation efficiency. COO (Coordinate Format): Structure: Three arrays: row indices, column indices, data. Example: row=[0,1,2], col=[2,0,1], data=[5,3,7] represents matrix with (0,2)=5, (1,0)=3, (2,1)=7. Memory: 3×nnz (row, col, data). Pros: (1) Simple to construct—just append (row, col, val) triplets. (2) Good for building matrices incrementally. (3) Easy to understand. Cons: (1) No random access—must scan all entries to find element. (2) No efficient arithmetic (addition, multiplication). (3) Duplicate entries allowed (must sum on conversion). When to use: Initial construction, data loading, converting to other formats. Never use for arithmetic. CSR (Compressed Sparse Row): Structure: Three arrays: data (nnz values), indices (nnz column indices), indptr (n_rows+1 row pointers). Example: indptr=[0,2,3,5] means row 0 has elements at data[0:2], row 1 at data[2:3], row 2 at data[3:5]. Memory: 2×nnz + (n_rows+1). Slightly more efficient than COO. Pros: (1) Fast row slicing: row i is data[indptr[i]:indptr[i+1]]. O(nnz_row) not O(nnz). (2) Efficient matrix-vector product: y[i] = Σⱼ A[i,j]*x[j] iterates through row i. (3) Fast row operations (row sums, etc.). (4) Standard format for most sparse libraries. Cons: (1) Column slicing slow—must scan all rows. (2) Changing sparsity structure expensive (inserting/deleting non-zeros requires shifting arrays). (3) Construction slower than COO. When to use: Default format for arithmetic, matrix-vector products, iterative solvers. Especially when accessing rows. CSC (Compressed Sparse Column): Structure: Like CSR but column-oriented. data, indices (now row indices), indptr (now column pointers). Memory: Same as CSR. Pros: (1) Fast column slicing. (2) Efficient operations like Aᵀx (transpose-vector product) or column sums. Cons: (1) Row slicing slow. (2) Same structure change issues as CSR. When to use: When primarily accessing columns, or when computing Aᵀx frequently. Comparing CSR vs CSC: CSR: Prefer for A @ x (matrix-vector). CSC: Prefer for Aᵀ @ x (transpose-vector, common in optimization). Some libraries (like scipy) can convert between them, but conversion is O(nnz log nnz) (sorting required). Trade-off summary: COO → CSR/CSC: O(nnz log nnz) sorting. Worth it if doing multiple operations. CSR ↔ CSC: O(nnz log nnz). Expensive, avoid frequent conversion. Changing structure (insert/delete): Convert to LIL (List of Lists) or DOK (Dictionary of Keys), modify, convert back. Or rebuild from COO. Practical workflow: (1) Build with COO (simple). (2) Convert to CSR/CSC once. (3) Perform arithmetic in CSR/CSC. (4) Never modify structure of CSR/CSC directly. In deep learning: PyTorch sparse tensors primarily use COO. TensorFlow uses CSR/CSC. Choice depends on typical access patterns.',
          keyPoints: [
            'COO: (row, col, value) triplets, easy construction; CSR: fast row ops, mat-vec',
            'CSC: fast column ops; Trade-off: construction time vs operation speed',
            'Choose based on operations: CSR for sklearn models, CSC for matrix factorization',
          ],
        },
        {
          id: 'sparse-d2',
          question:
            'Discuss "fill-in" in sparse matrix factorization. Why does LU decomposition of a sparse matrix often produce dense factors? How do iterative solvers avoid this problem, and when should you use each approach?',
          sampleAnswer:
            "Fill-in is a major challenge in sparse linear algebra: factorizing sparse A often produces much denser factors. What is fill-in? Start with sparse A (most entries zero). Compute LU: A = LU. Often, L and U have many more non-zeros than A! Example: Tridiagonal matrix (3 diagonals, O(n) non-zeros). LU factors: L and U are lower/upper triangular with O(n²) non-zeros. We destroyed sparsity! Why fill-in occurs: LU elimination modifies matrix entries. Zero at (i,j) can become non-zero if elimination affects it. Specifically: A[i,j] → A[i,j] - A[i,k] * A[k,j] / A[k,k]. If A[i,k] ≠ 0 and A[k,j] ≠ 0, then A[i,j] becomes non-zero (even if originally zero). Geometric interpretation: Sparsity pattern reflects graph structure. Fill-in occurs when graph becomes more connected during elimination. Example: Sparse matrix for 2D grid (5-point stencil, nnz = O(n)). LU factors have nnz = O(n^{3/2}) for 2D grid. For 3D: A has O(n) but LU has O(n^{4/3}). Practical impact: Million-dimensional sparse system: A has ~10⁶ non-zeros (feasible). LU might have ~10¹² non-zeros (1 TB memory, infeasible). This makes direct solvers unusable for large sparse systems. Strategies to reduce fill-in: (1) Reordering: Permute rows/columns to minimize fill. Algorithms: Minimum degree, nested dissection, Cuthill-McKee. Can reduce fill significantly but doesn't eliminate it. (2) Incomplete factorizations: ILU (Incomplete LU) discards small entries during factorization, maintaining sparsity. Used as preconditioner for iterative methods. (3) Iterative solvers: Avoid factorization entirely! Iterative Methods (Krylov subspace): Instead of factoring A, iteratively improve solution. Conjugate Gradient (CG): For symmetric positive definite A. Iteration: xₖ₊₁ = xₖ + αₖpₖ (move along search direction pₖ). Each iteration requires matvec Apₖ, which is O(nnz). Never forms A⁻¹ or LU! Converges in ≤n iterations (theory), often much faster (practice). GMRES: For general (non-symmetric) A. Similar iteration with different search directions. Advantages of iterative: (1) No fill-in—always work with sparse A. (2) Memory: O(nnz) vs O(n²) for factors. (3) Can stop early (approximate solution often sufficient). (4) Embarrassingly parallelizable (matvec scales well). Disadvantages: (1) Convergence depends on conditioning. Ill-conditioned A requires many iterations. (2) No direct solution—only approximation. (3) Preconditioning often necessary (incomplete LU, multigrid). When to use direct (LU/Cholesky): (1) Small to medium systems (n < 10⁴). (2) Need exact solution. (3) Solve multiple systems with same A (factorize once, reuse). (4) Dense or low fill-in. When to use iterative: (1) Large sparse systems (n > 10⁵). (2) Approximate solution sufficient. (3) Good preconditioner available. (4) Memory constrained. Hybrid approach: Use iterative with direct preconditioner. ILU factors approximate A, accelerate CG/GMRES convergence. Deep learning context: Most DL systems are overdetermined (least squares), use normal equations (AᵀA)x = Aᵀb or QR (iterative often not needed for moderate size). For very large problems (billions of parameters), use stochastic gradient methods—never form Hessian! Takeaway: Fill-in makes direct sparse solvers impractical for large systems. Iterative methods are essential for scalability. Understanding trade-offs helps choose the right solver.",
          keyPoints: [
            'NLP: TF-IDF (10k-100k dims, <1% non-zero); Graphs: adjacency matrix (sparse)',
            'Recommenders: user-item matrix (millions users/items, sparse ratings)',
            'Sparse storage O(nnz) vs dense O(n²); operations O(nnz) vs O(n²)',
          ],
        },
        {
          id: 'sparse-d3',
          question:
            "In deep learning, neural network pruning creates sparse weight matrices. Discuss the benefits and challenges of sparse neural networks. Why haven't they completely replaced dense networks despite potential for huge speedups?",
          sampleAnswer:
            "Sparse neural networks promise massive compression and speedup by setting most weights to zero. Yet they haven't dominated. Why? Benefits of sparse networks: (1) Compression: 90-99% sparsity → 10-100× fewer parameters. A 100MB model becomes 1-10MB. Crucial for mobile/edge devices. (2) Theoretical speedup: O(nnz) vs O(n²) operations. Should be 10-100× faster! (3) Regularization: Sparsity can reduce overfitting (fewer parameters to overfit with). (4) Interpretability: Sparse = few connections, easier to understand. (5) Lottery Ticket Hypothesis: Sparse subnetworks exist that match dense performance. Finding them enables training smaller models from scratch. Challenges and why sparse isn't winning: (1) Hardware efficiency: Modern GPUs/TPUs optimized for dense matrix multiplication (cuBLAS, tensor cores). Dense matmul achieves >80% peak FLOPS. Sparse operations are memory-bound, achieve <20% peak. Even with 10× fewer operations, sparse can be slower than dense on GPU! Example: Dense 1000×1000 @ 1000 vector: 1ms (optimized). Sparse (10% nnz) same operation: 0.5ms (2× speedup, not 10× due to overhead). (2) Irregular memory access: Sparse = random access patterns, poor cache utilization. GPUs rely on coalesced memory access (threads access contiguous memory). Sparse breaks this. (3) Software support: PyTorch/TensorFlow sparse operations immature. Limited layer types (mostly linear), poor autograd support, bugs. Dense has decades of optimization (BLAS, cuDNN). (4) Structured vs unstructured sparsity: Unstructured: Any weight can be zero. Hard to exploit on hardware (irregular). Structured: Entire rows/cols/blocks zero. Easier to implement (prune neurons, not weights). But requires more sparsity for same speedup. (5) Training dynamics: Sparse networks harder to train. Dead neurons (zero gradient) never recover. Dynamic sparsity (change pattern during training) complex. Pruning: Train dense → prune → fine-tune. Requires training dense first! (6) Precision: Sparse often requires lower precision (int8) for real speedup. Combining sparsity + quantization complex. When sparse works well: (1) Extreme sparsity (>99%): At 99.9% sparsity, even inefficient sparse ops win. E.g., embeddings (vocab size 1M, most words rare). (2) CPU inference: CPUs have less memory bandwidth, benefit more from reduced mem access. (3) Specialized hardware: Dedicated sparse accelerators (Google Sparse Core, Cerebras, etc.). (4) Natural sparsity: Some domains inherently sparse (text sparse features, knowledge graphs). Lottery Ticket & Pruning Strategies: Lottery Ticket: Sparse subnetwork exists from random init that can train to same accuracy. Implication: Could skip dense training if we found the subnetwork. Challenge: Finding it requires training dense model (catch-22). Pruning methods: (1) Magnitude: Remove weights with smallest |w|. Simple, effective. (2) Gradient-based: Remove weights with smallest |w·∇w| (impact on loss). (3) Structured: Prune entire neurons/channels/layers. (4) Lottery Ticket Rewinding: Prune, reset to early weights (not init), retrain. Iterative Magnitude Pruning (IMP): Train → prune 20% → fine-tune → repeat. Can reach 90%+ sparsity with <1% accuracy loss. But still requires full dense training each cycle. Future of sparse: (1) Hardware: Specialized accelerators (Cerebras, Graphcore) improve sparse performance. (2) Algorithms: Sparse from scratch (RigL, SET—dynamically grow/prune during training). (3) Structured sparsity: Block-sparse, 2:4 sparsity (hardware-friendly). (4) Hybrid: Sparse backbone + dense heads. Current state: Sparse research active, but production models mostly dense. Exceptions: Embeddings (naturally sparse), MoE (Mixture of Experts—structured sparsity). Takeaway: Sparse neural nets theoretically compelling, practically challenging. Hardware and software need to catch up. Structured sparsity and specialized accelerators are promising directions. For now, dense wins in most scenarios due to mature infrastructure.",
          keyPoints: [
            'Iterative solvers: CG (symmetric positive definite), GMRES (general systems)',
            'Avoid dense ops: matrix multiply often destroys sparsity (fill-in problem)',
            'SciPy: scipy.sparse.linalg.cg, spsolve; scales to millions of variables',
          ],
        },
      ],
    },
  ],
  keyTakeaways: [
    'Vectors represent data points, features, and parameters in ML',
    'Dot product measures similarity and is fundamental to neural networks',
    'Matrices enable compact representation of datasets and transformations',
    'Matrix multiplication is the core operation in neural networks',
    'Different norms (L1, L2, L∞) serve different purposes in ML',
    'Eigenvalues and eigenvectors reveal important data structure',
    'SVD is a powerful decomposition used in dimensionality reduction',
    'PCA uses linear algebra to find principal components',
    'Sparse matrices enable efficient large-scale computation',
    'Understanding linear algebra is essential for deep learning',
  ],
  learningObjectives: [
    'Understand vectors and their geometric interpretation',
    'Perform vector operations: dot product, norms, distances',
    'Master matrix operations and multiplication',
    'Apply linear transformations using matrices',
    'Compute eigenvalues, eigenvectors, and matrix decompositions',
    'Implement PCA for dimensionality reduction',
    'Work with sparse matrices efficiently',
    'Use linear algebra in machine learning algorithms',
    'Implement neural network operations with matrices',
    'Debug common linear algebra errors in ML code',
  ],
  prerequisites: [
    'Basic algebra and functions',
    'Python programming fundamentals',
    'NumPy basics',
  ],
};

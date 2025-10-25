/**
 * Linear Transformations Section
 */

export const lineartransformationsSection = {
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
A = np.array([[np.cos (theta), -np.sin (theta)],
              [np.sin (theta), np.cos (theta)]])

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

def rotation_matrix_2d (theta):
    """Create 2D rotation matrix."""
    return np.array([[np.cos (theta), -np.sin (theta)],
                     [np.sin (theta), np.cos (theta)]])

# Rotate vector [1, 0] by 90 degrees
R_90 = rotation_matrix_2d (np.pi / 2)
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
u_normalized = u / np.linalg.norm (u)
P_line = np.outer (u_normalized, u_normalized)

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
R = rotation_matrix_2d (np.pi / 4)
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
R = rotation_matrix_2d (np.pi / 6)
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
print(f"Recovered original: {np.allclose (v, v_recovered)}")
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

def rotation_matrix_3d_z (theta):
    """Rotate around z-axis."""
    return np.array([[np.cos (theta), -np.sin (theta), 0],
                     [np.sin (theta), np.cos (theta), 0],
                     [0, 0, 1]])

def rotation_matrix_3d_x (theta):
    """Rotate around x-axis."""
    return np.array([[1, 0, 0],
                     [0, np.cos (theta), -np.sin (theta)],
                     [0, np.sin (theta), np.cos (theta)]])

# Rotate around z-axis
R_z = rotation_matrix_3d_z (np.pi / 4)
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
print(f"Match: {np.allclose (x_standard, x_recovered)}")
\`\`\`

### 4. Dimensionality Reduction

Project high-dimensional data to lower dimensions.

\`\`\`python
print("\\n=== Application: Dimensionality Reduction ===")

# 3D data
X_3d = np.random.randn(100, 3)

# Project onto first 2 principal components
# (In practice, use PCA from sklearn)
U, S, Vt = np.linalg.svd(X_3d - X_3d.mean (axis=0), full_matrices=False)
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
    "Rotation (Orthogonal)": rotation_matrix_2d (np.pi/4),
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
    print(f"  Area scaling factor: |det| = {abs (det_T):.2f}")
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
- **Range**: dim (range) = rank(**A**)
- **Null space**: dim (null) = n - rank(**A**)
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
};

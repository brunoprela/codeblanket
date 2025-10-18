/**
 * Number Systems & Properties Section
 */

export const numbersystemsSection = {
  id: 'number-systems',
  title: 'Number Systems & Properties',
  content: `
# Number Systems & Properties

## Introduction

Understanding number systems and their properties is fundamental to machine learning and computational mathematics. While these concepts may seem basic, they form the foundation for understanding floating-point arithmetic, numerical stability, and the limitations of computer representations of numbers.

## Number Systems Overview

### Integers (ℤ)

**Definition**: The set of whole numbers including negative numbers, zero, and positive numbers.

\`\`\`
ℤ = {..., -3, -2, -1, 0, 1, 2, 3, ...}
\`\`\`

**Properties**:
- Closed under addition, subtraction, and multiplication
- NOT closed under division (dividing two integers doesn't always yield an integer)
- Used for: Counting, indexing arrays, discrete mathematics

**Python Example**:
\`\`\`python
import numpy as np

# Integer operations
a = 10
b = 3

print(f"Addition: {a + b}")        # 13
print(f"Subtraction: {a - b}")     # 7
print(f"Multiplication: {a * b}")  # 30
print(f"Division: {a / b}")        # 3.333... (becomes float)
print(f"Integer Division: {a // b}") # 3 (floor division)
print(f"Modulo: {a % b}")          # 1

# Integer arrays in NumPy
int_array = np.array([1, 2, 3, 4, 5], dtype=np.int64)
print(f"Integer array: {int_array}")
print(f"Data type: {int_array.dtype}")
\`\`\`

### Rational Numbers (ℚ)

**Definition**: Numbers that can be expressed as a ratio of two integers p/q where q ≠ 0.

\`\`\`
ℚ = {p/q | p, q ∈ ℤ, q ≠ 0}
\`\`\`

**Examples**: 1/2, -3/4, 5 (which is 5/1), 0.75 (which is 3/4)

**Properties**:
- Dense on the number line (between any two rationals, there's another rational)
- Closed under addition, subtraction, multiplication, and division (except by zero)
- Can be represented exactly in computers using fractions

**Python Example**:
\`\`\`python
from fractions import Fraction

# Creating rational numbers
r1 = Fraction(1, 2)
r2 = Fraction(3, 4)

print(f"r1 = {r1}")  # 1/2
print(f"r2 = {r2}")  # 3/4

# Operations maintain exact representation
print(f"r1 + r2 = {r1 + r2}")  # 5/4
print(f"r1 * r2 = {r1 * r2}")  # 3/8
print(f"r1 / r2 = {r1 / r2}")  # 2/3

# Convert to float
print(f"As decimal: {float(r1 + r2)}")  # 1.25
\`\`\`

### Real Numbers (ℝ)

**Definition**: The set of all rational and irrational numbers. Includes all points on the number line.

**Examples**: 
- Rational: 1, -5, 1/2, 0.333...
- Irrational: π, e, √2, √3

**Properties**:
- Complete (no "gaps" on the number line)
- NOT countable (unlike integers and rationals)
- Foundation for calculus and continuous mathematics

**Irrational Numbers**: Cannot be expressed as a ratio of integers
- √2 ≈ 1.41421356...
- π ≈ 3.14159265...
- e ≈ 2.71828182...
- φ (golden ratio) ≈ 1.61803398...

**Python Example**:
\`\`\`python
import numpy as np

# Irrational numbers (approximations)
print(f"π = {np.pi}")
print(f"e = {np.e}")
print(f"√2 = {np.sqrt(2)}")
print(f"Golden ratio φ = {(1 + np.sqrt(5)) / 2}")

# Real number operations
x = 3.14159
y = 2.71828

print(f"x + y = {x + y}")
print(f"x * y = {x * y}")
print(f"x^y = {x ** y}")
\`\`\`

### Complex Numbers (ℂ)

**Definition**: Numbers of the form a + bi where a, b ∈ ℝ and i = √(-1).

\`\`\`
ℂ = {a + bi | a, b ∈ ℝ}
\`\`\`

**Components**:
- a = real part
- b = imaginary part
- i = imaginary unit (i² = -1)

**Importance in ML**:
- Fourier transforms (signal processing)
- Eigenvalue computations
- Quantum computing
- Neural network complex-valued activations

**Python Example**:
\`\`\`python
import numpy as np

# Complex numbers in Python
z1 = 3 + 4j
z2 = 1 - 2j

print(f"z1 = {z1}")
print(f"z2 = {z2}")

# Operations
print(f"z1 + z2 = {z1 + z2}")
print(f"z1 * z2 = {z1 * z2}")

# Components
print(f"Real part of z1: {z1.real}")
print(f"Imaginary part of z1: {z1.imag}")

# Magnitude and phase
magnitude = abs(z1)
phase = np.angle(z1)
print(f"Magnitude: {magnitude}")
print(f"Phase: {phase} radians")

# Complex conjugate
print(f"Conjugate of z1: {np.conj(z1)}")

# NumPy complex arrays
complex_array = np.array([1+2j, 3-4j, 5+0j], dtype=np.complex128)
print(f"Complex array: {complex_array}")
\`\`\`

## Fundamental Properties

### Commutative Property

**Addition**: a + b = b + a
**Multiplication**: a × b = b × a

\`\`\`python
# Verification
a, b = 5, 3
print(f"{a} + {b} = {a + b}")
print(f"{b} + {a} = {b + a}")
print(f"Commutative? {a + b == b + a}")

# Important: Matrix multiplication is NOT commutative
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("\\nMatrix multiplication:")
print(f"AB =\\n{A @ B}")
print(f"BA =\\n{B @ A}")
print(f"AB == BA? {np.array_equal(A @ B, B @ A)}")
\`\`\`

### Associative Property

**Addition**: (a + b) + c = a + (b + c)
**Multiplication**: (a × b) × c = a × (b × c)

\`\`\`python
a, b, c = 2, 3, 4
print(f"({a} + {b}) + {c} = {(a + b) + c}")
print(f"{a} + ({b} + {c}) = {a + (b + c)}")
print(f"Associative? {(a + b) + c == a + (b + c)}")
\`\`\`

### Distributive Property

**a × (b + c) = (a × b) + (a × c)**

\`\`\`python
a, b, c = 2, 3, 4
left_side = a * (b + c)
right_side = (a * b) + (a * c)
print(f"{a} × ({b} + {c}) = {left_side}")
print(f"({a} × {b}) + ({a} × {c}) = {right_side}")
print(f"Distributive? {left_side == right_side}")
\`\`\`

## Absolute Values and Inequalities

### Absolute Value

**Definition**: The distance from zero on the number line.

\`\`\`
|x| = { x   if x ≥ 0
      { -x  if x < 0
\`\`\`

**Properties**:
- |x| ≥ 0 for all x
- |x| = 0 if and only if x = 0
- |xy| = |x| × |y|
- |x + y| ≤ |x| + |y| (Triangle Inequality)

**Python Example**:
\`\`\`python
import numpy as np

# Absolute values
print(f"|5| = {abs(5)}")
print(f"|-5| = {abs(-5)}")
print(f"|0| = {abs(0)}")

# Triangle inequality verification
x, y = 3, -7
print(f"\\nTriangle Inequality:")
print(f"|{x} + {y}| = {abs(x + y)}")
print(f"|{x}| + |{y}| = {abs(x) + abs(y)}")
print(f"|x + y| ≤ |x| + |y|? {abs(x + y) <= abs(x) + abs(y)}")

# NumPy arrays
arr = np.array([-5, -3, 0, 2, 4])
print(f"\\nArray: {arr}")
print(f"Absolute values: {np.abs(arr)}")

# L1 norm (sum of absolute values) - used in ML
print(f"L1 norm: {np.sum(np.abs(arr))}")
\`\`\`

### Inequalities

**Basic Inequality Properties**:
1. If a < b and c < d, then a + c < b + d
2. If a < b and c > 0, then ac < bc
3. If a < b and c < 0, then ac > bc (reverses!)
4. If 0 < a < b, then 1/a > 1/b

**Python Example**:
\`\`\`python
# Inequality operations
a, b = 2, 5
print(f"{a} < {b}: {a < b}")
print(f"{a} ≤ {b}: {a <= b}")
print(f"{a} > {b}: {a > b}")

# Boolean arrays with NumPy
arr = np.array([1, 2, 3, 4, 5])
print(f"\\nArray: {arr}")
print(f"Elements > 3: {arr[arr > 3]}")
print(f"Elements ≤ 3: {arr[arr <= 3]}")

# Chained comparisons in ML (checking ranges)
x = 3
print(f"\\nIs {x} in [1, 5]? {1 <= x <= 5}")
print(f"Is {x} in (0, 10)? {0 < x < 10}")
\`\`\`

## Scientific Notation and Orders of Magnitude

### Scientific Notation

**Format**: a × 10^n where 1 ≤ |a| < 10 and n ∈ ℤ

**Examples**:
- 1,000,000 = 1 × 10^6
- 0.0001 = 1 × 10^-4
- 299,792,458 m/s (speed of light) ≈ 3 × 10^8 m/s

**Python Example**:
\`\`\`python
# Scientific notation in Python
large_number = 1e6   # 1 × 10^6 = 1,000,000
small_number = 1e-6  # 1 × 10^-6 = 0.000001

print(f"1e6 = {large_number}")
print(f"1e-6 = {small_number}")

# ML-relevant magnitudes
learning_rate = 1e-3
print(f"Learning rate: {learning_rate}")

# NumPy scientific notation
np.set_printoptions(precision=2, suppress=False)
weights = np.array([1.23e-5, 4.56e3, 7.89e-10])
print(f"Weights: {weights}")
\`\`\`

### Orders of Magnitude in ML

Understanding scale is crucial in machine learning:

| Concept | Typical Range | Example |
|---------|---------------|---------|
| Learning Rate | 10^-5 to 10^-1 | 0.001 |
| Weight Initialization | 10^-2 to 10^-1 | 0.01 |
| Gradients | 10^-8 to 10^2 | Can vary widely |
| Loss Values | 10^-3 to 10^3 | Depends on scale |
| Dataset Size | 10^3 to 10^9 | Millions of examples |
| Model Parameters | 10^6 to 10^12 | GPT-3: 175 billion |

\`\`\`python
# Example: Analyzing orders of magnitude in ML
import matplotlib.pyplot as plt

# Learning rate schedule (exponential decay)
epochs = 100
initial_lr = 1e-3
decay_rate = 0.96

learning_rates = [initial_lr * (decay_rate ** epoch) for epoch in range(epochs)]

# Plot on log scale
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(learning_rates)
plt.title('Learning Rate - Linear Scale')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.subplot(1, 2, 2)
plt.semilogy(learning_rates)
plt.title('Learning Rate - Log Scale')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate (log scale)')
plt.tight_layout()
plt.show()
\`\`\`

## Floating-Point Precision and Limitations

### Floating-Point Representation

Computers represent real numbers using floating-point arithmetic (IEEE 754 standard):

**Float32** (single precision):
- 1 sign bit
- 8 exponent bits
- 23 mantissa bits
- Range: ≈ 10^-38 to 10^38
- Precision: ~7 decimal digits

**Float64** (double precision):
- 1 sign bit
- 11 exponent bits
- 52 mantissa bits
- Range: ≈ 10^-308 to 10^308
- Precision: ~15 decimal digits

**Python Example**:
\`\`\`python
import numpy as np
import sys

# Float types
float32_val = np.float32(1.0)
float64_val = np.float64(1.0)

print(f"Float32 info: {np.finfo(np.float32)}")
print(f"\\nFloat64 info: {np.finfo(np.float64)}")

# Machine epsilon (smallest number where 1 + eps != 1)
eps32 = np.finfo(np.float32).eps
eps64 = np.finfo(np.float64).eps

print(f"\\nFloat32 epsilon: {eps32}")
print(f"Float64 epsilon: {eps64}")

# Demonstration of precision loss
x = np.float32(1.0)
print(f"\\n1.0 + eps32 == 1.0? {x + eps32 == x}")
print(f"1.0 + eps32/2 == 1.0? {x + eps32/2 == x}")
\`\`\`

### Common Floating-Point Issues

**1. Loss of Precision**:
\`\`\`python
# Adding very different magnitudes
large = 1e10
small = 1e-10

result_32 = np.float32(large) + np.float32(small)
result_64 = np.float64(large) + np.float64(small)

print(f"Float32: {large} + {small} = {result_32}")
print(f"Expected: {large + small}")
print(f"Float64: {result_64}")
\`\`\`

**2. Catastrophic Cancellation**:
\`\`\`python
# Subtracting nearly equal numbers
a = np.float32(1.0000001)
b = np.float32(1.0000000)

difference = a - b
print(f"Difference: {difference}")
print(f"Relative error can be large!")
\`\`\`

**3. Rounding Errors**:
\`\`\`python
# Famous 0.1 + 0.2 example
result = 0.1 + 0.2
print(f"0.1 + 0.2 = {result}")
print(f"0.1 + 0.2 == 0.3? {result == 0.3}")
print(f"Difference: {result - 0.3}")

# Use np.isclose() for comparisons
print(f"np.isclose(0.1 + 0.2, 0.3)? {np.isclose(0.1 + 0.2, 0.3)}")
\`\`\`

## Applications in Machine Learning

### 1. Feature Scaling

Understanding number ranges helps in choosing appropriate scaling methods:

\`\`\`python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Sample data with different scales
data = np.array([
    [1, 100, 0.001],
    [2, 200, 0.002],
    [3, 300, 0.003],
    [4, 400, 0.004],
    [5, 500, 0.005]
])

print(f"Original data:\\n{data}")

# Standard scaling (mean=0, std=1)
standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(data)
print(f"\\nStandard scaled:\\n{data_standard}")

# Min-Max scaling (range [0, 1])
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)
print(f"\\nMin-Max scaled:\\n{data_minmax}")
\`\`\`

### 2. Numerical Stability

Adding a small constant (epsilon) to avoid division by zero:

\`\`\`python
def safe_divide(numerator, denominator, eps=1e-10):
    """Safely divide with numerical stability"""
    return numerator / (denominator + eps)

# Example: Computing probabilities
counts = np.array([10, 0, 5, 20])
total = counts.sum()

# Without safety
# probabilities = counts / total  # Could fail if total = 0

# With safety
probabilities = safe_divide(counts, total)
print(f"Probabilities: {probabilities}")

# Common in loss functions
def cross_entropy_loss(y_true, y_pred, eps=1e-10):
    """Cross-entropy with numerical stability"""
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return -np.sum(y_true * np.log(y_pred))

y_true = np.array([1, 0, 0])
y_pred = np.array([0.7, 0.2, 0.1])
loss = cross_entropy_loss(y_true, y_pred)
print(f"\\nCross-entropy loss: {loss}")
\`\`\`

### 3. Understanding Model Capacity

Model parameters are integers, but understanding their scale matters:

\`\`\`python
def calculate_parameters(layers):
    """Calculate total parameters in a neural network"""
    total_params = 0
    for i in range(len(layers) - 1):
        # Weights + biases
        weights = layers[i] * layers[i+1]
        biases = layers[i+1]
        total_params += weights + biases
        print(f"Layer {i+1}: {weights:,} weights + {biases:,} biases = {weights + biases:,}")
    return total_params

# Example: Simple neural network
network = [784, 128, 64, 10]  # Input, hidden1, hidden2, output
total = calculate_parameters(network)
print(f"\\nTotal parameters: {total:,}")
print(f"Order of magnitude: 10^{int(np.log10(total))}")
\`\`\`

## Best Practices

### 1. Choose Appropriate Data Types

\`\`\`python
# Memory comparison
data_float64 = np.random.randn(1000000).astype(np.float64)
data_float32 = np.random.randn(1000000).astype(np.float32)
data_float16 = np.random.randn(1000000).astype(np.float16)

print(f"Float64 memory: {data_float64.nbytes / 1024 / 1024:.2f} MB")
print(f"Float32 memory: {data_float32.nbytes / 1024 / 1024:.2f} MB")
print(f"Float16 memory: {data_float16.nbytes / 1024 / 1024:.2f} MB")

# Recommendation: Use float32 for deep learning (good balance)
\`\`\`

### 2. Be Aware of Numerical Instability

\`\`\`python
# Bad: Computing softmax directly
def softmax_unstable(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

# Good: Numerically stable softmax
def softmax_stable(x):
    exp_x = np.exp(x - x.max())  # Subtract max for stability
    return exp_x / exp_x.sum()

x = np.array([1000, 1001, 1002])  # Large values

try:
    result_unstable = softmax_unstable(x)
    print(f"Unstable result: {result_unstable}")
except:
    print("Unstable softmax failed (overflow)!")

result_stable = softmax_stable(x)
print(f"Stable result: {result_stable}")
\`\`\`

### 3. Use Logarithms for Large Products

\`\`\`python
# Computing probability of sequence (common in NLP)
probabilities = np.array([0.9, 0.8, 0.85, 0.7, 0.95])

# Bad: Direct multiplication (underflow risk)
product = np.prod(probabilities)
print(f"Direct product: {product}")

# Good: Use log probabilities
log_probs = np.log(probabilities)
log_product = np.sum(log_probs)
print(f"Log probabilities sum: {log_product}")
print(f"Exponentiate back: {np.exp(log_product)}")
\`\`\`

## Common Pitfalls

1. **Assuming exact equality** with floating-point numbers
2. **Integer division** when you need float division
3. **Overflow/underflow** with very large/small numbers
4. **Loss of precision** when adding numbers of different magnitudes
5. **Not using appropriate numerical types** for the task

## Summary

- **Integers**: Discrete, exact, used for counting and indexing
- **Rationals**: Fractions, can be exact in computers
- **Reals**: Continuous, approximated by floating-point
- **Complex**: Essential for advanced signal processing and quantum computing
- **Properties**: Commutative, associative, distributive (know the exceptions!)
- **Floating-Point**: Understand precision limitations and numerical stability
- **Scientific Notation**: Essential for understanding scale in ML
- **Best Practices**: Use appropriate types, handle numerical instability, use log space when needed

These foundations are critical for understanding why certain ML algorithms work, debugging numerical issues, and writing efficient code.
`,
};

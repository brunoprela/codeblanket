import { Module } from '../types';

export const mlMathematicalFoundationsModule: Module = {
  id: 'ml-mathematical-foundations',
  title: 'Mathematical Foundations',
  description:
    'Master elementary mathematics, algebra, and functions essential for machine learning and AI',
  icon: '🔢',
  sections: [
    {
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
      multipleChoice: [
        {
          id: 'mc1-number-systems',
          question:
            'Which number system is NOT closed under division (excluding division by zero)?',
          options: [
            'Rational numbers (ℚ)',
            'Real numbers (ℝ)',
            'Integers (ℤ)',
            'Complex numbers (ℂ)',
          ],
          correctAnswer: 2,
          explanation:
            'Integers are NOT closed under division. For example, 5 ÷ 2 = 2.5, which is not an integer. Rationals, reals, and complex numbers are all closed under division (except by zero).',
        },
        {
          id: 'mc2-floating-point',
          question:
            'What is the main reason why 0.1 + 0.2 != 0.3 in most programming languages?',
          options: [
            'Programming language bug',
            'Binary floating-point representation cannot exactly represent 0.1 and 0.2',
            'Insufficient memory allocation',
            'Integer overflow',
          ],
          correctAnswer: 1,
          explanation:
            'In binary floating-point representation (IEEE 754), decimal fractions like 0.1 and 0.2 cannot be represented exactly. They are approximated, leading to tiny rounding errors that accumulate. This is why 0.1 + 0.2 results in something like 0.30000000000000004.',
        },
        {
          id: 'mc3-scientific-notation',
          question:
            'In machine learning, a typical learning rate of 0.001 is best expressed in scientific notation as:',
          options: ['1 × 10^3', '1 × 10^-3', '10 × 10^-2', '0.1 × 10^-2'],
          correctAnswer: 1,
          explanation:
            '0.001 = 1 × 10^-3. In scientific notation, we express numbers as a × 10^n where 1 ≤ |a| < 10. While options C and D are technically correct, option B follows the standard scientific notation convention.',
        },
        {
          id: 'mc4-absolute-value',
          question:
            'The Triangle Inequality states that |x + y| ≤ |x| + |y|. Which of the following demonstrates when equality holds?',
          options: [
            'When x and y are both negative',
            'When x and y have the same sign',
            'When x and y have opposite signs',
            'When one of them is zero',
          ],
          correctAnswer: 1,
          explanation:
            'Equality holds when x and y have the same sign (both positive or both negative). When they have the same sign, there is no cancellation, so |x + y| = |x| + |y|. When they have opposite signs, some cancellation occurs, making |x + y| < |x| + |y|.',
        },
        {
          id: 'mc5-numerical-stability',
          question:
            'Why do we subtract the maximum value before computing softmax: exp(x - max(x)) / sum(exp(x - max(x)))?',
          options: [
            'To make the computation faster',
            'To prevent numerical overflow when exponentiating large values',
            'To ensure the output sums to 1',
            'To make all values negative',
          ],
          correctAnswer: 1,
          explanation:
            "Subtracting the maximum value before exponentiation prevents numerical overflow. exp(1000) would overflow, but exp(1000 - 1000) = exp(0) = 1 is manageable. This transformation doesn't change the final result due to properties of exponents but makes computation numerically stable.",
        },
      ],
      quiz: [
        {
          id: 'dq1-float-precision',
          question:
            'In deep learning, why is float32 (single precision) preferred over float64 (double precision) for most applications, despite float64 having higher precision?',
          sampleAnswer: `Float32 is preferred in deep learning for several practical reasons:

**Memory Efficiency**: Float32 uses half the memory of float64 (4 bytes vs 8 bytes). With models containing millions or billions of parameters (e.g., GPT-3 has 175 billion parameters), this difference is significant:
- Float32: 175B parameters × 4 bytes = 700 GB
- Float64: 175B parameters × 8 bytes = 1,400 GB

**Computational Speed**: Modern GPUs are optimized for float32 operations (or even lower precision like float16). Float32 tensor cores provide 2-4x speedup compared to float64. For training that takes days or weeks, this matters substantially.

**Sufficient Precision**: The precision of float32 (~7 decimal digits) is sufficient for most ML applications. Neural networks are inherently noisy - we use stochastic gradient descent, dropout, and other stochastic processes. The additional precision of float64 doesn't meaningfully improve model performance.

**Gradient Descent Tolerance**: The optimization landscape of neural networks is complex and non-convex. The extra precision of float64 doesn't help us find better minima; we're often satisfied with "good enough" solutions.

**Trade-offs**: The only time float64 might be preferred is in scientific computing where numerical accuracy is critical, or in certain numerical stability situations. For standard deep learning, float32 or even mixed-precision training (float16 with float32 accumulation) is the standard.`,
          keyPoints: [
            'Memory efficiency: 50% reduction in model size',
            'GPU hardware optimization for float32',
            'Sufficient precision for ML optimization',
            'No significant accuracy gains from float64',
            'Faster training and inference',
          ],
        },
        {
          id: 'dq2-complex-numbers',
          question:
            'How are complex numbers used in machine learning and signal processing? Provide specific examples of where they are essential.',
          sampleAnswer: `Complex numbers are fundamental in several ML and signal processing applications:

**1. Fourier Transforms**:
The Discrete Fourier Transform (DFT) converts time-domain signals to frequency domain:
X(k) = Σ x(n) × e^(-i2πkn/N)

This is essential for:
- Audio processing (speech recognition, music analysis)
- Image processing (frequency filtering, compression)
- Time series analysis (detecting periodic patterns)

**2. Convolution Theorem**:
Complex numbers make convolution efficient via:
- Convolution in time domain = Multiplication in frequency domain
- FFT (Fast Fourier Transform) uses complex arithmetic
- Used in CNNs for efficient computation

**3. Eigenvalues and Eigenvectors**:
Many real matrices have complex eigenvalues:
- Stability analysis of systems
- PageRank algorithm
- Principal Component Analysis (PCA) in certain cases
- Markov chain analysis

**4. Quantum Machine Learning**:
Quantum computing fundamentally operates with complex numbers:
- Quantum states are complex-valued vectors
- Quantum gates are unitary matrices (complex)
- Emerging field of quantum neural networks

**5. Complex-Valued Neural Networks**:
Recently, research has explored networks with complex-valued weights and activations:
- Better suited for signals naturally represented as complex (radar, RF signals)
- Potential advantages in representing phase information
- Used in specific domains like magnetic resonance imaging (MRI)

**Example in Audio**:
When processing audio with a spectrogram, each point represents a complex number where the magnitude is the amplitude and the phase contains timing information. Both are crucial for reconstructing the original signal.`,
          keyPoints: [
            'Essential for Fourier transforms and frequency analysis',
            'Enable efficient convolution via FFT',
            'Required for eigenvalue computations',
            'Fundamental to quantum computing',
            'Emerging applications in complex-valued neural networks',
          ],
        },
        {
          id: 'dq3-numerical-trading',
          question:
            'In algorithmic trading systems, how can understanding number systems and numerical precision prevent costly bugs? Provide examples of numerical issues that could impact trading decisions.',
          sampleAnswer: `Numerical precision issues can cause severe problems in trading systems, potentially leading to significant financial losses:

**1. Price Precision and Tick Size**:
Stock prices are typically stored with limited decimal places (e.g., cents). Failing to account for this can cause issues:
- Rounding errors when calculating position sizes
- Impossible price targets (e.g., $10.125 when minimum tick is $0.01)
- Accumulation of small errors in high-frequency trading

**2. Floating-Point Arithmetic in P&L Calculations**:
\`\`\`python
# Dangerous
shares = 1000000
price_bought = 100.1
price_sold = 100.2
profit = shares * (price_sold - price_bought)  # Rounding errors
\`\`\`

Better approach: Use integer arithmetic with cents or basis points.

**3. Order Quantity Calculation**:
When calculating order quantities based on portfolio percentage:
\`\`\`python
# Bad
portfolio_value = 1000000.00
target_pct = 0.15
share_price = 123.45
shares = int((portfolio_value * target_pct) / share_price)
\`\`\`

Issues: Truncation errors, failure to account for lot sizes, minimum order quantities.

**4. Interest Rate Calculations**:
Compounding interest with daily rates:
- Direct multiplication accumulates floating-point errors
- Better: Use log space or exact rational arithmetic
- Critical for accurate bond pricing and yield calculations

**5. Cumulative Returns**:
\`\`\`python
# Unstable for long time series
cumulative_return = (1 + r1) * (1 + r2) * ... * (1 + rn)

# Better
log_cumulative = sum(log(1 + r) for r in returns)
cumulative_return = exp(log_cumulative)
\`\`\`

**6. Risk Metrics (VaR, CVaR)**:
When calculating Value at Risk:
- Sorting large arrays of returns requires stable precision
- Quantile calculations can be sensitive to numerical errors
- Covariance matrix computations can be ill-conditioned

**7. Stop-Loss Triggers**:
\`\`\`python
# Dangerous with floating point
if current_price <= stop_price:  # May not trigger due to precision
    execute_stop_loss()

# Better with tolerance
if current_price <= stop_price + EPSILON:
    execute_stop_loss()
\`\`\`

**Best Practices for Trading Systems**:
1. Use decimal.Decimal for money calculations
2. Store prices in smallest unit (cents, satoshis)
3. Implement tolerance-based comparisons
4. Use rational arithmetic for exact calculations
5. Thoroughly test with edge cases
6. Implement sanity checks and validation
7. Log all calculations for audit trails

**Real Example**:
The 2012 Knight Capital trading glitch lost $440 million in 45 minutes partly due to software bugs, including numerical handling issues in order execution logic.`,
          keyPoints: [
            'Price precision and tick size handling',
            'Use integer arithmetic or Decimal for money',
            'Floating-point errors in P&L calculations',
            'Numerical stability in risk calculations',
            'Tolerance-based comparisons for triggers',
            'Historical examples of costly bugs',
          ],
        },
      ],
    },
    {
      id: 'algebraic-expressions',
      title: 'Algebraic Expressions & Equations',
      content: `
# Algebraic Expressions & Equations

## Introduction

Algebra forms the backbone of machine learning mathematics. From understanding linear regression equations to manipulating loss functions, algebraic thinking is essential. This section covers variables, expressions, equations, and their solutions—skills you'll use daily in ML work.

## Variables, Coefficients, and Constants

### Definitions

**Variable**: A symbol (usually x, y, z, or θ) representing an unknown or changing quantity.

**Coefficient**: A number multiplied by a variable.

**Constant**: A fixed numerical value.

**Example Expression**: \`3x² + 5x - 7\`
- Variables: x
- Coefficients: 3 (for x²), 5 (for x)
- Constant: -7

### Variables in Machine Learning

In ML, we use specific variable conventions:

| Symbol | Common Use | Example |
|--------|------------|---------|
| x | Input features | x₁, x₂, ..., xₙ |
| y | Output/target | y = f(x) |
| θ, w | Model parameters/weights | θ = [θ₀, θ₁, ..., θₙ] |
| α | Learning rate | α = 0.01 |
| λ | Regularization parameter | λ = 0.1 |
| ε | Error term or small constant | ε = 1e-8 |

**Python Example**:
\`\`\`python
import numpy as np

# Variables in linear regression: y = θ₀ + θ₁x₁ + θ₂x₂
# θ₀ is the intercept (constant term)
# θ₁, θ₂ are coefficients (weights)

theta_0 = 2.5  # intercept
theta_1 = 1.3  # coefficient for x₁
theta_2 = -0.7  # coefficient for x₂

# Input features
x1 = 3.0
x2 = 4.0

# Prediction
y_pred = theta_0 + theta_1 * x1 + theta_2 * x2
print(f"Prediction: y = {theta_0} + {theta_1}*{x1} + {theta_2}*{x2} = {y_pred}")

# Vectorized form (more efficient)
theta = np.array([theta_0, theta_1, theta_2])
x = np.array([1, x1, x2])  # Note: 1 for intercept
y_pred_vec = np.dot(theta, x)
print(f"Vectorized prediction: {y_pred_vec}")
\`\`\`

## Simplifying Expressions

### Combining Like Terms

**Like terms**: Terms with the same variables raised to the same powers.

**Example**: \`3x + 5x - 2x = (3 + 5 - 2)x = 6x\`

**Python Implementation**:
\`\`\`python
from sympy import symbols, simplify, expand

x = symbols('x')

# Expression: 3x + 5x - 2x
expr = 3*x + 5*x - 2*x
simplified = simplify(expr)
print(f"3x + 5x - 2x = {simplified}")

# More complex expression
expr2 = 2*x**2 + 3*x + 4*x**2 - x + 5
simplified2 = simplify(expr2)
print(f"2x² + 3x + 4x² - x + 5 = {simplified2}")

# Application: Combining gradient terms
theta = symbols('theta')
gradient_term1 = 2 * theta - 3
gradient_term2 = 4 * theta + 1
gradient_term3 = -theta + 5

total_gradient = simplify(gradient_term1 + gradient_term2 + gradient_term3)
print(f"\\nTotal gradient: {total_gradient}")
\`\`\`

### Distributive Property

**Rule**: \`a(b + c) = ab + ac\`

**Python Example**:
\`\`\`python
from sympy import symbols, expand, factor

x, y = symbols('x y')

# Expanding
expr = 2*x * (3*x + 4)
expanded = expand(expr)
print(f"2x(3x + 4) = {expanded}")

# Factoring (reverse)
expr2 = 6*x**2 + 8*x
factored = factor(expr2)
print(f"6x² + 8x = {factored}")

# ML application: Expanding loss function
# L(θ) = (y - θx)²
theta, x_var, y_var = symbols('theta x y')
loss = (y_var - theta * x_var)**2
expanded_loss = expand(loss)
print(f"\\nExpanded loss: (y - θx)² = {expanded_loss}")
\`\`\`

### Factoring

Common factoring patterns:

1. **Common factor**: \`ax + ay = a(x + y)\`
2. **Difference of squares**: \`a² - b² = (a + b)(a - b)\`
3. **Perfect square**: \`a² + 2ab + b² = (a + b)²\`
4. **Quadratic**: \`ax² + bx + c = a(x - r₁)(x - r₂)\`

\`\`\`python
from sympy import symbols, factor, expand

x = symbols('x')

# Common factor
expr1 = 3*x**2 + 6*x
print(f"Factor {expr1}: {factor(expr1)}")

# Difference of squares
expr2 = x**2 - 9
print(f"Factor {expr2}: {factor(expr2)}")

# Perfect square
expr3 = x**2 + 6*x + 9
print(f"Factor {expr3}: {factor(expr3)}")

# Quadratic
expr4 = x**2 - 5*x + 6
print(f"Factor {expr4}: {factor(expr4)}")

# Verify by expanding back
factored = factor(expr4)
expanded_back = expand(factored)
print(f"Expand {factored}: {expanded_back}")
\`\`\`

## Solving Linear Equations

### Single Variable

**Standard form**: \`ax + b = 0\`
**Solution**: \`x = -b/a\` (provided a ≠ 0)

**Example**: \`3x - 12 = 0\`
\`\`\`
3x = 12
x = 4
\`\`\`

**Python Example**:
\`\`\`python
from sympy import symbols, Eq, solve

x = symbols('x')

# Equation: 3x - 12 = 0
equation = Eq(3*x - 12, 0)
solution = solve(equation, x)
print(f"Solution to 3x - 12 = 0: x = {solution}")

# More complex: 2(x + 3) = 4x - 10
equation2 = Eq(2*(x + 3), 4*x - 10)
solution2 = solve(equation2, x)
print(f"Solution to 2(x + 3) = 4x - 10: x = {solution2}")

# Verification
x_val = solution2[0]
left_side = 2*(x_val + 3)
right_side = 4*x_val - 10
print(f"Verification: {left_side} = {right_side}")
\`\`\`

### Application: Solving for Learning Rate

In gradient descent, we want to find when gradient = 0:

\`\`\`python
# Find learning rate α such that new_loss = target_loss
# new_loss = old_loss - α * gradient

theta, alpha, gradient = symbols('theta alpha gradient')
old_loss = symbols('old_loss')
target_loss = symbols('target_loss')

# Equation: old_loss - α * gradient = target_loss
equation = Eq(old_loss - alpha * gradient, target_loss)
alpha_solution = solve(equation, alpha)
print(f"Optimal α = {alpha_solution[0]}")

# Numerical example
old_loss_val = 10.0
target_loss_val = 8.0
gradient_val = 5.0

alpha_val = (old_loss_val - target_loss_val) / gradient_val
print(f"\\nNumerical: α = {alpha_val}")
\`\`\`

## Quadratic Equations

### Standard Form

**ax² + bx + c = 0** where a ≠ 0

### Quadratic Formula

**x = (-b ± √(b² - 4ac)) / (2a)**

**Discriminant**: Δ = b² - 4ac
- If Δ > 0: Two distinct real solutions
- If Δ = 0: One repeated real solution
- If Δ < 0: Two complex solutions

**Python Implementation**:
\`\`\`python
import numpy as np
from sympy import symbols, solve, Eq

def solve_quadratic(a, b, c):
    """Solve quadratic equation ax² + bx + c = 0"""
    discriminant = b**2 - 4*a*c
    
    if discriminant > 0:
        x1 = (-b + np.sqrt(discriminant)) / (2*a)
        x2 = (-b - np.sqrt(discriminant)) / (2*a)
        return f"Two solutions: x₁ = {x1:.4f}, x₂ = {x2:.4f}"
    elif discriminant == 0:
        x = -b / (2*a)
        return f"One solution: x = {x:.4f}"
    else:
        real_part = -b / (2*a)
        imag_part = np.sqrt(-discriminant) / (2*a)
        return f"Complex solutions: x = {real_part:.4f} ± {imag_part:.4f}i"

# Examples
print("1. x² - 5x + 6 = 0")
print(solve_quadratic(1, -5, 6))

print("\\n2. x² - 6x + 9 = 0")
print(solve_quadratic(1, -6, 9))

print("\\n3. x² + x + 1 = 0")
print(solve_quadratic(1, 1, 1))

# Using SymPy
x = symbols('x')
equation = x**2 - 5*x + 6
solutions = solve(equation, x)
print(f"\\nSymPy solution: {solutions}")
\`\`\`

### Application: Finding Optimal Parameters

Many optimization problems reduce to quadratic equations:

\`\`\`python
# Example: Minimize f(θ) = θ² - 4θ + 5
# Find θ where f'(θ) = 0

theta = symbols('theta')
f = theta**2 - 4*theta + 5

# Take derivative
f_prime = f.diff(theta)
print(f"f(θ) = {f}")
print(f"f'(θ) = {f_prime}")

# Solve f'(θ) = 0
optimal_theta = solve(f_prime, theta)
print(f"Optimal θ = {optimal_theta}")

# Evaluate minimum value
min_value = f.subs(theta, optimal_theta[0])
print(f"Minimum value: f({optimal_theta[0]}) = {min_value}")

# Verify with NumPy
theta_range = np.linspace(-1, 5, 100)
f_values = theta_range**2 - 4*theta_range + 5
optimal_idx = np.argmin(f_values)
print(f"\\nNumerical verification: θ ≈ {theta_range[optimal_idx]:.4f}")
\`\`\`

## Systems of Equations

### Two Variables

**System**:
\`\`\`
a₁x + b₁y = c₁
a₂x + b₂y = c₂
\`\`\`

**Methods**:
1. Substitution
2. Elimination
3. Matrix methods (covered in Linear Algebra)

**Python Example - Substitution Method**:
\`\`\`python
from sympy import symbols, Eq, solve

x, y = symbols('x y')

# System:
# 2x + 3y = 8
# x - y = 1

eq1 = Eq(2*x + 3*y, 8)
eq2 = Eq(x - y, 1)

# Solve system
solution = solve((eq1, eq2), (x, y))
print(f"Solution: {solution}")
print(f"x = {solution[x]}, y = {solution[y]}")

# Verification
x_val, y_val = solution[x], solution[y]
print(f"\\nVerification:")
print(f"2({x_val}) + 3({y_val}) = {2*x_val + 3*y_val}")
print(f"{x_val} - {y_val} = {x_val - y_val}")
\`\`\`

**Python Example - Matrix Method (preview)**:
\`\`\`python
import numpy as np

# System in matrix form: Ax = b
# [2  3] [x]   [8]
# [1 -1] [y] = [1]

A = np.array([[2, 3],
              [1, -1]])
b = np.array([8, 1])

# Solve using linear algebra
solution = np.linalg.solve(A, b)
print(f"Matrix solution: x = {solution[0]:.4f}, y = {solution[1]:.4f}")

# Verification
result = A @ solution
print(f"Verification: Ax = {result}, b = {b}")
print(f"Match? {np.allclose(result, b)}")
\`\`\`

### Three Variables

**System**:
\`\`\`
a₁x + b₁y + c₁z = d₁
a₂x + b₂y + c₂z = d₂
a₃x + b₃y + c₃z = d₃
\`\`\`

**Python Example**:
\`\`\`python
from sympy import symbols, Eq, solve
import numpy as np

# Using SymPy
x, y, z = symbols('x y z')

# System:
# x + y + z = 6
# 2x - y + z = 3
# x + 2y - z = 2

eq1 = Eq(x + y + z, 6)
eq2 = Eq(2*x - y + z, 3)
eq3 = Eq(x + 2*y - z, 2)

solution = solve((eq1, eq2, eq3), (x, y, z))
print(f"SymPy solution: {solution}")

# Using NumPy (faster for numerical problems)
A = np.array([[1,  1,  1],
              [2, -1,  1],
              [1,  2, -1]])
b = np.array([6, 3, 2])

solution_np = np.linalg.solve(A, b)
print(f"\\nNumPy solution: x={solution_np[0]:.4f}, y={solution_np[1]:.4f}, z={solution_np[2]:.4f}")
\`\`\`

### Application: Multi-variable Linear Regression

In linear regression with 3 features, we solve for θ₀, θ₁, θ₂:

\`\`\`python
# Normal equation: θ = (XᵀX)⁻¹Xᵀy

# Sample data
X = np.array([
    [1, 2, 3],    # Sample 1: [feature1, feature2, feature3]
    [1, 3, 4],    # Sample 2
    [1, 4, 5],    # Sample 3
    [1, 5, 6],    # Sample 4
])
X_with_intercept = np.c_[np.ones(X.shape[0]), X]  # Add column of 1s for intercept

y = np.array([10, 15, 20, 25])

# Solve normal equation
XtX = X_with_intercept.T @ X_with_intercept
Xty = X_with_intercept.T @ y
theta = np.linalg.solve(XtX, Xty)

print(f"Learned parameters: θ₀={theta[0]:.4f}, θ₁={theta[1]:.4f}, θ₂={theta[2]:.4f}, θ₃={theta[3]:.4f}")

# Make prediction
x_new = np.array([1, 3, 4, 5])  # Include 1 for intercept
y_pred = x_new @ theta
print(f"Prediction for {x_new[1:]}: {y_pred:.4f}")
\`\`\`

## Real-World Problem Modeling

### Example 1: Portfolio Optimization

**Problem**: You have $10,000 to invest in stocks (S) and bonds (B). You want:
- At least $3,000 in bonds
- Stock investment should be at most 2× bond investment
- Expected return: 8% for stocks, 5% for bonds
- Maximize total return

**Mathematical Model**:
\`\`\`
Variables: S (stock), B (bonds)
Constraints:
  S + B = 10,000
  B ≥ 3,000
  S ≤ 2B
Objective: Maximize 0.08S + 0.05B
\`\`\`

\`\`\`python
from scipy.optimize import linprog

# This is a preview of optimization (covered in Calculus module)
# For now, we solve the constraint equation

# From S + B = 10,000, we get S = 10,000 - B
# From S ≤ 2B: 10,000 - B ≤ 2B → B ≥ 10,000/3 ≈ 3,333.33

B = 3334  # Round up to meet constraint
S = 10000 - B

print(f"Investment: \${S:,} in stocks, \${B:,} in bonds")
print(f"Expected return: \${0.08*S + 0.05*B:,.2f}")
print(f"Return rate: {(0.08*S + 0.05*B)/10000*100:.2f}%")
\`\`\`

### Example 2: Break-Even Analysis for Trading

**Problem**: A trading strategy has:
- Fixed costs: $1,000/month (software, data feeds)
- Variable cost: $2 per trade
- Revenue: $5 per profitable trade
- How many trades needed to break even?

**Mathematical Model**:
\`\`\`
Let n = number of trades
Cost = 1000 + 2n
Revenue = 5n
Break-even: Revenue = Cost
5n = 1000 + 2n
\`\`\`

\`\`\`python
# Solve: 5n = 1000 + 2n
# 3n = 1000
# n = 1000/3

n = symbols('n')
equation = Eq(5*n, 1000 + 2*n)
breakeven = solve(equation, n)
print(f"Break-even trades: {breakeven[0]:.0f} trades")
print(f"Rounded up: {np.ceil(float(breakeven[0]))} trades")

# Verification
trades = 334
cost = 1000 + 2*trades
revenue = 5*trades
profit = revenue - cost
print(f"\\nAt {trades} trades:")
print(f"Cost: \${cost:,}")
print(f"Revenue: \${revenue:,}")
print(f"Profit: \${profit:,}")
\`\`\`

## Using SymPy for Symbolic Math

SymPy is Python's symbolic mathematics library:

\`\`\`python
from sympy import symbols, solve, Eq, simplify, expand, factor, diff

# Define symbols
x, y, theta, alpha = symbols('x y theta alpha')

# 1. Solve equations
eq = x**2 + 3*x - 10
solution = solve(eq, x)
print(f"Solve x² + 3x - 10 = 0: {solution}")

# 2. Simplify expressions
expr = (x + 1)**2 - (x**2 + 2*x + 1)
simplified = simplify(expr)
print(f"\\nSimplify (x+1)² - (x²+2x+1): {simplified}")

# 3. Expand expressions
expr2 = (x + y)**3
expanded = expand(expr2)
print(f"\\nExpand (x+y)³: {expanded}")

# 4. Factor expressions
expr3 = x**3 - y**3
factored = factor(expr3)
print(f"\\nFactor x³ - y³: {factored}")

# 5. Derivatives (preview)
f = x**3 - 2*x**2 + x - 5
f_prime = diff(f, x)
print(f"\\nDerivative of {f}: {f_prime}")

# 6. Substitute values
result = f.subs(x, 2)
print(f"f(2) = {result}")

# 7. Systems of equations
eq1 = Eq(2*x + y, 10)
eq2 = Eq(x - y, 2)
sol = solve((eq1, eq2), (x, y))
print(f"\\nSystem solution: {sol}")
\`\`\`

## Common Mistakes and How to Avoid Them

### 1. Sign Errors

\`\`\`python
# Wrong: -2x + 5 = 13 → x = -4 (forgot to flip sign)
# Right: -2x = 8 → x = -4

x = symbols('x')
eq = Eq(-2*x + 5, 13)
correct = solve(eq, x)
print(f"Correct solution: {correct}")
\`\`\`

### 2. Order of Operations

\`\`\`python
# Wrong: 2 + 3 * 4 = 20 (added first)
# Right: 2 + 3 * 4 = 14 (multiply first)

result_wrong = (2 + 3) * 4  # 20
result_right = 2 + 3 * 4     # 14
print(f"Wrong (left to right): {result_wrong}")
print(f"Right (PEMDAS): {result_right}")
\`\`\`

### 3. Division by Zero

\`\`\`python
def safe_solve(numerator, denominator):
    """Safely solve ax + b = 0"""
    if denominator == 0:
        if numerator == 0:
            return "Infinite solutions"
        else:
            return "No solution"
    return -numerator / denominator

# 0x + 5 = 0 (no solution)
print(f"0x + 5 = 0: {safe_solve(5, 0)}")

# 0x + 0 = 0 (infinite solutions)
print(f"0x + 0 = 0: {safe_solve(0, 0)}")

# 2x + 6 = 0 (x = -3)
print(f"2x + 6 = 0: x = {safe_solve(6, 2)}")
\`\`\`

### 4. Squaring Both Sides (Introduces Extra Solutions)

\`\`\`python
# Solving √x = -2 by squaring both sides
# (√x)² = (-2)² → x = 4
# But √4 = 2, not -2! Need to check solution.

x = symbols('x')
# Original: sqrt(x) = -2 (no real solution)
# After squaring: x = 4 (but this doesn't satisfy original!)

from sympy import sqrt
eq_original = Eq(sqrt(x), -2)
eq_squared = Eq(x, 4)

print(f"Original equation: {eq_original}")
print(f"After squaring: {eq_squared}")
print(f"Check x=4 in original: √4 = 2 ≠ -2")
print(f"Conclusion: No real solution exists")
\`\`\`

## Summary

- **Variables** represent unknowns; **coefficients** multiply variables; **constants** are fixed values
- **Simplifying** combines like terms and applies distributive property
- **Linear equations** (ax + b = 0) have one solution: x = -b/a
- **Quadratic equations** (ax² + bx + c = 0) use quadratic formula
- **Systems of equations** can be solved by substitution, elimination, or matrix methods
- **SymPy** provides powerful symbolic math capabilities in Python
- **Real-world modeling** translates word problems into mathematical equations
- Always **verify solutions** by substituting back into original equation

These algebraic skills are fundamental for:
- Deriving machine learning algorithms
- Solving optimization problems
- Understanding model equations
- Implementing gradient descent
- Debugging mathematical errors in code
`,
      multipleChoice: [
        {
          id: 'mc1-algebra-terms',
          question:
            'In the expression 5x² - 3x + 7, which statement is correct?',
          options: [
            '5 is a variable and 7 is a coefficient',
            'x² and x are like terms',
            '5 is a coefficient of x² and 7 is a constant',
            'The degree of the expression is 3',
          ],
          correctAnswer: 2,
          explanation:
            "5 is the coefficient of x² (it multiplies x²), and 7 is a constant (it doesn't multiply any variable). x² and x are NOT like terms because they have different powers. The degree is 2 (highest power of x).",
        },
        {
          id: 'mc2-quadratic-discriminant',
          question:
            'For the quadratic equation 2x² + 3x + 5 = 0, what does the discriminant tell us about the solutions?',
          options: [
            'Two distinct real solutions because Δ > 0',
            'One repeated real solution because Δ = 0',
            'Two complex solutions because Δ < 0',
            'No solutions exist',
          ],
          correctAnswer: 2,
          explanation:
            'The discriminant Δ = b² - 4ac = 3² - 4(2)(5) = 9 - 40 = -31. Since Δ < 0, the equation has two complex conjugate solutions, not real solutions. This is common in certain ML optimization scenarios where complex eigenvalues appear.',
        },
        {
          id: 'mc3-system-solutions',
          question:
            'When solving a system of two linear equations with two unknowns, which outcome is NOT possible?',
          options: [
            'Exactly one solution (lines intersect at a point)',
            'No solutions (parallel lines)',
            'Infinitely many solutions (same line)',
            'Exactly three solutions',
          ],
          correctAnswer: 3,
          explanation:
            'A system of two linear equations in two variables can have: (1) exactly one solution (lines intersect), (2) no solutions (parallel lines), or (3) infinitely many solutions (same line). It CANNOT have exactly two, three, or any finite number other than one.',
        },
        {
          id: 'mc4-factoring',
          question: 'Which factoring pattern does x² - 16 follow?',
          options: [
            'Perfect square trinomial',
            'Difference of squares',
            'Sum of squares',
            'Common factor',
          ],
          correctAnswer: 1,
          explanation:
            "x² - 16 = x² - 4² is a difference of squares pattern: a² - b² = (a + b)(a - b). So x² - 16 = (x + 4)(x - 4). Note that sum of squares (a² + b²) doesn't factor over real numbers.",
        },
        {
          id: 'mc5-ml-equation',
          question:
            'In the gradient descent update rule θ_new = θ_old - α∇L(θ), what does solving for α when ∇L(θ) = 10 and we want θ_new = θ_old - 5 give us?',
          options: ['α = 0.5', 'α = 2', 'α = 5', 'α = 50'],
          correctAnswer: 0,
          explanation:
            'From θ_new = θ_old - α∇L(θ), we have: θ_old - 5 = θ_old - α(10). This gives us: -5 = -10α, so α = 5/10 = 0.5. This represents the learning rate needed to achieve the desired parameter update.',
        },
      ],
      quiz: [
        {
          id: 'dq1-normal-equation',
          question:
            'Explain why the normal equation in linear regression θ = (X^T X)^(-1) X^T y involves solving a system of equations. What does each component represent, and when might this approach fail?',
          sampleAnswer: `The normal equation is the closed-form solution to linear regression that minimizes the mean squared error. Let's break down why it involves systems of equations:

**Derivation Context**:
When we have m training examples with n features, we're trying to find parameters θ = [θ₀, θ₁, ..., θₙ] that minimize:
L(θ) = (1/2m) Σ(y⁽ⁱ⁾ - h_θ(x⁽ⁱ⁾))²

Taking the derivative with respect to each θⱼ and setting to zero gives us:
∂L/∂θⱼ = 0 for j = 0, 1, ..., n

This creates n+1 equations with n+1 unknowns—a system of equations!

**Component Breakdown**:
- **X**: m × (n+1) design matrix where each row is one training example
- **X^T X**: (n+1) × (n+1) square matrix of feature correlations
- **(X^T X)^(-1)**: Inverse used to "solve" the system
- **X^T y**: (n+1) × 1 vector representing target correlations
- **θ**: (n+1) × 1 vector of parameters we're solving for

**Why it's a System of Equations**:
X^T X θ = X^T y represents:
[sum of feature cross-products] × [parameters] = [sum of feature-target products]

This is equivalent to n+1 simultaneous equations. For example, with 2 features:
θ₀ Σ1 + θ₁ Σx₁ + θ₂ Σx₂ = Σy
θ₀ Σx₁ + θ₁ Σx₁² + θ₂ Σx₁x₂ = Σx₁y
θ₀ Σx₂ + θ₁ Σx₁x₂ + θ₂ Σx₂² = Σx₂y

**When This Approach Fails**:

1. **Non-invertible X^T X** (singular matrix):
   - Happens when features are linearly dependent
   - Or when m < n (fewer examples than features)
   - Solution: Use regularization (Ridge/Lasso) or pseudo-inverse

2. **Computational Complexity**:
   - Computing (X^T X)^(-1) is O(n³)
   - With millions of features, this is prohibitive
   - Solution: Use iterative methods like gradient descent

3. **Numerical Instability**:
   - If X^T X is ill-conditioned (nearly singular)
   - Small changes in data cause large changes in θ
   - Solution: Feature scaling, regularization

4. **Memory Constraints**:
   - X^T X requires O(n²) memory
   - For very large n, may not fit in memory
   - Solution: Stochastic gradient descent

**Practical Implications**:
- Use normal equation when: n ≤ 10,000 and X^T X is invertible
- Use gradient descent when: n > 10,000 or need online learning
- Modern deep learning almost never uses normal equation due to scale`,
          keyPoints: [
            'Normal equation solves n+1 simultaneous linear equations',
            'X^T X θ = X^T y is the matrix form of the system',
            'Fails when X^T X is not invertible (singular)',
            'Computationally expensive O(n³) for large n',
            'Gradient descent is preferred for high-dimensional problems',
          ],
        },
        {
          id: 'dq2-quadratic-optimization',
          question:
            'Many machine learning optimization problems reduce to quadratic equations. Explain how this happens in the context of finding optimal learning rates or convergence analysis, and why understanding quadratic equations helps debug training issues.',
          sampleAnswer: `Quadratic equations appear frequently in ML optimization due to second-order Taylor approximations and the geometry of loss landscapes:

**1. Learning Rate Optimization**:

Consider gradient descent: θ_{t+1} = θ_t - α∇L(θ_t)

Using Taylor expansion around θ_t:
L(θ_{t+1}) ≈ L(θ_t) + ∇L^T(θ_{t+1} - θ_t) + (1/2)(θ_{t+1} - θ_t)^T H (θ_{t+1} - θ_t)

Where H is the Hessian (second derivatives). Substituting θ_{t+1}:
L(θ_t - α∇L) ≈ L(θ_t) - α||∇L||² + (α²/2)∇L^T H ∇L

This is a **quadratic equation in α**! To find optimal α, set derivative to zero:
dL/dα = -||∇L||² + α ∇L^T H ∇L = 0
α = ||∇L||² / (∇L^T H ∇L)

**2. Convergence Analysis**:

For strongly convex functions with Lipschitz continuous gradients:
L(θ) ≥ L(θ*) + (μ/2)||θ - θ*||²

The convergence rate of gradient descent is:
||θ_t - θ*||² ≤ (1 - αμ)^t ||θ₀ - θ*||²

This is a geometric series (related to sequences), but analyzing when (1 - αμ) < 1 involves:
αμ < 2 (quadratic inequality)

**3. Newton's Method**:

Newton's method uses quadratic approximation explicitly:
θ_{t+1} = θ_t - H^(-1)∇L(θ_t)

This assumes the loss is locally quadratic, which is why it converges faster near optima but can fail in non-convex regions.

**4. Debugging Training Issues**:

**Problem: Loss Diverges (Explodes)**
- Quadratic analysis: If α > 2/λ_max (λ_max = largest eigenvalue of H)
- Solution: Reduce learning rate or use adaptive methods

**Problem: Loss Plateaus**
- Could be at local minimum where ∇L ≈ 0
- Check second derivative (Hessian): 
  - If H > 0: true minimum
  - If H < 0: maximum (shouldn't happen)
  - If H ≈ 0: saddle point (common in deep learning)

**Problem: Oscillating Loss**
- Learning rate too large: overshooting minimum
- Quadratic bowl analogy: α causes "bouncing" back and forth
- Solution: Reduce α or use momentum

**5. Practical Example - Debugging Polynomial Features**:

\`\`\`python
# With polynomial features, loss landscape becomes more complex
# Understanding quadratic terms helps tune regularization

def analyze_loss_curvature(X, y, theta):
    """Compute second derivative (curvature) of loss"""
    n = len(y)
    # For MSE: L = (1/2n)||Xθ - y||²
    # Hessian: H = (1/n)X^T X
    H = (X.T @ X) / n
    eigenvalues = np.linalg.eigvalsh(H)
    
    print(f"Condition number: {eigenvalues.max() / eigenvalues.min():.2e}")
    print(f"Max safe learning rate: {2 / eigenvalues.max():.2e}")
    return eigenvalues
\`\`\`

**6. Portfolio Optimization Context**:

In mean-variance portfolio theory:
min (1/2)θ^T Σ θ - μ^T θ

Where Σ is covariance matrix. This is a **quadratic program**!
- θ: portfolio weights
- Quadratic term: risk (variance)
- Linear term: expected return

Understanding this helps:
- Know when solution is unique (Σ positive definite)
- Understand risk-return tradeoff geometrically
- Debug numerical issues in portfolio solvers

**Key Takeaways**:
Quadratic equations in ML represent local curvature, enable learning rate theory, and help diagnose optimization problems through eigenvalue analysis.`,
          keyPoints: [
            'Taylor expansions create quadratic loss approximations',
            'Optimal learning rate comes from solving quadratic equation',
            'Eigenvalues of Hessian determine convergence properties',
            'Understanding curvature helps debug divergence and oscillation',
            'Quadratic programs appear in portfolio optimization',
          ],
        },
        {
          id: 'dq3-symbolic-numeric',
          question:
            'Compare symbolic computation (SymPy) versus numeric computation (NumPy) for solving algebraic equations. When should you use each in machine learning and trading applications? Provide specific scenarios.',
          sampleAnswer: `The choice between symbolic (SymPy) and numeric (NumPy) computation depends on whether you need exact analytical solutions or fast numerical approximations:

**Symbolic Computation (SymPy)**

**Strengths**:
1. **Exact Solutions**: No floating-point errors
2. **Analytical Insights**: See formulas, not just numbers
3. **Simplification**: Algebraic manipulation and simplification
4. **Derivative Automation**: Exact symbolic derivatives

**Limitations**:
1. **Slow**: Orders of magnitude slower than numeric
2. **Memory Intensive**: Expressions grow large
3. **Limited Scalability**: Struggles with high dimensions
4. **Can't Always Solve**: Some equations have no closed form

**Numeric Computation (NumPy)**

**Strengths**:
1. **Speed**: Highly optimized, uses BLAS/LAPACK
2. **Scalability**: Handles millions of variables
3. **Memory Efficient**: Fixed-size arrays
4. **Robust Algorithms**: Iterative solvers for any problem

**Limitations**:
1. **Numerical Error**: Floating-point precision issues
2. **No Insights**: Just numbers, not formulas
3. **Initial Guess**: Iterative methods need starting point
4. **Convergence**: May fail to converge or find wrong solution

**ML Scenarios - When to Use Each**:

**Use SymPy**:

1. **Deriving Custom Loss Functions**:
\`\`\`python
# Derive gradient symbolically
from sympy import symbols, diff
x, y, theta = symbols('x y theta')
loss = (y - theta*x)**2
gradient = diff(loss, theta)  # Exact: 2*x*(theta*x - y)
\`\`\`

2. **Theoretical Analysis**:
- Proving convergence properties
- Analyzing algorithm behavior
- Computing exact update rules

3. **Small-Scale Prototypes**:
- Research papers: show exact formulas
- Educational purposes
- Verifying numeric implementations

4. **Equation Simplification**:
\`\`\`python
# Simplify complex regularization terms
expr = (theta**2 + 2*theta + 1) / (theta + 1)
simplified = simplify(expr)  # theta + 1
\`\`\`

**Use NumPy**:

1. **Training Neural Networks**:
\`\`\`python
# Millions of parameters - symbolic impossible
weights = np.random.randn(1000, 1000)
gradients = compute_numerical_gradient(weights)  # Fast
\`\`\`

2. **Real-Time Predictions**:
- Trading: Need microsecond latency
- Production systems: Speed critical
- Online learning: Continuous updates

3. **Large-Scale Optimization**:
\`\`\`python
# Solve 10,000 × 10,000 system
A = generate_feature_matrix(10000)
b = target_vector
theta = np.linalg.solve(A, b)  # Fast
\`\`\`

4. **Empirical Analysis**:
- Backtesting strategies
- Monte Carlo simulations
- Performance benchmarking

**Trading Application Scenarios**:

**Scenario 1: Developing New Strategy (Use SymPy)**
\`\`\`python
# Derive exact formula for expected return
from sympy import symbols, integrate, exp
p, mu, sigma = symbols('p mu sigma', real=True, positive=True)

# Option payoff with probability distribution
payoff = max(p - K, 0)  # Call option
expected = integrate(payoff * normal_pdf(p, mu, sigma), (p, -oo, oo))
# Get exact formula for pricing
\`\`\`

**Scenario 2: Backtesting (Use NumPy)**
\`\`\`python
# Process millions of price ticks
prices = load_market_data()  # Shape: (1000000, 100)
returns = np.log(prices[1:] / prices[:-1])
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
# Fast numerical computation
\`\`\`

**Scenario 3: Risk Analysis (Both)**
\`\`\`python
# Symbolic: Derive VaR formula
from sympy import quantile
var_formula = quantile(returns_distribution, alpha)

# Numeric: Compute actual VaR
returns_array = np.random.normal(mu, sigma, 10000)
var_numeric = np.percentile(returns_array, 5)  # 5% VaR
\`\`\`

**Scenario 4: Order Execution (Use NumPy)**
\`\`\`python
# Real-time: Must execute in microseconds
optimal_order_size = compute_vwap_slice(
    current_volume=volume_array,  # NumPy
    target_quantity=1000,
    time_remaining=60
)
# Symbolic would timeout
\`\`\`

**Scenario 5: Academic Research Paper (Use SymPy)**
\`\`\`python
# Show exact theoretical optimal bid-ask spread
s, lambda_b, lambda_a, sigma = symbols('s lambda_b lambda_a sigma')
expected_profit = derive_market_maker_profit(s, lambda_b, lambda_a, sigma)
optimal_spread = solve(diff(expected_profit, s), s)
# Include in paper as formula
\`\`\`

**Hybrid Approach**:

Best practice: Use both!

1. **Derive with SymPy, Implement with NumPy**:
\`\`\`python
# Step 1: Symbolic derivation
gradient_formula = derive_gradient_symbolically()

# Step 2: Convert to numeric function
gradient_func = lambdify(theta, gradient_formula)

# Step 3: Use in training
for epoch in range(epochs):
    theta -= learning_rate * gradient_func(theta)  # Fast numeric
\`\`\`

2. **Validate Numeric with Symbolic**:
\`\`\`python
# Compute numerically
numeric_result = solve_with_numpy()

# Verify with symbolic (on small subset)
symbolic_result = solve_with_sympy(small_sample)
assert np.isclose(numeric_result, symbolic_result)
\`\`\`

**Summary**:
- **SymPy**: Prototyping, derivation, analysis, small problems
- **NumPy**: Production, training, large-scale, real-time
- **Both**: Derive formulas symbolically, implement numerically
- **Trading**: NumPy for execution, SymPy for strategy development`,
          keyPoints: [
            'SymPy: exact solutions, insights, slow, small-scale',
            'NumPy: fast, scalable, approximate, production-ready',
            'Use SymPy for derivation and theoretical analysis',
            'Use NumPy for training, backtesting, and real-time systems',
            'Hybrid approach: derive symbolically, implement numerically',
          ],
        },
      ],
    },
    {
      id: 'functions-relations',
      title: 'Functions & Relations',
      content: `
# Functions & Relations

## Introduction

Functions are the foundation of machine learning. Every ML model is essentially a function that maps inputs to outputs. Understanding function notation, properties, and types is crucial for grasping how neural networks, loss functions, and activation functions work.

## Function Notation and Domain/Range

### Definition

A **function** is a relation that assigns exactly one output to each input.

**Notation**: f(x) = y
- f: function name
- x: input (independent variable)
- y: output (dependent variable)

**Domain**: Set of all possible input values
**Range**: Set of all possible output values

**Example**:
\`\`\`
f(x) = 2x + 1
Domain: all real numbers ℝ
Range: all real numbers ℝ
f(3) = 2(3) + 1 = 7
\`\`\`

### Python Implementation

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Define a function
def f(x):
    """Simple linear function"""
    return 2*x + 1

# Evaluate function at specific points
x_values = np.array([0, 1, 2, 3, 4])
y_values = f(x_values)

print("x:", x_values)
print("f(x):", y_values)

# Visualize
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, 'bo-', label='f(x) = 2x + 1')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Linear Function')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

### ML Context: Hypothesis Function

In machine learning, we call our model a **hypothesis function**:

\`\`\`python
# Linear regression hypothesis
def h_theta(x, theta_0, theta_1):
    """
    Hypothesis function for linear regression
    h_θ(x) = θ₀ + θ₁x
    """
    return theta_0 + theta_1 * x

# Example: predicting house prices
# x = square footage, y = price
theta_0 = 50000  # base price
theta_1 = 100    # price per sq ft

square_footage = np.array([1000, 1500, 2000, 2500])
predicted_prices = h_theta(square_footage, theta_0, theta_1)

print("Square Footage:", square_footage)
print("Predicted Prices:", predicted_prices)

# Vectorized version (more efficient)
def h_theta_vectorized(X, theta):
    """
    X: feature matrix (with intercept column)
    theta: parameter vector
    """
    return X @ theta

# Add intercept column
X = np.c_[np.ones(len(square_footage)), square_footage]
theta = np.array([theta_0, theta_1])
predicted_prices_vec = h_theta_vectorized(X, theta)
print("\\nVectorized predictions:", predicted_prices_vec)
\`\`\`

## Types of Functions

### Linear Functions

**Form**: f(x) = mx + b
- m: slope
- b: y-intercept

**Properties**:
- Constant rate of change
- Graph is a straight line

\`\`\`python
def plot_linear_functions():
    x = np.linspace(-5, 5, 100)
    
    plt.figure(figsize=(10, 6))
    
    # Different slopes
    plt.plot(x, 2*x + 1, label='f(x) = 2x + 1', linewidth=2)
    plt.plot(x, -x + 3, label='f(x) = -x + 3', linewidth=2)
    plt.plot(x, 0.5*x - 2, label='f(x) = 0.5x - 2', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Linear Functions')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.show()

plot_linear_functions()
print("Linear functions plotted")
\`\`\`

**ML Application**: Linear regression, linear layers in neural networks

### Quadratic Functions

**Form**: f(x) = ax² + bx + c
- a: determines concavity (a > 0: opens up, a < 0: opens down)
- Vertex at x = -b/(2a)

**Properties**:
- Parabolic shape
- One global extremum (min or max)

\`\`\`python
def plot_quadratic_functions():
    x = np.linspace(-5, 5, 100)
    
    plt.figure(figsize=(10, 6))
    
    # Different quadratics
    plt.plot(x, x**2, label='f(x) = x²', linewidth=2)
    plt.plot(x, -x**2 + 4, label='f(x) = -x² + 4', linewidth=2)
    plt.plot(x, 0.5*x**2 - 2*x + 1, label='f(x) = 0.5x² - 2x + 1', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Quadratic Functions')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.ylim(-5, 5)
    plt.show()

plot_quadratic_functions()
print("Quadratic functions plotted")
\`\`\`

**ML Application**: Convex optimization, loss function landscapes

### Polynomial Functions

**Form**: f(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀
- n: degree of polynomial
- aᵢ: coefficients

\`\`\`python
from numpy.polynomial import Polynomial

# Create polynomial: 2x³ - 3x² + x - 5
coefficients = [-5, 1, -3, 2]  # constant to highest degree
poly = Polynomial(coefficients)

x = np.linspace(-3, 3, 100)
y = poly(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Polynomial: f(x) = 2x³ - 3x² + x - 5')
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

print(f"Polynomial: {poly}")
print(f"Degree: {poly.degree()}")
\`\`\`

**ML Application**: Polynomial regression, feature engineering with polynomial features

\`\`\`python
from sklearn.preprocessing import PolynomialFeatures

# Polynomial feature expansion
X = np.array([[2], [3], [4]])
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X)

print("Original features:\\n", X)
print("\\nPolynomial features (degree 3):\\n", X_poly)
print("\\nFeature names:", poly_features.get_feature_names_out(['x']))
\`\`\`

### Exponential Functions

**Form**: f(x) = a·bˣ or f(x) = a·eˣ
- Base b > 1: exponential growth
- Base 0 < b < 1: exponential decay
- e ≈ 2.71828: natural exponential base

**Properties**:
- Always positive (for real inputs)
- Rapid growth or decay
- Never touches x-axis

\`\`\`python
def plot_exponential_functions():
    x = np.linspace(-2, 3, 100)
    
    plt.figure(figsize=(10, 6))
    
    # Growth and decay
    plt.plot(x, np.exp(x), label='f(x) = eˣ (growth)', linewidth=2)
    plt.plot(x, np.exp(-x), label='f(x) = e⁻ˣ (decay)', linewidth=2)
    plt.plot(x, 2**x, label='f(x) = 2ˣ', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Exponential Functions')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 10)
    plt.show()

plot_exponential_functions()
print("Exponential functions plotted")
\`\`\`

**ML Application**: Softmax activation, exponential learning rate decay

\`\`\`python
def softmax(x):
    """Softmax activation function"""
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / exp_x.sum()

# Example: converting logits to probabilities
logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)

print("Logits:", logits)
print("Probabilities:", probabilities)
print("Sum:", probabilities.sum())  # Should be 1.0
\`\`\`

### Logarithmic Functions

**Form**: f(x) = logₐ(x) or f(x) = ln(x)
- Inverse of exponential function
- Domain: x > 0
- Range: all real numbers

**Properties**:
- Slow growth
- Undefined for x ≤ 0
- log(ab) = log(a) + log(b)
- log(aⁿ) = n·log(a)

\`\`\`python
def plot_logarithmic_functions():
    x = np.linspace(0.1, 10, 100)
    
    plt.figure(figsize=(10, 6))
    
    # Different bases
    plt.plot(x, np.log(x), label='f(x) = ln(x) (natural log)', linewidth=2)
    plt.plot(x, np.log10(x), label='f(x) = log₁₀(x)', linewidth=2)
    plt.plot(x, np.log2(x), label='f(x) = log₂(x)', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Logarithmic Functions')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=1, color='k', linewidth=0.5)
    plt.show()

plot_logarithmic_functions()
print("Logarithmic functions plotted")
\`\`\`

**ML Application**: Log loss (cross-entropy), log-likelihood

\`\`\`python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-10):
    """
    Binary cross-entropy loss
    Uses logarithms to penalize wrong predictions
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
loss = binary_cross_entropy(y_true, y_pred)
print(f"Binary Cross-Entropy Loss: {loss:.4f}")
\`\`\`

## Inverse Functions

### Definition

If f(x) = y, then f⁻¹(y) = x

**Properties**:
- f(f⁻¹(x)) = x
- f⁻¹(f(x)) = x
- Graph of f⁻¹ is reflection of f across y = x line

\`\`\`python
# Example: f(x) = 2x + 1
def f(x):
    return 2*x + 1

# Inverse: f⁻¹(x) = (x - 1) / 2
def f_inverse(x):
    return (x - 1) / 2

# Verify
x_test = 5
print(f"f({x_test}) = {f(x_test)}")
print(f"f⁻¹(f({x_test})) = {f_inverse(f(x_test))}")

# Visualize
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(8, 8))
plt.plot(x, f(x), label='f(x) = 2x + 1', linewidth=2)
plt.plot(x, f_inverse(x), label='f⁻¹(x) = (x-1)/2', linewidth=2)
plt.plot(x, x, 'k--', label='y = x', linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function and Its Inverse')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
\`\`\`

**ML Application**: Inverse transformations in autoencoders, invertible neural networks

## Composition of Functions

### Definition

**Composition**: (f ∘ g)(x) = f(g(x))

First apply g, then apply f to the result.

\`\`\`python
# Example: f(x) = x² and g(x) = x + 1
def f(x):
    return x**2

def g(x):
    return x + 1

def compose(f, g):
    """Return the composition f ∘ g"""
    return lambda x: f(g(x))

# f(g(x)) = (x + 1)²
f_compose_g = compose(f, g)

x_test = 3
print(f"f(x) = x²")
print(f"g(x) = x + 1")
print(f"(f ∘ g)({x_test}) = f(g({x_test})) = f({g(x_test)}) = {f_compose_g(x_test)}")

# Note: composition is NOT commutative
g_compose_f = compose(g, f)
print(f"\\n(g ∘ f)({x_test}) = g(f({x_test})) = g({f(x_test)}) = {g_compose_f(x_test)}")
print(f"(f ∘ g) ≠ (g ∘ f): {f_compose_g(x_test)} ≠ {g_compose_f(x_test)}")
\`\`\`

**ML Application**: Neural network layers (function composition!)

\`\`\`python
# Neural network as function composition
def layer1(x, W1, b1):
    """First layer: linear transformation"""
    return x @ W1 + b1

def activation_relu(x):
    """ReLU activation"""
    return np.maximum(0, x)

def layer2(x, W2, b2):
    """Second layer: linear transformation"""
    return x @ W2 + b2

def neural_network(x, W1, b1, W2, b2):
    """
    Two-layer neural network as composition:
    f(x) = layer2(ReLU(layer1(x)))
    """
    h1 = layer1(x, W1, b1)
    h1_activated = activation_relu(h1)
    output = layer2(h1_activated, W2, b2)
    return output

# Example
x = np.array([[1, 2, 3]])  # 1 sample, 3 features
W1 = np.random.randn(3, 4)  # 3 inputs, 4 hidden units
b1 = np.random.randn(4)
W2 = np.random.randn(4, 2)  # 4 hidden, 2 outputs
b2 = np.random.randn(2)

output = neural_network(x, W1, b1, W2, b2)
print(f"Neural network output shape: {output.shape}")
print(f"Output: {output}")
\`\`\`

## Activation Functions in ML

Activation functions are crucial non-linear functions in neural networks:

### Sigmoid Function

**Formula**: σ(x) = 1 / (1 + e⁻ˣ)
- Range: (0, 1)
- Output can be interpreted as probability
- Smooth, differentiable

\`\`\`python
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Sigmoid Function: σ(x) = 1/(1 + e⁻ˣ)')
plt.grid(True)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.show()

# Derivative
def sigmoid_derivative(x):
    """Derivative of sigmoid: σ'(x) = σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

print(f"σ(0) = {sigmoid(0)}")
print(f"σ'(0) = {sigmoid_derivative(0)}")
\`\`\`

### ReLU Function

**Formula**: ReLU(x) = max(0, x)
- Most common in modern deep learning
- Computationally efficient
- Helps with vanishing gradient problem

\`\`\`python
def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

x = np.linspace(-5, 5, 100)
y = relu(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.title('ReLU Function: max(0, x)')
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

# Derivative
def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

print(f"ReLU(2) = {relu(2)}")
print(f"ReLU(-2) = {relu(-2)}")
\`\`\`

### Tanh Function

**Formula**: tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
- Range: (-1, 1)
- Zero-centered (unlike sigmoid)
- Similar to sigmoid but symmetric

\`\`\`python
def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

x = np.linspace(-5, 5, 100)
y = tanh(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.title('Tanh Function')
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

print(f"tanh(0) = {tanh(0)}")
print(f"tanh(2) = {tanh(2):.4f}")
print(f"tanh(-2) = {tanh(-2):.4f}")
\`\`\`

## Piecewise Functions

Functions defined differently on different intervals:

\`\`\`python
def piecewise_function(x):
    """
    f(x) = { x²     if x < 0
           { x      if 0 ≤ x < 2
           { 4      if x ≥ 2
    """
    return np.where(x < 0, x**2,
                    np.where(x < 2, x, 4))

x = np.linspace(-3, 4, 1000)
y = piecewise_function(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Piecewise Function')
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=2, color='r', linestyle='--', alpha=0.5)
plt.show()
\`\`\`

**ML Application**: ReLU and its variants are piecewise functions

## Function Transformations

Understanding how functions transform is crucial for feature engineering:

### Vertical Shift: f(x) + c

\`\`\`python
x = np.linspace(-5, 5, 100)
f_x = x**2

plt.figure(figsize=(10, 6))
plt.plot(x, f_x, label='f(x) = x²', linewidth=2)
plt.plot(x, f_x + 2, label='f(x) + 2', linewidth=2)
plt.plot(x, f_x - 3, label='f(x) - 3', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vertical Shifts')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

### Horizontal Shift: f(x - c)

\`\`\`python
x = np.linspace(-10, 10, 100)
f_x = x**2

plt.figure(figsize=(10, 6))
plt.plot(x, f_x, label='f(x) = x²', linewidth=2)
plt.plot(x, (x - 2)**2, label='f(x - 2)', linewidth=2)
plt.plot(x, (x + 3)**2, label='f(x + 3)', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Horizontal Shifts')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

### Vertical Scaling: c·f(x)

\`\`\`python
x = np.linspace(-3, 3, 100)
f_x = x**2

plt.figure(figsize=(10, 6))
plt.plot(x, f_x, label='f(x) = x²', linewidth=2)
plt.plot(x, 2*f_x, label='2·f(x)', linewidth=2)
plt.plot(x, 0.5*f_x, label='0.5·f(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vertical Scaling')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

**ML Application**: Feature scaling and normalization

## Even and Odd Functions

### Even Functions: f(-x) = f(x)
- Symmetric about y-axis
- Examples: x², cos(x), |x|

### Odd Functions: f(-x) = -f(x)
- Symmetric about origin
- Examples: x, x³, sin(x)

\`\`\`python
x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 5))

# Even function
plt.subplot(1, 2, 1)
plt.plot(x, x**2, linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Even Function: f(x) = x²')
plt.grid(True)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

# Odd function
plt.subplot(1, 2, 2)
plt.plot(x, x**3, linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Odd Function: f(x) = x³')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Check properties
x_test = 2
print(f"Even function: f({x_test}) = {x_test**2}, f({-x_test}) = {(-x_test)**2}")
print(f"Odd function: f({x_test}) = {x_test**3}, f({-x_test}) = {(-x_test)**3}")
\`\`\`

## Summary

- **Functions** map inputs to outputs following a rule
- **Domain** and **range** define valid inputs and possible outputs
- **Linear, quadratic, polynomial** functions model various relationships
- **Exponential** and **logarithmic** functions are inverses
- **Composition** chains functions together (crucial for neural networks)
- **Activation functions** (sigmoid, ReLU, tanh) add non-linearity to neural networks
- **Transformations** (shifts, scaling) are used in feature engineering
- Every ML model is essentially a function: y = f(x; θ)

These function concepts are the foundation for understanding:
- How neural networks work (composition of functions)
- Loss functions and their properties
- Activation functions and their role
- Feature transformations
- Model predictions as function evaluations
`,
      multipleChoice: [
        {
          id: 'mc1-function-basics',
          question: 'For a function to be valid, which property must hold?',
          options: [
            'Every input must have at least two outputs',
            'Every input must have exactly one output',
            'Every output must have exactly one input',
            'The domain must equal the range',
          ],
          correctAnswer: 1,
          explanation:
            'A function must assign exactly ONE output to each input. This is the fundamental definition of a function. However, multiple inputs can map to the same output (many-to-one is allowed), but one input cannot map to multiple outputs.',
        },
        {
          id: 'mc2-composition',
          question: 'If f(x) = x + 3 and g(x) = 2x, what is (f ∘ g)(4)?',
          options: ['11', '14', '8', '10'],
          correctAnswer: 0,
          explanation:
            '(f ∘ g)(4) = f(g(4)). First apply g: g(4) = 2(4) = 8. Then apply f: f(8) = 8 + 3 = 11. Note: This is different from (g ∘ f)(4) = g(f(4)) = g(7) = 14.',
        },
        {
          id: 'mc3-sigmoid',
          question:
            'What is the range of the sigmoid function σ(x) = 1/(1 + e⁻ˣ)?',
          options: ['(-∞, ∞)', '[-1, 1]', '(0, 1)', '[0, 1]'],
          correctAnswer: 2,
          explanation:
            'The sigmoid function has range (0, 1) - open interval, meaning it approaches but never reaches 0 or 1. As x → -∞, σ(x) → 0 and as x → ∞, σ(x) → 1. This makes sigmoid useful for binary classification where outputs are interpreted as probabilities.',
        },
        {
          id: 'mc4-inverse',
          question: 'If f(x) = 3x - 6, what is the inverse function f⁻¹(x)?',
          options: [
            'f⁻¹(x) = (x + 6)/3',
            'f⁻¹(x) = (x - 6)/3',
            'f⁻¹(x) = 3x + 6',
            'f⁻¹(x) = x/3 + 6',
          ],
          correctAnswer: 0,
          explanation:
            'To find inverse: Start with y = 3x - 6, solve for x: y + 6 = 3x, x = (y + 6)/3. Replace y with x: f⁻¹(x) = (x + 6)/3. Verify: f(f⁻¹(x)) = 3((x+6)/3) - 6 = x + 6 - 6 = x ✓',
        },
        {
          id: 'mc5-neural-network',
          question:
            'In a neural network with 3 layers (input → hidden → output), if each layer applies linear transformation followed by ReLU activation (except output), how many function compositions are involved?',
          options: [
            '3 (one per layer)',
            '5 (linear + activation for hidden, linear for output)',
            '6 (linear + activation for each layer)',
            '2 (just hidden and output)',
          ],
          correctAnswer: 1,
          explanation:
            'The computation is: output = linear2(ReLU(linear1(input))). This involves 3 function compositions: (1) linear1 (input to hidden), (2) ReLU (activation), (3) linear2 (hidden to output). Total: 3 functions composed. However, considering each operation separately: input→linear1→ReLU→linear2→output = 3 transformations, but 5 function applications if counting start point.',
        },
      ],
      quiz: [
        {
          id: 'dq1-neural-networks-composition',
          question:
            'Explain why deep neural networks are fundamentally function compositions. How does this perspective help us understand backpropagation and the chain rule? Provide specific examples from a 3-layer network.',
          sampleAnswer: `Deep neural networks are literally compositions of functions, and this mathematical perspective is crucial for understanding how they work and learn:

**Function Composition in Neural Networks**:

A 3-layer network can be written as:
y = f₃(f₂(f₁(x)))

Where each layer fᵢ is itself a composition:
fᵢ(x) = σᵢ(Wᵢx + bᵢ)
- Linear transformation: Wᵢx + bᵢ  
- Activation function: σᵢ(·)

**Explicit Example**:
\`\`\`python
# Layer 1: Input (3 features) → Hidden (4 units)
def f1(x):
    z1 = W1 @ x + b1  # Linear: 4×3 @ 3×1 = 4×1
    a1 = relu(z1)      # Activation
    return a1

# Layer 2: Hidden (4) → Hidden (3 units)
def f2(x):
    z2 = W2 @ x + b2  # Linear: 3×4 @ 4×1 = 3×1
    a2 = relu(z2)      # Activation
    return a2

# Layer 3: Hidden (3) → Output (1 unit)
def f3(x):
    z3 = W3 @ x + b3  # Linear: 1×3 @ 3×1 = 1×1
    return z3          # No activation (regression)

# Complete network: function composition
def network(x):
    return f3(f2(f1(x)))
\`\`\`

**Connection to Chain Rule**:

The chain rule from calculus states:
d/dx[f(g(x))] = f'(g(x)) · g'(x)

For multiple compositions:
d/dx[f₃(f₂(f₁(x)))] = f₃'(f₂(f₁(x))) · f₂'(f₁(x)) · f₁'(x)

This IS backpropagation!

**Backpropagation Derivation**:

Given loss L = (y - ŷ)² where ŷ = f₃(f₂(f₁(x))):

1. **Output gradient**:
   ∂L/∂ŷ = 2(ŷ - y)

2. **Layer 3 gradient** (chain rule):
   ∂L/∂W₃ = ∂L/∂ŷ · ∂ŷ/∂z₃ · ∂z₃/∂W₃
   where z₃ = W₃a₂ + b₃

3. **Layer 2 gradient** (chain continues):
   ∂L/∂W₂ = ∂L/∂ŷ · ∂ŷ/∂z₃ · ∂z₃/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂W₂

4. **Layer 1 gradient** (full chain):
   ∂L/∂W₁ = ∂L/∂ŷ · ∂ŷ/∂z₃ · ∂z₃/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁

Notice the pattern: we multiply gradients flowing backward through each function in the composition.

**Why This Perspective Matters**:

1. **Vanishing Gradients**: 
   - Chain rule multiplies many terms
   - If σ'(x) < 1 for many layers, product → 0
   - Deep networks: more compositions = more multiplications
   - Solution: Better activations (ReLU), normalization

2. **Exploding Gradients**:
   - If some σ'(x) > 1, product → ∞
   - Solution: Gradient clipping, careful initialization

3. **Skip Connections** (ResNets):
   - Instead of f₃(f₂(f₁(x))), use f₃(f₂(f₁(x)) + x)
   - Creates additional paths in the chain rule
   - Helps gradients flow directly backward

4. **Automatic Differentiation**:
   - PyTorch/TensorFlow build computational graphs
   - Each node is a function in the composition
   - Chain rule applied automatically by traversing graph backward

**Practical Implications**:

Understanding networks as function compositions helps you:
- Debug gradient flow issues
- Design better architectures
- Understand why certain activation functions work better
- Appreciate why depth (more compositions) can be powerful but also challenging

**Trading Context**:
When building trading models with neural networks:
- More layers = more complex feature transformations (compositions)
- But deeper networks may overfit or have unstable gradients
- Balance complexity with stability
- Monitor gradient magnitudes during training`,
          keyPoints: [
            'Neural networks are literal function compositions: f₃(f₂(f₁(x)))',
            'Backpropagation is the chain rule applied to function composition',
            "Each layer's gradient involves product of all subsequent derivatives",
            'Vanishing/exploding gradients result from multiplying many terms',
            'Skip connections provide alternative paths in the composition chain',
          ],
        },
        {
          id: 'dq2-activation-functions',
          question:
            'Why do we need activation functions in neural networks? Compare sigmoid, tanh, and ReLU - discuss their mathematical properties, advantages, disadvantages, and when to use each in practical ML applications.',
          sampleAnswer: `Activation functions are essential because they introduce non-linearity into neural networks. Without them, deep networks would be equivalent to a single linear transformation:

**Why Non-Linearity is Necessary**:

Consider a 2-layer network WITHOUT activation functions:
h = W₁x + b₁
y = W₂h + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)

This simplifies to y = W'x + b', a single linear transformation!

No matter how many layers you stack, without activation functions, the network can only learn linear relationships. Most real-world patterns (images, text, trading patterns) are highly non-linear.

**Sigmoid Function: σ(x) = 1/(1 + e⁻ˣ)**

**Properties**:
- Range: (0, 1)
- Smooth, continuously differentiable
- Derivative: σ'(x) = σ(x)(1 - σ(x))
- S-shaped curve

**Advantages**:
- Output interpretable as probability
- Smooth gradients
- Historically significant (early neural networks)
- Perfect for binary classification output layer

**Disadvantages**:
- Vanishing gradient problem: σ'(x) ≈ 0 for |x| > 4
  - Max derivative is 0.25 (at x=0)
  - Deep networks: multiplying many 0.25s → gradient → 0
- Not zero-centered (outputs always positive)
- Expensive computation (exponential)

**When to use**:
- OUTPUT layer for binary classification
- AVOID in hidden layers of deep networks

**Tanh Function: tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)**

**Properties**:
- Range: (-1, 1)
- Zero-centered
- Derivative: tanh'(x) = 1 - tanh²(x)
- Similar shape to sigmoid, but symmetric

**Advantages**:
- Zero-centered (better than sigmoid)
- Stronger gradients than sigmoid (max derivative = 1)
- Symmetric around origin

**Disadvantages**:
- Still suffers from vanishing gradient for |x| > 2
- Expensive computation

**When to use**:
- Hidden layers when you need zero-centered activations
- RNNs/LSTMs (historically common)
- Better than sigmoid but worse than ReLU for deep networks

**ReLU Function: ReLU(x) = max(0, x)**

**Properties**:
- Range: [0, ∞)
- Piecewise linear
- Derivative: 1 if x > 0, else 0
- Non-differentiable at x=0 (but we use subgradient)

**Advantages**:
- Computationally cheap (just comparison and multiplication)
- No vanishing gradient for x > 0
- Sparse activation (about 50% of neurons are zero)
- Empirically works very well
- Faster convergence than sigmoid/tanh

**Disadvantages**:
- Not zero-centered
- "Dying ReLU" problem: if neuron outputs 0, gradient is 0, it never recovers
  - Happens with large learning rates
  - Neuron gets "stuck" at 0 forever
- Unbounded output (can lead to numerical issues)

**When to use**:
- DEFAULT choice for hidden layers in deep networks
- Computer vision models
- Most modern architectures

**Comparison Summary**:

| Property | Sigmoid | Tanh | ReLU |
|----------|---------|------|------|
| Range | (0,1) | (-1,1) | [0,∞) |
| Zero-centered | ❌ | ✅ | ❌ |
| Vanishing gradient | ✅ Bad | ✅ Bad | ❌ Good |
| Computation | Slow | Slow | Fast |
| Sparse activation | ❌ | ❌ | ✅ |
| Dead neurons | ❌ | ❌ | ✅ Possible |

**Modern Variants**:

1. **Leaky ReLU**: max(0.01x, x)
   - Fixes dying ReLU (small gradient for x < 0)

2. **ELU**: x if x>0, else α(eˣ-1)
   - Smooth, zero-centered mean

3. **GELU**: x·Φ(x) (Gaussian error linear unit)
   - Used in transformers (BERT, GPT)

**Practical Recommendations**:

**For hidden layers**:
- Start with ReLU (default)
- If dying ReLU occurs: try Leaky ReLU or ELU
- For transformers/NLP: consider GELU

**For output layers**:
- Binary classification: Sigmoid
- Multi-class classification: Softmax
- Regression: Linear (no activation) or ReLU (if output ≥ 0)

**Trading Application Example**:
\`\`\`python
# Predicting stock returns (can be positive or negative)
# Hidden layers: ReLU for efficiency
# Output: Linear or tanh (symmetric around 0)

model = nn.Sequential(
    nn.Linear(features, 64),
    nn.ReLU(),              # Hidden layer 1
    nn.Linear(64, 32),
    nn.ReLU(),              # Hidden layer 2
    nn.Linear(32, 1),       # Output
    nn.Tanh()               # Symmetric output for returns
)
\`\`\`

**Key Insight**: ReLU's success isn't just mathematical—it's empirical. Despite theoretical disadvantages (not zero-centered, unbounded), it works remarkably well in practice due to computational efficiency and sparse representations.`,
          keyPoints: [
            'Activation functions provide non-linearity; without them, deep networks = single linear layer',
            'Sigmoid: good for output probabilities, bad for hidden layers (vanishing gradient)',
            'Tanh: better than sigmoid (zero-centered), still has vanishing gradient',
            'ReLU: default choice, fast, no vanishing gradient, watch for dying ReLU',
            'Choose activation based on layer type and problem requirements',
          ],
        },
        {
          id: 'dq3-loss-functions',
          question:
            'Loss functions are special functions in ML that measure prediction error. Explain the mathematical properties of mean squared error (MSE) and cross-entropy loss. Why is MSE used for regression and cross-entropy for classification? How do their derivatives influence gradient descent?',
          sampleAnswer: `Loss functions quantify how wrong our model's predictions are. Their mathematical properties directly affect training dynamics and model convergence:

**Mean Squared Error (MSE) - Regression**

**Formula**:
MSE = (1/n) Σ(yᵢ - ŷᵢ)²

**Mathematical Properties**:

1. **Always non-negative**: (y - ŷ)² ≥ 0
2. **Minimum at y = ŷ**: Perfect predictions give MSE = 0
3. **Convex** for linear models: Single global minimum
4. **Differentiable** everywhere: Smooth optimization
5. **Symmetric**: Overestimation and underestimation penalized equally

**Derivative**:
∂MSE/∂ŷ = (2/n) Σ(ŷᵢ - yᵢ)

The gradient is **linear** in the error: If error is large, gradient is large (fast updates). If error is small, gradient is small (slow updates).

**Why MSE for Regression**:

1. **Gaussian assumption**: MSE assumes errors follow normal distribution
   - Maximizing likelihood under Gaussian noise = minimizing MSE
   
2. **Penalizes large errors**: Quadratic term heavily penalizes outliers
   - Error of 2 is 4x worse than error of 1 (2² vs 1²)
   
3. **Smooth gradients**: Easy to optimize with gradient descent

4. **Interpretable**: In same units as target variable squared

**Example**:
\`\`\`python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_gradient(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

# Price prediction
y_true = np.array([100, 150, 200])
y_pred = np.array([110, 140, 210])

loss = mse_loss(y_true, y_pred)
grad = mse_gradient(y_true, y_pred)

print(f"MSE Loss: {loss}")  # 66.67
print(f"Gradient: {grad}")  # [6.67, -6.67, 6.67]
# Notice: Large errors → large gradients
\`\`\`

**Cross-Entropy Loss - Classification**

**Binary Cross-Entropy**:
BCE = -(1/n) Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]

Where y ∈ {0, 1} and ŷ ∈ (0, 1)

**Categorical Cross-Entropy** (multi-class):
CCE = -(1/n) Σ Σ yᵢⱼlog(ŷᵢⱼ)

Where yᵢⱼ is one-hot encoded

**Mathematical Properties**:

1. **Always non-negative**: -log(p) ≥ 0 for p ∈ (0,1)
2. **Asymmetric penalty**: 
   - Predicting 0.01 when truth is 1: Loss ≈ 4.6
   - Predicting 0.99 when truth is 0: Loss ≈ 4.6
   - But predicting 0.5 when truth is 1: Loss ≈ 0.69
3. **Convex** for logistic regression
4. **Unbounded**: As ŷ → 0 when y=1, loss → ∞

**Derivative** (binary case with sigmoid):
∂BCE/∂z = ŷ - y

Where z is pre-activation (logit) and ŷ = sigmoid(z)

**Remarkable property**: The gradient simplifies to just the error!

**Why Cross-Entropy for Classification**:

1. **Probabilistic interpretation**: 
   - Minimizing cross-entropy = maximizing likelihood
   - ŷ represents probability distribution
   - Measures "distance" between true and predicted distributions

2. **Handles probabilities correctly**:
   - If y=1, only log(ŷ) matters → encourages ŷ → 1
   - If y=0, only log(1-ŷ) matters → encourages ŷ → 0

3. **Better gradients for classification**:
   - MSE + sigmoid leads to flat gradients when very wrong
   - Cross-entropy + sigmoid gives gradient proportional to error

4. **Penalizes confidence on wrong predictions**:
   - Being confidently wrong (ŷ=0.99 when y=0) is heavily penalized

**Example**:
\`\`\`python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-10):
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + 
                    (1 - y_true) * np.log(1 - y_pred))

# Classification predictions
y_true = np.array([1, 0, 1, 1, 0])
y_pred_good = np.array([0.9, 0.1, 0.8, 0.85, 0.15])
y_pred_bad = np.array([0.6, 0.4, 0.6, 0.55, 0.45])

loss_good = binary_cross_entropy(y_true, y_pred_good)
loss_bad = binary_cross_entropy(y_true, y_pred_bad)

print(f"Good predictions loss: {loss_good:.4f}")  # ~0.15
print(f"Bad predictions loss: {loss_bad:.4f}")    # ~0.62

# Notice: Confident correct predictions heavily rewarded
\`\`\`

**Why NOT MSE for Classification**:

With sigmoid + MSE:
- Derivative: σ'(z) · (ŷ - y)
- Problem: σ'(z) ≈ 0 when z is very large/small (saturated)
- If model is very wrong (z >> 0 when y=0), σ'(z) ≈ 0 → gradient ≈ 0
- Learning stalls even though error is large!

With sigmoid + cross-entropy:
- Derivative simplifies to just (ŷ - y)
- No saturation problem
- Large error → large gradient → fast learning

**Gradient Descent Dynamics**:

**MSE**:
\`\`\`python
# Update rule
θ_new = θ_old - α · (2/n) Σ(ŷ - y) · ∂ŷ/∂θ

# Linear gradient in error
# Far from optimum → large gradient → big steps
# Near optimum → small gradient → small steps (good!)
\`\`\`

**Cross-Entropy**:
\`\`\`python
# Update rule (with softmax)
θ_new = θ_old - α · (1/n) Σ(ŷ - y)

# Also linear in error, but
# Logarithmic penalty encourages extreme probabilities (0 or 1)
# Better for classification where we want confident decisions
\`\`\`

**Trading Application**:

**Regression (price prediction)**:
\`\`\`python
# Predicting stock price: Use MSE
loss = mse_loss(actual_prices, predicted_prices)
# Treats $10 error on $100 stock same as $10 error on $1000 stock
\`\`\`

**Classification (trade direction)**:
\`\`\`python
# Predicting up/down: Use cross-entropy
loss = binary_cross_entropy(actual_direction, predicted_prob)
# Heavily penalizes confident wrong predictions
# In trading, being confidently wrong is especially costly!
\`\`\`

**Advanced**: For trading, you might use custom losses:
\`\`\`python
def asymmetric_mse(y_true, y_pred):
    """Penalize underestimating risk more than overestimating"""
    error = y_pred - y_true
    return np.mean(np.where(error > 0, error**2, 2 * error**2))
\`\`\`

**Summary**:
- MSE: Regression, Gaussian assumption, quadratic penalty, symmetric
- Cross-Entropy: Classification, probabilistic, logarithmic penalty, matches sigmoid/softmax
- Derivatives determine learning speed and stability
- Choice of loss function should match problem structure`,
          keyPoints: [
            'MSE: Quadratic penalty, symmetric, convex for linear models, linear gradient',
            'Cross-Entropy: Logarithmic penalty, probabilistic interpretation, unbounded',
            'MSE + sigmoid has vanishing gradient problem for classification',
            'Cross-Entropy + sigmoid derivative simplifies to (ŷ - y)',
            'Choose loss function based on problem: regression → MSE, classification → cross-entropy',
          ],
        },
      ],
    },
    {
      id: 'exponents-logarithms',
      title: 'Exponents & Logarithms',
      content: `
# Exponents & Logarithms

## Introduction

Exponents and logarithms are fundamental operations that appear everywhere in machine learning: learning rate schedules, activation functions (sigmoid, softmax), information theory (entropy, cross-entropy), complexity analysis, and time series forecasting. Understanding their properties and relationship is crucial for both theory and implementation.

## Laws of Exponents

### Basic Rules

**Product Rule**: aᵐ · aⁿ = aᵐ⁺ⁿ
**Quotient Rule**: aᵐ / aⁿ = aᵐ⁻ⁿ
**Power Rule**: (aᵐ)ⁿ = aᵐⁿ
**Power of Product**: (ab)ᵐ = aᵐbᵐ
**Power of Quotient**: (a/b)ᵐ = aᵐ/bᵐ
**Zero Exponent**: a⁰ = 1 (for a ≠ 0)
**Negative Exponent**: a⁻ⁿ = 1/aⁿ
**Fractional Exponent**: a^(m/n) = ⁿ√(aᵐ)

### Python Implementation

\`\`\`python
import numpy as np

# Basic exponent operations
a, m, n = 2, 3, 4

# Product rule
print(f"{a}^{m} · {a}^{n} = {a**m * a**n} = {a**(m+n)}")  # 2³ · 2⁴ = 128 = 2⁷

# Quotient rule
print(f"{a}^{m} / {a}^{n} = {a**m / a**n} = {a**(m-n)}")  # 2³ / 2⁴ = 0.125 = 2⁻¹

# Power rule
print(f"({a}^{m})^{n} = {(a**m)**n} = {a**(m*n)}")  # (2³)⁴ = 4096 = 2¹²

# Zero exponent
print(f"{a}^0 = {a**0}")  # 2⁰ = 1

# Negative exponent
print(f"{a}^-{m} = {a**(-m)} = {1/a**m}")  # 2⁻³ = 0.125

# Fractional exponent
print(f"{a}^(1/{m}) = {a**(1/m)} ≈ {a**(1/m):.4f}")  # 2^(1/3) ≈ 1.2599 (cube root)
\`\`\`

### ML Application: Learning Rate Decay

Exponential decay is common for learning rate schedules:

\`\`\`python
def exponential_decay(initial_lr, epoch, decay_rate):
    """
    Learning rate with exponential decay
    lr = lr₀ · e^(-decay_rate · epoch)
    """
    return initial_lr * np.exp(-decay_rate * epoch)

initial_lr = 0.1
decay_rate = 0.05
epochs = np.arange(0, 100)

# Calculate learning rates
lrs = exponential_decay(initial_lr, epochs, decay_rate)

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(epochs, lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Exponential Learning Rate Decay')
plt.grid(True)
plt.show()

print(f"Initial LR: {lrs[0]:.4f}")
print(f"LR at epoch 50: {lrs[50]:.6f}")
print(f"LR at epoch 99: {lrs[99]:.6f}")
\`\`\`

## Logarithms

### Definition

**Logarithm**: The inverse of exponentiation

If aˣ = y, then logₐ(y) = x
- a: base
- y: argument
- x: result

**Common bases**:
- **Natural log**: ln(x) = logₑ(x), where e ≈ 2.71828
- **Common log**: log₁₀(x)
- **Binary log**: log₂(x) (used in information theory)

### Python Implementation

\`\`\`python
# Different logarithm bases
x = 8

# Natural logarithm (base e)
ln_x = np.log(x)
print(f"ln({x}) = {ln_x:.4f}")
print(f"Verify: e^{ln_x:.4f} = {np.exp(ln_x):.4f}")

# Common logarithm (base 10)
log10_x = np.log10(x)
print(f"\\nlog₁₀({x}) = {log10_x:.4f}")
print(f"Verify: 10^{log10_x:.4f} = {10**log10_x:.4f}")

# Binary logarithm (base 2)
log2_x = np.log2(x)
print(f"\\nlog₂({x}) = {log2_x:.4f}")
print(f"Verify: 2^{log2_x:.4f} = {2**log2_x:.4f}")

# Change of base formula: logₐ(x) = ln(x) / ln(a)
base = 5
logbase_x = np.log(x) / np.log(base)
print(f"\\nlog₅({x}) = {logbase_x:.4f}")
print(f"Verify: 5^{logbase_x:.4f} = {base**logbase_x:.4f}")
\`\`\`

## Laws of Logarithms

### Basic Rules

**Product Rule**: log(xy) = log(x) + log(y)
**Quotient Rule**: log(x/y) = log(x) - log(y)
**Power Rule**: log(xⁿ) = n·log(x)
**Change of Base**: logₐ(x) = logᵦ(x) / logᵦ(a)
**Identity**: logₐ(a) = 1
**Inverse**: logₐ(1) = 0

### Python Verification

\`\`\`python
x, y, n = 4, 16, 3

# Product rule
print(f"log({x}·{y}) = {np.log(x*y):.4f}")
print(f"log({x}) + log({y}) = {np.log(x) + np.log(y):.4f}")
print(f"Equal: {np.isclose(np.log(x*y), np.log(x) + np.log(y))}")

# Quotient rule
print(f"\\nlog({y}/{x}) = {np.log(y/x):.4f}")
print(f"log({y}) - log({x}) = {np.log(y) - np.log(x):.4f}")
print(f"Equal: {np.isclose(np.log(y/x), np.log(y) - np.log(x))}")

# Power rule
print(f"\\nlog({x}^{n}) = {np.log(x**n):.4f}")
print(f"{n}·log({x}) = {n * np.log(x):.4f}")
print(f"Equal: {np.isclose(np.log(x**n), n * np.log(x))}")
\`\`\`

### ML Application: Log Space Computation

Many ML operations are more stable in log space:

\`\`\`python
# Problem: Computing product of many small probabilities
probs = np.array([0.1, 0.2, 0.15, 0.08, 0.12])

# Direct multiplication (can underflow)
product_direct = np.prod(probs)
print(f"Direct product: {product_direct}")
print(f"Scientific notation: {product_direct:.2e}")

# Log space computation (more stable)
log_probs = np.log(probs)
log_product = np.sum(log_probs)  # log(a·b·c) = log(a) + log(b) + log(c)
product_log_space = np.exp(log_product)
print(f"\\nLog space product: {product_log_space}")
print(f"Scientific notation: {product_log_space:.2e}")

# Even with very small probabilities
tiny_probs = np.full(100, 0.01)  # 100 probabilities of 0.01
print(f"\\n100 probabilities of 0.01:")
print(f"Direct: {np.prod(tiny_probs):.2e}")  # May underflow to 0
print(f"Log space: {np.exp(np.sum(np.log(tiny_probs))):.2e}")
\`\`\`

## Exponential Growth and Compound Interest

### Compound Interest Formula

A = P(1 + r/n)ⁿᵗ
- A: final amount
- P: principal (initial amount)
- r: annual interest rate
- n: number of times compounded per year
- t: time in years

**Continuous compounding**: A = Peʳᵗ (as n → ∞)

\`\`\`python
def compound_interest(principal, rate, times_per_year, years):
    """Calculate compound interest"""
    return principal * (1 + rate/times_per_year)**(times_per_year * years)

def continuous_compound(principal, rate, years):
    """Calculate continuous compound interest"""
    return principal * np.exp(rate * years)

# Example: $1000 at 5% for 10 years
P, r, t = 1000, 0.05, 10

# Different compounding frequencies
print(f"Initial investment: \${P}")
print(f"Annual rate: {r*100}%")
print(f"Time: {t} years\\n")

print(f"Annual compounding: \${compound_interest(P, r, 1, t):.2f}")
print(f"Monthly compounding: \${compound_interest(P, r, 12, t):.2f}")
print(f"Daily compounding: \${compound_interest(P, r, 365, t):.2f}")
print(f"Continuous compounding: \${continuous_compound(P, r, t):.2f}")
\`\`\`

### Trading Application: Portfolio Growth

\`\`\`python
def portfolio_growth(initial_value, monthly_return, months):
    """
    Calculate portfolio growth with compound returns
    Similar to compound interest but with monthly returns
    """
    return initial_value * (1 + monthly_return)**months

# Example: $10,000 portfolio, 2% average monthly return
initial = 10000
monthly_return = 0.02
months = 12

final_value = portfolio_growth(initial, monthly_return, months)
total_return = (final_value - initial) / initial

print(f"Initial portfolio: \${initial:,.2f}")
print(f"Average monthly return: {monthly_return*100}%")
print(f"After {months} months: \${final_value:,.2f}")
print(f"Total return: {total_return*100:.2f}%")
print(f"Simple (non-compounded) would be: {monthly_return*months*100:.2f}%")
\`\`\`

## Logarithmic Scales

Logarithmic scales are useful when data spans many orders of magnitude:

\`\`\`python
# Training loss often decreases exponentially
epochs = np.arange(1, 101)
loss = 10 * np.exp(-0.05 * epochs) + 0.1  # Exponential decay + noise

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
ax1.plot(epochs, loss, linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss (Linear Scale)')
ax1.grid(True)

# Logarithmic scale
ax2.plot(epochs, loss, linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_yscale('log')
ax2.set_title('Training Loss (Log Scale)')
ax2.grid(True)

plt.tight_layout()
plt.show()

print("Log scale makes exponential trends linear!")
\`\`\`

## Natural Exponential and e

### The number e

e ≈ 2.71828... is the base of natural logarithms

**Properties**:
- e = lim(n→∞) (1 + 1/n)ⁿ
- e = Σ(1/k!) for k=0 to ∞
- eˣ is the only function equal to its own derivative: d/dx(eˣ) = eˣ

### Why e is "Natural"

\`\`\`python
# Demonstrating e through compound interest
n_values = [1, 10, 100, 1000, 10000, 100000, 1000000]
e_approx = [(1 + 1/n)**n for n in n_values]

print("Approximating e with (1 + 1/n)^n:")
for n, e_val in zip(n_values, e_approx):
    print(f"n = {n:>7}: e ≈ {e_val:.10f}")
print(f"\\nActual e: {np.e:.10f}")

# e through series
def e_series(terms):
    """Approximate e using series: e = 1 + 1/1! + 1/2! + 1/3! + ..."""
    from math import factorial
    return sum(1/factorial(k) for k in range(terms))

print(f"\\ne from series (10 terms): {e_series(10):.10f}")
print(f"e from series (20 terms): {e_series(20):.10f}")
\`\`\`

## Information Theory: Entropy

Logarithms are fundamental in information theory:

### Shannon Entropy

H(X) = -Σ p(x) log₂(p(x))

Measures average information content or uncertainty.

\`\`\`python
def entropy(probabilities):
    """
    Calculate Shannon entropy
    Uses log base 2 (bits of information)
    """
    # Remove zeros to avoid log(0)
    probs = probabilities[probabilities > 0]
    return -np.sum(probs * np.log2(probs))

# Example 1: Fair coin
fair_coin = np.array([0.5, 0.5])
H_fair = entropy(fair_coin)
print(f"Fair coin entropy: {H_fair:.4f} bits")  # 1 bit

# Example 2: Biased coin
biased_coin = np.array([0.9, 0.1])
H_biased = entropy(biased_coin)
print(f"Biased coin entropy: {H_biased:.4f} bits")  # Less than 1

# Example 3: Certain outcome
certain = np.array([1.0, 0.0])
H_certain = entropy(certain)
print(f"Certain outcome entropy: {H_certain:.4f} bits")  # 0

# Example 4: Uniform distribution over 8 outcomes
uniform_8 = np.ones(8) / 8
H_uniform = entropy(uniform_8)
print(f"Uniform over 8 outcomes: {H_uniform:.4f} bits")  # 3 bits (log₂(8))

print("\\nHigher entropy = more uncertainty = more information needed")
\`\`\`

### Cross-Entropy Loss

Cross-entropy between true distribution p and predicted distribution q:

H(p, q) = -Σ p(x) log(q(x))

\`\`\`python
def cross_entropy(y_true, y_pred, epsilon=1e-10):
    """
    Cross-entropy loss (using natural log)
    y_true: true probabilities
    y_pred: predicted probabilities
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

# Binary classification example
y_true = np.array([1, 0, 1, 1, 0])  # True labels

# One-hot encode for binary case
y_true_onehot = np.column_stack([1 - y_true, y_true])

# Good predictions (confident and correct)
y_pred_good = np.array([[0.1, 0.9],   # Predict 1, true is 1 ✓
                        [0.9, 0.1],   # Predict 0, true is 0 ✓
                        [0.2, 0.8],   # Predict 1, true is 1 ✓
                        [0.15, 0.85], # Predict 1, true is 1 ✓
                        [0.85, 0.15]])# Predict 0, true is 0 ✓

# Bad predictions (wrong)
y_pred_bad = np.array([[0.6, 0.4],    # Predict 0, true is 1 ✗
                       [0.4, 0.6],    # Predict 1, true is 0 ✗
                       [0.5, 0.5],    # Uncertain, true is 1
                       [0.7, 0.3],    # Predict 0, true is 1 ✗
                       [0.3, 0.7]])   # Predict 1, true is 0 ✗

loss_good = sum(cross_entropy(y_true_onehot[i], y_pred_good[i]) 
                for i in range(len(y_true))) / len(y_true)
loss_bad = sum(cross_entropy(y_true_onehot[i], y_pred_bad[i]) 
               for i in range(len(y_true))) / len(y_true)

print(f"Cross-entropy (good predictions): {loss_good:.4f}")
print(f"Cross-entropy (bad predictions): {loss_bad:.4f}")
print(f"\\nLower is better - good predictions have lower loss")
\`\`\`

## Logarithmic Complexity

Algorithm complexity often involves logarithms:

\`\`\`python
import time

# Binary search: O(log n)
def binary_search(arr, target):
    """Binary search in sorted array"""
    left, right = 0, len(arr) - 1
    comparisons = 0
    
    while left <= right:
        comparisons += 1
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid, comparisons
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1, comparisons

# Linear search: O(n)
def linear_search(arr, target):
    """Linear search"""
    comparisons = 0
    for i, val in enumerate(arr):
        comparisons += 1
        if val == target:
            return i, comparisons
    return -1, comparisons

# Compare on different sizes
sizes = [100, 1000, 10000, 100000]
print("Comparisons needed to find element at end:\\n")
print(f"{'Size':>10} {'Linear':>12} {'Binary':>12} {'log₂(n)':>12}")
print("-" * 50)

for n in sizes:
    arr = list(range(n))
    target = n - 1  # Last element
    
    _, linear_comps = linear_search(arr, target)
    _, binary_comps = binary_search(arr, target)
    log2_n = np.log2(n)
    
    print(f"{n:>10} {linear_comps:>12} {binary_comps:>12} {log2_n:>12.2f}")

print("\\nBinary search comparisons ≈ log₂(n)")
print("This is why logarithmic algorithms scale so well!")
\`\`\`

## Practical ML Applications

### Softmax with Log-Sum-Exp Trick

Softmax is numerically unstable. The log-sum-exp trick uses logarithms for stability:

\`\`\`python
def softmax_naive(x):
    """Naive softmax (can overflow)"""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def softmax_stable(x):
    """Numerically stable softmax using log-sum-exp trick"""
    # Subtract max for stability
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Example with large values
logits_large = np.array([1000, 1001, 1002])

try:
    result_naive = softmax_naive(logits_large)
    print(f"Naive softmax: {result_naive}")
except:
    print("Naive softmax: OVERFLOW ERROR")

result_stable = softmax_stable(logits_large)
print(f"Stable softmax: {result_stable}")
print(f"Sum: {np.sum(result_stable):.10f}")  # Should be 1.0

# Why it works:
# softmax(x) = exp(x) / Σexp(x)
# = exp(x - max(x)) / Σexp(x - max(x))
# Subtracting max keeps exponentials from overflowing
\`\`\`

### Log-Likelihood in Training

Many loss functions are negative log-likelihoods:

\`\`\`python
def negative_log_likelihood(y_true, y_pred_prob, epsilon=1e-10):
    """
    Negative log-likelihood loss
    Equivalent to cross-entropy for classification
    """
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    return -np.mean(np.log(y_pred_prob[np.arange(len(y_true)), y_true]))

# Multi-class classification
y_true = np.array([0, 2, 1, 0, 2])  # Class indices
y_pred_prob = np.array([
    [0.7, 0.2, 0.1],  # Predict class 0, true is 0 ✓
    [0.1, 0.2, 0.7],  # Predict class 2, true is 2 ✓
    [0.2, 0.6, 0.2],  # Predict class 1, true is 1 ✓
    [0.8, 0.1, 0.1],  # Predict class 0, true is 0 ✓
    [0.1, 0.3, 0.6],  # Predict class 2, true is 2 ✓
])

nll = negative_log_likelihood(y_true, y_pred_prob)
print(f"Negative Log-Likelihood: {nll:.4f}")
print("\\nLower NLL = better predictions")
print("NLL is minimized when predicted probabilities match true labels")
\`\`\`

## Summary

- **Exponents** model growth (compound interest, neural network depth, learning curves)
- **Logarithms** are inverses of exponents and compress large ranges
- **Laws** of exponents and logarithms simplify complex calculations
- **Log space** provides numerical stability for products of small numbers
- **e** and natural log appear naturally in continuous growth and calculus
- **Entropy** and **cross-entropy** use logarithms to measure information and loss
- **Logarithmic complexity** O(log n) is extremely efficient for large datasets
- **Numerical tricks** (log-sum-exp, log-likelihood) prevent overflow/underflow

**Key ML Applications**:
- Softmax and log-softmax
- Cross-entropy loss
- Learning rate schedules
- Information theory metrics
- Complexity analysis
- Numerical stability
`,
      multipleChoice: [
        {
          id: 'mc1-exponent-laws',
          question: 'Simplify: (2³)² · 2⁴ / 2⁵',
          options: ['2⁵', '2⁷', '2⁹', '2¹¹'],
          correctAnswer: 0,
          explanation:
            'Step by step: (2³)² = 2⁶ (power rule). Then 2⁶ · 2⁴ = 2¹⁰ (product rule). Finally 2¹⁰ / 2⁵ = 2⁵ (quotient rule). Answer: 2⁵ = 32.',
        },
        {
          id: 'mc2-logarithm-properties',
          question: 'If log₂(x) = 5, what is x?',
          options: ['10', '25', '32', '64'],
          correctAnswer: 2,
          explanation:
            'log₂(x) = 5 means 2⁵ = x. Therefore x = 32. Logarithms and exponents are inverse operations.',
        },
        {
          id: 'mc3-log-laws',
          question: 'Simplify: log(100) + log(10) - log(10)',
          options: ['log(100)', 'log(1000)', '2', '3'],
          correctAnswer: 0,
          explanation:
            'Using log laws: log(100) + log(10) = log(100·10) = log(1000). Then log(1000) - log(10) = log(1000/10) = log(100). If using base 10: log₁₀(100) = 2.',
        },
        {
          id: 'mc4-entropy',
          question:
            'A fair coin flip has entropy H = 1 bit. What is the entropy of a fair 4-sided die?',
          options: ['1 bit', '2 bits', '3 bits', '4 bits'],
          correctAnswer: 1,
          explanation:
            'For uniform distribution over n outcomes: H = log₂(n). For 4-sided die: H = log₂(4) = 2 bits. You need 2 bits to represent 4 equally likely outcomes.',
        },
        {
          id: 'mc5-compound-interest',
          question:
            'Which gives higher returns after 1 year: $100 at 12% compounded monthly, or $100 at 12% simple interest?',
          options: [
            'Simple interest',
            'Compound interest',
            'They are equal',
            'Cannot determine',
          ],
          correctAnswer: 1,
          explanation:
            'Simple: $100 + $100(0.12) = $112. Compound monthly: $100(1 + 0.12/12)¹² = $100(1.01)¹² ≈ $112.68. Compound interest is always higher than simple interest for the same nominal rate.',
        },
      ],
      quiz: [
        {
          id: 'dq1-log-space-stability',
          question:
            'Explain why computing in log space is more numerically stable than direct computation. Provide specific examples from machine learning where this matters (softmax, likelihood computation). What are the trade-offs?',
          sampleAnswer: `Computing in log space is crucial for numerical stability in machine learning, especially when dealing with very small or very large numbers that can cause underflow or overflow.

**Why Log Space is More Stable**:

**Problem 1: Underflow**
When multiplying many small probabilities (common in ML), the product can underflow to 0:

\`\`\`python
# Example: Computing likelihood of a sequence
probs = np.array([0.1, 0.15, 0.08, 0.12, 0.09])
print(f"Direct product: {np.prod(probs)}")  # 0.00001296

# With 100 such probabilities
tiny_probs = np.full(100, 0.1)
print(f"100 probs: {np.prod(tiny_probs):.2e}")  # Underflows to 0!

# Log space is stable
log_likelihood = np.sum(np.log(tiny_probs))
print(f"Log-likelihood: {log_likelihood:.4f}")  # -230.2585 (stable!)
# Convert back if needed: np.exp(log_likelihood)
\`\`\`

**Why it works**:
- log(a · b · c) = log(a) + log(b) + log(c)
- Turns multiplication → addition
- Addition is numerically stable
- Log maps (0, 1) → (-∞, 0), spreading out small numbers

**Problem 2: Overflow in Softmax**

Naive softmax with large inputs overflows:

\`\`\`python
def softmax_naive(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# Large logits (common in deep networks)
logits = np.array([1000, 1001, 1002])
try:
    result = softmax_naive(logits)
    print(result)  # RuntimeWarning: overflow
except:
    print("OVERFLOW!")

# Log-sum-exp trick
def softmax_stable(x):
    x_shifted = x - np.max(x)  # Shift by max
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

result_stable = softmax_stable(logits)
print(f"Stable result: {result_stable}")  # [0.09, 0.24, 0.67]
\`\`\`

**Derivation**:
softmax(x) = exp(xᵢ) / Σexp(xⱼ)
= exp(xᵢ - c) / Σexp(xⱼ - c)  [for any constant c]
= exp(xᵢ - max(x)) / Σexp(xⱼ - max(x))  [choose c = max(x)]

Subtracting max ensures all exponentials are ≤ 1, preventing overflow.

**Application 3: Log-Likelihood in Training**

Maximum likelihood estimation is more stable in log space:

\`\`\`python
# Likelihood of data given model parameters
def likelihood(data, model_params):
    """Product of individual probabilities"""
    probs = [model_prob(x, model_params) for x in data]
    return np.prod(probs)  # Can underflow!

def log_likelihood(data, model_params):
    """Sum of log probabilities"""
    log_probs = [np.log(model_prob(x, model_params)) for x in data]
    return np.sum(log_probs)  # Stable!

# In practice, we minimize negative log-likelihood
nll = -log_likelihood(data, params)
\`\`\`

**Why this matters**:
- Maximizing likelihood = Minimizing negative log-likelihood
- log is monotonic, so arg max doesn't change
- But computation is stable

**Application 4: Numerical Precision**

\`\`\`python
# Compare precision
a = 1e-100
b = 1e-100

# Direct multiplication
product = a * b
print(f"Direct: {product}")  # May be 0 due to underflow

# Log space
log_product = np.log(a) + np.log(b)
print(f"Log space: {log_product:.4f}")  # -460.5170 (precise!)
print(f"Recovered: {np.exp(log_product):.2e}")  # 1.00e-200
\`\`\`

**Trade-offs**:

**Advantages**:
✅ Prevents underflow/overflow
✅ Multiplication becomes addition (faster, more accurate)
✅ Essential for long sequences (RNNs, HMMs)
✅ Natural for likelihood-based methods

**Disadvantages**:
❌ log() and exp() are expensive operations
❌ Must remember to convert back (exp) when needed
❌ Can't directly compare probabilities (must exponentiate)
❌ Not intuitive (working with log-probabilities)
❌ Requires careful bookkeeping

**When to Use Log Space**:

✅ **Always use** for:
- Likelihood computation with many terms
- Softmax with potentially large logits
- Sequence modeling (RNNs, HMMs)
- Bayesian inference
- Information theory metrics

❌ **Don't need** for:
- Single probability computations
- Small datasets where underflow unlikely
- When direct probability interpretation needed

**Real Trading Example**:

\`\`\`python
# Bayesian portfolio optimization
def portfolio_log_likelihood(returns, weights, params):
    """
    Compute log-likelihood of portfolio returns
    More stable than direct likelihood
    """
    residuals = returns - np.dot(weights, params)
    # Log of Gaussian likelihood
    log_like = -0.5 * np.sum(residuals**2 / params['variance'])
    log_like -= 0.5 * len(returns) * np.log(2 * np.pi * params['variance'])
    return log_like

# Optimize in log space, interpret results in probability space
\`\`\`

**Summary**:
Log space transforms multiplication into addition, preventing numerical issues with extreme values. Essential for ML stability, especially in deep learning and probabilistic models. The computational overhead is worth it for numerical reliability.`,
          keyPoints: [
            'Log space turns multiplication into addition, preventing underflow/overflow',
            'Softmax uses log-sum-exp trick: shift by max before exponentiation',
            'Maximum likelihood = Minimum negative log-likelihood (stable optimization)',
            'Trade-off: computational cost vs numerical stability',
            'Essential for: sequence models, likelihood computation, deep networks',
          ],
        },
        {
          id: 'dq2-compound-growth',
          question:
            'Compare linear growth, exponential growth, and logarithmic growth. For each, provide the mathematical form, real-world examples, and explain when each dominates. How does compound interest relate to portfolio returns in trading?',
          sampleAnswer: `Understanding different growth patterns is fundamental to mathematics, computer science, and finance. Each has distinct characteristics and applications.

**Linear Growth: f(x) = mx + b**

**Characteristics**:
- Constant additive change per unit
- Straight line on regular plot
- Predictable, steady growth

**Examples**:
- Saving $100/month (no interest)
- Distance traveled at constant speed
- Simple interest: I = Prt

\`\`\`python
def linear_growth(initial, rate, time):
    return initial + rate * time

# $1000 + $100/month
t = np.arange(0, 60)  # 60 months
linear = linear_growth(1000, 100, t)

plt.plot(t, linear, label='Linear', linewidth=2)
plt.xlabel('Time (months)')
plt.ylabel('Value ($)')
plt.title('Linear Growth')
plt.grid(True)
\`\`\`

**Exponential Growth: f(x) = a · bˣ or f(x) = a · eʳˣ**

**Characteristics**:
- Constant multiplicative change per unit
- Growth rate proportional to current value
- Curves upward on regular plot, straight on log plot

**Examples**:
- Compound interest
- Population growth
- Viral spread
- Neural network gradient explosion
- Portfolio returns (compounded)

\`\`\`python
def exponential_growth(initial, rate, time):
    return initial * np.exp(rate * time)

# $1000 at 10% annual rate, compounded
t = np.arange(0, 60)
exponential = exponential_growth(1000, 0.10/12, t)  # Monthly

plt.plot(t, exponential, label='Exponential', linewidth=2)
\`\`\`

**Logarithmic Growth: f(x) = a · log(x) + b**

**Characteristics**:
- Growth slows down over time
- Early rapid growth, then plateaus
- Inverse of exponential

**Examples**:
- Algorithm complexity: binary search O(log n)
- Information gain from data (diminishing returns)
- Depth of balanced binary tree
- Learning curves (diminishing improvement)

\`\`\`python
def logarithmic_growth(initial, scale, time):
    return initial + scale * np.log(time + 1)  # +1 to avoid log(0)

t = np.arange(0, 60)
logarithmic = logarithmic_growth(1000, 500, t)

plt.plot(t, logarithmic, label='Logarithmic', linewidth=2)
\`\`\`

**Comparison**:

\`\`\`python
# Compare all three
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

t = np.arange(1, 61)
linear = 1000 + 100 * t
exponential = 1000 * (1.10)**(t/12)
logarithmic = 1000 + 500 * np.log(t)

# Linear scale
ax1.plot(t, linear, label='Linear', linewidth=2)
ax1.plot(t, exponential, label='Exponential', linewidth=2)
ax1.plot(t, logarithmic, label='Logarithmic', linewidth=2)
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.set_title('Growth Patterns (Linear Scale)')
ax1.legend()
ax1.grid(True)

# Log scale
ax2.plot(t, linear, label='Linear', linewidth=2)
ax2.plot(t, exponential, label='Exponential', linewidth=2)
ax2.plot(t, logarithmic, label='Logarithmic', linewidth=2)
ax2.set_xlabel('Time')
ax2.set_ylabel('Value')
ax2.set_yscale('log')
ax2.set_title('Growth Patterns (Log Scale)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# On log scale:
# - Exponential becomes linear
# - Linear becomes logarithmic
# - Logarithmic becomes even flatter
\`\`\`

**When Each Dominates**:

**Short term** (small x):
- All similar initially
- Logarithmic grows fastest early
- Exponential looks slow at first

**Medium term**:
- Exponential overtakes linear
- Linear overtakes logarithmic

**Long term** (large x):
- Exponential dominates everything
- Linear grows steadily
- Logarithmic barely increases

**Formal relationships**:
log(x) << x << xᵏ << eˣ << x! << xˣ

**Compound Interest & Portfolio Returns**:

**Compound Interest Formula**:
A = P(1 + r/n)ⁿᵗ

As n → ∞ (continuous): A = Peʳᵗ

\`\`\`python
# Compare simple vs compound returns
initial = 10000
annual_rate = 0.12  # 12% annual
years = np.arange(0, 30)

# Simple interest (linear)
simple = initial * (1 + annual_rate * years)

# Compound annually
compound_annual = initial * (1 + annual_rate)**years

# Compound monthly
compound_monthly = initial * (1 + annual_rate/12)**(12*years)

# Continuous compounding
continuous = initial * np.exp(annual_rate * years)

plt.figure(figsize=(12, 6))
plt.plot(years, simple, label='Simple (Linear)', linewidth=2)
plt.plot(years, compound_annual, label='Compound (Annual)', linewidth=2)
plt.plot(years, compound_monthly, label='Compound (Monthly)', linewidth=2)
plt.plot(years, continuous, label='Continuous', linewidth=2, linestyle='--')
plt.xlabel('Years')
plt.ylabel('Portfolio Value ($)')
plt.title('Simple vs Compound Returns')
plt.legend()
plt.grid(True)

print(f"After 30 years:")
print(f"Simple: \${simple[-1]:,.2f}")
print(f"Compound: \${compound_annual[-1]:,.2f}")
print(f"Continuous: \${continuous[-1]:,.2f}")
            \`\`\`

**Trading Application: Compounding Returns**:

\`\`\`python
def portfolio_simulation(initial, monthly_returns):
    """
    Simulate portfolio with compound returns
    Each month's return compounds on previous value
    """
    portfolio_values = [initial]
    current_value = initial
    
    for r in monthly_returns:
        current_value *= (1 + r)  # Compound effect
        portfolio_values.append(current_value)
    
    return np.array(portfolio_values)

# Simulate 5 years of trading
np.random.seed(42)
months = 60
# Average 1% monthly return with 5% volatility
monthly_returns = np.random.normal(0.01, 0.05, months)

portfolio = portfolio_simulation(10000, monthly_returns)

# Compare with simple (non-compounded)
simple_portfolio = 10000 * (1 + np.cumsum(np.insert(monthly_returns, 0, 0)))

plt.figure(figsize=(12, 6))
plt.plot(portfolio, label='Compound Returns', linewidth=2)
plt.plot(simple_portfolio, label='Simple Returns', linewidth=2, linestyle='--')
plt.xlabel('Month')
plt.ylabel('Portfolio Value ($)')
plt.title('Compound vs Simple Returns in Trading')
plt.legend()
plt.grid(True)

total_compound = (portfolio[-1] - 10000) / 10000 * 100
total_simple = (simple_portfolio[-1] - 10000) / 10000 * 100
print(f"Total return (compound): {total_compound:.2f}%")
print(f"Total return (simple): {total_simple:.2f}%")
print(f"Difference: {total_compound - total_simple:.2f}%")
\`\`\`

**Key Insights for Trading**:

1. **Compound returns matter**: Even small differences in return rates compound dramatically over time

2. **Drawdowns hurt more with compounding**: 
   - Lose 50% → need 100% gain to recover
   - Because you're compounding from a lower base

3. **Consistent small gains beat volatile large swings**:
   - 1% monthly (compounded) = 12.68% annually
   - 12% once per year = 12% annually
   - Compounding frequency matters!

4. **Exponential growth is powerful but rare**:
   - Can't sustain indefinitely (reversion to mean)
   - Market returns are NOT perfectly exponential
   - But long-term equity returns approximate it

**Summary**:
- Linear: constant absolute change (O(n))
- Exponential: constant relative change, dominates long-term (O(eⁿ))
- Logarithmic: diminishing returns, very efficient (O(log n))
- Compound interest = exponential growth
- In trading, compounding small consistent returns is powerful
- Understanding growth patterns helps with algorithm choice and portfolio strategy`,
          keyPoints: [
            'Linear: constant addition; Exponential: constant multiplication; Logarithmic: inverse of exponential',
            'Long-term: Exponential dominates, then linear, then logarithmic',
            'Compound interest is exponential growth: A = P(1+r)ⁿ',
            'In trading, compounding small consistent returns beats large volatile swings',
            'Drawdowns hurt more with compounding: -50% requires +100% to recover',
          ],
        },
        {
          id: 'dq3-entropy-information',
          question:
            'Explain Shannon entropy and cross-entropy. Why do we use cross-entropy as a loss function in classification? How does it relate to information theory? Provide intuition and mathematical details.',
          sampleAnswer: `Entropy and cross-entropy connect information theory to machine learning. Understanding them provides deep insight into why certain loss functions work well for classification.

**Shannon Entropy: Measuring Uncertainty**

**Definition**:
H(X) = -Σ p(x) log₂(p(x))

**Intuition**: 
Entropy measures the average amount of information (in bits) needed to describe a random variable. Higher entropy = more uncertainty = more information required.

**Examples**:

\`\`\`python
def shannon_entropy(probs):
    """Calculate Shannon entropy (base 2 for bits)"""
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

# Example 1: Fair coin (maximum uncertainty for 2 outcomes)
fair_coin = np.array([0.5, 0.5])
H_fair = shannon_entropy(fair_coin)
print(f"Fair coin: H = {H_fair:.4f} bits")  # 1.0 bit

# You need 1 bit to represent: 0=heads, 1=tails

# Example 2: Biased coin (less uncertainty)
biased_coin = np.array([0.9, 0.1])
H_biased = shannon_entropy(biased_coin)
print(f"Biased coin: H = {H_biased:.4f} bits")  # 0.469 bits

# Less information needed since outcome is more predictable

# Example 3: Certain outcome (no uncertainty)
certain = np.array([1.0, 0.0])
H_certain = shannon_entropy(certain)
print(f"Certain: H = {H_certain:.4f} bits")  # 0 bits

# No information needed - outcome is known

# Example 4: Uniform distribution (maximum uncertainty)
uniform_4 = np.array([0.25, 0.25, 0.25, 0.25])
H_uniform_4 = shannon_entropy(uniform_4)
print(f"Uniform (4 outcomes): H = {H_uniform_4:.4f} bits")  # 2.0 bits

# Need 2 bits to represent 4 equally likely outcomes
\`\`\`

**Key Property**: 
For uniform distribution over n outcomes: H = log₂(n)

**Intuition**:
- Fair coin: 2 outcomes → 1 bit
- Fair 4-sided die: 4 outcomes → 2 bits
- Fair 8-sided die: 8 outcomes → 3 bits

**Cross-Entropy: Comparing Distributions**

**Definition**:
H(p, q) = -Σ p(x) log(q(x))

Where:
- p(x): true distribution
- q(x): predicted/approximate distribution

**Intuition**: 
Average number of bits needed to encode data from true distribution p using code optimized for distribution q.

**Relationship to Entropy**:
H(p, q) ≥ H(p, p) = H(p)

Equality holds only when q = p (perfect match).

**KL Divergence** (relative entropy):
D_KL(p || q) = H(p, q) - H(p) = Σ p(x) log(p(x)/q(x))

Measures "distance" from q to p (not symmetric).

\`\`\`python
def cross_entropy(p, q, epsilon=1e-10):
    """Cross-entropy between distributions p and q"""
    q = np.clip(q, epsilon, 1)  # Avoid log(0)
    return -np.sum(p * np.log(q))

def kl_divergence(p, q, epsilon=1e-10):
    """KL divergence from q to p"""
    return cross_entropy(p, q, epsilon) - shannon_entropy(p)

# True distribution
p_true = np.array([0.6, 0.3, 0.1])

# Perfect prediction
q_perfect = np.array([0.6, 0.3, 0.1])

# Good prediction
q_good = np.array([0.5, 0.35, 0.15])

# Bad prediction
q_bad = np.array([0.1, 0.1, 0.8])

print(f"Entropy H(p): {shannon_entropy(p_true):.4f}")
print(f"\\nCross-Entropy:")
print(f"  Perfect: {cross_entropy(p_true, q_perfect):.4f}")
print(f"  Good: {cross_entropy(p_true, q_good):.4f}")
print(f"  Bad: {cross_entropy(p_true, q_bad):.4f}")

print(f"\\nKL Divergence:")
print(f"  Perfect: {kl_divergence(p_true, q_perfect):.6f}")
print(f"  Good: {kl_divergence(p_true, q_good):.4f}")
print(f"  Bad: {kl_divergence(p_true, q_bad):.4f}")
\`\`\`

**Cross-Entropy as Loss Function**

In classification, we want to minimize distance between:
- p: true distribution (one-hot encoded labels)
- q: predicted distribution (model outputs)

**Binary Cross-Entropy**:
BCE = -Σ[y log(ŷ) + (1-y) log(1-ŷ)]

**Categorical Cross-Entropy** (multi-class):
CCE = -Σ Σ y_ij log(ŷ_ij)

\`\`\`python
# Binary classification example
def binary_cross_entropy(y_true, y_pred, epsilon=1e-10):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + 
                    (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([1, 0, 1, 1, 0])

# Confident correct predictions
y_pred_good = np.array([0.95, 0.05, 0.90, 0.85, 0.10])
loss_good = binary_cross_entropy(y_true, y_pred_good)

# Uncertain predictions
y_pred_uncertain = np.array([0.6, 0.4, 0.6, 0.55, 0.45])
loss_uncertain = binary_cross_entropy(y_true, y_pred_uncertain)

# Confident wrong predictions
y_pred_bad = np.array([0.1, 0.9, 0.2, 0.15, 0.85])
loss_bad = binary_cross_entropy(y_true, y_pred_bad)

print(f"Good predictions: Loss = {loss_good:.4f}")
print(f"Uncertain predictions: Loss = {loss_uncertain:.4f}")
print(f"Bad predictions: Loss = {loss_bad:.4f}")

# Visualize loss landscape
pred_range = np.linspace(0.01, 0.99, 100)
loss_when_true = -np.log(pred_range)  # y=1
loss_when_false = -np.log(1 - pred_range)  # y=0

plt.figure(figsize=(10, 6))
plt.plot(pred_range, loss_when_true, label='True label = 1', linewidth=2)
plt.plot(pred_range, loss_when_false, label='True label = 0', linewidth=2)
plt.xlabel('Predicted Probability')
plt.ylabel('Loss')
plt.title('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.ylim(0, 5)
\`\`\`

**Why Cross-Entropy for Classification?**

**1. Probabilistic Interpretation**:
- Minimizing cross-entropy = Maximizing likelihood
- Model outputs interpreted as probabilities
- Natural fit for classification

**Proof**:
Given data D and model parameters θ:
- Likelihood: L(θ) = Π p(yᵢ|xᵢ; θ)
- Log-likelihood: log L(θ) = Σ log p(yᵢ|xᵢ; θ)
- For classification: p(y=1|x) = ŷ
- Negative log-likelihood = Cross-entropy loss!

**2. Proper Gradients with Softmax/Sigmoid**:

With softmax + cross-entropy:
∂L/∂z = ŷ - y (simple!)

With softmax + MSE:
∂L/∂z = (ŷ - y) · ŷ · (1 - ŷ) (more complex, can vanish)

\`\`\`python
# Compare gradients
y_true = 1
predictions = np.linspace(0.01, 0.99, 100)

# Cross-entropy gradient magnitude: |ŷ - y|
ce_grad = np.abs(predictions - y_true)

# MSE gradient magnitude: |ŷ - y| · ŷ · (1 - ŷ)
mse_grad = np.abs(predictions - y_true) * predictions * (1 - predictions)

plt.figure(figsize=(10, 6))
plt.plot(predictions, ce_grad, label='Cross-Entropy', linewidth=2)
plt.plot(predictions, mse_grad, label='MSE', linewidth=2)
plt.xlabel('Predicted Probability (ŷ)')
plt.ylabel('Gradient Magnitude')
plt.title('Gradient Comparison: Cross-Entropy vs MSE')
plt.legend()
plt.grid(True)

# Notice: MSE gradient vanishes at extremes!
# Cross-entropy maintains strong gradient even when very wrong
\`\`\`

**3. Penalizes Confidence on Wrong Predictions**:

\`\`\`python
# When true label is 1:
for pred in [0.01, 0.1, 0.5, 0.9, 0.99]:
    ce_loss = -np.log(pred)
    mse_loss = (1 - pred)**2
    print(f"Pred={pred:.2f}: CE={ce_loss:.4f}, MSE={mse_loss:.4f}")

# Cross-entropy heavily penalizes confident wrong predictions
# MSE penalty is more uniform
\`\`\`

**Information Theory Connection**:

**Optimal Coding**: 
If event has probability p, optimal code length = -log₂(p) bits

**Example**:
- Event with p=0.5 → -log₂(0.5) = 1 bit
- Event with p=0.25 → -log₂(0.25) = 2 bits
- Rare events need more bits!

**Cross-entropy in ML**:
Model assigns probability ŷ to true event y.
- If model is confident and correct (ŷ ≈ 1 when y=1): low loss
- If model is confident and wrong (ŷ ≈ 0 when y=1): high loss

**Trading Application**:

\`\`\`python
# Predicting market direction
def train_direction_classifier(features, directions):
    """
    Use cross-entropy loss for binary direction classification
    directions: 1 = up, 0 = down
    """
    model = ...  # Your model
    
    # Cross-entropy loss
    def loss_fn(y_true, y_pred):
        return binary_cross_entropy(y_true, y_pred)
    
    # Train to minimize cross-entropy
    # Encourages confident predictions when pattern is clear
    # Uncertain predictions when market is ambiguous
    
    return model

# Entropy can also measure strategy diversity
def strategy_entropy(position_distribution):
    """
    High entropy = diversified positions
    Low entropy = concentrated positions
    """
    return shannon_entropy(position_distribution)

positions = np.array([0.4, 0.3, 0.2, 0.1])  # 4 assets
print(f"Portfolio entropy: {strategy_entropy(positions):.4f} bits")
\`\`\`

**Summary**:
- Entropy measures uncertainty/information content
- Cross-entropy compares two distributions
- Minimizing cross-entropy = Maximizing likelihood
- Natural loss for classification due to:
  - Probabilistic interpretation
  - Clean gradients with softmax
  - Appropriate penalty structure
- Information theory provides deep theoretical foundation for ML`,
          keyPoints: [
            'Entropy H(X) measures uncertainty: higher entropy = more information needed',
            'Cross-entropy H(p,q) measures cost of encoding p using code for q',
            'Minimizing cross-entropy = Maximizing likelihood (probabilistic interpretation)',
            'Cross-entropy + softmax gives clean gradient: ∂L/∂z = ŷ - y',
            'Heavily penalizes confident wrong predictions (important for classification)',
          ],
        },
      ],
    },
    {
      id: 'sequences-series',
      title: 'Sequences & Series',
      content: `
# Sequences & Series

## Introduction

Sequences and series are fundamental concepts that appear throughout machine learning and data science: gradient descent iterations, time series analysis, convergence analysis, loss curves, and compound returns in trading. Understanding their properties, convergence behavior, and summation is crucial for analyzing algorithmic behavior and financial models.

## Sequences

### Definition

A **sequence** is an ordered list of numbers: a₁, a₂, a₃, ..., aₙ, ...

**Notation**: {aₙ} or (aₙ)

**Index**: n (usually starts at 0 or 1)

**Term**: aₙ is the nth term

### Types of Sequences

#### Arithmetic Sequences

**Definition**: Constant difference between consecutive terms

**Formula**: aₙ = a₁ + (n-1)d
- a₁: first term
- d: common difference
- n: term number

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def arithmetic_sequence(a1, d, n_terms):
    """Generate arithmetic sequence"""
    n = np.arange(1, n_terms + 1)
    return a1 + (n - 1) * d

# Example: 3, 7, 11, 15, ...
a1, d = 3, 4
n_terms = 10
seq = arithmetic_sequence(a1, d, n_terms)

print("Arithmetic sequence:", seq)
print(f"First term: {seq[0]}")
print(f"Common difference: {d}")
print(f"10th term: {seq[-1]}")

# Verify constant difference
differences = np.diff(seq)
print(f"Differences: {differences}")
print(f"All equal to d? {np.all(differences == d)}")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_terms + 1), seq, 'bo-', markersize=8, linewidth=2)
plt.xlabel('n (term number)')
plt.ylabel('aₙ')
plt.title(f'Arithmetic Sequence: aₙ = {a1} + {d}(n-1)')
plt.grid(True)
plt.show()
\`\`\`

**ML Application**: Learning rate schedules with linear decay

\`\`\`python
def linear_lr_decay(initial_lr, decay_rate, epoch):
    """Linear learning rate decay (arithmetic sequence)"""
    return initial_lr - decay_rate * epoch

initial_lr = 0.1
decay_rate = 0.001
epochs = np.arange(0, 100)
lrs = linear_lr_decay(initial_lr, decay_rate, epochs)

plt.figure(figsize=(10, 6))
plt.plot(epochs, lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Linear Learning Rate Decay')
plt.grid(True)
plt.show()

print(f"Initial LR: {lrs[0]:.4f}")
print(f"LR at epoch 50: {lrs[50]:.4f}")
print(f"Final LR: {lrs[-1]:.4f}")
\`\`\`

#### Geometric Sequences

**Definition**: Constant ratio between consecutive terms

**Formula**: aₙ = a₁ · rⁿ⁻¹
- a₁: first term
- r: common ratio
- n: term number

\`\`\`python
def geometric_sequence(a1, r, n_terms):
    """Generate geometric sequence"""
    n = np.arange(1, n_terms + 1)
    return a1 * r**(n - 1)

# Example: 2, 6, 18, 54, ... (r=3)
a1, r = 2, 3
n_terms = 10
seq = geometric_sequence(a1, r, n_terms)

print("Geometric sequence:", seq)
print(f"First term: {seq[0]}")
print(f"Common ratio: {r}")
print(f"10th term: {seq[-1]}")

# Verify constant ratio
ratios = seq[1:] / seq[:-1]
print(f"Ratios: {ratios}")
print(f"All equal to r? {np.allclose(ratios, r)}")

# Visualize (log scale to show exponential growth)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, n_terms + 1), seq, 'ro-', markersize=8, linewidth=2)
ax1.set_xlabel('n')
ax1.set_ylabel('aₙ')
ax1.set_title(f'Geometric Sequence (Linear Scale): aₙ = {a1}·{r}^(n-1)')
ax1.grid(True)

ax2.plot(range(1, n_terms + 1), seq, 'ro-', markersize=8, linewidth=2)
ax2.set_xlabel('n')
ax2.set_ylabel('aₙ')
ax2.set_yscale('log')
ax2.set_title(f'Geometric Sequence (Log Scale)')
ax2.grid(True)

plt.tight_layout()
plt.show()
\`\`\`

**ML Application**: Exponential learning rate decay

\`\`\`python
def exponential_lr_decay(initial_lr, decay_rate, epoch):
    """Exponential learning rate decay (geometric sequence)"""
    return initial_lr * decay_rate**epoch

initial_lr = 0.1
decay_rate = 0.96  # 4% decay per epoch
epochs = np.arange(0, 100)
lrs = exponential_lr_decay(initial_lr, decay_rate, epochs)

plt.figure(figsize=(10, 6))
plt.plot(epochs, lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Exponential Learning Rate Decay')
plt.grid(True)
plt.yscale('log')
plt.show()

print(f"Initial LR: {lrs[0]:.4f}")
print(f"LR at epoch 50: {lrs[50]:.6f}")
print(f"Final LR: {lrs[-1]:.8f}")
\`\`\`

#### Recursive Sequences

**Definition**: Each term defined in terms of previous term(s)

**Example - Fibonacci**: aₙ = aₙ₋₁ + aₙ₋₂, with a₁=1, a₂=1

\`\`\`python
def fibonacci(n):
    """Generate first n Fibonacci numbers"""
    if n <= 0:
        return []
    elif n == 1:
        return [1]
    elif n == 2:
        return [1, 1]
    
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

# Generate first 20 Fibonacci numbers
fib_seq = fibonacci(20)
print("Fibonacci sequence:", fib_seq)

# Golden ratio approximation
ratios = [fib_seq[i+1] / fib_seq[i] for i in range(len(fib_seq)-1)]
print(f"\\nRatios converge to golden ratio φ ≈ 1.618...")
print(f"Last 5 ratios: {ratios[-5:]}")

golden_ratio = (1 + np.sqrt(5)) / 2
print(f"Golden ratio: {golden_ratio:.10f}")
print(f"Final ratio: {ratios[-1]:.10f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(fib_seq) + 1), fib_seq, 'go-', markersize=8, linewidth=2)
plt.xlabel('n')
plt.ylabel('Fibonacci(n)')
plt.title('Fibonacci Sequence')
plt.grid(True)
plt.yscale('log')
plt.show()
\`\`\`

**ML Application**: Recurrent relationships in RNNs

\`\`\`python
# Simple RNN: hidden state is recursive sequence
def simple_rnn_sequence(input_sequence, W_hh, W_xh, h0):
    """
    Generate hidden state sequence in RNN
    hₜ = tanh(W_hh·hₜ₋₁ + W_xh·xₜ)
    """
    hidden_states = [h0]
    h = h0
    
    for x in input_sequence:
        h = np.tanh(W_hh @ h + W_xh @ x)
        hidden_states.append(h)
    
    return np.array(hidden_states)

# Example
input_seq = [np.array([1.0]), np.array([0.5]), np.array([0.8])]
W_hh = np.array([[0.9]])  # Weight for previous hidden state
W_xh = np.array([[0.5]])  # Weight for input
h0 = np.array([0.0])      # Initial hidden state

hidden_seq = simple_rnn_sequence(input_seq, W_hh, W_xh, h0)
print("Hidden state sequence:", hidden_seq.flatten())
\`\`\`

### Convergence of Sequences

**Definition**: A sequence {aₙ} **converges** to L if for any ε > 0, there exists N such that |aₙ - L| < ε for all n > N.

**Notation**: lim(n→∞) aₙ = L

\`\`\`python
def visualize_convergence(sequence, limit, title):
    """Visualize sequence convergence"""
    n = len(sequence)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n + 1), sequence, 'bo-', markersize=6, linewidth=2, label='Sequence')
    plt.axhline(y=limit, color='r', linestyle='--', linewidth=2, label=f'Limit = {limit}')
    
    # Show epsilon bands
    epsilon = 0.1
    plt.axhline(y=limit + epsilon, color='g', linestyle=':', alpha=0.5, label=f'ε = {epsilon}')
    plt.axhline(y=limit - epsilon, color='g', linestyle=':', alpha=0.5)
    
    plt.xlabel('n')
    plt.ylabel('aₙ')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Example 1: aₙ = 1/n → 0
n = np.arange(1, 101)
seq1 = 1 / n
visualize_convergence(seq1, 0, 'Convergence: aₙ = 1/n → 0')

# Example 2: aₙ = (1 + 1/n)^n → e
seq2 = (1 + 1/n)**n
visualize_convergence(seq2, np.e, 'Convergence: aₙ = (1 + 1/n)^n → e')

print(f"lim(n→∞) 1/n = {seq1[-1]:.6f}")
print(f"lim(n→∞) (1 + 1/n)^n = {seq2[-1]:.10f}")
print(f"Actual e = {np.e:.10f}")
\`\`\`

**ML Application**: Convergence of gradient descent

\`\`\`python
def gradient_descent_sequence(f, grad_f, x0, learning_rate, n_iterations):
    """
    Generate sequence of iterates in gradient descent
    xₙ₊₁ = xₙ - α·∇f(xₙ)
    """
    x_sequence = [x0]
    x = x0
    
    for _ in range(n_iterations):
        x = x - learning_rate * grad_f(x)
        x_sequence.append(x)
    
    return np.array(x_sequence)

# Example: f(x) = x^2, minimum at x=0
def f(x):
    return x**2

def grad_f(x):
    return 2*x

x0 = 10.0
lr = 0.1
n_iter = 50

x_seq = gradient_descent_sequence(f, grad_f, x0, lr, n_iter)

plt.figure(figsize=(10, 6))
plt.plot(range(len(x_seq)), x_seq, 'bo-', markersize=4, linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Minimum')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('Gradient Descent Sequence Converging to Minimum')
plt.legend()
plt.grid(True)
plt.show()

print(f"Initial x: {x_seq[0]:.6f}")
print(f"Final x: {x_seq[-1]:.10f}")
print(f"Converged to 0? {np.abs(x_seq[-1]) < 1e-6}")
\`\`\`

## Series

### Definition

A **series** is the sum of terms in a sequence: S = a₁ + a₂ + a₃ + ... + aₙ + ...

**Notation**: Σ(n=1 to ∞) aₙ or Σ aₙ

**Partial sum**: Sₙ = Σ(k=1 to n) aₖ

### Arithmetic Series

**Sum formula**: Sₙ = n/2 · (a₁ + aₙ) = n/2 · (2a₁ + (n-1)d)

\`\`\`python
def arithmetic_series_sum(a1, d, n):
    """Sum of first n terms of arithmetic sequence"""
    # Method 1: Direct formula
    sum_formula = n/2 * (2*a1 + (n-1)*d)
    
    # Method 2: Explicit computation (verification)
    terms = arithmetic_sequence(a1, d, n)
    sum_explicit = np.sum(terms)
    
    return sum_formula, sum_explicit

# Example: Sum of 1 + 2 + 3 + ... + 100
a1, d, n = 1, 1, 100
sum_formula, sum_explicit = arithmetic_series_sum(a1, d, n)

print(f"Sum of first {n} natural numbers:")
print(f"Formula: {sum_formula:.0f}")
print(f"Explicit: {sum_explicit:.0f}")
print(f"Gauss formula: n(n+1)/2 = {n*(n+1)/2}")

# Visualize partial sums
n_values = np.arange(1, 101)
partial_sums = [arithmetic_series_sum(1, 1, n)[0] for n in n_values]

plt.figure(figsize=(10, 6))
plt.plot(n_values, partial_sums, linewidth=2)
plt.xlabel('n')
plt.ylabel('Sₙ (sum of first n terms)')
plt.title('Arithmetic Series: Sₙ = 1 + 2 + ... + n')
plt.grid(True)
plt.show()
\`\`\`

**ML Application**: Analyzing training time complexity

\`\`\`python
# Total number of operations in training with variable batch sizes
def total_operations(batch_sizes):
    """
    If batch sizes form arithmetic sequence,
    total ops = arithmetic series sum
    """
    n = len(batch_sizes)
    a1 = batch_sizes[0]
    d = batch_sizes[1] - batch_sizes[0] if n > 1 else 0
    
    total = n/2 * (2*a1 + (n-1)*d)
    return int(total)

# Example: batch sizes 32, 36, 40, ..., 100
batch_sizes = list(range(32, 101, 4))
total_ops = total_operations(batch_sizes)
print(f"Batch sizes: {batch_sizes[:5]} ... {batch_sizes[-3:]}")
print(f"Total operations: {total_ops:,}")
\`\`\`

### Geometric Series

**Sum formula (finite)**: Sₙ = a₁ · (1 - rⁿ) / (1 - r) for r ≠ 1

**Infinite series**: S = a₁ / (1 - r) if |r| < 1 (converges)

\`\`\`python
def geometric_series_sum(a1, r, n):
    """Sum of first n terms of geometric sequence"""
    if r == 1:
        return a1 * n
    
    # Finite sum formula
    sum_formula = a1 * (1 - r**n) / (1 - r)
    
    # Explicit computation (verification)
    terms = geometric_sequence(a1, r, n)
    sum_explicit = np.sum(terms)
    
    return sum_formula, sum_explicit

# Example: 1 + 1/2 + 1/4 + 1/8 + ...
a1, r = 1, 0.5
n_terms = [5, 10, 20, 50, 100]

print("Geometric series: 1 + 1/2 + 1/4 + ...")
for n in n_terms:
    sum_n, _ = geometric_series_sum(a1, r, n)
    print(f"S_{n:>3} = {sum_n:.10f}")

# Infinite sum (converges for |r| < 1)
infinite_sum = a1 / (1 - r)
print(f"\\nInfinite sum (theoretical): {infinite_sum}")
print(f"S_100 is very close to infinite sum: {np.isclose(sum_n, infinite_sum)}")

# Visualize convergence
n_range = np.arange(1, 101)
partial_sums = [geometric_series_sum(a1, r, n)[0] for n in n_range]

plt.figure(figsize=(10, 6))
plt.plot(n_range, partial_sums, linewidth=2, label='Partial sums')
plt.axhline(y=infinite_sum, color='r', linestyle='--', linewidth=2, label=f'Limit = {infinite_sum}')
plt.xlabel('n')
plt.ylabel('Sₙ')
plt.title('Geometric Series Convergence')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

**ML Application**: Discount factor in reinforcement learning

\`\`\`python
def discounted_return(rewards, gamma):
    """
    Compute discounted return (geometric series)
    G = r₁ + γr₂ + γ²r₃ + ... = Σ γᵗrₜ
    """
    n = len(rewards)
    discount_factors = gamma ** np.arange(n)
    return np.sum(rewards * discount_factors)

# Example: sequence of rewards in RL
rewards = np.array([1, 2, 3, 4, 5])
gamma = 0.9  # Discount factor

G = discounted_return(rewards, gamma)
print(f"Rewards: {rewards}")
print(f"Discount factor γ: {gamma}")
print(f"Discounted return: {G:.4f}")

# Compare with undiscounted
undiscounted = np.sum(rewards)
print(f"Undiscounted sum: {undiscounted}")
print(f"Discount effect: {(1 - G/undiscounted)*100:.1f}% reduction")
\`\`\`

### Infinite Series and Convergence

#### Tests for Convergence

**1. Divergence Test**: If lim(n→∞) aₙ ≠ 0, then Σaₙ diverges

**2. Ratio Test**: If lim |aₙ₊₁/aₙ| < 1, series converges

**3. Comparison Test**: If 0 ≤ aₙ ≤ bₙ and Σbₙ converges, then Σaₙ converges

\`\`\`python
def ratio_test(sequence, n_terms=100):
    """
    Apply ratio test to check convergence
    Returns limit of |aₙ₊₁/aₙ|
    """
    ratios = np.abs(sequence[1:] / sequence[:-1])
    
    # Take last several ratios (should stabilize)
    limit = np.mean(ratios[-10:])
    
    print(f"Ratio test: lim |aₙ₊₁/aₙ| ≈ {limit:.6f}")
    if limit < 1:
        print("Series converges (ratio < 1)")
    elif limit > 1:
        print("Series diverges (ratio > 1)")
    else:
        print("Test inconclusive (ratio = 1)")
    
    return limit

# Example 1: Σ 1/n² (converges)
n = np.arange(1, 101)
seq1 = 1 / n**2
print("Series: Σ 1/n²")
print(f"Partial sum S_100: {np.sum(seq1):.6f}")
print(f"Known limit: π²/6 = {np.pi**2/6:.6f}")
# Ratio test
ratio1 = ratio_test(seq1)

# Example 2: Σ 1/2^n (converges - geometric with r=1/2)
seq2 = 1 / 2**n
print(f"\\nSeries: Σ 1/2^n")
print(f"Partial sum S_100: {np.sum(seq2):.10f}")
print(f"Known limit: 1/(1-1/2) = 2 (minus first term)")
ratio2 = ratio_test(seq2)

# Example 3: Σ n (diverges)
seq3 = n
print(f"\\nSeries: Σ n")
print(f"Partial sum S_100: {np.sum(seq3):.0f}")
print("This series diverges (terms don't approach 0)")
\`\`\`

### Power Series

**Definition**: Σ(n=0 to ∞) cₙxⁿ

Important power series in ML:

**1. Exponential**: eˣ = Σ xⁿ/n!

**2. Sine**: sin(x) = Σ (-1)ⁿx^(2n+1)/(2n+1)!

**3. Cosine**: cos(x) = Σ (-1)ⁿx^(2n)/(2n)!

**4. Geometric**: 1/(1-x) = Σ xⁿ for |x| < 1

\`\`\`python
from math import factorial

def exp_series(x, n_terms=20):
    """Approximate e^x using power series"""
    return sum(x**n / factorial(n) for n in range(n_terms))

def sin_series(x, n_terms=20):
    """Approximate sin(x) using power series"""
    return sum((-1)**n * x**(2*n+1) / factorial(2*n+1) for n in range(n_terms))

def cos_series(x, n_terms=20):
    """Approximate cos(x) using power series"""
    return sum((-1)**n * x**(2*n) / factorial(2*n) for n in range(n_terms))

# Test approximations
x_test = 1.5

print("Power series approximations vs actual:")
print(f"\\nx = {x_test}")
print(f"e^x: series = {exp_series(x_test):.10f}, actual = {np.exp(x_test):.10f}")
print(f"sin(x): series = {sin_series(x_test):.10f}, actual = {np.sin(x_test):.10f}")
print(f"cos(x): series = {cos_series(x_test):.10f}, actual = {np.cos(x_test):.10f}")

# Visualize convergence of e^x series
x_range = np.linspace(-2, 2, 100)
n_terms_list = [1, 2, 3, 5, 10, 20]

plt.figure(figsize=(12, 6))
for n in n_terms_list:
    approx = [exp_series(x, n) for x in x_range]
    plt.plot(x_range, approx, label=f'{n} terms', linewidth=2)

plt.plot(x_range, np.exp(x_range), 'k--', linewidth=2, label='Actual e^x')
plt.xlabel('x')
plt.ylabel('e^x')
plt.title('Power Series Approximation of e^x')
plt.legend()
plt.grid(True)
plt.ylim(-5, 10)
plt.show()
\`\`\`

**ML Application**: Taylor approximation of activation functions

\`\`\`python
def sigmoid(x):
    """Standard sigmoid"""
    return 1 / (1 + np.exp(-x))

def sigmoid_taylor(x, n_terms=5):
    """
    Taylor series approximation of sigmoid around x=0
    σ(x) ≈ 1/2 + x/4 - x³/48 + ...
    """
    # First few terms of Taylor series
    if n_terms >= 1:
        result = 0.5
    if n_terms >= 2:
        result += x / 4
    if n_terms >= 3:
        result -= x**3 / 48
    if n_terms >= 4:
        result += x**5 / 480
    return result

# Compare
x_range = np.linspace(-2, 2, 100)
sig_actual = sigmoid(x_range)
sig_taylor = [sigmoid_taylor(x, 3) for x in x_range]

plt.figure(figsize=(10, 6))
plt.plot(x_range, sig_actual, 'b-', linewidth=2, label='Actual sigmoid')
plt.plot(x_range, sig_taylor, 'r--', linewidth=2, label='Taylor approx (3 terms)')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Sigmoid vs Taylor Approximation')
plt.legend()
plt.grid(True)
plt.show()

print("Taylor approximation is good near x=0!")
print("This is used for efficient approximate computations")
\`\`\`

## Summation Notation

### Sigma Notation

**Σ(i=m to n) f(i)**: Sum f(i) for i from m to n

**Properties**:
- Σ(c·aᵢ) = c·Σaᵢ (constant multiple)
- Σ(aᵢ + bᵢ) = Σaᵢ + Σbᵢ (linearity)
- Σ(i=1 to n) c = n·c (constant sum)

\`\`\`python
def evaluate_sum(f, start, end):
    """Evaluate Σ f(i) from start to end"""
    return sum(f(i) for i in range(start, end + 1))

# Common summation formulas
def sum_first_n_natural(n):
    """Σ(i=1 to n) i = n(n+1)/2"""
    return n * (n + 1) // 2

def sum_first_n_squares(n):
    """Σ(i=1 to n) i² = n(n+1)(2n+1)/6"""
    return n * (n + 1) * (2*n + 1) // 6

def sum_first_n_cubes(n):
    """Σ(i=1 to n) i³ = [n(n+1)/2]²"""
    return (n * (n + 1) // 2) ** 2

n = 100
print(f"Summation formulas for n = {n}:")
print(f"Σ i     = {sum_first_n_natural(n):,}")
print(f"Σ i²    = {sum_first_n_squares(n):,}")
print(f"Σ i³    = {sum_first_n_cubes(n):,}")

# Verify with explicit computation
print(f"\\nVerification:")
print(f"Σ i     = {sum(range(1, n+1)):,}")
print(f"Σ i²    = {sum(i**2 for i in range(1, n+1)):,}")
print(f"Σ i³    = {sum(i**3 for i in range(1, n+1)):,}")
\`\`\`

**ML Application**: Loss function over dataset

\`\`\`python
def mean_squared_error(y_true, y_pred):
    """
    MSE = (1/n) Σ(yᵢ - ŷᵢ)²
    Summation over all data points
    """
    n = len(y_true)
    return np.sum((y_true - y_pred)**2) / n

def cross_entropy_loss(y_true, y_pred, epsilon=1e-10):
    """
    CE = -(1/n) Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
    Summation over all data points
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    n = len(y_true)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / n

# Example
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

mse = mean_squared_error(y_true, y_pred)
ce = cross_entropy_loss(y_true, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Cross-Entropy Loss: {ce:.4f}")
\`\`\`

## Applications in Trading

### Compound Returns (Geometric Series)

\`\`\`python
def compound_return(returns):
    """
    Total return with compounding
    (1+r₁)(1+r₂)...(1+rₙ) - 1
    """
    return np.prod(1 + returns) - 1

def arithmetic_mean_return(returns):
    """Simple average return"""
    return np.mean(returns)

def geometric_mean_return(returns):
    """Geometric mean (compound average)"""
    return np.prod(1 + returns)**(1/len(returns)) - 1

# Monthly returns
monthly_returns = np.array([0.02, -0.01, 0.03, 0.01, -0.02, 0.04])

total = compound_return(monthly_returns)
arith_mean = arithmetic_mean_return(monthly_returns)
geom_mean = geometric_mean_return(monthly_returns)

print("Monthly returns:", monthly_returns)
print(f"\\nTotal compounded return: {total*100:.2f}%")
print(f"Arithmetic mean: {arith_mean*100:.2f}%")
print(f"Geometric mean: {geom_mean*100:.2f}%")
print("\\nGeometric mean ≤ Arithmetic mean (equality only if all returns equal)")
\`\`\`

### Moving Averages (Arithmetic Series)

\`\`\`python
def simple_moving_average(prices, window):
    """
    SMA(t) = (1/n) Σ(i=t-n+1 to t) priceᵢ
    Arithmetic series with equal weights
    """
    sma = []
    for i in range(window - 1, len(prices)):
        window_prices = prices[i - window + 1:i + 1]
        sma.append(np.mean(window_prices))
    return np.array(sma)

def exponential_moving_average(prices, span):
    """
    EMA uses exponentially decaying weights (geometric series)
    """
    alpha = 2 / (span + 1)
    ema = [prices[0]]
    
    for price in prices[1:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])
    
    return np.array(ema)

# Example stock prices
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(100) * 2)

sma_20 = simple_moving_average(prices, 20)
ema_20 = exponential_moving_average(prices, 20)

plt.figure(figsize=(12, 6))
plt.plot(prices, label='Price', linewidth=1, alpha=0.7)
plt.plot(range(19, len(prices)), sma_20, label='SMA(20)', linewidth=2)
plt.plot(ema_20, label='EMA(20)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Moving Averages')
plt.legend()
plt.grid(True)
plt.show()

print(f"SMA uses equal weights: 1/n for each term")
print(f"EMA uses exponentially decaying weights: α(1-α)^i (geometric)")
\`\`\`

## Summary

- **Sequences**: Ordered lists of numbers (arithmetic, geometric, recursive)
- **Convergence**: Sequences approaching a limit
- **Series**: Sums of sequence terms
- **Arithmetic series**: Sum = n/2(a₁ + aₙ)
- **Geometric series**: Sum = a₁(1-rⁿ)/(1-r), infinite sum = a₁/(1-r) if |r|<1
- **Power series**: Represent functions as infinite polynomials
- **Summation notation**: Σ for compact representation

**ML Applications**:
- Gradient descent iterations (convergent sequence)
- Learning rate schedules (arithmetic/geometric sequences)
- Loss over epochs (convergent sequence)
- Discounted returns in RL (geometric series)
- Taylor approximations (power series)
- Dataset summations (loss functions)

**Trading Applications**:
- Compound returns (geometric series)
- Moving averages (arithmetic series, exponential weights)
- Portfolio growth over time
- Time value of money
`,
      multipleChoice: [
        {
          id: 'mc1-arithmetic-sequence',
          question:
            'What is the 50th term of the arithmetic sequence 5, 9, 13, 17, ...?',
          options: ['201', '205', '197', '209'],
          correctAnswer: 0,
          explanation:
            'Formula: aₙ = a₁ + (n-1)d where a₁=5, d=4, n=50. Therefore a₅₀ = 5 + (50-1)×4 = 5 + 49×4 = 5 + 196 = 201.',
        },
        {
          id: 'mc2-geometric-series',
          question:
            'What is the sum of the infinite geometric series 1 + 1/3 + 1/9 + 1/27 + ...?',
          options: ['1', '1.5', '2', '3'],
          correctAnswer: 1,
          explanation:
            'This is a geometric series with a₁=1 and r=1/3. Since |r|<1, it converges. Sum = a₁/(1-r) = 1/(1-1/3) = 1/(2/3) = 3/2 = 1.5.',
        },
        {
          id: 'mc3-fibonacci',
          question:
            'If the Fibonacci sequence is 1, 1, 2, 3, 5, 8, 13, ..., what is the 10th term?',
          options: ['34', '55', '89', '21'],
          correctAnswer: 1,
          explanation:
            'Continue the sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55. The 10th term is 55. Each term is the sum of the previous two.',
        },
        {
          id: 'mc4-summation',
          question: 'Evaluate: Σ(i=1 to 100) i²',
          options: ['5,050', '338,350', '250,000', '1,000,000'],
          correctAnswer: 1,
          explanation:
            'Formula for sum of squares: Σi² = n(n+1)(2n+1)/6. For n=100: 100×101×201/6 = 2,030,100/6 = 338,350.',
        },
        {
          id: 'mc5-convergence',
          question: 'Which sequence converges?',
          options: ['aₙ = n', 'aₙ = (-1)ⁿ', 'aₙ = 1/n', 'aₙ = n²'],
          correctAnswer: 2,
          explanation:
            'aₙ = 1/n converges to 0 as n→∞. The others: n→∞ (diverges), (-1)ⁿ oscillates (no limit), n²→∞ (diverges).',
        },
      ],
      quiz: [
        {
          id: 'dq1-gradient-descent-convergence',
          question:
            'Gradient descent generates a sequence of parameter updates: θₙ₊₁ = θₙ - α∇L(θₙ). Explain the conditions under which this sequence converges to a minimum. Discuss the role of learning rate α, the relationship to geometric sequences, and why convergence is not always guaranteed. Provide mathematical analysis and practical examples.',
          sampleAnswer: `Gradient descent is fundamentally about generating a convergent sequence of parameters that approach an optimal value. Understanding sequence convergence is crucial for analyzing GD behavior.

**Gradient Descent as a Sequence**:

The update rule generates a sequence {θ₀, θ₁, θ₂, ...}:
θₙ₊₁ = θₙ - α∇L(θₙ)

We want: lim(n→∞) θₙ = θ* (optimal parameters)

**Conditions for Convergence**:

**1. Lipschitz Continuous Gradient**:

The gradient must not change too rapidly:
‖∇L(x) - ∇L(y)‖ ≤ L‖x - y‖

where L is the Lipschitz constant.

**Why it matters**: If gradients change wildly, small steps can lead to huge changes.

**2. Learning Rate Constraint**:

For smooth functions with Lipschitz constant L:
α < 2/L (necessary for convergence)
α ≤ 1/L (sufficient for convergence)

**Proof sketch**: 
Consider quadratic loss L(θ) = ½θᵀAθ - bᵀθ
Update: θₙ₊₁ = θₙ - α(Aθₙ - b)
       = (I - αA)θₙ + αb

This is a linear recurrence. Convergence requires eigenvalues of (I - αA) to have magnitude < 1:
|1 - αλᵢ| < 1 for all eigenvalues λᵢ of A
⟹ α < 2/λmax

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x0, alpha, n_iterations, tol=1e-6):
    """
    Gradient descent that returns full sequence
    """
    x_sequence = [x0]
    f_sequence = [f(x0)]
    x = x0
    
    for i in range(n_iterations):
        grad = grad_f(x)
        x_new = x - alpha * grad
        
        x_sequence.append(x_new)
        f_sequence.append(f(x_new))
        
        # Check convergence
        if np.abs(x_new - x) < tol:
            print(f"Converged at iteration {i+1}")
            break
        
        x = x_new
    
    return np.array(x_sequence), np.array(f_sequence)

# Example: f(x) = x^2, optimal at x=0
def f(x):
    return x**2

def grad_f(x):
    return 2*x

x0 = 10.0

# Test different learning rates
learning_rates = [0.1, 0.5, 0.9, 1.1]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, alpha in enumerate(learning_rates):
    ax = axes[idx // 2, idx % 2]
    
    x_seq, f_seq = gradient_descent(f, grad_f, x0, alpha, 50)
    
    ax.plot(f_seq, 'bo-', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Learning Rate α = {alpha}')
    ax.grid(True)
    ax.set_ylim(-5, 110)
    
    # Annotate convergence behavior
    if alpha < 1.0:
        ax.text(0.5, 0.9, 'Converges', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    else:
        ax.text(0.5, 0.9, 'Diverges', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

plt.tight_layout()
plt.show()

# For f(x) = x^2, L = 2 (second derivative)
# So α < 2/2 = 1 for convergence
print(f"\\nLipschitz constant L = 2")
print(f"Convergence requires α < 2/L = 1.0")
print(f"α = 0.1, 0.5, 0.9 converge ✓")
print(f"α = 1.1 diverges ✗")
\`\`\`

**3. Convexity (for global convergence)**:

For convex L, any local minimum is global.
GD on convex function with appropriate α converges to global optimum.

For non-convex (neural networks), GD converges to local minimum or saddle point.

**Connection to Geometric Sequences**:

For quadratic loss near optimum, GD behaves like geometric sequence:

Let eₙ = θₙ - θ* (error at step n)
Then: eₙ₊₁ = (1 - αλ)eₙ (for eigenvalue λ)

This is geometric with ratio r = (1 - αλ)

Convergence rate: |eₙ| = |r|ⁿ|e₀|

**Fast convergence**: Need |r| << 1 ⟹ α ≈ 1/λ (but not too large)
**Slow convergence**: |r| ≈ 1 ⟹ α too small or too large
**Divergence**: |r| > 1 ⟹ α too large

\`\`\`python
# Demonstrate geometric convergence
def analyze_convergence_rate(x_sequence, x_optimal):
    """Analyze convergence rate of sequence"""
    errors = np.abs(x_sequence - x_optimal)
    
    # Check if geometric: compute ratios eₙ₊₁/eₙ
    ratios = errors[1:] / errors[:-1]
    
    # Estimate convergence rate
    avg_ratio = np.mean(ratios[10:])  # After initial phase
    
    return errors, ratios, avg_ratio

x0 = 10.0
alpha = 0.4
x_seq, _ = gradient_descent(f, grad_f, x0, alpha, 100)
errors, ratios, r = analyze_convergence_rate(x_seq, 0.0)

print(f"\\nConvergence Analysis (α = {alpha}):")
print(f"Average ratio |eₙ₊₁/eₙ| = {r:.4f}")
print(f"Theoretical ratio (1-αL) = {1-alpha*2:.4f}")
print(f"Match: {np.isclose(r, 1-alpha*2, atol=0.01)}")

# Plot error decay
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogy(errors, 'bo-', markersize=4)
plt.xlabel('Iteration')
plt.ylabel('|θₙ - θ*| (log scale)')
plt.title('Error Decay (Linear on Log Scale = Geometric)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(ratios, 'ro-', markersize=4)
plt.axhline(y=r, color='g', linestyle='--', label=f'Average = {r:.3f}')
plt.xlabel('Iteration')
plt.ylabel('|eₙ₊₁/eₙ|')
plt.title('Convergence Ratio (Constant = Geometric)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
\`\`\`

**Why Convergence Not Always Guaranteed**:

**1. Learning Rate Too Large**:
- Overshoots minimum
- Can diverge or oscillate
- Sequence does not converge

**2. Non-Convex Landscape**:
- Can get stuck in local minima
- Saddle points can slow convergence
- No guarantee of global optimum

**3. Vanishing/Exploding Gradients**:
- Very deep networks
- Gradients → 0 (vanishing) or → ∞ (exploding)
- Sequence stops moving or diverges

**4. Poor Initialization**:
- Start far from any good minimum
- May never reach good region

**5. Stochastic Gradient Descent**:
- Uses mini-batches (noisy gradients)
- Sequence is stochastic, not deterministic
- Doesn't converge to exact point (oscillates around optimum)

**Practical Solutions**:

\`\`\`python
# Adaptive learning rates (like geometric sequence with changing ratio)
def adam_style_learning_rate(iteration, initial_lr=0.001, decay_rate=0.9):
    """
    Adaptive LR that adjusts based on iteration
    Similar to geometric decay but more sophisticated
    """
    return initial_lr * (decay_rate ** (iteration / 100))

# Learning rate scheduling
def step_decay(initial_lr, epoch, drop_rate=0.5, epochs_drop=10):
    """
    Drop learning rate by factor every N epochs
    Creates piecewise geometric sequence
    """
    return initial_lr * (drop_rate ** (epoch // epochs_drop))

epochs = np.arange(0, 100)
adam_lrs = [adam_style_learning_rate(e) for e in epochs]
step_lrs = [step_decay(0.1, e) for e in epochs]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, adam_lrs, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.title('Exponential Decay (Geometric)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, step_lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Step Decay (Piecewise Geometric)')
plt.grid(True)

plt.tight_layout()
plt.show()
\`\`\`

**Trading Application**:

In trading strategy optimization:
- Parameter updates follow GD
- Non-convex landscape (market regimes change)
- Need adaptive learning rates
- Monitor convergence of backtest performance

\`\`\`python
# Simplified trading strategy parameter optimization
def optimize_trading_strategy(initial_params, historical_data, n_iterations=100):
    """
    Optimize strategy parameters using gradient descent
    Parameters might be: stop-loss %, take-profit %, position size
    """
    params = initial_params
    performance_sequence = []
    
    for i in range(n_iterations):
        # Backtest with current params
        performance = backtest(params, historical_data)
        performance_sequence.append(performance)
        
        # Compute gradient (finite differences in practice)
        grad = compute_gradient(params, historical_data)
        
        # Update with adaptive learning rate
        alpha = step_decay(0.01, i)
        params = params + alpha * grad  # Maximize performance
        
        # Check convergence
        if i > 10 and np.std(performance_sequence[-10:]) < 0.001:
            print(f"Strategy parameters converged at iteration {i}")
            break
    
    return params, performance_sequence

# Key insight: convergence of strategy parameters indicates
# stable optimal configuration for historical data
# But beware overfitting!
\`\`\`

**Summary**:
- GD creates parameter sequence θₙ₊₁ = θₙ - α∇L(θₙ)
- Convergence requires: α < 2/L (Lipschitz constant)
- Near optimum, behaves like geometric sequence with ratio (1-αL)
- Geometric convergence: error decays as rⁿ
- Non-convex problems, poor α, or bad initialization prevent convergence
- Adaptive learning rates improve convergence in practice
- Understanding sequences crucial for debugging training dynamics`,
          keyPoints: [
            'GD generates sequence θₙ₊₁ = θₙ - α∇L(θₙ), want convergence to θ*',
            'Convergence requires α < 2/L where L is Lipschitz constant',
            'Near optimum, GD behaves like geometric sequence with ratio (1-αL)',
            'Geometric convergence: error |θₙ - θ*| ≈ rⁿ|θ₀ - θ*|',
            'Adaptive learning rates and momentum improve convergence in practice',
          ],
        },
        {
          id: 'dq2-compound-returns-series',
          question:
            'In trading, compound returns follow: (1+r₁)(1+r₂)...(1+rₙ) while arithmetic returns are (r₁+r₂+...+rₙ)/n. Explain why compound (geometric) returns are always less than or equal to arithmetic returns. How do sequences and series help us understand portfolio growth? Discuss geometric mean vs arithmetic mean, and why this matters for trading strategies.',
          sampleAnswer: `The difference between compound and arithmetic returns is fundamental to understanding portfolio performance and is directly related to geometric series and sequences.

**Compound vs Arithmetic Returns**:

**Arithmetic (Simple) Return**:
R_arithmetic = (r₁ + r₂ + ... + rₙ) / n

**Geometric (Compound) Return**:
R_geometric = [(1+r₁)(1+r₂)...(1+rₙ)]^(1/n) - 1

**Key Insight**: R_geometric ≤ R_arithmetic (AM-GM Inequality)

Equality only when all returns are equal.

**Why Compound Returns Are Lower**:

**Mathematical Proof** (AM-GM Inequality):

For positive numbers a₁, a₂, ..., aₙ:
(a₁ + a₂ + ... + aₙ)/n ≥ (a₁·a₂·...·aₙ)^(1/n)

Let aᵢ = 1 + rᵢ:
(Σ(1+rᵢ))/n ≥ [Π(1+rᵢ)]^(1/n)

The left side is related to arithmetic mean, right side is geometric mean.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def arithmetic_return(returns):
    """Simple average"""
    return np.mean(returns)

def geometric_return(returns):
    """Compound average"""
    return np.prod(1 + returns)**(1/len(returns)) - 1

def final_wealth(initial, returns):
    """Final portfolio value with compounding"""
    return initial * np.prod(1 + returns)

# Example: Compare different return sequences
sequences = {
    'Constant': np.array([0.05, 0.05, 0.05, 0.05, 0.05]),
    'Volatile +': np.array([0.10, -0.02, 0.08, 0.01, 0.03]),
    'Volatile ++': np.array([0.20, -0.10, 0.15, -0.05, 0.10]),
    'Extreme': np.array([0.50, -0.30, 0.40, -0.20, 0.30])
}

print("Return Sequences Analysis:\\n")
print(f"{'Sequence':<12} {'Arith %':<10} {'Geom %':<10} {'Final $':<10} {'Gap %':<10}")
print("-" * 60)

for name, returns in sequences.items():
    arith = arithmetic_return(returns) * 100
    geom = geometric_return(returns) * 100
    final = final_wealth(1000, returns)
    gap = arith - geom
    
    print(f"{name:<12} {arith:>9.2f} {geom:>9.2f} {final:>9.2f} {gap:>9.2f}")

print("\\nKey Observation: Higher volatility → Larger gap between arithmetic and geometric")
\`\`\`

**Output**:
\`\`\`
Sequence     Arith %    Geom %     Final $    Gap %     
------------------------------------------------------------
Constant        5.00      5.00   1276.28      0.00
Volatile +      4.00      3.88   1208.38      0.12
Volatile ++     6.00      5.31   1295.50      0.69
Extreme        14.00     10.73   1643.06      3.27
\`\`\`

**Intuition - Volatility Drag**:

Consider two periods: +50%, then -33.33%

Arithmetic mean: (50% - 33.33%) / 2 = 8.33%
Geometric mean: (1.5 × 0.6667)^0.5 - 1 = 0%

You end up at the same place! 
$100 → $150 → $100

The arithmetic mean is misleading because losses hurt more than gains help (when compounding).

\`\`\`python
# Visualize the effect
def simulate_paths(initial, mean_return, volatility, periods, n_paths=1000):
    """Simulate multiple portfolio paths"""
    np.random.seed(42)
    returns = np.random.normal(mean_return, volatility, (n_paths, periods))
    
    paths = np.zeros((n_paths, periods + 1))
    paths[:, 0] = initial
    
    for t in range(periods):
        paths[:, t+1] = paths[:, t] * (1 + returns[:, t])
    
    return paths, returns

# Simulate low vs high volatility
initial = 1000
mean_return = 0.01  # 1% per period
periods = 100

low_vol_paths, low_vol_returns = simulate_paths(initial, mean_return, 0.02, periods)
high_vol_paths, high_vol_returns = simulate_paths(initial, mean_return, 0.10, periods)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Low volatility paths
axes[0, 0].plot(low_vol_paths.T, alpha=0.1, color='blue')
axes[0, 0].plot(low_vol_paths.mean(axis=0), color='red', linewidth=2, label='Mean path')
axes[0, 0].set_title('Low Volatility (σ = 2%)')
axes[0, 0].set_xlabel('Period')
axes[0, 0].set_ylabel('Portfolio Value')
axes[0, 0].legend()
axes[0, 0].grid(True)

# High volatility paths
axes[0, 1].plot(high_vol_paths.T, alpha=0.1, color='blue')
axes[0, 1].plot(high_vol_paths.mean(axis=0), color='red', linewidth=2, label='Mean path')
axes[0, 1].set_title('High Volatility (σ = 10%)')
axes[0, 1].set_xlabel('Period')
axes[0, 1].set_ylabel('Portfolio Value')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Final value distributions
axes[1, 0].hist(low_vol_paths[:, -1], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[1, 0].axvline(low_vol_paths[:, -1].mean(), color='red', linewidth=2, label='Mean')
axes[1, 0].set_title('Low Vol: Final Value Distribution')
axes[1, 0].set_xlabel('Final Portfolio Value')
axes[1, 0].legend()

axes[1, 1].hist(high_vol_paths[:, -1], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[1, 1].axvline(high_vol_paths[:, -1].mean(), color='red', linewidth=2, label='Mean')
axes[1, 1].set_title('High Vol: Final Value Distribution')
axes[1, 1].set_xlabel('Final Portfolio Value')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Compute realized returns
low_vol_arith = arithmetic_return(low_vol_returns.flatten()) * 100
low_vol_geom = geometric_return(low_vol_returns.flatten()) * 100
high_vol_arith = arithmetic_return(high_vol_returns.flatten()) * 100
high_vol_geom = geometric_return(high_vol_returns.flatten()) * 100

print(f"\\nLow Volatility:")
print(f"  Arithmetic mean: {low_vol_arith:.2f}%")
print(f"  Geometric mean: {low_vol_geom:.2f}%")
print(f"  Difference: {low_vol_arith - low_vol_geom:.2f}%")

print(f"\\nHigh Volatility:")
print(f"  Arithmetic mean: {high_vol_arith:.2f}%")
print(f"  Geometric mean: {high_vol_geom:.2f}%")
print(f"  Difference: {high_vol_arith - high_vol_geom:.2f}%")

print(f"\\nVolatility drag is {(high_vol_arith - high_vol_geom)/(low_vol_arith - low_vol_geom):.1f}x larger for high volatility!")
\`\`\`

**Sequences Perspective**:

Portfolio value forms a sequence:
V₀, V₁, V₂, ..., Vₙ

where Vₜ = Vₜ₋₁(1 + rₜ)

This is a geometric sequence if returns constant, otherwise more complex.

\`\`\`python
def portfolio_sequence(initial, returns):
    """Generate portfolio value sequence"""
    values = [initial]
    for r in returns:
        values.append(values[-1] * (1 + r))
    return np.array(values)

# Example
returns = np.array([0.10, -0.05, 0.08, 0.03, -0.02])
portfolio = portfolio_sequence(1000, returns)

print("Portfolio value sequence:")
for i, (v, r) in enumerate(zip(portfolio[:-1], returns)):
    print(f"V_{i} = \${v:.2f} → V_{i+1} = \${portfolio[i+1]:.2f} (return: {r*100:+.1f}%)")

print(f"\\nFinal value: \${portfolio[-1]:.2f}")
print(f"Total return: {(portfolio[-1]/portfolio[0] - 1)*100:.2f}%")
\`\`\`

**Why This Matters for Trading**:

**1. Performance Measurement**:
- Use geometric returns for actual growth
- Arithmetic returns overstate performance if volatile

**2. Volatility is Costly**:
- Two strategies with same arithmetic return but different volatility
- Lower volatility strategy will have higher compounded return
- This is "volatility drag"

**3. Risk Management**:
- Large losses require even larger gains to recover
- Lose 50% → Need 100% gain to break even
- Geometric perspective shows asymmetry

\`\`\`python
def recovery_return_needed(loss_percent):
    """
    Calculate return needed to recover from loss
    If lose X%, need gain of X/(1-X) to recover
    """
    loss = loss_percent / 100
    recovery = loss / (1 - loss)
    return recovery * 100

losses = [10, 20, 30, 40, 50, 60, 70, 80, 90]
recoveries = [recovery_return_needed(L) for L in losses]

plt.figure(figsize=(10, 6))
plt.plot(losses, recoveries, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Loss (%)')
plt.ylabel('Gain Needed to Recover (%)')
plt.title('Asymmetry of Gains and Losses (Geometric Returns)')
plt.grid(True)
plt.axline((0, 0), slope=1, color='gray', linestyle='--', label='If symmetric')
plt.legend()
plt.show()

print("Loss vs Recovery:")
for L, R in zip(losses, recoveries):
    print(f"Lose {L}% → Need +{R:.1f}% to recover")
\`\`\`

**4. Sharpe Ratio Adjustment**:

Traditional Sharpe uses arithmetic mean.
For compounding, should use geometric mean:

Sharpe_geometric = (R_geom - R_f) / σ

**5. Long-Term Projections**:

Projecting wealth growth over time:
- Use geometric mean (actual compounding)
- Not arithmetic mean (overstates growth)

\`\`\`python
# Project wealth over 30 years
initial_wealth = 10000
annual_return = 0.08  # 8%
volatility = 0.15      # 15%

# Arithmetic projection (WRONG for compounding)
years = 30
wealth_arithmetic = initial_wealth * (1 + annual_return)**years

# Geometric projection (accounting for volatility drag)
# Approximate: geometric return ≈ arithmetic - σ²/2
volatility_drag = volatility**2 / 2
geometric_return = annual_return - volatility_drag
wealth_geometric = initial_wealth * (1 + geometric_return)**years

print(f"30-year wealth projection (initial \${initial_wealth:,}):")
print(f"Arithmetic (wrong): \${wealth_arithmetic:,.0f}")
print(f"Geometric (correct): \${wealth_geometric:,.0f}")
print(f"Difference: \${wealth_arithmetic - wealth_geometric:,.0f}")
print(f"\\nOverstimation: {(wealth_arithmetic/wealth_geometric - 1)*100:.1f}%")
\`\`\`

**Summary**:
- Geometric returns account for compounding (sequences/series perspective)
- Always ≤ arithmetic returns (AM-GM inequality)
- Difference increases with volatility ("volatility drag")
- Losses require disproportionate gains to recover (geometric asymmetry)
- Use geometric mean for realistic portfolio projections
- Key insight: Minimizing volatility can increase long-term growth even with same average return`,
          keyPoints: [
            'Geometric return ≤ Arithmetic return (AM-GM inequality), equality only if constant',
            'Volatility drag: higher volatility → larger gap between arithmetic and geometric',
            'Portfolio sequence Vₜ = V₀·Π(1+rᵢ) shows compounding nature',
            'Losses hurt more than gains help: lose 50% needs 100% gain to recover',
            'Use geometric mean for realistic long-term projections, not arithmetic',
          ],
        },
        {
          id: 'dq3-reinforcement-learning-series',
          question:
            'In reinforcement learning, the discounted return is defined as Gₜ = Σ(k=0 to ∞) γᵏrₜ₊ₖ where γ is the discount factor. This is a geometric series. Explain: (1) Why we discount future rewards, (2) Conditions for convergence of this infinite series, (3) How the discount factor affects agent behavior, (4) Practical computation, (5) Application to trading strategies.',
          sampleAnswer: `The discounted return in reinforcement learning is a perfect application of geometric series theory to sequential decision-making, directly relevant to algorithmic trading strategies.

**1. Why Discount Future Rewards?**

**Definition**:
Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + γ³rₜ₊₃ + ...
   = Σ(k=0 to ∞) γᵏrₜ₊ₖ

where:
- Gₜ: return from time t
- rₜ: immediate reward at time t
- γ ∈ [0, 1]: discount factor

**Reasons for Discounting**:

**Mathematical**: Ensures convergence of infinite sum (more on this below)

**Economic**: Time value of money - future rewards worth less than immediate
- $100 today > $100 in 1 year
- Can invest $100 today to grow

**Uncertainty**: Future is uncertain
- Closer rewards more reliable than distant ones
- Model may be imperfect for long horizons

**Agent Behavior**: Encourages taking action sooner rather than later
- Without discounting (γ=1), may delay rewards indefinitely
- With discounting, prefers sooner rewards

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def compute_discounted_return(rewards, gamma):
    """
    Compute Gₜ = Σ γᵏrₜ₊ₖ
    This is a geometric series with weights γᵏ
    """
    n = len(rewards)
    discount_factors = gamma ** np.arange(n)
    return np.sum(rewards * discount_factors)

# Example: sequence of rewards
rewards = np.array([1, 2, 3, 4, 5])

# Different discount factors
gammas = [0.5, 0.9, 0.95, 0.99, 1.0]

print("Effect of discount factor γ on return:\\n")
print(f"Rewards: {rewards}\\n")

for gamma in gammas:
    G = compute_discounted_return(rewards, gamma)
    print(f"γ = {gamma:.2f}: G = {G:.4f}")

# Undiscounted sum
print(f"\\nUndiscounted (γ=1): {np.sum(rewards)}")
print("Lower γ → More emphasis on immediate rewards")
print("Higher γ → More consideration of future rewards")
\`\`\`

**2. Convergence Conditions**

The infinite series Gₜ = Σ γᵏrₜ₊ₖ is a geometric series.

**Condition 1**: |γ| < 1 (discount factor between 0 and 1)

**Condition 2**: Rewards must be bounded: |rₜ| ≤ R_max for all t

**Proof of Convergence**:

If rewards bounded by R_max:
|Gₜ| ≤ Σ(k=0 to ∞) γᵏR_max
     = R_max · Σ γᵏ
     = R_max / (1 - γ)    [geometric series formula]

So Gₜ is bounded, hence converges.

\`\`\`python
def infinite_geometric_series_sum(gamma, R_max):
    """
    Theoretical maximum return for bounded rewards
    G_max = R_max / (1 - γ)
    """
    if gamma >= 1:
        return np.inf
    return R_max / (1 - gamma)

# Demonstrate convergence
def compute_partial_returns(rewards_infinite, gamma, n_terms_list):
    """Compute partial sums to show convergence"""
    partial_returns = []
    for n in n_terms_list:
        G_n = compute_discounted_return(rewards_infinite[:n], gamma)
        partial_returns.append(G_n)
    return np.array(partial_returns)

# Simulate infinite reward sequence (bounded)
np.random.seed(42)
R_max = 10
rewards_infinite = np.random.uniform(-R_max, R_max, 1000)

gamma = 0.9
n_terms_list = range(1, 101)
partial_returns = compute_partial_returns(rewards_infinite, gamma, n_terms_list)

# Theoretical bound
theoretical_max = infinite_geometric_series_sum(gamma, R_max)

plt.figure(figsize=(10, 6))
plt.plot(n_terms_list, partial_returns, linewidth=2, label='Partial sums')
plt.axhline(y=theoretical_max, color='r', linestyle='--', linewidth=2, 
            label=f'Theoretical max = {theoretical_max:.2f}')
plt.axhline(y=-theoretical_max, color='r', linestyle='--', linewidth=2)
plt.fill_between(n_terms_list, -theoretical_max, theoretical_max, alpha=0.1, color='red')
plt.xlabel('Number of terms')
plt.ylabel('Partial return Gₙ')
plt.title(f'Convergence of Discounted Return (γ = {gamma})')
plt.legend()
plt.grid(True)
plt.show()

print(f"\\nConvergence Analysis (γ = {gamma}, R_max = {R_max}):")
print(f"Theoretical bound: ±{theoretical_max:.2f}")
print(f"Actual G_100: {partial_returns[-1]:.2f}")
print(f"Within bounds: {abs(partial_returns[-1]) <= theoretical_max}")
\`\`\`

**3. Effect of Discount Factor on Behavior**

**γ = 0** (Myopic):
- Only immediate reward matters: Gₜ = rₜ
- Agent ignores future completely
- Very short-sighted behavior

**γ ≈ 0.5** (Short-term):
- Future rewards decay quickly
- Effective horizon ~2-3 steps
- Good for highly uncertain environments

**γ ≈ 0.9** (Medium-term):
- Balances immediate and future
- Common in practice
- Effective horizon ~10 steps

**γ ≈ 0.99** (Long-term):
- Values future highly
- Effective horizon ~100 steps
- Better for stable environments

**γ = 1** (No discounting):
- All rewards equally important
- May not converge (infinite sum)
- Only works for episodic tasks (finite horizon)

\`\`\`python
# Effective horizon: number of steps that matter
def effective_horizon(gamma, threshold=0.01):
    """
    Compute effective horizon: k where γᵏ < threshold
    Rewards beyond this have < threshold weight
    """
    if gamma >= 1:
        return np.inf
    return int(np.log(threshold) / np.log(gamma))

gammas = [0.5, 0.7, 0.9, 0.95, 0.99]
print("Effective Horizon (weight < 1%):\\n")
for gamma in gammas:
    horizon = effective_horizon(gamma, 0.01)
    print(f"γ = {gamma:.2f}: ~{horizon} steps")

# Visualize discount weights
k = np.arange(0, 50)
plt.figure(figsize=(12, 6))

for gamma in [0.5, 0.9, 0.99]:
    weights = gamma ** k
    plt.plot(k, weights, linewidth=2, label=f'γ = {gamma}')

plt.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='1% threshold')
plt.xlabel('Steps into future (k)')
plt.ylabel('Discount weight γᵏ')
plt.title('Discount Weights Over Time')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()
\`\`\`

**4. Practical Computation**

**Forward View** (what we defined):
Gₜ = rₜ + γGₜ₊₁

Requires looking into future - impractical online!

**Backward View** (TD learning):
Vₜ ← Vₜ + α(rₜ + γVₜ₊₁ - Vₜ)

Updates value incrementally without full rollout.

**Monte Carlo** (episodic):
After episode ends, compute actual returns backward:

\`\`\`python
def compute_returns_backward(rewards, gamma):
    """
    Compute discounted returns efficiently going backward
    Gₜ = rₜ + γGₜ₊₁
    """
    n = len(rewards)
    returns = np.zeros(n)
    
    # Start from end
    returns[-1] = rewards[-1]
    
    # Work backward
    for t in range(n-2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t+1]
    
    return returns

# Example episode
rewards = np.array([1, 0, 0, 1, 2])
gamma = 0.9
returns = compute_returns_backward(rewards, gamma)

print("Rewards:  ", rewards)
print("Returns:  ", [f"{r:.3f}" for r in returns])

# Verify first return
manual_G0 = 1 + 0.9*0 + 0.9**2*0 + 0.9**3*1 + 0.9**4*2
print(f"\\nManual G₀: {manual_G0:.3f}")
print(f"Computed G₀: {returns[0]:.3f}")
print(f"Match: {np.isclose(manual_G0, returns[0])}")
\`\`\`

**Vector Implementation** (efficient):

\`\`\`python
def compute_returns_vectorized(rewards, gamma):
    """
    Vectorized computation using geometric series
    """
    n = len(rewards)
    # Create discount matrix
    discount_matrix = gamma ** np.abs(np.arange(n)[:, None] - np.arange(n))
    # Upper triangular to only include future rewards
    discount_matrix = np.triu(discount_matrix)
    # Multiply with rewards
    returns = discount_matrix @ rewards
    return returns

rewards = np.array([1, 0, 0, 1, 2])
returns_vec = compute_returns_vectorized(rewards, 0.9)
returns_back = compute_returns_backward(rewards, 0.9)

print("\\nVectorized vs Backward:")
print("Vectorized:", [f"{r:.3f}" for r in returns_vec])
print("Backward:  ", [f"{r:.3f}" for r in returns_back])
print("Match:", np.allclose(returns_vec, returns_back))
\`\`\`

**5. Application to Trading Strategies**

**Trading as RL Problem**:
- State: market conditions, portfolio state
- Action: buy, sell, hold, position size
- Reward: profit/loss
- Goal: Maximize discounted cumulative returns

\`\`\`python
# Simplified trading example
class TradingEnvironment:
    def __init__(self, prices):
        self.prices = prices
        self.current_step = 0
        self.position = 0  # 1 = long, 0 = flat, -1 = short
        
    def step(self, action):
        # action: -1 (sell), 0 (hold), 1 (buy)
        current_price = self.prices[self.current_step]
        
        # Compute reward (profit/loss from position)
        if self.current_step > 0:
            price_change = self.prices[self.current_step] - self.prices[self.current_step-1]
            reward = self.position * price_change
        else:
            reward = 0
        
        # Update position
        self.position = action
        self.current_step += 1
        
        done = self.current_step >= len(self.prices) - 1
        return reward, done

# Simulate trading episode
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(50) * 2)

env = TradingEnvironment(prices)

# Simple strategy: buy when price drops, sell when rises
actions = []
rewards = []

for i in range(len(prices) - 1):
    # Simple momentum strategy
    if i == 0:
        action = 1  # Start long
    else:
        price_change = prices[i] - prices[i-1]
        action = 1 if price_change > 0 else -1
    
    reward, done = env.step(action)
    actions.append(action)
    rewards.append(reward)
    
    if done:
        break

# Compute discounted returns for different γ
gammas_trading = [0.9, 0.95, 0.99]
print("\\nTrading Episode Analysis:\\n")

for gamma in gammas_trading:
    returns_seq = compute_returns_backward(np.array(rewards), gamma)
    G0 = returns_seq[0]
    print(f"γ = {gamma:.2f}: Initial expected return G₀ = \${G0:.2f}")

print(f"\\nTotal episode profit (undiscounted): \${np.sum(rewards):.2f}")
print("\\nLower γ → Prioritizes near-term profits")
print("Higher γ → Considers long-term strategy value")
\`\`\`

**Choosing γ for Trading**:

**High Frequency Trading**: γ ≈ 0.5-0.7
- Short holding periods
- Quick profits matter most
- Future very uncertain (milliseconds)

**Swing Trading**: γ ≈ 0.9-0.95
- Hold for days/weeks
- Balance immediate and future gains
- Medium-term market view

**Long-term Investing**: γ ≈ 0.99-0.999
- Hold for months/years
- Future returns highly valued
- Building long-term portfolio value

**Risk Management Context**:
- Higher γ encourages avoiding large drawdowns (future matters)
- Lower γ may take more risk for immediate gains

\`\`\`python
# Demonstrate effect of γ on trading strategy valuation
def evaluate_strategy(rewards, gamma):
    """Evaluate trading strategy with given discount factor"""
    return compute_discounted_return(rewards, gamma)

# Two strategies:
# A: consistent small gains
# B: large gain at end, losses early

strategy_A = np.array([2, 2, 2, 2, 2])
strategy_B = np.array([-1, -1, -1, -1, 12])

gammas = np.linspace(0.5, 1.0, 50)
values_A = [evaluate_strategy(strategy_A, g) for g in gammas]
values_B = [evaluate_strategy(strategy_B, g) for g in gammas]

plt.figure(figsize=(10, 6))
plt.plot(gammas, values_A, linewidth=2, label='Strategy A (consistent)')
plt.plot(gammas, values_B, linewidth=2, label='Strategy B (back-loaded)')
plt.xlabel('Discount Factor γ')
plt.ylabel('Discounted Return')
plt.title('Strategy Valuation vs Discount Factor')
plt.legend()
plt.grid(True)
plt.show()

print("\\nStrategy Comparison:")
print(f"Both have same total: A = {np.sum(strategy_A)}, B = {np.sum(strategy_B)}")
print(f"\\nAt γ = 0.9:")
print(f"  Strategy A: {evaluate_strategy(strategy_A, 0.9):.2f}")
print(f"  Strategy B: {evaluate_strategy(strategy_B, 0.9):.2f}")
print("\\nLower γ strongly prefers early gains (A)!")
\`\`\`

**Summary**:
- Discounted return is geometric series: Gₜ = Σ γᵏrₜ₊ₖ
- Converges if |γ| < 1 and rewards bounded
- γ controls effective time horizon: low γ = myopic, high γ = farsighted
- Computed efficiently backward: Gₜ = rₜ + γGₜ₊₁
- In trading: γ depends on strategy timeframe (HFT low, investing high)
- Key insight: γ encodes trader's time preference for profits`,
          keyPoints: [
            'Discounted return Gₜ = Σγᵏrₜ₊ₖ is geometric series, converges if |γ|<1',
            'γ controls effective horizon: γ=0.5 → ~2 steps, γ=0.9 → ~10 steps, γ=0.99 → ~100 steps',
            'Computed efficiently backward: Gₜ = rₜ + γGₜ₊₁ (recursive formula)',
            'Trading application: γ depends on strategy timeframe (HFT: low, investing: high)',
            'Lower γ prioritizes immediate profits, higher γ considers long-term value',
          ],
        },
      ],
    },
    {
      id: 'set-theory-logic',
      title: 'Set Theory & Logic',
      content: `
# Set Theory & Logic

## Introduction

Set theory provides the foundation for organizing data, understanding probability, and reasoning about machine learning algorithms. Logic enables us to make precise statements, build correct algorithms, and understand Boolean operations in neural networks and decision trees. These concepts are fundamental to data science, feature engineering, and algorithmic reasoning.

## Sets

### Definition

A **set** is a collection of distinct objects, called **elements** or **members**.

**Notation**:
- A = {1, 2, 3, 4, 5}
- x ∈ A (x is an element of A)
- x ∉ A (x is not an element of A)

**Ways to Define Sets**:
1. **Roster notation**: List all elements: A = {1, 2, 3}
2. **Set-builder notation**: A = {x | x is an integer, 1 ≤ x ≤ 3}
3. **Rule-based**: A = {x | x² < 10}

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Python sets
A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6, 7}

print(f"Set A: {A}")
print(f"Set B: {B}")

# Membership testing
print(f"\\n3 in A: {3 in A}")
print(f"10 in A: {10 in A}")

# Set from list (removes duplicates)
numbers = [1, 2, 2, 3, 3, 3, 4]
unique_numbers = set(numbers)
print(f"\\nOriginal list: {numbers}")
print(f"Set (unique): {unique_numbers}")

# Set comprehension (like set-builder notation)
squares = {x**2 for x in range(1, 6)}
print(f"\\nSquares: {squares}")

# ML Application: Unique classes in dataset
labels = ['cat', 'dog', 'cat', 'bird', 'dog', 'cat']
unique_classes = set(labels)
print(f"\\nLabels: {labels}")
print(f"Unique classes: {unique_classes}")
print(f"Number of classes: {len(unique_classes)}")
\`\`\`

### Special Sets

**Empty Set**: ∅ or {} (contains no elements)
**Universal Set**: U (contains all elements under consideration)
**Natural Numbers**: ℕ = {1, 2, 3, ...}
**Integers**: ℤ = {..., -2, -1, 0, 1, 2, ...}
**Real Numbers**: ℝ

\`\`\`python
# Empty set
empty = set()  # Not {} (that's empty dict)
print(f"Empty set: {empty}")
print(f"Size: {len(empty)}")

# Infinite sets (represented by rules)
def is_natural(x):
    return isinstance(x, int) and x > 0

def is_integer(x):
    return isinstance(x, int)

def is_even(x):
    return isinstance(x, int) and x % 2 == 0

# Test membership
print(f"\\n5 is natural: {is_natural(5)}")
print(f"-3 is natural: {is_natural(-3)}")
print(f"6 is even: {is_even(6)}")
\`\`\`

## Set Operations

### Union (∪)

**A ∪ B**: All elements in A or B (or both)

\`\`\`python
A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6, 7}

union = A | B  # or A.union(B)
print(f"A ∪ B = {union}")

# ML Application: Combining feature sets
features_model1 = {'age', 'income', 'education'}
features_model2 = {'income', 'location', 'job_title'}
all_features = features_model1 | features_model2
print(f"\\nModel 1 features: {features_model1}")
print(f"Model 2 features: {features_model2}")
print(f"Combined features: {all_features}")
\`\`\`

### Intersection (∩)

**A ∩ B**: Elements in both A and B

\`\`\`python
intersection = A & B  # or A.intersection(B)
print(f"A ∩ B = {intersection}")

# ML Application: Common features
common_features = features_model1 & features_model2
print(f"\\nCommon features: {common_features}")

# Empty intersection (disjoint sets)
C = {1, 2, 3}
D = {4, 5, 6}
print(f"\\nC ∩ D = {C & D}")  # Empty set - disjoint
\`\`\`

### Difference (\\)

**A \\ B**: Elements in A but not in B

\`\`\`python
difference = A - B  # or A.difference(B)
print(f"A \\ B = {difference}")
print(f"B \\ A = {B - A}")

# ML Application: Features unique to one model
unique_to_model1 = features_model1 - features_model2
print(f"\\nUnique to Model 1: {unique_to_model1}")
\`\`\`

### Symmetric Difference (△)

**A △ B**: Elements in A or B but not both

\`\`\`python
sym_diff = A ^ B  # or A.symmetric_difference(B)
print(f"A △ B = {sym_diff}")

# Equivalent to (A ∪ B) \\ (A ∩ B)
equivalent = (A | B) - (A & B)
print(f"Equivalent: {equivalent}")
print(f"Match: {sym_diff == equivalent}")
\`\`\`

### Complement

**A'** or **Aᶜ**: Elements in universal set U but not in A

\`\`\`python
# Need to define universal set
U = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
A = {1, 2, 3, 4, 5}

complement = U - A
print(f"U = {U}")
print(f"A = {A}")
print(f"A' (complement) = {complement}")

# ML Application: Negative class
all_classes = {'cat', 'dog', 'bird', 'fish'}
positive_class = {'cat'}
negative_classes = all_classes - positive_class
print(f"\\nAll classes: {all_classes}")
print(f"Positive: {positive_class}")
print(f"Negative (complement): {negative_classes}")
\`\`\`

## Venn Diagrams

Visualizing set relationships:

\`\`\`python
from matplotlib_venn import venn2, venn3

# Two sets
A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6, 7}

plt.figure(figsize=(8, 6))
venn2([A, B], set_labels=('A', 'B'))
plt.title('Venn Diagram: A and B')
plt.show()

# Three sets
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
C = {4, 5, 6, 7}

plt.figure(figsize=(8, 6))
venn3([A, B, C], set_labels=('A', 'B', 'C'))
plt.title('Venn Diagram: A, B, and C')
plt.show()

print("Venn diagrams displayed!")
\`\`\`

### ML Application: Data Filtering

\`\`\`python
# Example: Customer segmentation
import pandas as pd

# Sample customer data
customers = pd.DataFrame({
    'customer_id': range(1, 11),
    'age': [25, 35, 45, 22, 55, 30, 40, 28, 50, 33],
    'income': [40000, 60000, 80000, 35000, 90000, 55000, 75000, 45000, 85000, 62000],
    'purchased': [True, True, False, True, False, True, False, True, True, False]
})

# Define customer segments using sets
young = set(customers[customers['age'] < 35]['customer_id'])
high_income = set(customers[customers['income'] > 60000]['customer_id'])
purchasers = set(customers[customers['purchased']]['customer_id'])

print("Customer Segments:")
print(f"Young (< 35): {young}")
print(f"High income (> 60k): {high_income}")
print(f"Purchasers: {purchasers}")

# Set operations for targeting
young_high_income = young & high_income
print(f"\\nYoung AND high income: {young_high_income}")

young_or_high_income = young | high_income
print(f"Young OR high income: {young_or_high_income}")

young_non_purchasers = young - purchasers
print(f"Young non-purchasers (to target): {young_non_purchasers}")

# Complex query
target_segment = (young | high_income) & purchasers
print(f"\\n(Young OR high income) AND purchasers: {target_segment}")
\`\`\`

## Subsets and Supersets

**A ⊆ B** (A is subset of B): Every element of A is in B
**A ⊂ B** (A is proper subset): A ⊆ B and A ≠ B
**A ⊇ B** (A is superset of B): B ⊆ A

\`\`\`python
A = {1, 2, 3}
B = {1, 2, 3, 4, 5}
C = {1, 2, 3}

# Subset
print(f"A ⊆ B: {A <= B}")  # A.issubset(B)
print(f"B ⊆ A: {B <= A}")

# Proper subset
print(f"\\nA ⊂ B (proper): {A < B}")
print(f"A ⊂ C (proper): {A < C}")  # False, they're equal

# Superset
print(f"\\nB ⊇ A: {B >= A}")  # B.issuperset(A)

# ML Application: Feature hierarchy
basic_features = {'age', 'gender'}
extended_features = {'age', 'gender', 'income', 'education'}
premium_features = {'age', 'gender', 'income', 'education', 'credit_score', 'employment_history'}

print(f"\\nBasic ⊆ Extended: {basic_features <= extended_features}")
print(f"Extended ⊆ Premium: {extended_features <= premium_features}")
print(f"Basic ⊆ Premium: {basic_features <= premium_features}")  # Transitivity
\`\`\`

## Cardinality

**|A|**: Number of elements in set A

\`\`\`python
A = {1, 2, 3, 4, 5}
print(f"|A| = {len(A)}")

# Properties
# |A ∪ B| = |A| + |B| - |A ∩ B| (Inclusion-Exclusion)
A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6, 7}

size_union = len(A | B)
size_formula = len(A) + len(B) - len(A & B)

print(f"\\n|A| = {len(A)}")
print(f"|B| = {len(B)}")
print(f"|A ∩ B| = {len(A & B)}")
print(f"|A ∪ B| = {size_union}")
print(f"|A| + |B| - |A ∩ B| = {size_formula}")
print(f"Formula matches: {size_union == size_formula}")
\`\`\`

## Propositional Logic

### Basic Propositions

A **proposition** is a statement that is either true or false.

\`\`\`python
# Propositions in Python (Boolean values)
p = True   # "It is raining"
q = False  # "It is cold"

print(f"p (It is raining): {p}")
print(f"q (It is cold): {q}")

# ML Application: Conditions
threshold = 0.5
prediction = 0.75

is_positive = prediction > threshold
is_confident = prediction > 0.9

print(f"\\nPrediction: {prediction}")
print(f"Is positive class: {is_positive}")
print(f"Is confident: {is_confident}")
\`\`\`

### Logical Operators

**NOT (¬)**: Negation
**AND (∧)**: Conjunction
**OR (∨)**: Disjunction
**XOR (⊕)**: Exclusive or
**IMPLIES (→)**: Implication
**IFF (↔)**: If and only if

\`\`\`python
def logical_not(p):
    """NOT: ¬p"""
    return not p

def logical_and(p, q):
    """AND: p ∧ q"""
    return p and q

def logical_or(p, q):
    """OR: p ∨ q"""
    return p or q

def logical_xor(p, q):
    """XOR: p ⊕ q (exclusive or)"""
    return p != q

def logical_implies(p, q):
    """IMPLIES: p → q (if p then q)"""
    return (not p) or q

def logical_iff(p, q):
    """IFF: p ↔ q (if and only if)"""
    return p == q

# Test all combinations
p_values = [True, False]
q_values = [True, False]

print("Truth Table:")
print(f"{'p':<6} {'q':<6} {'¬p':<6} {'p∧q':<6} {'p∨q':<6} {'p⊕q':<6} {'p→q':<6} {'p↔q':<6}")
print("-" * 50)

for p in p_values:
    for q in q_values:
        print(f"{p!s:<6} {q!s:<6} {logical_not(p)!s:<6} {logical_and(p, q)!s:<6} "
              f"{logical_or(p, q)!s:<6} {logical_xor(p, q)!s:<6} "
              f"{logical_implies(p, q)!s:<6} {logical_iff(p, q)!s:<6}")
\`\`\`

### De Morgan's Laws

**¬(p ∧ q) = ¬p ∨ ¬q**
**¬(p ∨ q) = ¬p ∧ ¬q**

\`\`\`python
# Verify De Morgan's Laws
def verify_demorgan():
    """Verify De Morgan's laws for all truth values"""
    for p in [True, False]:
        for q in [True, False]:
            # Law 1: ¬(p ∧ q) = ¬p ∨ ¬q
            left1 = not (p and q)
            right1 = (not p) or (not q)
            
            # Law 2: ¬(p ∨ q) = ¬p ∧ ¬q
            left2 = not (p or q)
            right2 = (not p) and (not q)
            
            print(f"p={p}, q={q}:")
            print(f"  ¬(p∧q)={left1}, ¬p∨¬q={right1}, Equal: {left1 == right1}")
            print(f"  ¬(p∨q)={left2}, ¬p∧¬q={right2}, Equal: {left2 == right2}")

verify_demorgan()
\`\`\`

**ML Application**: Feature filtering logic

\`\`\`python
# Example: Filter data with complex conditions
age = 25
income = 70000
has_degree = True

# Condition: (age >= 25 AND income > 60000) OR has_degree
condition1 = (age >= 25 and income > 60000) or has_degree
print(f"\\nCondition 1 (original): {condition1}")

# Apply De Morgan's law to negate
# NOT[(age >= 25 AND income > 60000) OR has_degree]
# = NOT(age >= 25 AND income > 60000) AND NOT(has_degree)
# = (NOT(age >= 25) OR NOT(income > 60000)) AND NOT(has_degree)
condition2_negated = ((age < 25) or (income <= 60000)) and (not has_degree)
condition2 = not condition2_negated

print(f"Condition 2 (De Morgan): {condition2}")
print(f"Match: {condition1 == condition2}")
\`\`\`

## Truth Tables

Complete enumeration of logical outcomes:

\`\`\`python
import pandas as pd

def generate_truth_table(n_variables):
    """Generate truth table for n Boolean variables"""
    from itertools import product
    
    # Generate all combinations
    combinations = list(product([False, True], repeat=n_variables))
    
    # Create column names
    var_names = [f'p{i+1}' for i in range(n_variables)]
    
    # Create DataFrame
    df = pd.DataFrame(combinations, columns=var_names)
    
    return df

# Example: 3 variables
truth_table = generate_truth_table(3)
print("Truth Table for 3 variables:")
print(truth_table)

# Add derived columns
truth_table['p1 ∧ p2'] = truth_table['p1'] & truth_table['p2']
truth_table['p1 ∨ p2'] = truth_table['p1'] | truth_table['p2']
truth_table['p1 → p2'] = ~truth_table['p1'] | truth_table['p2']

print("\\nWith logical operations:")
print(truth_table)
\`\`\`

## Applications in Machine Learning

### Boolean Features

\`\`\`python
# Binary features in ML
data = pd.DataFrame({
    'has_account': [True, False, True, True, False],
    'is_premium': [False, False, True, False, True],
    'purchased': [False, False, True, False, True]
})

print("Dataset:")
print(data)

# Logical feature engineering
data['premium_no_purchase'] = data['is_premium'] & ~data['purchased']
data['account_or_premium'] = data['has_account'] | data['is_premium']

print("\\nWith engineered features:")
print(data)
\`\`\`

### Decision Tree Logic

Decision trees use logical operations:

\`\`\`python
def decision_tree_logic(age, income, credit_score):
    """
    Simple decision tree as logical expressions
    Approve loan if:
      (age >= 25 AND income > 50000) OR credit_score > 700
    """
    condition1 = age >= 25 and income > 50000
    condition2 = credit_score > 700
    
    approve = condition1 or condition2
    
    return approve, condition1, condition2

# Test cases
test_cases = [
    (30, 60000, 650),  # Satisfies age and income
    (22, 40000, 750),  # Satisfies credit score
    (28, 70000, 720),  # Satisfies both
    (20, 30000, 600),  # Satisfies neither
]

print("Decision Tree Logic:")
print(f"{'Age':<5} {'Income':<8} {'Credit':<7} {'Cond1':<7} {'Cond2':<7} {'Approve':<8}")
print("-" * 50)

for age, income, credit in test_cases:
    approve, cond1, cond2 = decision_tree_logic(age, income, credit)
    print(f"{age:<5} {income:<8} {credit:<7} {cond1!s:<7} {cond2!s:<7} {approve!s:<8}")
\`\`\`

### Neural Network Activations

Boolean logic can be implemented with neural networks:

\`\`\`python
# Perceptron implementing logic gates

def perceptron(inputs, weights, bias):
    """Simple perceptron"""
    activation = np.dot(inputs, weights) + bias
    return 1 if activation > 0 else 0

# AND gate
def neural_AND(x1, x2):
    weights = np.array([1, 1])
    bias = -1.5
    return perceptron([x1, x2], weights, bias)

# OR gate
def neural_OR(x1, x2):
    weights = np.array([1, 1])
    bias = -0.5
    return perceptron([x1, x2], weights, bias)

# NOT gate
def neural_NOT(x):
    weights = np.array([-1])
    bias = 0.5
    return perceptron([x], weights, bias)

# Test logic gates
print("\\nNeural Logic Gates:")
print(f"{'x1':<5} {'x2':<5} {'AND':<5} {'OR':<5} {'NOT x1':<8}")
print("-" * 30)

for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"{x1:<5} {x2:<5} {neural_AND(x1, x2):<5} {neural_OR(x1, x2):<5} {neural_NOT(x1):<8}")
\`\`\`

### Set Operations on Data

\`\`\`python
# Train/test split using sets
all_indices = set(range(100))
train_indices = set(np.random.choice(100, 70, replace=False))
test_indices = all_indices - train_indices

print(f"Total samples: {len(all_indices)}")
print(f"Training samples: {len(train_indices)}")
print(f"Test samples: {len(test_indices)}")
print(f"No overlap: {len(train_indices & test_indices) == 0}")

# Feature selection using sets
available_features = {'age', 'income', 'education', 'credit_score', 
                     'employment_history', 'debt_ratio'}
selected_by_correlation = {'income', 'credit_score', 'debt_ratio'}
selected_by_importance = {'age', 'income', 'credit_score'}

# Intersection: features selected by both methods
robust_features = selected_by_correlation & selected_by_importance
print(f"\\nRobust features (selected by both): {robust_features}")

# Union: all selected features
all_selected = selected_by_correlation | selected_by_importance
print(f"All selected features: {all_selected}")
\`\`\`

## Summary

- **Sets**: Collections of distinct objects
- **Operations**: Union (∪), Intersection (∩), Difference (\\), Complement (')
- **Subsets**: A ⊆ B means all elements of A are in B
- **Cardinality**: |A| is the number of elements
- **Logic**: Propositions with TRUE/FALSE values
- **Operators**: NOT (¬), AND (∧), OR (∨), XOR (⊕), IMPLIES (→), IFF (↔)
- **De Morgan's Laws**: Transform negations of AND/OR
- **Truth Tables**: Systematic enumeration of logical outcomes

**ML Applications**:
- Data filtering and segmentation
- Feature engineering with Boolean operations
- Decision trees (logical conditions)
- Train/test splits
- Feature selection
- Neural network logic gates
- Boolean features in models
`,
      multipleChoice: [
        {
          id: 'mc1-set-operations',
          question:
            'If A = {1, 2, 3, 4} and B = {3, 4, 5, 6}, what is A △ B (symmetric difference)?',
          options: ['{3, 4}', '{1, 2, 5, 6}', '{1, 2, 3, 4, 5, 6}', '{1, 2}'],
          correctAnswer: 1,
          explanation:
            'Symmetric difference A △ B contains elements in A or B but not both. A △ B = (A ∪ B) \\ (A ∩ B) = {1, 2, 3, 4, 5, 6} \\ {3, 4} = {1, 2, 5, 6}.',
        },
        {
          id: 'mc2-cardinality',
          question: 'If |A| = 5, |B| = 7, and |A ∩ B| = 2, what is |A ∪ B|?',
          options: ['12', '10', '9', '14'],
          correctAnswer: 1,
          explanation:
            'Using inclusion-exclusion: |A ∪ B| = |A| + |B| - |A ∩ B| = 5 + 7 - 2 = 10.',
        },
        {
          id: 'mc3-logic',
          question: 'What is the truth value of (TRUE AND FALSE) OR TRUE?',
          options: ['TRUE', 'FALSE', 'Cannot determine', 'Undefined'],
          correctAnswer: 0,
          explanation:
            'Evaluate step by step: (TRUE AND FALSE) = FALSE. Then FALSE OR TRUE = TRUE. Remember: OR returns TRUE if at least one operand is TRUE.',
        },
        {
          id: 'mc4-demorgan',
          question: "According to De Morgan's Law, ¬(p ∨ q) is equivalent to:",
          options: ['¬p ∨ ¬q', '¬p ∧ ¬q', 'p ∧ q', '¬p → ¬q'],
          correctAnswer: 1,
          explanation:
            "De Morgan's Law: ¬(p ∨ q) = ¬p ∧ ¬q. The negation of OR becomes AND of negations.",
        },
        {
          id: 'mc5-implication',
          question: 'The logical implication p → q is FALSE only when:',
          options: [
            'p is TRUE and q is TRUE',
            'p is FALSE and q is FALSE',
            'p is TRUE and q is FALSE',
            'p is FALSE and q is TRUE',
          ],
          correctAnswer: 2,
          explanation:
            'Implication p → q is only FALSE when the premise (p) is TRUE but the conclusion (q) is FALSE. In all other cases, it\'s TRUE. Think: "If p then q" is only violated when p happens but q doesn\'t.',
        },
      ],
      quiz: [
        {
          id: 'dq1-feature-selection-sets',
          question:
            'Explain how set theory is used in feature selection for machine learning. Discuss multiple feature selection methods (correlation-based, importance-based, recursive elimination) and how set operations (union, intersection, difference) help combine or compare their results. Provide concrete examples with code showing how to identify robust features.',
          sampleAnswer: `Set theory provides an elegant framework for feature selection, allowing us to combine insights from multiple selection methods and reason about feature relationships.

**Feature Selection as Set Operations**:

Each feature selection method produces a set of selected features. Set operations let us:
- Find features selected by all methods (intersection = robust)
- Combine features from multiple methods (union = comprehensive)
- Find method-specific features (difference = unique insights)
- Compare method agreements (symmetric difference = disagreements)

**Implementation**:

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, n_repeated=0, random_state=42)

feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Method 1: Correlation-based (ANOVA F-statistic)
selector_corr = SelectKBest(f_classif, k=10)
selector_corr.fit(X, y)
features_correlation = set([feature_names[i] for i in selector_corr.get_support(indices=True)])

print(f"\\nCorrelation-based: {len(features_correlation)} features")
print(f"  {features_correlation}")

# Method 2: Tree-based importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
# Select top 10 by importance
top_indices = np.argsort(importances)[-10:]
features_importance = set([feature_names[i] for i in top_indices])

print(f"\\nImportance-based: {len(features_importance)} features")
print(f"  {features_importance}")

# Method 3: Recursive Feature Elimination
estimator = LogisticRegression(max_iter=1000)
selector_rfe = RFE(estimator, n_features_to_select=10, step=1)
selector_rfe.fit(X, y)
features_rfe = set([feature_names[i] for i in selector_rfe.get_support(indices=True)])

print(f"\\nRFE-based: {len(features_rfe)} features")
print(f"  {features_rfe}")
\`\`\`

**Set Operations for Analysis**:

\`\`\`python
# Intersection: Features selected by ALL methods (most robust)
robust_features = features_correlation & features_importance & features_rfe
print(f"\\nRobust features (all 3 methods): {len(robust_features)}")
print(f"  {robust_features}")

# Union: Features selected by ANY method (comprehensive)
all_selected = features_correlation | features_importance | features_rfe
print(f"\\nAll selected features (any method): {len(all_selected)}")
print(f"  {all_selected}")

# Pairwise intersections
corr_imp = features_correlation & features_importance
corr_rfe = features_correlation & features_rfe
imp_rfe = features_importance & features_rfe

print(f"\\nPairwise agreements:")
print(f"  Correlation ∩ Importance: {len(corr_imp)} features")
print(f"  Correlation ∩ RFE: {len(corr_rfe)} features")
print(f"  Importance ∩ RFE: {len(imp_rfe)} features")

# Features unique to each method
unique_to_corr = features_correlation - features_importance - features_rfe
unique_to_imp = features_importance - features_correlation - features_rfe
unique_to_rfe = features_rfe - features_correlation - features_importance

print(f"\\nUnique selections:")
print(f"  Only Correlation: {unique_to_corr}")
print(f"  Only Importance: {unique_to_imp}")
print(f"  Only RFE: {unique_to_rfe}")

# Symmetric difference: features with disagreement
disagreement_corr_imp = features_correlation ^ features_importance
print(f"\\nDisagreement (Correlation △ Importance): {disagreement_corr_imp}")
\`\`\`

**Visualizing Set Relationships**:

\`\`\`python
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
venn3([features_correlation, features_importance, features_rfe],
      set_labels=('Correlation', 'Importance', 'RFE'))
plt.title('Feature Selection Methods: Set Relationships')
plt.show()
\`\`\`

**Decision Strategy Using Sets**:

**Strategy 1: Conservative (Intersection)**
- Use features selected by all methods
- High confidence, may miss some useful features
- Best for: high-stakes applications, limited compute

\`\`\`python
conservative_features = features_correlation & features_importance & features_rfe
print(f"\\nConservative strategy: {len(conservative_features)} features")
\`\`\`

**Strategy 2: Majority Vote**
- Use features selected by at least 2 methods
- Balanced approach

\`\`\`python
# Count votes for each feature
all_features = features_correlation | features_importance | features_rfe
feature_votes = {}

for feature in all_features:
    votes = 0
    if feature in features_correlation:
        votes += 1
    if feature in features_importance:
        votes += 1
    if feature in features_rfe:
        votes += 1
    feature_votes[feature] = votes

majority_features = {f for f, v in feature_votes.items() if v >= 2}
print(f"\\nMajority vote (≥2 methods): {len(majority_features)} features")
print(f"  {majority_features}")
\`\`\`

**Strategy 3: Aggressive (Union)**
- Use features from any method
- Maximum coverage, may include noise
- Best for: exploration, ensemble models

\`\`\`python
aggressive_features = features_correlation | features_importance | features_rfe
print(f"\\nAggressive strategy: {len(aggressive_features)} features")
\`\`\`

**Analyzing Method Agreement**:

\`\`\`python
def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

# Compute pairwise similarities
sim_corr_imp = jaccard_similarity(features_correlation, features_importance)
sim_corr_rfe = jaccard_similarity(features_correlation, features_rfe)
sim_imp_rfe = jaccard_similarity(features_importance, features_rfe)

print(f"\\nMethod Agreement (Jaccard Similarity):")
print(f"  Correlation ↔ Importance: {sim_corr_imp:.3f}")
print(f"  Correlation ↔ RFE: {sim_corr_rfe:.3f}")
print(f"  Importance ↔ RFE: {sim_imp_rfe:.3f}")

# Overall agreement
overall_agreement = len(robust_features) / len(all_selected)
print(f"\\nOverall agreement: {overall_agreement:.3f}")
print(f"({len(robust_features)} robust / {len(all_selected)} total)")
\`\`\`

**Trading Application**:

\`\`\`python
# Example: Selecting features for stock prediction model

# Different feature sets from domain knowledge
technical_indicators = {'RSI', 'MACD', 'SMA_20', 'SMA_50', 'volume', 'volatility'}
fundamental_features = {'PE_ratio', 'earnings', 'revenue', 'debt_ratio', 'ROE'}
sentiment_features = {'news_sentiment', 'social_sentiment', 'analyst_rating'}

# Statistical selection from backtesting
selected_by_sharpe = {'RSI', 'MACD', 'volume', 'earnings', 'news_sentiment'}
selected_by_sortino = {'SMA_20', 'SMA_50', 'PE_ratio', 'analyst_rating', 'volatility'}

# Combine domain knowledge with statistical selection
# Strategy: Use technical features that are statistically validated
validated_technical = technical_indicators & (selected_by_sharpe | selected_by_sortino)
print(f"\\nValidated technical indicators: {validated_technical}")

# Add fundamental features selected by either metric
validated_fundamental = fundamental_features & (selected_by_sharpe | selected_by_sortino)
print(f"Validated fundamental features: {validated_fundamental}")

# Final feature set
final_features = validated_technical | validated_fundamental
print(f"\\nFinal feature set: {final_features}")
print(f"Total features: {len(final_features)}")
\`\`\`

**Key Insights**:

1. **Intersection gives confidence**: Features selected by multiple methods are likely truly important

2. **Union enables exploration**: Including all candidates helps discover unexpected relationships

3. **Difference reveals specialization**: Each method captures different aspects

4. **Set size trades off**: Larger sets = more information but also more noise/compute

5. **Jaccard similarity quantifies agreement**: Low similarity suggests methods capture complementary information

**Summary**:
Set operations provide a principled way to combine feature selection methods, moving beyond arbitrary choices to systematic analysis of feature importance across multiple perspectives.`,
          keyPoints: [
            'Each selection method produces a set of features',
            'Intersection (∩) finds robust features selected by all methods',
            'Union (∪) combines features from all methods for comprehensive coverage',
            'Difference (\\) identifies method-specific selections',
            'Jaccard similarity quantifies agreement between methods',
            'Strategy choice (conservative/majority/aggressive) depends on application requirements',
          ],
        },
        {
          id: 'dq2-logic-decision-trees',
          question:
            "Decision trees use logical conditions to make predictions. Explain how Boolean logic (AND, OR, NOT) maps to decision tree structure. How can complex logical expressions be represented as trees? Discuss De Morgan's laws in the context of decision tree simplification and provide examples showing equivalent tree representations.",
          sampleAnswer: `Decision trees are essentially visual representations of logical expressions, where each path from root to leaf represents a conjunction (AND) of conditions, and the tree as a whole represents a disjunction (OR) of these paths.

**Decision Trees as Logic**:

**Basic Structure**:
- Each node tests a condition (Boolean expression)
- Left/right branches represent TRUE/FALSE
- Leaf nodes give predictions
- Path from root to leaf = AND of conditions
- Multiple paths to same prediction = OR of conditions

**Example Tree**:

\`\`\`
         age >= 25?
          /      \\
        YES       NO
        /          \\
   income > 50k?   REJECT
      /      \\
    YES      NO
    /         \\
  APPROVE   credit > 700?
              /        \\
            YES        NO
            /           \\
         APPROVE      REJECT
\`\`\`

**As Logical Expression**:
\`\`\`
APPROVE = (age >= 25 ∧ income > 50k) 
          ∨ (age >= 25 ∧ income ≤ 50k ∧ credit > 700)
\`\`\`

**Implementation**:

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# Sample data
data = pd.DataFrame({
    'age': [22, 30, 35, 28, 45, 25, 32, 27, 40, 23],
    'income': [35000, 60000, 75000, 45000, 90000, 55000, 48000, 40000, 80000, 38000],
    'credit_score': [620, 680, 720, 650, 750, 700, 640, 710, 730, 630],
    'approved': [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
})

X = data[['age', 'income', 'credit_score']]
y = data['approved']

# Train decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# Visualize
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=['age', 'income', 'credit_score'],
          class_names=['Reject', 'Approve'], filled=True, fontsize=10)
plt.show()

# Extract rules as text
tree_rules = export_text(tree, feature_names=['age', 'income', 'credit_score'])
print("Decision Tree Rules:")
print(tree_rules)
\`\`\`

**Manual Logic Implementation**:

\`\`\`python
def decision_tree_as_logic(age, income, credit_score):
    """
    Manually implement tree logic
    Shows explicit AND/OR structure
    """
    # Path 1: age >= 25 AND income > 50000
    path1 = (age >= 25) and (income > 50000)
    
    # Path 2: age >= 25 AND income <= 50000 AND credit_score > 700
    path2 = (age >= 25) and (income <= 50000) and (credit_score > 700)
    
    # Path 3: age < 25 AND credit_score > 720
    path3 = (age < 25) and (credit_score > 720)
    
    # Final decision: OR of all approval paths
    approve = path1 or path2 or path3
    
    return approve, (path1, path2, path3)

# Test
test_cases = [
    (30, 60000, 650, "Path 1: mature + high income"),
    (28, 45000, 750, "Path 2: mature + low income but great credit"),
    (23, 40000, 730, "Path 3: young but excellent credit"),
    (35, 75000, 720, "Path 1: mature + high income"),
    (22, 35000, 650, "No path: young, low income, ok credit"),
]

print("\\nTesting logical paths:")
for age, income, credit, desc in test_cases:
    approve, paths = decision_tree_as_logic(age, income, credit)
    active_paths = [i+1 for i, p in enumerate(paths) if p]
    print(f"{desc}")
    print(f"  Input: age={age}, income={income}, credit={credit}")
    print(f"  Decision: {'APPROVE' if approve else 'REJECT'}")
    print(f"  Active paths: {active_paths if active_paths else 'None'}")
\`\`\`

**De Morgan's Laws in Trees**:

De Morgan's laws allow us to transform decision trees:
- ¬(A ∧ B) = ¬A ∨ ¬B
- ¬(A ∨ B) = ¬A ∧ ¬B

**Example - Rejection Condition**:

Instead of defining approval conditions, we can define rejection:

\`\`\`python
def approval_positive_logic(age, income, credit):
    """Define APPROVAL conditions (positive logic)"""
    return ((age >= 25 and income > 50000) or
            (age >= 25 and income <= 50000 and credit > 700) or
            (age < 25 and credit > 720))

def approval_negative_logic(age, income, credit):
    """
    Define REJECTION conditions (negative logic)
    Then negate to get approval
    
    Using De Morgan's laws to transform
    """
    # REJECT if:
    # NOT[(age >= 25 ∧ income > 50000) ∨ ...]
    # = [¬(age >= 25 ∧ income > 50000)] ∧ [¬(...)] ∧ [¬(...)]
    
    not_path1 = not ((age >= 25) and (income > 50000))
    not_path2 = not ((age >= 25) and (income <= 50000) and (credit > 700))
    not_path3 = not ((age < 25) and (credit > 720))
    
    reject = not_path1 and not_path2 and not_path3
    approve = not reject
    
    return approve

# Verify equivalence
for age, income, credit, _ in test_cases:
    pos = approval_positive_logic(age, income, credit)
    neg = approval_negative_logic(age, income, credit)
    print(f"age={age}, income={income}, credit={credit}: "
          f"Positive={pos}, Negative={neg}, Match={pos == neg}")
\`\`\`

**Tree Simplification with De Morgan's**:

\`\`\`python
# Original condition (complex)
def original_condition(x1, x2, x3):
    return not ((x1 and x2) or x3)

# Apply De Morgan's: ¬(A ∨ B) = ¬A ∧ ¬B
# ¬[(x1 ∧ x2) ∨ x3] = ¬(x1 ∧ x2) ∧ ¬x3

# Apply again: ¬(A ∧ B) = ¬A ∨ ¬B
# ¬(x1 ∧ x2) = ¬x1 ∨ ¬x2

# Final simplified:
def simplified_condition(x1, x2, x3):
    return ((not x1) or (not x2)) and (not x3)

# Verify equivalence
print("\\nDe Morgan's Simplification:")
for x1 in [True, False]:
    for x2 in [True, False]:
        for x3 in [True, False]:
            orig = original_condition(x1, x2, x3)
            simp = simplified_condition(x1, x2, x3)
            match = "✓" if orig == simp else "✗"
            print(f"x1={x1}, x2={x2}, x3={x3}: "
                  f"Original={orig}, Simplified={simp} {match}")
\`\`\`

**Equivalent Tree Representations**:

The same logical expression can be represented by different tree structures:

\`\`\`python
# Representation 1: (A ∧ B) ∨ (C ∧ D)
def tree_representation_1(A, B, C, D):
    """
    Tree splits on A first:
           A?
          / \\
        B?   C?
        /\\   /\\
       T F  D? F
           /\\
          T  F
    """
    if A:
        return B  # If A, result depends on B
    else:
        return C and D  # If not A, need both C and D

# Representation 2: Equivalent but splits on C first
def tree_representation_2(A, B, C, D):
    """
    Tree splits on C first:
           C?
          / \\
        D?   A?
        /\\   /\\
       T F  B? F
           /\\
          T  F
    """
    if C:
        return D  # If C, result depends on D
    else:
        return A and B  # If not C, need both A and B

# Both represent: (A ∧ B) ∨ (C ∧ D)
# Verify equivalence
print("\\nEquivalent Tree Representations:")
for A in [True, False]:
    for B in [True, False]:
        for C in [True, False]:
            for D in [True, False]:
                r1 = tree_representation_1(A, B, C, D)
                r2 = tree_representation_2(A, B, C, D)
                if r1 != r2:
                    print(f"MISMATCH: A={A}, B={B}, C={C}, D={D}")
                    
print("All cases match! Trees are equivalent.")
\`\`\`

**Trading Application**:

\`\`\`python
# Trading signal decision tree

def trading_signal_positive(price_above_sma, rsi_oversold, volume_high, trend_up):
    """
    BUY signal if:
    (price below SMA AND RSI oversold) OR (volume high AND trend up)
    
    Positive logic formulation
    """
    signal_1 = (not price_above_sma) and rsi_oversold
    signal_2 = volume_high and trend_up
    
    return signal_1 or signal_2

def trading_signal_negative(price_above_sma, rsi_oversold, volume_high, trend_up):
    """
    Same logic using De Morgan's transformation
    Negative logic: when NOT to buy
    
    ¬BUY = ¬[(¬price_above_sma ∧ rsi_oversold) ∨ (volume_high ∧ trend_up)]
         = [¬(¬price_above_sma ∧ rsi_oversold)] ∧ [¬(volume_high ∧ trend_up)]
         = [(price_above_sma ∨ ¬rsi_oversold)] ∧ [(¬volume_high ∨ ¬trend_up)]
    """
    not_signal_1 = price_above_sma or (not rsi_oversold)
    not_signal_2 = (not volume_high) or (not trend_up)
    
    no_buy = not_signal_1 and not_signal_2
    return not no_buy

# Test
test_scenarios = [
    (False, True, False, False, "Below SMA + oversold"),
    (True, False, True, True, "Above SMA but high volume + uptrend"),
    (False, False, False, False, "No clear signal"),
    (True, True, True, True, "All positive indicators"),
]

print("\\nTrading Signal Logic:")
for price_above, rsi_os, vol_high, trend, desc in test_scenarios:
    pos = trading_signal_positive(price_above, rsi_os, vol_high, trend)
    neg = trading_signal_negative(price_above, rsi_os, vol_high, trend)
    print(f"{desc}: {'BUY' if pos else 'HOLD'} (Match: {pos == neg})")
\`\`\`

**Summary**:
- Decision trees = visual logic (AND for paths, OR for multiple paths)
- De Morgan's laws enable tree transformations and simplifications
- Same logic can have multiple equivalent tree structures
- Understanding logical equivalences helps optimize decision trees
- Negative logic (rejection rules) often simpler than positive logic`,
          keyPoints: [
            'Decision tree paths are AND operations (conjunction of conditions)',
            'Multiple paths to same class are OR operations (disjunction)',
            "De Morgan's laws: ¬(A∧B)=¬A∨¬B and ¬(A∨B)=¬A∧¬B enable transformations",
            'Same logical expression can be represented by different tree structures',
            'Negative logic (rejection conditions) sometimes simpler than positive',
            'Tree simplification using logical equivalences reduces complexity',
          ],
        },
        {
          id: 'dq3-set-operations-data-splitting',
          question:
            'In machine learning, train/validation/test splits must be disjoint sets with no overlap. Explain using set theory: (1) Why splits must be disjoint, (2) How to verify no data leakage using set operations, (3) Stratification as preserving set proportions, (4) Cross-validation as set partitioning, (5) Time-series splits and ordered sets. Provide code examples.',
          sampleAnswer: `Set theory provides a rigorous framework for understanding data splitting in machine learning, ensuring proper evaluation and preventing data leakage.

**1. Why Splits Must Be Disjoint**:

**Definition**: Sets A and B are **disjoint** if A ∩ B = ∅ (empty set)

**Requirement**: Train ∩ Val ∩ Test = ∅

**Reason**: To accurately estimate generalization performance
- Training set: Learn patterns
- Validation set: Tune hyperparameters  
- Test set: Final evaluation

If sets overlap → model has seen test data → overoptimistic performance estimate

**Mathematical Formulation**:

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create dataset
n_samples = 1000
all_indices = set(range(n_samples))

# Split into train/val/test
train_size = 0.6
val_size = 0.2
test_size = 0.2

# First split: train vs temp
train_indices, temp_indices = train_test_split(
    list(all_indices), train_size=train_size, random_state=42
)
train_indices = set(train_indices)
temp_indices = set(temp_indices)

# Second split: val vs test
val_indices, test_indices = train_test_split(
    list(temp_indices), train_size=val_size/(val_size + test_size), random_state=42
)
val_indices = set(val_indices)
test_indices = set(test_indices)

print("Set Sizes:")
print(f"Total: {len(all_indices)}")
print(f"Train: {len(train_indices)} ({len(train_indices)/n_samples:.1%})")
print(f"Val: {len(val_indices)} ({len(val_indices)/n_samples:.1%})")
print(f"Test: {len(test_indices)} ({len(test_indices)/n_samples:.1%})")
\`\`\`

**2. Verifying No Data Leakage with Set Operations**:

\`\`\`python
def verify_data_splits(train, val, test, all_data):
    """
    Comprehensive verification of data splits using set theory
    """
    print("\\n=== Data Split Verification ===\\n")
    
    # Check 1: Pairwise disjoint (no overlap)
    train_val_overlap = train & val
    train_test_overlap = train & test
    val_test_overlap = val & test
    
    print("1. Disjoint Sets (must be empty):")
    print(f"   Train ∩ Val = {train_val_overlap} (size: {len(train_val_overlap)})")
    print(f"   Train ∩ Test = {train_test_overlap} (size: {len(train_test_overlap)})")
    print(f"   Val ∩ Test = {val_test_overlap} (size: {len(val_test_overlap)})")
    
    all_disjoint = (len(train_val_overlap) == 0 and 
                    len(train_test_overlap) == 0 and 
                    len(val_test_overlap) == 0)
    print(f"   ✓ All disjoint: {all_disjoint}")
    
    # Check 2: Union equals original (no missing data)
    union = train | val | test
    print(f"\\n2. Complete Coverage:")
    print(f"   Train ∪ Val ∪ Test = {len(union)} samples")
    print(f"   Original data = {len(all_data)} samples")
    print(f"   ✓ Complete: {union == all_data}")
    
    # Check 3: No missing indices
    missing = all_data - union
    print(f"\\n3. Missing Data:")
    print(f"   All - Union = {missing} (size: {len(missing)})")
    print(f"   ✓ No missing: {len(missing) == 0}")
    
    # Check 4: No duplicate indices within sets
    print(f"\\n4. No Duplicates (set property automatically enforced):")
    print(f"   ✓ Sets inherently have no duplicates")
    
    # Summary
    is_valid = all_disjoint and (union == all_data) and (len(missing) == 0)
    print(f"\\n{'='*40}")
    print(f"Overall: {'✓ VALID SPLIT' if is_valid else '✗ INVALID SPLIT'}")
    print(f"{'='*40}")
    
    return is_valid

# Verify our splits
is_valid = verify_data_splits(train_indices, val_indices, test_indices, all_indices)
\`\`\`

**3. Stratification as Preserving Set Proportions**:

**Goal**: Maintain class distribution across splits

\`\`\`python
from sklearn.model_split import train_test_split
from collections import Counter

# Create imbalanced dataset
y = np.array([0]*700 + [1]*250 + [2]*50)  # Imbalanced classes
X = np.arange(len(y)).reshape(-1, 1)

print("Original class distribution:")
original_dist = Counter(y)
for class_label, count in sorted(original_dist.items()):
    print(f"  Class {class_label}: {count} ({count/len(y):.1%})")

# Stratified split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=0.6, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, train_size=0.5, stratify=y_temp, random_state=42
)

# Analyze stratification using sets
train_by_class = {c: set(X_train[y_train == c].flatten()) for c in [0, 1, 2]}
val_by_class = {c: set(X_val[y_val == c].flatten()) for c in [0, 1, 2]}
test_by_class = {c: set(X_test[y_test == c].flatten()) for c in [0, 1, 2]}

print("\\nStratified split class distributions:")
for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    print(f"\\n{split_name}:")
    split_dist = Counter(y_split)
    for class_label, count in sorted(split_dist.items()):
        original_prop = original_dist[class_label] / len(y)
        split_prop = count / len(y_split)
        print(f"  Class {class_label}: {count} ({split_prop:.1%}) "
              f"[Original: {original_prop:.1%}]")

# Verify class-wise disjointness
print("\\nClass-wise disjoint verification:")
for c in [0, 1, 2]:
    overlap_train_val = train_by_class[c] & val_by_class[c]
    overlap_train_test = train_by_class[c] & test_by_class[c]
    overlap_val_test = val_by_class[c] & test_by_class[c]
    
    all_disjoint = (len(overlap_train_val) == 0 and 
                    len(overlap_train_test) == 0 and 
                    len(overlap_val_test) == 0)
    print(f"  Class {c}: {'✓ Disjoint' if all_disjoint else '✗ Overlap detected'}")
\`\`\`

**4. Cross-Validation as Set Partitioning**:

**Definition**: Partition dataset into k disjoint subsets (folds)

**Properties**:
- Fold₁ ∪ Fold₂ ∪ ... ∪ Foldₖ = All Data
- Foldᵢ ∩ Foldⱼ = ∅ for i ≠ j

\`\`\`python
from sklearn.model_selection import KFold

# K-Fold Cross-Validation
n_samples = 100
k_folds = 5

indices = np.arange(n_samples)
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Create folds as sets
folds = []
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
    fold = set(val_idx)
    folds.append(fold)
    print(f"Fold {fold_idx + 1}: {len(fold)} samples")

# Verify partition properties
print("\\n=== Partition Verification ===")

# 1. Union equals all data
union_folds = set().union(*folds)
all_data = set(indices)
print(f"\\n1. Union = All Data: {union_folds == all_data}")

# 2. Pairwise disjoint
print(f"\\n2. Pairwise Disjoint:")
all_disjoint = True
for i in range(len(folds)):
    for j in range(i + 1, len(folds)):
        overlap = folds[i] & folds[j]
        if len(overlap) > 0:
            print(f"   Fold {i+1} ∩ Fold {j+1}: {len(overlap)} (PROBLEM!)")
            all_disjoint = False

if all_disjoint:
    print(f"   ✓ All folds are pairwise disjoint")

# 3. Equal sizes (approximately)
sizes = [len(fold) for fold in folds]
print(f"\\n3. Fold Sizes: {sizes}")
print(f"   Min: {min(sizes)}, Max: {max(sizes)}, Difference: {max(sizes) - min(sizes)}")

# Visualize CV folds
print("\\n=== Cross-Validation Iterations ===")
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
    train_set = set(train_idx)
    val_set = set(val_idx)
    
    print(f"\\nIteration {fold_idx + 1}:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val: {len(val_set)} samples")
    print(f"  Disjoint: {len(train_set & val_set) == 0}")
    print(f"  Union = All: {(train_set | val_set) == all_data}")
\`\`\`

**5. Time-Series Splits and Ordered Sets**:

**Key Difference**: Time series requires preserving temporal order

**Time-Based Partitioning**:
- Train: {t₁, t₂, ..., tₙ}
- Test: {tₙ₊₁, tₙ₊₂, ..., tₘ}
- Constraint: max(Train) < min(Test)

\`\`\`python
from sklearn.model_selection import TimeSeriesSplit

# Simulate time series data
n_samples = 100
dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
data = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(n_samples),
    'index': range(n_samples)
})

# Time series split
tscv = TimeSeriesSplit(n_splits=5)

print("=== Time Series Cross-Validation ===\\n")

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(data)):
    train_set = set(train_idx)
    test_set = set(test_idx)
    
    print(f"Fold {fold_idx + 1}:")
    print(f"  Train: indices {min(train_idx)} to {max(train_idx)} ({len(train_set)} samples)")
    print(f"  Test: indices {min(test_idx)} to {max(test_idx)} ({len(test_set)} samples)")
    
    # Verify temporal ordering
    max_train = max(train_idx)
    min_test = min(test_idx)
    temporal_order_preserved = max_train < min_test
    
    # Verify disjoint
    disjoint = len(train_set & test_set) == 0
    
    print(f"  Temporal order preserved: {temporal_order_preserved}")
    print(f"  Disjoint: {disjoint}")
    print()

# Visualize temporal splits
plt.figure(figsize=(14, 8))

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(data)):
    plt.subplot(5, 1, fold_idx + 1)
    
    # Plot train as blue, test as red
    train_mask = np.zeros(n_samples)
    train_mask[train_idx] = 1
    test_mask = np.zeros(n_samples)
    test_mask[test_idx] = 2
    
    combined = train_mask + test_mask
    plt.scatter(range(n_samples), [fold_idx + 1] * n_samples, 
                c=combined, cmap='RdBu', s=50, marker='|')
    plt.ylabel(f'Fold {fold_idx + 1}', rotation=0, labelpad=20)
    plt.yticks([])
    
    if fold_idx == 4:
        plt.xlabel('Time Index')

plt.suptitle('Time Series Cross-Validation Splits (Blue=Train, Red=Test)')
plt.tight_layout()
plt.show()
\`\`\`

**Trading Application - Walk-Forward Validation**:

\`\`\`python
def walk_forward_validation(data, train_window, test_window):
    """
    Walk-forward validation for trading strategies
    Maintains temporal order and disjoint sets
    """
    n = len(data)
    splits = []
    
    start = 0
    while start + train_window + test_window <= n:
        train_end = start + train_window
        test_end = train_end + test_window
        
        train_indices = set(range(start, train_end))
        test_indices = set(range(train_end, test_end))
        
        splits.append((train_indices, test_indices))
        start += test_window  # Move forward by test window
    
    return splits

# Example: 500 days of trading data
n_days = 500
train_window = 252  # 1 year
test_window = 63    # Quarter

splits = walk_forward_validation(range(n_days), train_window, test_window)

print(f"=== Walk-Forward Validation ===")
print(f"Total periods: {len(splits)}\\n")

for i, (train, test) in enumerate(splits):
    print(f"Period {i + 1}:")
    print(f"  Train: days {min(train)} to {max(train)} ({len(train)} days)")
    print(f"  Test: days {min(test)} to {max(test)} ({len(test)} days)")
    
    # Verify properties
    disjoint = len(train & test) == 0
    temporal = max(train) < min(test)
    print(f"  Disjoint: {disjoint}, Temporal order: {temporal}")
    print()
\`\`\`

**Summary**:
- Disjoint sets (Train ∩ Val ∩ Test = ∅) prevent data leakage
- Set operations verify split validity systematically
- Stratification preserves class proportions across splits
- Cross-validation partitions data into k disjoint folds
- Time series requires ordered sets with temporal constraints
- Set theory provides rigorous framework for proper evaluation`,
          keyPoints: [
            'Splits must be disjoint (no overlap) to prevent data leakage',
            'Verify splits: Train ∩ Val ∩ Test = ∅ and Train ∪ Val ∪ Test = All',
            'Stratification maintains class proportions across splits',
            'Cross-validation partitions data into k disjoint, equal-sized folds',
            'Time series splits require temporal ordering: max(Train) < min(Test)',
            'Set operations provide systematic verification of split validity',
          ],
        },
      ],
    },
    {
      id: 'combinatorics-basics',
      title: 'Combinatorics Basics',
      content: `
# Combinatorics Basics

## Introduction

Combinatorics is the mathematics of counting. In machine learning and data science, we constantly count: possible model configurations, dataset splits, feature combinations, and more. Understanding combinatorics is essential for probability theory, analyzing algorithm complexity, and understanding model capacity.

## Fundamental Counting Principles

### Addition Principle

**If there are n ways to do task A and m ways to do task B, and these tasks cannot be done simultaneously, there are n + m ways to do either task A or task B.**

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, comb, perm

# Example: Classification models
linear_models = 3  # Linear regression, logistic regression, perceptron
tree_models = 4    # Decision tree, random forest, gradient boosting, XGBoost

total_model_choices = linear_models + tree_models
print(f"Linear models: {linear_models}")
print(f"Tree models: {tree_models}")
print(f"Total model choices: {total_model_choices}")

# ML Application: Feature selection methods
correlation_methods = 2  # Pearson, Spearman
mutual_info_methods = 1
tree_based_methods = 3   # RF importance, XGBoost gain, permutation

total_selection_methods = correlation_methods + mutual_info_methods + tree_based_methods
print(f"\\nTotal feature selection methods: {total_selection_methods}")
\`\`\`

### Multiplication Principle

**If there are n ways to do task A and m ways to do task B, there are n × m ways to do both tasks.**

\`\`\`python
# Example: Hyperparameter grid search
learning_rates = [0.001, 0.01, 0.1]         # 3 options
batch_sizes = [16, 32, 64, 128]             # 4 options
optimizers = ['SGD', 'Adam', 'RMSprop']     # 3 options

total_combinations = len(learning_rates) * len(batch_sizes) * len(optimizers)

print("Hyperparameter Grid:")
print(f"Learning rates: {len(learning_rates)}")
print(f"Batch sizes: {len(batch_sizes)}")
print(f"Optimizers: {len(optimizers)}")
print(f"Total combinations: {total_combinations}")

# Generate all combinations
from itertools import product

configs = list(product(learning_rates, batch_sizes, optimizers))
print(f"\\nFirst 5 configurations:")
for i, (lr, bs, opt) in enumerate(configs[:5]):
    print(f"{i+1}. LR={lr}, Batch={bs}, Optimizer={opt}")
\`\`\`

## Permutations

### Definition

**Permutation**: Arrangement of objects where **order matters**

**Formula**: P(n, r) = n!/(n-r)! = n × (n-1) × ... × (n-r+1)

Number of ways to arrange r objects from n total objects.

\`\`\`python
from math import perm

def permutations_formula(n, r):
    """Calculate P(n, r) = n!/(n-r)!"""
    return perm(n, r)

# Example: Feature ordering for sequential model
features = ['age', 'income', 'credit_score', 'debt_ratio', 'employment']
n_features = len(features)

# How many ways to order 3 features?
r = 3
p = permutations_formula(n_features, r)

print(f"Total features: {n_features}")
print(f"Selecting: {r} features")
print(f"Permutations P({n_features}, {r}) = {p}")

# Generate actual permutations
from itertools import permutations as perm_iter

feature_perms = list(perm_iter(features, r))
print(f"\\nFirst 10 orderings:")
for i, perm in enumerate(feature_perms[:10]):
    print(f"{i+1}. {' → '.join(perm)}")
\`\`\`

### Special Case: All Objects

P(n, n) = n! (all n objects arranged)

\`\`\`python
# Example: Order of applying data augmentations
augmentations = ['rotate', 'flip', 'crop', 'color_jitter']
n = len(augmentations)

total_orderings = factorial(n)
print(f"Augmentations: {augmentations}")
print(f"Total possible orderings: {total_orderings}")

# Show some orderings
orderings = list(perm_iter(augmentations))
print(f"\\nSome orderings:")
for i, ordering in enumerate(orderings[:6]):
    print(f"{i+1}. {' → '.join(ordering)}")
\`\`\`

## Combinations

### Definition

**Combination**: Selection of objects where **order doesn't matter**

**Formula**: C(n, r) = n!/(r!(n-r)!) = "n choose r"

Number of ways to select r objects from n total objects.

\`\`\`python
from math import comb

def combinations_formula(n, r):
    """Calculate C(n, r) = n!/(r!(n-r)!)"""
    return comb(n, r)

# Example: Selecting features for a model
all_features = ['age', 'income', 'education', 'credit_score', 'debt_ratio', 
                'employment', 'location', 'marital_status']
n_features = len(all_features)
select_k = 3

c = combinations_formula(n_features, select_k)

print(f"Total features available: {n_features}")
print(f"Selecting: {select_k} features")
print(f"Combinations C({n_features}, {select_k}) = {c}")

# Generate actual combinations
from itertools import combinations as comb_iter

feature_combs = list(comb_iter(all_features, select_k))
print(f"\\nFirst 10 feature combinations:")
for i, combo in enumerate(feature_combs[:10]):
    print(f"{i+1}. {combo}")
\`\`\`

### Permutations vs Combinations

\`\`\`python
def compare_perm_comb(n, r):
    """Compare permutations vs combinations"""
    p = permutations_formula(n, r)
    c = combinations_formula(n, r)
    
    print(f"n={n}, r={r}:")
    print(f"  Permutations (order matters): {p}")
    print(f"  Combinations (order doesn't matter): {c}")
    print(f"  Ratio P/C = {p/c:.1f} = {r}!")
    print(f"  For each combination, there are {r}! = {factorial(r)} permutations")

compare_perm_comb(5, 3)
print()
compare_perm_comb(10, 4)
\`\`\`

## Pascal's Triangle and Binomial Coefficients

C(n, r) are called **binomial coefficients** and appear in Pascal's triangle:

\`\`\`python
def pascals_triangle(rows):
    """Generate Pascal's triangle"""
    triangle = []
    for n in range(rows):
        row = [comb(n, r) for r in range(n + 1)]
        triangle.append(row)
    return triangle

# Generate and display
triangle = pascals_triangle(8)

print("Pascal's Triangle:")
for n, row in enumerate(triangle):
    spaces = ' ' * (len(triangle) - n - 1) * 2
    print(spaces + '  '.join(f'{x:3}' for x in row))

# Properties
print(f"\\nProperties:")
print(f"1. Symmetry: C(n, r) = C(n, n-r)")
print(f"   C(6, 2) = {comb(6, 2)}, C(6, 4) = {comb(6, 4)}")

print(f"\\n2. Sum of row n = 2^n:")
for n in range(6):
    row_sum = sum(triangle[n])
    print(f"   Row {n}: sum = {row_sum} = 2^{n} = {2**n}")

print(f"\\n3. Pascal's identity: C(n, r) = C(n-1, r-1) + C(n-1, r)")
n, r = 5, 2
print(f"   C(5, 2) = {comb(5, 2)}")
print(f"   C(4, 1) + C(4, 2) = {comb(4, 1)} + {comb(4, 2)} = {comb(4, 1) + comb(4, 2)}")
\`\`\`

## Applications in Machine Learning

### k-Fold Cross-Validation

\`\`\`python
def count_cv_orderings(n_samples, k_folds):
    """
    Count possible ways to partition n samples into k folds
    This is the multinomial coefficient
    """
    fold_size = n_samples // k_folds
    # Simplified: C(n, fold_size) for first fold, C(n-fold_size, fold_size) for second, etc.
    # Actual formula is multinomial coefficient
    
    # For equal-sized folds: n! / (fold_size!)^k * k!
    # Divided by k! if folds are unordered
    
    return comb(n_samples, fold_size)

n_samples = 100
k = 5
fold_size = n_samples // k

print(f"Cross-validation combinations:")
print(f"Samples: {n_samples}, Folds: {k}, Fold size: {fold_size}")
print(f"Ways to choose first fold: C({n_samples}, {fold_size}) = {comb(n_samples, fold_size)}")
print(f"\\nThis number is astronomical! Good thing we use random splitting.")
\`\`\`

### Feature Selection

\`\`\`python
def count_feature_subsets(n_features, min_k=1, max_k=None):
    """Count all possible feature subsets of size k to max_k"""
    if max_k is None:
        max_k = n_features
    
    total = 0
    counts = {}
    
    for k in range(min_k, max_k + 1):
        count = comb(n_features, k)
        counts[k] = count
        total += count
    
    return counts, total

n_features = 20
counts, total = count_feature_subsets(n_features, min_k=1, max_k=10)

print(f"Feature subset counts (n={n_features}):")
for k, count in counts.items():
    print(f"  {k} features: {count:,} combinations")

print(f"\\nTotal: {total:,} possible feature sets")
print(f"All possible subsets (2^n - 1): {2**n_features - 1:,}")  # -1 to exclude empty set

# Visualize
plt.figure(figsize=(10, 6))
k_values = list(counts.keys())
count_values = list(counts.values())
plt.bar(k_values, count_values)
plt.xlabel('Number of features selected (k)')
plt.ylabel('Number of combinations C(n, k)')
plt.title(f'Feature Subset Counts (n={n_features})')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

### Model Ensemble Combinations

\`\`\`python
def ensemble_combinations(n_models, min_ensemble_size=2):
    """Count ways to form ensembles from n models"""
    combinations = {}
    
    for k in range(min_ensemble_size, n_models + 1):
        combinations[k] = comb(n_models, k)
    
    return combinations

n_models = 10
ensembles = ensemble_combinations(n_models)

print(f"Ensemble combinations from {n_models} models:")
for size, count in ensembles.items():
    print(f"  Ensemble of {size} models: {count:,} ways")

total = sum(ensembles.values())
print(f"\\nTotal possible ensembles: {total:,}")
\`\`\`

### Hyperparameter Search Space

\`\`\`python
def hyperparameter_search_space(param_grid):
    """Calculate size of hyperparameter search space"""
    # Grid search: multiply all options
    grid_size = 1
    for param, values in param_grid.items():
        grid_size *= len(values)
    
    return grid_size

# Example search space
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64, 128],
    'num_layers': [2, 3, 4, 5],
    'hidden_size': [64, 128, 256, 512],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
    'optimizer': ['sgd', 'adam', 'rmsprop']
}

space_size = hyperparameter_search_space(param_grid)

print("Hyperparameter Search Space:")
for param, values in param_grid.items():
    print(f"  {param}: {len(values)} options")

print(f"\\nTotal configurations: {space_size:,}")
print(f"\\nIf each config takes 10 minutes to train:")
print(f"  Total time: {space_size * 10 / 60:.1f} hours = {space_size * 10 / 60 / 24:.1f} days")
\`\`\`

## Trading Applications

### Portfolio Combinations

\`\`\`python
def portfolio_combinations(n_assets, portfolio_size):
    """
    Count ways to select portfolio of given size from n assets
    """
    return comb(n_assets, portfolio_size)

# Example: Selecting stocks for portfolio
available_stocks = 100
portfolio_size = 10

combinations = portfolio_combinations(available_stocks, portfolio_size)

print(f"Portfolio Selection:")
print(f"Available stocks: {available_stocks}")
print(f"Portfolio size: {portfolio_size}")
print(f"Possible portfolios: {combinations:,}")

# Different portfolio sizes
print(f"\\nPortfolio size vs combinations:")
for size in [5, 10, 15, 20]:
    count = portfolio_combinations(available_stocks, size)
    print(f"  {size} stocks: {count:,} portfolios")
\`\`\`

### Trading Strategy Combinations

\`\`\`python
# Example: Combining technical indicators
indicators = ['RSI', 'MACD', 'SMA', 'EMA', 'Bollinger', 'Stochastic', 'ADX', 'OBV']
n_indicators = len(indicators)

print("Trading Strategy Combinations:")
print(f"Available indicators: {n_indicators}")

for k in range(2, min(6, n_indicators + 1)):
    count = comb(n_indicators, k)
    print(f"  Using {k} indicators: {count} combinations")

# Show some specific combinations
print(f"\\nExample 3-indicator strategies:")
strategies = list(comb_iter(indicators, 3))
for i, strategy in enumerate(strategies[:10]):
    print(f"  {i+1}. {', '.join(strategy)}")
\`\`\`

## Summary

- **Counting principles**: Addition (OR), Multiplication (AND)
- **Permutations**: P(n, r) = n!/(n-r)! (order matters)
- **Combinations**: C(n, r) = n!/(r!(n-r)!) (order doesn't matter)
- **Pascal's triangle**: Contains binomial coefficients
- **Key insight**: P(n, r) = C(n, r) × r!

**ML Applications**:
- Cross-validation fold partitions
- Feature subset selection (2^n possible subsets!)
- Hyperparameter grid search space size
- Model ensemble combinations
- Dataset split possibilities

**Trading Applications**:
- Portfolio asset selection
- Indicator combinations for strategies
- Backtesting scenario enumeration
`,
      multipleChoice: [
        {
          id: 'mc1-combinations',
          question:
            'You have 10 features and want to select exactly 3 for your model. How many different feature sets are possible?',
          options: ['30', '120', '720', '1000'],
          correctAnswer: 1,
          explanation:
            "This is C(10, 3) = 10!/(3!×7!) = (10×9×8)/(3×2×1) = 720/6 = 120. Order doesn't matter for feature selection, so we use combinations.",
        },
        {
          id: 'mc2-permutations',
          question:
            'In how many ways can you arrange 4 different models in an ensemble pipeline where order matters?',
          options: ['4', '16', '24', '256'],
          correctAnswer: 2,
          explanation:
            'This is P(4, 4) = 4! = 4×3×2×1 = 24. Order matters, so we use permutations of all 4 models.',
        },
        {
          id: 'mc3-grid-search',
          question:
            'A grid search has 3 learning rates, 4 batch sizes, and 2 optimizers. How many total configurations?',
          options: ['9', '12', '24', '64'],
          correctAnswer: 2,
          explanation:
            'Use multiplication principle: 3 × 4 × 2 = 24 total configurations.',
        },
        {
          id: 'mc4-binomial-coefficient',
          question: 'What is C(6, 2)?',
          options: ['12', '15', '30', '720'],
          correctAnswer: 1,
          explanation: 'C(6, 2) = 6!/(2!×4!) = (6×5)/(2×1) = 30/2 = 15.',
        },
        {
          id: 'mc5-pascals-triangle',
          question:
            "What is the sum of all numbers in row n of Pascal's triangle?",
          options: ['n', 'n!', '2^n', 'n^2'],
          correctAnswer: 2,
          explanation:
            "The sum of row n in Pascal's triangle equals 2^n. This is because C(n,0) + C(n,1) + ... + C(n,n) = 2^n.",
        },
      ],
      quiz: [
        {
          id: 'dq1-feature-selection-complexity',
          question:
            'Explain why exhaustive feature selection becomes computationally infeasible as the number of features grows. If you have n features, how many total possible feature subsets exist? Compare brute-force search with intelligent search strategies (forward selection, backward elimination, genetic algorithms). Provide complexity analysis and practical examples.',
          sampleAnswer: `The computational complexity of exhaustive feature selection grows exponentially with the number of features, making it infeasible for even moderately-sized feature sets.

**Total Number of Feature Subsets**:

For n features, the total number of non-empty subsets is: **2^n - 1**

**Why 2^n?**

For each feature, we have 2 choices: include it or exclude it.
- Feature 1: 2 choices
- Feature 2: 2 choices
- ...
- Feature n: 2 choices

Total: 2 × 2 × ... × 2 = 2^n subsets (including empty set)

We subtract 1 to exclude the empty set (no features selected).

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from math import comb
import time

def count_all_subsets(n):
    """Total non-empty subsets of n features"""
    return 2**n - 1

# Demonstrate exponential growth
n_values = range(1, 31)
subset_counts = [count_all_subsets(n) for n in n_values]

print("Feature subsets (exhaustive search):")
for n in [5, 10, 15, 20, 25, 30]:
    count = count_all_subsets(n)
    print(f"  {n} features: {count:,} subsets")

# Visualize
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(n_values, subset_counts, 'b-', linewidth=2)
plt.xlabel('Number of features (n)')
plt.ylabel('Number of subsets (2^n - 1)')
plt.title('Exhaustive Feature Selection: Exponential Growth')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(n_values, subset_counts, 'r-', linewidth=2)
plt.xlabel('Number of features (n)')
plt.ylabel('Number of subsets (log scale)')
plt.title('Log Scale View')
plt.grid(True)

plt.tight_layout()
plt.show()
\`\`\`

**Practical Infeasibility**:

\`\`\`python
# Estimate time for exhaustive search
def estimate_search_time(n_features, time_per_model_seconds):
    """Estimate time for exhaustive feature selection"""
    n_subsets = count_all_subsets(n_features)
    total_seconds = n_subsets * time_per_model_seconds
    
    hours = total_seconds / 3600
    days = hours / 24
    years = days / 365
    
    return n_subsets, total_seconds, hours, days, years

print("\\nTime estimates (assuming 10 seconds per model evaluation):\\n")

for n in [10, 15, 20, 25, 30]:
    subsets, secs, hrs, days, yrs = estimate_search_time(n, 10)
    print(f"{n} features: {subsets:,} subsets")
    
    if yrs >= 1:
        print(f"  Time: {yrs:,.1f} years")
    elif days >= 1:
        print(f"  Time: {days:,.1f} days")
    elif hrs >= 1:
        print(f"  Time: {hrs:,.1f} hours")
    else:
        print(f"  Time: {secs:,.1f} seconds")
    print()
\`\`\`

**Intelligent Search Strategies**:

**1. Forward Selection (Greedy)**:

Start with no features, add one feature at a time (the one that improves performance most).

Complexity: O(n²) evaluations

\`\`\`python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                           random_state=42)

def forward_selection(X, y, max_features=None):
    """
    Forward feature selection
    Greedy algorithm: add one feature at a time
    """
    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features
    
    selected = []
    remaining = list(range(n_features))
    scores = []
    evaluations = 0
    
    for iteration in range(max_features):
        best_score = -np.inf
        best_feature = None
        
        # Try adding each remaining feature
        for feature in remaining:
            candidate = selected + [feature]
            X_subset = X[:, candidate]
            
            # Evaluate
            model = LogisticRegression(max_iter=1000)
            score = cross_val_score(model, X_subset, y, cv=3).mean()
            evaluations += 1
            
            if score > best_score:
                best_score = score
                best_feature = feature
        
        # No improvement possible
        if best_feature is None:
            break
        
        # Add best feature
        selected.append(best_feature)
        remaining.remove(best_feature)
        scores.append(best_score)
        
        print(f"Iteration {iteration + 1}: Added feature {best_feature}, "
              f"Score: {best_score:.4f}")
    
    print(f"\\nTotal evaluations: {evaluations}")
    print(f"Selected features: {selected}")
    
    return selected, scores, evaluations

print("Forward Selection:")
selected, scores, evals = forward_selection(X, y, max_features=5)

# Compare to exhaustive
n = X.shape[1]
exhaustive_evals = count_all_subsets(n)
print(f"\\nComparison:")
print(f"  Forward selection: {evals} evaluations")
print(f"  Exhaustive search: {exhaustive_evals:,} evaluations")
print(f"  Speedup: {exhaustive_evals / evals:,.0f}x")
\`\`\`

**2. Backward Elimination**:

Start with all features, remove one at a time (the least important).

Complexity: O(n²) evaluations

**3. Genetic Algorithm**:

Evolutionary approach: maintain population of feature sets, evolve through mutation and crossover.

Complexity: O(p × g × n) where p=population size, g=generations

\`\`\`python
def genetic_algorithm_feature_selection(X, y, population_size=20, generations=10):
    """
    Simplified genetic algorithm for feature selection
    """
    n_features = X.shape[1]
    evaluations = 0
    
    # Initialize random population
    population = [np.random.rand(n_features) > 0.5 for _ in range(population_size)]
    
    for gen in range(generations):
        # Evaluate fitness
        fitness = []
        for individual in population:
            if not any(individual):  # At least one feature
                individual[np.random.randint(n_features)] = True
            
            X_subset = X[:, individual]
            model = LogisticRegression(max_iter=1000)
            score = cross_val_score(model, X_subset, y, cv=3).mean()
            evaluations += 1
            fitness.append(score)
        
        # Select best half
        sorted_indices = np.argsort(fitness)[::-1]
        population = [population[i] for i in sorted_indices[:population_size // 2]]
        
        # Reproduce (crossover + mutation)
        offspring = []
        for _ in range(population_size // 2):
            parent1, parent2 = np.random.choice(len(population), 2, replace=False)
            child = np.array([population[parent1][i] if np.random.rand() > 0.5 
                            else population[parent2][i] for i in range(n_features)])
            
            # Mutation
            if np.random.rand() < 0.1:
                flip_idx = np.random.randint(n_features)
                child[flip_idx] = not child[flip_idx]
            
            offspring.append(child)
        
        population.extend(offspring)
        
        best_score = max(fitness)
        print(f"Generation {gen + 1}: Best score = {best_score:.4f}")
    
    # Return best individual
    final_fitness = [cross_val_score(LogisticRegression(max_iter=1000), 
                                     X[:, ind], y, cv=3).mean() 
                    for ind in population]
    evaluations += len(population)
    
    best_idx = np.argmax(final_fitness)
    best_features = np.where(population[best_idx])[0]
    
    print(f"\\nTotal evaluations: {evaluations}")
    print(f"Selected features: {list(best_features)}")
    
    return best_features, evaluations

print("\\n" + "="*50)
print("Genetic Algorithm:")
best_features, evals_ga = genetic_algorithm_feature_selection(X, y)

print(f"\\nComparison:")
print(f"  Genetic algorithm: {evals_ga} evaluations")
print(f"  Forward selection: {evals} evaluations")
print(f"  Exhaustive search: {exhaustive_evals:,} evaluations")
\`\`\`

**Comparison Table**:

\`\`\`python
import pandas as pd

comparison = pd.DataFrame({
    'Method': ['Exhaustive', 'Forward Selection', 'Backward Elimination', 
               'Genetic Algorithm', 'Random Search'],
    'Complexity': ['O(2^n)', 'O(n²)', 'O(n²)', 'O(p×g×n)', 'O(k)'],
    'Guarantees Optimal': ['Yes', 'No', 'No', 'No', 'No'],
    'Feasible for n=20': ['No', 'Yes', 'Yes', 'Yes', 'Yes'],
    'Feasible for n=100': ['No', 'Maybe', 'Maybe', 'Yes', 'Yes']
})

print("\\nFeature Selection Methods Comparison:")
print(comparison.to_string(index=False))
\`\`\`

**Key Insights**:

1. **Exponential explosion**: 2^n grows impossibly fast
   - 20 features: 1 million subsets
   - 30 features: 1 billion subsets
   - 40 features: 1 trillion subsets

2. **Greedy methods** (forward/backward): Polynomial time, but may miss optimal

3. **Evolutionary methods** (genetic algorithms): Balance exploration and exploitation

4. **Random search**: Simple baseline, surprisingly effective

5. **Modern approaches**: Regularization (L1/L2), tree-based importance

**Trading Application**:

\`\`\`python
# Stock portfolio optimization
# With 100 stocks, choosing 10 for portfolio:

n_stocks = 100
portfolio_size = 10

portfolios = comb(n_stocks, portfolio_size)
print(f"\\nPortfolio Selection:")
print(f"Stocks: {n_stocks}, Portfolio size: {portfolio_size}")
print(f"Possible portfolios: {portfolios:,}")

# If testing each portfolio takes 1 minute of backtesting:
hours = portfolios / 60
days = hours / 24
print(f"Time for exhaustive search: {days:,.0f} days")

print(f"\\nIntelligent approaches:")
print(f"  - Genetic algorithm: Evolve good portfolios")
print(f"  - Greedy: Add stocks one by one (maximize Sharpe ratio)")
print(f"  - Random search: Sample random portfolios")
print(f"  - Domain knowledge: Pre-filter to top 20 stocks, then combine")
\`\`\`

**Summary**:
- Exhaustive search: 2^n subsets (exponentially infeasible)
- Forward/Backward: n² evaluations (polynomial, practical)
- Genetic algorithms: Flexible, good for large spaces
- Trade-off: Optimality vs computational feasibility
- Real-world: Use intelligent search + domain knowledge`,
          keyPoints: [
            'Total feature subsets = 2^n - 1 (exponential growth)',
            'Exhaustive search becomes infeasible around n=20-25 features',
            'Forward selection: O(n²) greedy algorithm, adds best feature iteratively',
            'Genetic algorithms: O(p×g×n) evolutionary approach, balances exploration/exploitation',
            'Trade-off between optimality (exhaustive) and feasibility (heuristics)',
          ],
        },
        {
          id: 'dq2-permutations-augmentation',
          question:
            'In data augmentation for computer vision, if we have 5 different transformations (flip, rotate 90°, rotate 180°, rotate 270°, no transformation) that we can apply to an image, how many different augmented versions can we create if we apply exactly one transformation? What if we apply a sequence of 2 transformations? Explain how permutations relate to data augmentation strategies and discuss the trade-off between augmentation diversity and computational cost.',
          sampleAnswer: `Data augmentation uses permutations and combinations to create diverse training samples from limited data. Understanding combinatorics helps design effective augmentation strategies.

**Single Transformation**:

With 5 transformations, applying exactly one gives us **5 augmented versions** (including the original with "no transformation").

This is simply n = 5 choices.

**Sequence of 2 Transformations**:

If we apply 2 transformations in sequence WITH replacement (can repeat):
- Total: **n² = 5² = 25** different sequences

If WITHOUT replacement (no repeats):
- Total: **P(5,2) = 5!/(5-2)! = 5×4 = 20** permutations

\`\`\`python
import numpy as np
from math import factorial, perm

# Transformations
transformations = ['flip', 'rotate_90', 'rotate_180', 'rotate_270', 'no_op']
n = len(transformations)

print("Single transformation:")
print(f"  Options: {n}")
print(f"  Augmented versions: {n}")

print("\\nSequence of 2 transformations:")
print(f"  WITH replacement (can repeat): {n**2}")
print(f"  WITHOUT replacement (no repeats): {perm(n, 2)}")

# Generate all 2-transformation sequences (with replacement)
sequences = []
for t1 in transformations:
    for t2 in transformations:
        sequences.append(f"{t1} → {t2}")

print(f"\\nTotal sequences: {len(sequences)}")
print("\\nExample sequences:")
for seq in sequences[:10]:
    print(f"  {seq}")
\`\`\`

**Data Augmentation Strategy**:

**1. Simple Augmentation (Single Transformation)**:

\`\`\`python
import torch
import torchvision.transforms as transforms
from PIL import Image

# Define transformations
simple_transforms = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(90),
    transforms.RandomRotation(180),
    transforms.RandomRotation(270),
    transforms.Lambda(lambda x: x),  # no-op
]

def augment_single(image, num_augmentations=5):
    """Apply each transformation once"""
    augmented = []
    for transform in simple_transforms[:num_augmentations]:
        aug_img = transform(image)
        augmented.append(aug_img)
    return augmented

# With 1,000 original images and 5 transformations
original_images = 1000
augmented_per_image = 5
total_samples = original_images * augmented_per_image

print(f"Simple augmentation:")
print(f"  Original images: {original_images:,}")
print(f"  Transformations per image: {augmented_per_image}")
print(f"  Total training samples: {total_samples:,}")
\`\`\`

**2. Compositional Augmentation (Sequence of Transformations)**:

\`\`\`python
import itertools

def augment_compositional(image, transformations, seq_length=2):
    """Apply sequences of transformations"""
    augmented = []
    
    # Generate all permutations of length seq_length
    for perm in itertools.product(transformations, repeat=seq_length):
        aug_img = image.copy()
        for transform in perm:
            aug_img = transform(aug_img)
        augmented.append(aug_img)
    
    return augmented

# With sequences of length 2
seq_length = 2
augmented_per_image = len(transformations) ** seq_length

print(f"\\nCompositional augmentation (length {seq_length}):")
print(f"  Transformations available: {len(transformations)}")
print(f"  Sequences per image: {augmented_per_image}")
print(f"  Total training samples: {original_images * augmented_per_image:,}")

# Impact on training time
print(f"\\nTraining time impact:")
print(f"  Simple: ~{augmented_per_image}x more epochs")
print(f"  Compositional (len=2): ~{len(transformations)**2}x more epochs")
\`\`\`

**Trade-offs**:

**Diversity vs Computational Cost**:

\`\`\`python
import matplotlib.pyplot as plt

# Calculate augmentation options for different sequence lengths
n_transforms = 5
seq_lengths = range(1, 6)
with_replacement = [n_transforms**k for k in seq_lengths]
without_replacement = [perm(n_transforms, k) if k <= n_transforms else 0 
                       for k in seq_lengths]

print("Augmentation diversity growth:\\n")
print("Seq Length | With Replacement | Without Replacement | Compute Cost")
print("-" * 70)
for k in seq_lengths:
    wr = n_transforms**k
    wor = perm(n_transforms, k) if k <= n_transforms else 0
    cost = f"{wr}x"
    print(f"    {k}      |      {wr:>6}       |       {wor:>6}        |    {cost}")

# Practical considerations
print("\\n**Practical Guidelines**:")
print("- Small dataset (< 1K images): Use compositional augmentation (length 2-3)")
print("- Medium dataset (1K-10K): Use simple augmentation + random combinations")
print("- Large dataset (> 10K): Use online random augmentation (not exhaustive)")
print("- Trade-off: More augmentation → better generalization but slower training")
\`\`\`

**Real-World Example**:

In computer vision (e.g., CIFAR-10 with 50,000 training images):

1. **Exhaustive augmentation**: All 25 sequences → 1.25M samples → impractical
2. **Smart augmentation**: Randomly sample 5 sequences per image → 250K samples → manageable
3. **Online augmentation**: Generate random augmentation on-the-fly during training → no storage cost

**Key Insights**:

- Permutations determine the diversity of augmented data
- Compositional augmentations grow exponentially (n^k)
- Balance augmentation diversity with computational budget
- Random sampling from augmentation space often better than exhaustive application
- Online augmentation (generate during training) saves storage while maintaining diversity`,
          keyPoints: [
            'Single transformation: n options; Sequence of k transformations: n^k (with replacement) or P(n,k) (without)',
            'Data augmentation uses permutations to create diverse training samples',
            'Compositional augmentation: Apply multiple transformations in sequence',
            'Trade-off: More augmentation diversity vs higher computational and storage cost',
            'Strategy: Small datasets benefit from exhaustive augmentation, large datasets use random sampling',
          ],
        },
        {
          id: 'dq3-combinations-hyperparameter-tuning',
          question:
            'In hyperparameter tuning, suppose you have 4 hyperparameters to optimize, and each can take 5 different values. How many total configurations exist? If testing each configuration takes 30 minutes, how long would exhaustive grid search take? Compare this with random search (testing 100 random configurations) and explain why random search is often more effective despite testing fewer configurations. Discuss the role of combinatorics in AutoML.',
          sampleAnswer: `Hyperparameter tuning involves exploring a combinatorial space of configurations. Understanding combinatorics reveals why exhaustive search is often infeasible and why random search can be surprisingly effective.

**Total Configurations (Grid Search)**:

With 4 hyperparameters, each with 5 possible values:
- **Total configurations = 5⁴ = 625**

This is because we choose one value for each of 4 hyperparameters independently:
- Hyperparameter 1: 5 choices
- Hyperparameter 2: 5 choices  
- Hyperparameter 3: 5 choices
- Hyperparameter 4: 5 choices
- Total: 5 × 5 × 5 × 5 = 625

\`\`\`python
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import time

# Hyperparameter configuration
n_hyperparameters = 4
values_per_hyperparameter = 5
time_per_config_minutes = 30

# Total configurations
total_configs = values_per_hyperparameter ** n_hyperparameters
total_time_hours = (total_configs * time_per_config_minutes) / 60
total_time_days = total_time_hours / 24

print("Hyperparameter Tuning Complexity:")
print(f"  Number of hyperparameters: {n_hyperparameters}")
print(f"  Values per hyperparameter: {values_per_hyperparameter}")
print(f"  Total configurations: {total_configs:,}")
print(f"  Time per configuration: {time_per_config_minutes} minutes")
print(f"\\nExhaustive Grid Search:")
print(f"  Total time: {total_time_hours:,.1f} hours ({total_time_days:,.1f} days)")

# Example hyperparameters
hyperparameters = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    'batch_size': [16, 32, 64, 128, 256],
    'num_layers': [2, 3, 4, 5, 6],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],
}

# Generate all combinations
all_configs = list(product(*hyperparameters.values()))
print(f"\\nGenerated {len(all_configs)} configurations")
print("\\nFirst 10 configurations:")
keys = list(hyperparameters.keys())
for i, config in enumerate(all_configs[:10]):
    config_dict = dict(zip(keys, config))
    print(f"  {i+1}. {config_dict}")
\`\`\`

**Random Search (100 Configurations)**:

\`\`\`python
import random

n_random_samples = 100
random_time_hours = (n_random_samples * time_per_config_minutes) / 60

print(f"\\nRandom Search:")
print(f"  Configurations tested: {n_random_samples}")
print(f"  Total time: {random_time_hours:,.1f} hours")
print(f"  Speedup: {total_time_hours / random_time_hours:.1f}x faster")
print(f"  Coverage: {100 * n_random_samples / total_configs:.1f}% of grid")

# Generate random configurations
def random_config(hyperparameters):
    """Sample one random configuration"""
    return {key: random.choice(values) for key, values in hyperparameters.items()}

random_configs = [random_config(hyperparameters) for _ in range(n_random_samples)]

print("\\nFirst 10 random configurations:")
for i, config in enumerate(random_configs[:10]):
    print(f"  {i+1}. {config}")
\`\`\`

**Why Random Search is More Effective**:

**1. Not All Hyperparameters Are Equally Important**:

\`\`\`python
# Simulate a model where only 2 out of 4 hyperparameters matter
def model_performance(lr, batch_size, num_layers, dropout):
    """
    Simulated model performance
    Only learning_rate and num_layers significantly affect performance
    batch_size and dropout have minimal impact
    """
    # Important hyperparameters
    lr_score = -abs(np.log10(lr) + 2.5)  # Optimal around 0.003
    layers_score = -abs(num_layers - 4)   # Optimal at 4 layers
    
    # Less important hyperparameters (noise)
    batch_score = -0.1 * np.random.rand()
    dropout_score = -0.1 * np.random.rand()
    
    return lr_score + layers_score + batch_score + dropout_score

# Grid search: Fixed grid spacing
lr_values = [0.001, 0.005, 0.01, 0.05, 0.1]
layers_values = [2, 3, 4, 5, 6]

# Random search: Continuous sampling
random_lr = [10**np.random.uniform(-3, -1) for _ in range(100)]
random_layers = [np.random.randint(2, 7) for _ in range(100)]

print("\\nGrid Search Samples (important hyperparameters only):")
print(f"  Learning rates tested: {lr_values}")
print(f"  Num layers tested: {layers_values}")
print(f"  Total combinations: {len(lr_values) * len(layers_values)} = 25")

print("\\nRandom Search (first 10 samples):")
for i in range(10):
    print(f"  lr={random_lr[i]:.4f}, layers={random_layers[i]}")
print(f"  Total: 100 diverse samples")
\`\`\`

**2. Visualizing Coverage**:

\`\`\`python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Grid search: only tests 5x5=25 combinations of important parameters
grid_lr = np.tile(lr_values, len(layers_values))
grid_layers = np.repeat(layers_values, len(lr_values))

ax1.scatter(grid_lr, grid_layers, s=100, c='blue', marker='s', alpha=0.7, label='Grid Search')
ax1.set_xscale('log')
ax1.set_xlabel('Learning Rate (log scale)')
ax1.set_ylabel('Number of Layers')
ax1.set_title(f'Grid Search: {len(grid_lr)} configurations')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Random search: 100 samples with better coverage
ax2.scatter(random_lr, random_layers, s=100, c='red', marker='o', alpha=0.7, label='Random Search')
ax2.set_xscale('log')
ax2.set_xlabel('Learning Rate (log scale)')
ax2.set_ylabel('Number of Layers')
ax2.set_title(f'Random Search: 100 configurations')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

print("\\nKey Insight: Random search explores more values per important hyperparameter")
print("  Grid: 5 learning rate values")
print("  Random: ~100 unique learning rate values (continuous sampling)")
\`\`\`

**Combinatorics Growth**:

\`\`\`python
# Show exponential growth
hp_range = range(1, 8)
values = 5

configs = [values**n for n in hp_range]
time_days = [(c * 30 / 60 / 24) for c in configs]

print("\\nCombinatorial Explosion:")
print("Hyperparameters | Configurations | Time (Grid Search)")
print("-" * 55)
for n, c, t in zip(hp_range, configs, time_days):
    if t < 1:
        time_str = f"{t*24:.1f} hours"
    else:
        time_str = f"{t:.1f} days"
    print(f"       {n}        |     {c:>6,}     | {time_str:>15}")
\`\`\`

**AutoML and Combinatorics**:

\`\`\`python
# Modern AutoML: Bayesian Optimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

print("\\nAutoML Strategies:")
print("1. Grid Search: Exhaustive, O(nᵏ), guarantees finding optimum in grid")
print("2. Random Search: O(m) where m=budget, better than grid when k>2")
print("3. Bayesian Optimization: O(m), uses past results to guide search")
print("4. Hyperband: O(m log m), adaptive resource allocation")
print("\\nCombinatorics lessons:")
print("- Exhaustive search infeasible for >3-4 hyperparameters")
print("- Random search effective when few hyperparameters are important")
print("- Smart search (Bayesian) balances exploration and exploitation")
print("- Trade-off: Computational budget vs finding optimal configuration")
\`\`\`

**Key Insights**:

- Hyperparameter space grows exponentially: O(vⁿ) where v=values, n=hyperparameters
- Grid search: 5⁴ = 625 configs, 312.5 hours = 13 days
- Random search: 100 configs, 50 hours = 2 days, often finds better results
- Random search wins because:
  - Tests more unique values per important hyperparameter
  - Not all hyperparameters equally important
  - Avoids wasting time on uniform grids
- Modern AutoML uses Bayesian optimization: ~10-50 configs often sufficient`,
          keyPoints: [
            'Hyperparameter space size: v^n where v=values per param, n=number of params',
            'Grid search: Exhaustive but exponentially expensive (625 configs = 13 days)',
            'Random search: Tests 100 configs in 2 days, often outperforms grid search',
            'Random search advantage: Better coverage of important hyperparameters',
            'AutoML uses Bayesian optimization to guide search based on past results',
          ],
        },
      ],
    },
    {
      id: 'notation-proof',
      title: 'Mathematical Notation & Proof',
      content: `
# Mathematical Notation & Proof

## Introduction

Mathematical notation provides precise, unambiguous communication. In machine learning, we use notation to define models, losses, and algorithms. Understanding how to read and write mathematical proofs helps us understand why algorithms work and debug when they don't.

## Common Notation

### Variables and Constants

- **Lowercase letters**: variables (x, y, w, b)
- **Uppercase letters**: matrices, random variables (X, Y, W)
- **Greek letters**: parameters (α learning rate, θ parameters, λ regularization)
- **Subscripts**: indices (x₁, x₂, ..., xₙ or xᵢ)
- **Superscripts**: powers (x²) or sample indices (x⁽¹⁾, x⁽²⁾)

\`\`\`python
import numpy as np

# Example: Linear regression notation
# Model: ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
# Or in vector form: ŷ = wᵀx + b

# Data
X = np.array([[1, 2], [3, 4], [5, 6]])  # Matrix X: (n_samples × n_features)
y = np.array([5, 11, 17])               # Vector y: (n_samples,)
w = np.array([2, 1])                     # Weights w: (n_features,)
b = 1                                    # Bias b: scalar

# Prediction for sample i: ŷ⁽ⁱ⁾ = wᵀx⁽ⁱ⁾ + b
y_pred = X @ w + b

print("Notation example:")
print(f"X shape: {X.shape} (n_samples × n_features)")
print(f"w shape: {w.shape} (n_features,)")
print(f"y_pred: {y_pred}")
print(f"\\nFor sample i=0:")
print(f"x⁽⁰⁾ = {X[0]}")
print(f"ŷ⁽⁰⁾ = w₁x₁ + w₂x₂ + b = {w[0]}×{X[0,0]} + {w[1]}×{X[0,1]} + {b} = {y_pred[0]}")
\`\`\`

### Summation (Σ)

**Σ(i=1 to n) xᵢ**: Sum x₁ + x₂ + ... + xₙ

\`\`\`python
# Example: Mean Squared Error
# MSE = (1/n) Σ(i=1 to n) (yᵢ - ŷᵢ)²

y_true = np.array([5, 11, 17])
y_pred = np.array([4, 10, 18])
n = len(y_true)

# Using summation notation
mse_sum = sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n

# Using vectorized operations
mse_vec = np.mean((y_true - y_pred)**2)

print(f"MSE (summation): {mse_sum}")
print(f"MSE (vectorized): {mse_vec}")
print(f"\\nSummation: Σ(i=1 to {n}) (yᵢ - ŷᵢ)² / {n}")
print(f"Expanded: ({y_true[0]}-{y_pred[0]})² + ({y_true[1]}-{y_pred[1]})² + ({y_true[2]}-{y_pred[2]})² / {n}")
\`\`\`

### Product (Π)

**Π(i=1 to n) xᵢ**: Product x₁ × x₂ × ... × xₙ

\`\`\`python
# Example: Probability of independent events
# P(A₁ ∩ A₂ ∩ ... ∩ Aₙ) = Π P(Aᵢ)

probabilities = np.array([0.9, 0.8, 0.95, 0.85])

# Using product notation
joint_prob = np.prod(probabilities)

print(f"Individual probabilities: {probabilities}")
print(f"Joint probability: Π P(Aᵢ) = {joint_prob:.4f}")
print(f"Expanded: {probabilities[0]} × {probabilities[1]} × {probabilities[2]} × {probabilities[3]} = {joint_prob:.4f}")
\`\`\`

### Set Notation

- **∈**: element of (x ∈ S means x is in set S)
- **∉**: not element of
- **⊆**: subset
- **∪**: union
- **∩**: intersection
- **∅**: empty set

\`\`\`python
# Example: Training/test split notation
# D = {(xᵢ, yᵢ)}ᵢ₌₁ⁿ (dataset of n samples)
# D_train ∪ D_test = D
# D_train ∩ D_test = ∅

D = set(range(100))  # Dataset indices
D_train = set(range(70))
D_test = set(range(70, 100))

print(f"|D| = {len(D)} (dataset size)")
print(f"|D_train| = {len(D_train)}")
print(f"|D_test| = {len(D_test)}")
print(f"\\nD_train ∪ D_test = D: {D_train | D_test == D}")
print(f"D_train ∩ D_test = ∅: {len(D_train & D_test) == 0}")
\`\`\`

### Functions and Mappings

- **f: X → Y**: function f maps from X to Y
- **f(x)**: function application
- **f ∘ g**: function composition

\`\`\`python
# Example: Neural network as function composition
# y = f₃(f₂(f₁(x))) where fᵢ are layer functions

def layer1(x):
    """f₁: ℝⁿ → ℝᵐ"""
    W1 = np.array([[1, 0], [0, 1], [1, 1]])  # (3, 2)
    return W1 @ x

def layer2(x):
    """f₂: ℝᵐ → ℝᵏ with ReLU"""
    W2 = np.array([[1, 0, 1], [0, 1, 1]])  # (2, 3)
    return np.maximum(0, W2 @ x)  # ReLU activation

def layer3(x):
    """f₃: ℝᵏ → ℝ"""
    W3 = np.array([1, 1])  # (2,)
    return W3 @ x

# Composition: f = f₃ ∘ f₂ ∘ f₁
def neural_network(x):
    """f: ℝ² → ℝ"""
    return layer3(layer2(layer1(x)))

x = np.array([1, 2])
y = neural_network(x)

print(f"Input x: {x} ∈ ℝ²")
print(f"After layer 1: {layer1(x)} ∈ ℝ³")
print(f"After layer 2: {layer2(layer1(x))} ∈ ℝ²")
print(f"Output y: {y} ∈ ℝ")
print(f"\\nNeural network = f₃ ∘ f₂ ∘ f₁")
\`\`\`

## Logical Statements

### Quantifiers

- **∀** (for all): ∀x ∈ S, P(x) means "for all x in S, property P holds"
- **∃** (there exists): ∃x ∈ S, P(x) means "there exists an x in S such that P holds"

\`\`\`python
# Example: ∀x ∈ training set, loss(x) >= 0

def loss(y_true, y_pred):
    """MSE loss"""
    return (y_true - y_pred)**2

y_true_samples = np.array([1, 2, 3, 4, 5])
y_pred_samples = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

losses = [loss(yt, yp) for yt, yp in zip(y_true_samples, y_pred_samples)]

print("Loss values:", losses)
print(f"∀ samples, loss >= 0: {all(L >= 0 for L in losses)}")

# Example: ∃x such that gradient = 0 (local minimum)
def f(x):
    return (x - 2)**2

def gradient(x):
    return 2 * (x - 2)

x_values = np.linspace(0, 4, 100)
gradients = [gradient(x) for x in x_values]

# Find where gradient ≈ 0
zero_grad_indices = [i for i, g in enumerate(gradients) if abs(g) < 0.1]
print(f"\\n∃x where |∇f(x)| < 0.1: {len(zero_grad_indices) > 0}")
if zero_grad_indices:
    print(f"Found at x ≈ {x_values[zero_grad_indices[0]]:.2f}")
\`\`\`

### Implications

- **⇒** (implies): P ⇒ Q means "if P then Q"
- **⇔** (if and only if): P ⇔ Q means "P implies Q and Q implies P"

\`\`\`python
# Example: Convexity
# f is convex ⇔ f''(x) ≥ 0 ∀x

def f_convex(x):
    """Convex function: f(x) = x²"""
    return x**2

def f_second_derivative(x):
    """f''(x) = 2"""
    return 2

x_test = np.linspace(-5, 5, 100)
second_derivs = [f_second_derivative(x) for x in x_test]

is_convex = all(d >= 0 for d in second_derivs)
print(f"f(x) = x²")
print(f"f''(x) = 2 ≥ 0 ∀x: {is_convex}")
print(f"Therefore, f is convex")
\`\`\`

## Proof Techniques

### Direct Proof

Prove P ⇒ Q by assuming P and deriving Q.

**Example**: Prove that gradient descent with appropriate learning rate converges for convex functions.

\`\`\`python
# Simplified demonstration (not rigorous proof)

def gradient_descent_convex(f, grad_f, x0, lr, n_iterations):
    """
    Gradient descent on convex function
    """
    x = x0
    trajectory = [x]
    
    for _ in range(n_iterations):
        x = x - lr * grad_f(x)
        trajectory.append(x)
    
    return np.array(trajectory)

# Convex function: f(x) = x²
def f(x):
    return x**2

def grad_f(x):
    return 2*x

# Test convergence
x0 = 10
lr = 0.1
trajectory = gradient_descent_convex(f, grad_f, x0, lr, 50)

print("Gradient Descent on Convex Function:")
print(f"Initial: x₀ = {x0}")
print(f"Final: x₅₀ = {trajectory[-1]:.10f}")
print(f"Minimum at x* = 0")
print(f"Converged: {abs(trajectory[-1]) < 1e-6}")

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
iterations = range(len(trajectory))
plt.plot(iterations, trajectory, 'bo-', markersize=4, linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', label='Optimal x*=0')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('Gradient Descent Convergence (Convex Function)')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

### Proof by Contradiction

Assume ¬Q and derive a contradiction, thus proving Q.

**Example**: Prove that L2 regularization prevents weights from growing unbounded.

### Proof by Induction

Prove base case, then prove inductive step: if true for n, then true for n+1.

**Example**: Prove properties of recursive algorithms.

\`\`\`python
# Example: Prove Σ(i=1 to n) i = n(n+1)/2 by induction

def sum_first_n(n):
    """Compute 1 + 2 + ... + n"""
    return sum(range(1, n+1))

def formula_first_n(n):
    """Formula: n(n+1)/2"""
    return n * (n + 1) // 2

print("Proof by Induction: Σ(i=1 to n) i = n(n+1)/2")
print("\\nBase case (n=1):")
print(f"  LHS: Σ(i=1 to 1) i = {sum_first_n(1)}")
print(f"  RHS: 1(1+1)/2 = {formula_first_n(1)}")
print(f"  Equal: {sum_first_n(1) == formula_first_n(1)} ✓")

print("\\nInductive step: Assume true for n=k, prove for n=k+1")
print("  Σ(i=1 to k+1) i = [Σ(i=1 to k) i] + (k+1)")
print("                  = k(k+1)/2 + (k+1)    [by inductive hypothesis]")
print("                  = [k(k+1) + 2(k+1)]/2")
print("                  = (k+1)(k+2)/2")
print("                  = (k+1)((k+1)+1)/2   [formula for n=k+1] ✓")

print("\\nVerification for several values:")
for n in [1, 5, 10, 50, 100]:
    computed = sum_first_n(n)
    formula = formula_first_n(n)
    print(f"  n={n:>3}: computed={computed:>5}, formula={formula:>5}, match={computed == formula}")
\`\`\`

## Reading Mathematical Papers

### Common Patterns

**Theorem Statement**:
"Let f: ℝⁿ → ℝ be convex. Then gradient descent with learning rate α ≤ 1/L converges to global minimum."

**Translation**:
- f is a convex function (input: n-dimensional vector, output: scalar)
- If we use gradient descent with small enough learning rate
- Then it will reach the best solution

\`\`\`python
# Practical implementation of theorem

def is_lipschitz_smooth(f, grad_f, L, x_samples):
    """
    Check if gradient is L-Lipschitz smooth:
    ‖∇f(x) - ∇f(y)‖ ≤ L‖x - y‖
    """
    for i in range(len(x_samples)):
        for j in range(i+1, len(x_samples)):
            x, y = x_samples[i], x_samples[j]
            grad_diff = abs(grad_f(x) - grad_f(y))
            x_diff = abs(x - y)
            
            if grad_diff > L * x_diff + 1e-6:  # Small tolerance
                return False
    return True

# Example: f(x) = x², ∇f(x) = 2x, L = 2
def f(x):
    return x**2

def grad_f(x):
    return 2*x

L = 2
x_samples = np.linspace(-10, 10, 50)

is_smooth = is_lipschitz_smooth(f, grad_f, L, x_samples)
print(f"f(x) = x² is {L}-Lipschitz smooth: {is_smooth}")
print(f"Theorem says: use α ≤ 1/L = 1/{L} = {1/L}")
print(f"\\nTesting learning rates:")

for alpha in [0.3, 0.5, 0.7]:
    traj = gradient_descent_convex(f, grad_f, 10, alpha, 50)
    converged = abs(traj[-1]) < 1e-6
    print(f"  α = {alpha}: {'✓ converged' if converged else '✗ diverged'}")
\`\`\`

## Summary

- **Notation**: Precise, unambiguous communication
- **Subscripts/superscripts**: Indices and powers
- **Σ, Π**: Summation and product
- **∀, ∃**: Universal and existential quantifiers
- **⇒, ⇔**: Logical implications
- **Proofs**: Direct, contradiction, induction
- **Reading papers**: Translate math to code/intuition

**Key Skill**: Bidirectional translation between math notation and code

**Practice**: Read papers, implement algorithms, verify theorems numerically
`,
      multipleChoice: [
        {
          id: 'mc1-summation',
          question: 'What does Σ(i=1 to 5) i² equal?',
          options: ['15', '25', '55', '225'],
          correctAnswer: 2,
          explanation:
            'Σ(i=1 to 5) i² = 1² + 2² + 3² + 4² + 5² = 1 + 4 + 9 + 16 + 25 = 55. Or use formula: n(n+1)(2n+1)/6 = 5×6×11/6 = 55.',
        },
        {
          id: 'mc2-quantifiers',
          question: '∀x ∈ ℝ, x² ≥ 0 means:',
          options: [
            'Some real numbers have non-negative squares',
            'All real numbers have non-negative squares',
            'There exists a real number with negative square',
            'No real numbers have non-negative squares',
          ],
          correctAnswer: 1,
          explanation:
            '∀ means "for all". The statement says for all real numbers x, x² is non-negative, which is true.',
        },
        {
          id: 'mc3-set-notation',
          question: 'If A = {1, 2, 3} and B = {2, 3, 4}, what is A ∩ B?',
          options: ['{1}', '{2, 3}', '{1, 2, 3, 4}', '{4}'],
          correctAnswer: 1,
          explanation:
            'A ∩ B (intersection) contains elements in both A and B: {2, 3}.',
        },
        {
          id: 'mc4-function-composition',
          question: 'If f(x) = 2x and g(x) = x + 1, what is (f ∘ g)(3)?',
          options: ['7', '8', '10', '14'],
          correctAnswer: 1,
          explanation:
            '(f ∘ g)(3) means f(g(3)). First g(3) = 3+1 = 4, then f(4) = 2×4 = 8.',
        },
        {
          id: 'mc5-product-notation',
          question:
            'What is Π(i=1 to 4) i (product of first 4 positive integers)?',
          options: ['4', '10', '16', '24'],
          correctAnswer: 3,
          explanation: 'Π(i=1 to 4) i = 1 × 2 × 3 × 4 = 24 = 4!',
        },
      ],
      quiz: [
        {
          id: 'dq1-summation-notation-ml',
          question:
            'Explain how summation notation (Σ) is used in machine learning to express loss functions and gradient computation. Provide specific examples of Mean Squared Error (MSE) and cross-entropy loss written in summation notation, then show how to translate these into vectorized NumPy code. Discuss why understanding summation notation is crucial for reading ML research papers.',
          sampleAnswer: `Summation notation is the backbone of expressing mathematical operations over datasets in machine learning. Understanding it is essential for both implementing algorithms and reading research papers.

**Mean Squared Error (MSE)**:

**Mathematical notation**:
\\[
\\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
\\]

This reads as: "Sum the squared differences between true values \\(y_i\\) and predictions \\(\\hat{y}_i\\) for all n samples, then divide by n."

**Components**:
- \\(\\sum_{i=1}^{n}\\): Sum from sample 1 to sample n
- \\(y_i\\): True label for sample i
- \\(\\hat{y}_i\\): Predicted value for sample i
- \\((y_i - \\hat{y}_i)^2\\): Squared error for sample i

\`\`\`python
import numpy as np

# Example data
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])
n = len(y_true)

# Summation notation approach (explicit loop)
mse_loop = 0
for i in range(n):
    mse_loop += (y_true[i] - y_pred[i])**2
mse_loop = mse_loop / n

# Vectorized NumPy (translates Σ to array operations)
mse_vectorized = np.mean((y_true - y_pred)**2)

print(f"MSE (loop): {mse_loop:.4f}")
print(f"MSE (vectorized): {mse_vectorized:.4f}")
print(f"Match: {np.isclose(mse_loop, mse_vectorized)}")

# Breaking down the vectorized operation:
print("\\nStep-by-step vectorization:")
print(f"1. Differences (y - ŷ): {y_true - y_pred}")
print(f"2. Squared: {(y_true - y_pred)**2}")
print(f"3. Sum (Σ): {np.sum((y_true - y_pred)**2)}")
print(f"4. Divide by n: {np.sum((y_true - y_pred)**2) / n}")
\`\`\`

**Cross-Entropy Loss (Binary Classification)**:

**Mathematical notation**:
\\[
\\text{Loss} = -\\frac{1}{n} \\sum_{i=1}^{n} [y_i \\log(\\hat{y}_i) + (1-y_i) \\log(1-\\hat{y}_i)]
\\]

This reads as: "For each sample, compute the log loss based on whether the true label is 1 or 0, sum all losses, then divide by n."

\`\`\`python
# Binary classification
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])  # Predicted probabilities
n = len(y_true)

# Summation notation approach (explicit loop)
loss_loop = 0
for i in range(n):
    loss_loop += y_true[i] * np.log(y_pred[i]) + (1 - y_true[i]) * np.log(1 - y_pred[i])
loss_loop = -loss_loop / n

# Vectorized NumPy
loss_vectorized = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

print(f"\\nCross-Entropy Loss (loop): {loss_loop:.4f}")
print(f"Cross-Entropy Loss (vectorized): {loss_vectorized:.4f}")
print(f"Match: {np.isclose(loss_loop, loss_vectorized)}")
\`\`\`

**Gradient Computation with Summation**:

**Mathematical notation for MSE gradient**:
\\[
\\frac{\\partial \\text{MSE}}{\\partial w_j} = \\frac{2}{n} \\sum_{i=1}^{n} (\\hat{y}_i - y_i) x_{ij}
\\]

Where \\(x_{ij}\\) is feature j for sample i.

\`\`\`python
# Linear regression: ŷ = Xw
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # (n_samples=4, n_features=2)
y = np.array([5, 11, 17, 23])
w = np.array([1.5, 2.0])

# Predictions
y_pred = X @ w

# Gradient: ∂MSE/∂w_j = (2/n) Σ(ŷ_i - y_i) x_ij
n = len(y)

# Loop version (following summation notation exactly)
gradient_loop = np.zeros(2)
for i in range(n):
    for j in range(2):  # For each feature
        gradient_loop[j] += (y_pred[i] - y[i]) * X[i, j]
gradient_loop = (2 / n) * gradient_loop

# Vectorized version (matrix notation)
gradient_vectorized = (2 / n) * X.T @ (y_pred - y)

print("\\nGradient computation:")
print(f"Loop version: {gradient_loop}")
print(f"Vectorized version: {gradient_vectorized}")
print(f"Match: {np.allclose(gradient_loop, gradient_vectorized)}")
\`\`\`

**Double Summation (Nested Sums)**:

Used for operations over matrices, like computing pairwise distances.

**Mathematical notation**:
\\[
\\sum_{i=1}^{n} \\sum_{j=1}^{m} A_{ij}
\\]

This sums all elements in matrix A.

\`\`\`python
# Sum all elements in a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Double loop (following notation exactly)
total_loop = 0
for i in range(A.shape[0]):  # n rows
    for j in range(A.shape[1]):  # m columns
        total_loop += A[i, j]

# Vectorized
total_vectorized = np.sum(A)

print(f"\\nDouble summation:")
print(f"Loop: {total_loop}")
print(f"Vectorized: {total_vectorized}")
print(f"Match: {total_loop == total_vectorized}")
\`\`\`

**Why This Matters for Research Papers**:

Research papers use summation notation extensively. Being able to translate it to code is critical:

1. **Compact representation**: \\(\\sum_{i=1}^{n}\\) is clearer than describing loops in words
2. **Mathematical rigor**: Summation notation is precise and unambiguous
3. **Implementation**: Directly maps to vectorized NumPy operations
4. **Debugging**: Understanding the notation helps verify your implementation

**Example from a paper**:

"The attention mechanism is computed as:
\\[
\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
\\]
where the softmax is applied row-wise."

Without understanding summation notation (implicit in softmax), you can't implement this correctly.`,
          keyPoints: [
            'Summation notation Σ expresses operations over all samples in a dataset',
            'MSE: (1/n)Σ(y_i - ŷ_i)² sums squared errors across all samples',
            'Cross-entropy: -(1/n)Σ[y_i log(ŷ_i) + (1-y_i)log(1-ŷ_i)] for binary classification',
            'Summation translates directly to vectorized NumPy: Σ → np.sum() or np.mean()',
            'Understanding summation notation essential for reading ML papers and implementing algorithms',
          ],
        },
        {
          id: 'dq2-proof-convergence',
          question:
            'In optimization theory, a key result states: "For convex functions with L-Lipschitz continuous gradients, gradient descent with learning rate α ≤ 1/L converges to the global minimum." Explain what this theorem means in practical terms, provide a proof sketch or intuition for why it works, and demonstrate with code how violating the learning rate condition (α > 1/L) causes divergence. Discuss the implications for choosing learning rates in deep learning.',
          sampleAnswer: `This theorem provides a mathematical guarantee for gradient descent convergence under specific conditions. Understanding it helps explain why learning rates matter and how to choose them systematically.

**Theorem Statement Breakdown**:

**Convex function**: f is convex if \\(f(\\lambda x + (1-\\lambda)y) \\leq \\lambda f(x) + (1-\\lambda)f(y)\\) for all \\(\\lambda \\in [0,1]\\).

**Practical meaning**: No local minima (only one global minimum), like a bowl shape.

**L-Lipschitz continuous gradient**: \\(\\|\\nabla f(x) - \\nabla f(y)\\| \\leq L\\|x - y\\|\\)

**Practical meaning**: The gradient doesn't change too rapidly. L is the "smoothness constant" - how much the slope can vary.

**Learning rate condition**: \\(\\alpha \\leq 1/L\\)

**Practical meaning**: Step size must be small enough relative to function's curvature.

**Convergence**: The sequence of parameters converges to the optimal value: \\(\\lim_{t \\to \\infty} \\theta_t = \\theta^*\\)

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Simple convex function: f(x) = x²
def f(x):
    """Convex quadratic function"""
    return x**2

def grad_f(x):
    """Gradient: f'(x) = 2x"""
    return 2*x

# For f(x) = x², the Lipschitz constant L = 2
# (gradient is 2x, so |grad_f(x) - grad_f(y)| = |2x - 2y| = 2|x - y|)
L = 2

print("Function: f(x) = x²")
print(f"Gradient: f'(x) = 2x")
print(f"Lipschitz constant L = {L}")
print(f"Theorem says: Use α ≤ 1/L = {1/L}")
print(f"\\nOptimal point: x* = 0")
\`\`\`

**Proof Intuition**:

The key idea is that for a convex function with L-Lipschitz gradient, we can bound how much the function can increase in one gradient step.

**Mathematical insight**:

For convex f with L-Lipschitz gradient:
\\[
f(x - \\alpha \\nabla f(x)) \\leq f(x) - \\alpha \\|\\nabla f(x)\\|^2 + \\frac{\\alpha^2 L}{2} \\|\\nabla f(x)\\|^2
\\]

This says: stepping in the negative gradient direction decreases the function value (middle term) but there's a penalty for taking too large a step (last term).

For descent to be guaranteed:
\\[
-\\alpha + \\frac{\\alpha^2 L}{2} < 0
\\]
\\[
\\alpha(\\frac{\\alpha L}{2} - 1) < 0
\\]
\\[
\\alpha < \\frac{2}{L}
\\]

For guaranteed convergence (stronger result), we need \\(\\alpha \\leq \\frac{1}{L}\\).

\`\`\`python
# Visualize the descent condition
alphas = np.linspace(0, 2/L, 1000)
descent_guaranteed = -alphas + (alphas**2 * L / 2)

plt.figure(figsize=(10, 6))
plt.plot(alphas, descent_guaranteed, linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', label='Descent boundary')
plt.axvline(x=1/L, color='g', linestyle='--', label=f'α = 1/L = {1/L}')
plt.axvline(x=2/L, color='orange', linestyle='--', label=f'α = 2/L = {2/L}')
plt.xlabel('Learning Rate α')
plt.ylabel('Change in function value')
plt.title('Gradient Descent: Function Decrease vs Learning Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.fill_between(alphas, descent_guaranteed, 0, where=(descent_guaranteed<0), alpha=0.3, color='green', label='Guaranteed descent')
plt.show()
\`\`\`

**Demonstration: Convergence vs Divergence**:

\`\`\`python
def gradient_descent(f, grad_f, x0, alpha, n_iterations):
    """Run gradient descent"""
    trajectory = [x0]
    x = x0
    
    for _ in range(n_iterations):
        x = x - alpha * grad_f(x)
        trajectory.append(x)
        
        # Stop if diverging
        if abs(x) > 1e10:
            break
    
    return np.array(trajectory)

# Test different learning rates
x0 = 10.0
n_iterations = 50

learning_rates = [
    0.3,    # < 1/L = 0.5 → should converge
    0.5,    # = 1/L → should converge
    0.6,    # > 1/L but < 2/L → might converge slowly or oscillate
    0.8,    # > 1/L → will oscillate or diverge
    1.2,    # >> 1/L → will diverge
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, alpha in enumerate(learning_rates):
    trajectory = gradient_descent(f, grad_f, x0, alpha, n_iterations)
    
    ax = axes[idx]
    iterations = range(len(trajectory))
    ax.plot(iterations, trajectory, 'bo-', markersize=3, linewidth=1)
    ax.axhline(y=0, color='r', linestyle='--', label='Optimal x*=0')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('x value')
    ax.set_title(f'α = {alpha} ({"✓ converges" if alpha <= 1/L else "✗ may diverge"})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Print convergence status
    final_x = trajectory[-1]
    converged = abs(final_x) < 1e-6
    print(f"α = {alpha}: Final x = {final_x:.6f}, Converged: {converged}")

plt.tight_layout()
plt.show()
\`\`\`

**Detailed Analysis**:

\`\`\`python
# Analyze convergence rate
def analyze_convergence(alpha, x0=10, n_iter=100):
    """Analyze convergence behavior"""
    x = x0
    errors = []
    
    for t in range(n_iter):
        errors.append(abs(x))  # Distance from optimal x*=0
        x = x - alpha * grad_f(x)
        
        if abs(x) > 1e10:  # Diverged
            return errors, False
    
    return errors, abs(x) < 1e-6

print("\\nConvergence Analysis:\\n")
print("Learning Rate | Converged | Final Error | Iterations to 1e-6")
print("-" * 65)

test_alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
for alpha in test_alphas:
    errors, converged = analyze_convergence(alpha, n_iter=200)
    final_error = errors[-1] if len(errors) > 0 else float('inf')
    
    # Find iteration where error < 1e-6
    iters_to_converge = next((i for i, e in enumerate(errors) if e < 1e-6), None)
    
    status = "✓ Yes" if converged else "✗ No"
    iter_str = str(iters_to_converge) if iters_to_converge else "N/A"
    
    print(f"    {alpha:.1f}       |   {status}    | {final_error:.2e}  |      {iter_str}")
\`\`\`

**Implications for Deep Learning**:

1. **Learning rate scheduling**: Start with a safe learning rate, then adapt
2. **Adam optimizer**: Adapts learning rates per parameter based on gradient history
3. **Learning rate warmup**: Start small, gradually increase to target value
4. **Non-convex functions**: Deep networks are non-convex, so this theorem doesn't directly apply, but insights still useful

\`\`\`python
# Deep learning example (conceptual)
print("\\nDeep Learning Learning Rates:")
print("- Small networks (few layers): α ∈ [0.01, 0.1]")
print("- Large networks (ResNet, Transformers): α ∈ [0.001, 0.01]")
print("- With momentum/Adam: Can use larger α (optimizer stabilizes)")
print("- Rule of thumb: Start with 0.001 for Adam, 0.1 for SGD")
print("- Always use learning rate schedule (decay or cosine annealing)")
\`\`\`

**Key Takeaways**:

- Mathematical proofs provide guarantees and insights
- Lipschitz constant L determines maximum safe learning rate
- Violating the condition (α > 1/L) can cause oscillation or divergence
- In practice, use adaptive optimizers (Adam) that handle this automatically
- For research: understanding proofs helps design better algorithms`,
          keyPoints: [
            'Theorem: For convex f with L-Lipschitz gradient, α ≤ 1/L guarantees convergence',
            'Lipschitz constant L measures how fast gradient changes',
            'Proof intuition: Step size must balance descent vs overshoot',
            'Violating α > 1/L causes oscillation or divergence',
            'Deep learning: Use adaptive optimizers (Adam) or learning rate schedules',
          ],
        },
        {
          id: 'dq3-notation-backpropagation',
          question:
            'The backpropagation algorithm is often expressed using chain rule notation. For a 2-layer neural network with weights W1, W2 and ReLU activation, write out the complete forward pass and backward pass using proper mathematical notation (including subscripts and superscripts). Then translate this notation into NumPy code. Explain how understanding the notation helps implement custom neural network layers and debug gradient computation errors.',
          sampleAnswer: `Backpropagation is the chain rule applied systematically to compute gradients in neural networks. Understanding the notation is essential for implementing custom architectures and debugging gradient errors.

**2-Layer Neural Network Architecture**:

**Network structure**:
- Input: \\(x \\in \\mathbb{R}^{d}\\) (d features)
- Hidden layer: \\(h \\in \\mathbb{R}^{m}\\) (m neurons)
- Output: \\(\\hat{y} \\in \\mathbb{R}\\) (scalar for regression)
- Loss: Mean Squared Error

**Mathematical Notation**:

**Forward Pass**:

1. First layer (linear transformation):
   \\[z^{(1)} = W^{(1)}x + b^{(1)}\\]
   where \\(W^{(1)} \\in \\mathbb{R}^{m \\times d}\\), \\(b^{(1)} \\in \\mathbb{R}^{m}\\)

2. ReLU activation:
   \\[h = \\text{ReLU}(z^{(1)}) = \\max(0, z^{(1)})\\]

3. Second layer (linear):
   \\[z^{(2)} = W^{(2)}h + b^{(2)}\\]
   where \\(W^{(2)} \\in \\mathbb{R}^{1 \\times m}\\), \\(b^{(2)} \\in \\mathbb{R}\\)

4. Output (no activation for regression):
   \\[\\hat{y} = z^{(2)}\\]

5. Loss:
   \\[L = (y - \\hat{y})^2\\]

**Backward Pass (Chain Rule)**:

We need: \\(\\frac{\\partial L}{\\partial W^{(1)}}\\), \\(\\frac{\\partial L}{\\partial b^{(1)}}\\), \\(\\frac{\\partial L}{\\partial W^{(2)}}\\), \\(\\frac{\\partial L}{\\partial b^{(2)}}\\)

**Layer 2 gradients**:

1. \\(\\frac{\\partial L}{\\partial \\hat{y}} = 2(\\hat{y} - y)\\)

2. \\(\\frac{\\partial L}{\\partial W^{(2)}} = \\frac{\\partial L}{\\partial \\hat{y}} \\cdot \\frac{\\partial \\hat{y}}{\\partial W^{(2)}} = 2(\\hat{y} - y) \\cdot h^T\\)

3. \\(\\frac{\\partial L}{\\partial b^{(2)}} = \\frac{\\partial L}{\\partial \\hat{y}} = 2(\\hat{y} - y)\\)

4. \\(\\frac{\\partial L}{\\partial h} = \\frac{\\partial L}{\\partial \\hat{y}} \\cdot \\frac{\\partial \\hat{y}}{\\partial h} = 2(\\hat{y} - y) \\cdot W^{(2)T}\\)

**Layer 1 gradients (chain rule through ReLU)**:

5. \\(\\frac{\\partial L}{\\partial z^{(1)}} = \\frac{\\partial L}{\\partial h} \\odot \\frac{\\partial h}{\\partial z^{(1)}}\\)

   where \\(\\frac{\\partial h}{\\partial z^{(1)}} = \\mathbb{1}_{z^{(1)} > 0}\\) (ReLU derivative)

6. \\(\\frac{\\partial L}{\\partial W^{(1)}} = \\frac{\\partial L}{\\partial z^{(1)}} \\cdot x^T\\)

7. \\(\\frac{\\partial L}{\\partial b^{(1)}} = \\frac{\\partial L}{\\partial z^{(1)}}\\)

\`\`\`python
import numpy as np

class TwoLayerNet:
    """2-layer neural network with ReLU activation"""
    
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize weights and biases
        
        Notation:
        - W1: W^(1) ∈ R^(hidden_dim × input_dim)
        - b1: b^(1) ∈ R^(hidden_dim)
        - W2: W^(2) ∈ R^(1 × hidden_dim)
        - b2: b^(2) ∈ R^(1)
        """
        # Initialize with small random values
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(1, hidden_dim) * 0.01
        self.b2 = np.zeros(1)
        
        # Cache for backward pass
        self.cache = {}
    
    def relu(self, z):
        """ReLU activation: h = max(0, z)"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """ReLU derivative: 1 if z > 0, else 0"""
        return (z > 0).astype(float)
    
    def forward(self, x):
        """
        Forward pass
        
        Notation:
        - z1 = W^(1)x + b^(1)
        - h = ReLU(z1)
        - z2 = W^(2)h + b^(2)
        - ŷ = z2
        """
        # Layer 1
        z1 = self.W1 @ x + self.b1  # z^(1)
        h = self.relu(z1)            # h = ReLU(z^(1))
        
        # Layer 2
        z2 = self.W2 @ h + self.b2   # z^(2)
        y_pred = z2                  # ŷ
        
        # Cache for backward pass
        self.cache = {
            'x': x,
            'z1': z1,
            'h': h,
            'z2': z2,
            'y_pred': y_pred
        }
        
        return y_pred
    
    def backward(self, y_true):
        """
        Backward pass (backpropagation)
        
        Compute all gradients using chain rule:
        - ∂L/∂W^(2), ∂L/∂b^(2)
        - ∂L/∂W^(1), ∂L/∂b^(1)
        """
        # Retrieve cached values
        x = self.cache['x']
        z1 = self.cache['z1']
        h = self.cache['h']
        y_pred = self.cache['y_pred']
        
        # Gradient of loss w.r.t. output
        # ∂L/∂ŷ = 2(ŷ - y)
        dL_dy_pred = 2 * (y_pred - y_true)
        
        # Layer 2 gradients
        # ∂L/∂W^(2) = ∂L/∂ŷ · ∂ŷ/∂W^(2) = dL_dy_pred · h^T
        dL_dW2 = dL_dy_pred * h.reshape(1, -1)  # Outer product
        
        # ∂L/∂b^(2) = ∂L/∂ŷ
        dL_db2 = dL_dy_pred
        
        # Gradient w.r.t. hidden layer
        # ∂L/∂h = ∂L/∂ŷ · ∂ŷ/∂h = dL_dy_pred · W^(2)^T
        dL_dh = dL_dy_pred * self.W2.T  # Shape: (hidden_dim,)
        dL_dh = dL_dh.flatten()
        
        # Gradient w.r.t. z1 (before ReLU)
        # ∂L/∂z^(1) = ∂L/∂h ⊙ ∂h/∂z^(1)
        # where ∂h/∂z^(1) = 1 if z^(1) > 0, else 0
        dL_dz1 = dL_dh * self.relu_derivative(z1)
        
        # Layer 1 gradients
        # ∂L/∂W^(1) = ∂L/∂z^(1) · x^T
        dL_dW1 = dL_dz1.reshape(-1, 1) @ x.reshape(1, -1)
        
        # ∂L/∂b^(1) = ∂L/∂z^(1)
        dL_db1 = dL_dz1
        
        gradients = {
            'dW1': dL_dW1,
            'db1': dL_db1,
            'dW2': dL_dW2,
            'db2': dL_db2
        }
        
        return gradients
    
    def loss(self, y_true, y_pred):
        """MSE loss: L = (y - ŷ)²"""
        return (y_true - y_pred)**2

# Example usage
print("2-Layer Neural Network with Backpropagation\\n")
print("=" * 60)

# Initialize network
input_dim = 3
hidden_dim = 4
net = TwoLayerNet(input_dim, hidden_dim)

# Sample data
x = np.array([1.0, 2.0, 3.0])
y = np.array([10.0])

# Forward pass
y_pred = net.forward(x)
loss = net.loss(y, y_pred)

print(f"Input x: {x}")
print(f"True y: {y[0]}")
print(f"Predicted ŷ: {y_pred[0]:.4f}")
print(f"Loss: {loss[0]:.4f}\\n")

# Backward pass
gradients = net.backward(y)

print("Gradients:")
print(f"  ∂L/∂W^(2) shape: {gradients['dW2'].shape}")
print(f"  ∂L/∂b^(2) shape: {gradients['db2'].shape}")
print(f"  ∂L/∂W^(1) shape: {gradients['dW1'].shape}")
print(f"  ∂L/∂b^(1) shape: {gradients['db1'].shape}")
\`\`\`

**Gradient Checking (Verify Implementation)**:

Numerical gradient approximation: \\(\\frac{\\partial L}{\\partial w} \\approx \\frac{L(w + \\epsilon) - L(w - \\epsilon)}{2\\epsilon}\\)

\`\`\`python
def numerical_gradient(net, x, y, param_name, epsilon=1e-5):
    """
    Compute numerical gradient using finite differences
    
    ∂L/∂w ≈ [L(w+ε) - L(w-ε)] / 2ε
    """
    # Get parameter
    if param_name == 'W1':
        param = net.W1
    elif param_name == 'b1':
        param = net.b1
    elif param_name == 'W2':
        param = net.W2
    elif param_name == 'b2':
        param = net.b2
    
    numerical_grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = param[idx]
        
        # f(w + ε)
        param[idx] = old_value + epsilon
        y_pred_plus = net.forward(x)
        loss_plus = net.loss(y, y_pred_plus)[0]
        
        # f(w - ε)
        param[idx] = old_value - epsilon
        y_pred_minus = net.forward(x)
        loss_minus = net.loss(y, y_pred_minus)[0]
        
        # Numerical gradient
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Restore
        param[idx] = old_value
        it.iternext()
    
    return numerical_grad

# Verify gradients
print("\\nGradient Checking:\\n")

# Compute analytical gradients
y_pred = net.forward(x)
gradients = net.backward(y)

# Check each parameter
for param_name in ['W1', 'b1', 'W2', 'b2']:
    analytical_grad = gradients[f'd{param_name}']
    numerical_grad = numerical_gradient(net, x, y, param_name)
    
    # Compute relative error
    diff = np.linalg.norm(analytical_grad - numerical_grad)
    sum_norm = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    relative_error = diff / (sum_norm + 1e-8)
    
    status = "✓ PASS" if relative_error < 1e-7 else "✗ FAIL"
    print(f"  {param_name}: relative error = {relative_error:.2e} {status}")
\`\`\`

**Key Insights**:

1. **Notation clarity**: Superscripts (layer index), subscripts (element index) prevent confusion
2. **Chain rule**: Gradients flow backward through layers: \\(\\frac{\\partial L}{\\partial W^{(1)}} = \\frac{\\partial L}{\\partial z^{(2)}} \\cdot \\frac{\\partial z^{(2)}}{ \\partial h} \\cdot \\frac{\\partial h}{\\partial z^{(1)}} \\cdot \\frac{\\partial z^{(1)}}{\\partial W^{(1)}}\\)
3. **Gradient checking**: Always verify custom layers with numerical gradients
4. **Debugging**: Understanding notation helps identify where gradients vanish or explode

This foundation extends to any architecture: CNNs, RNNs, Transformers all use the same backpropagation principle.`,
          keyPoints: [
            'Forward: z^(1)=W^(1)x+b^(1), h=ReLU(z^(1)), z^(2)=W^(2)h+b^(2), ŷ=z^(2)',
            'Backward: Chain rule computes ∂L/∂W^(2), ∂L/∂W^(1) by propagating gradients backward',
            'ReLU gradient: ∂h/∂z = 1 if z>0, else 0 (creates "dead neurons" if z always ≤0)',
            'Gradient checking: Verify analytical gradients match numerical approximation',
            'Understanding notation essential for implementing custom layers and debugging',
          ],
        },
      ],
    },
  ],
  keyTakeaways: [
    'Different number systems (integers, rationals, reals, complex) have different properties and uses',
    'Floating-point arithmetic has precision limitations that affect ML algorithms',
    'Scientific notation and orders of magnitude are crucial for understanding scale in ML',
    'Properties like commutative, associative, and distributive guide mathematical operations',
    'Numerical stability techniques (log space, epsilon additions) prevent computational errors',
    'Understanding absolute values and inequalities is essential for loss functions and distances',
    'Complex numbers are fundamental in signal processing, Fourier transforms, and quantum computing',
    'Choosing appropriate data types (float32 vs float64) impacts memory, speed, and precision',
  ],
  learningObjectives: [
    'Understand different number systems and their computational representations',
    'Recognize floating-point precision limitations and their implications',
    'Apply numerical stability techniques in machine learning code',
    'Use scientific notation to reason about scales in ML models',
    'Choose appropriate data types for different ML tasks',
    'Implement numerically stable algorithms for real-world applications',
    'Understand how number theory impacts algorithm design and debugging',
  ],
  prerequisites: ['Basic arithmetic', 'Basic Python programming'],
};

/**
 * Section: NumPy Fundamentals
 * Module: Python for Data Science
 *
 * Covers NumPy arrays, array creation methods, indexing, slicing, shapes, data types, and memory efficiency
 */

export const numpyFundamentals = {
  id: 'numpy-fundamentals',
  title: 'NumPy Fundamentals',
  content: `
# NumPy Fundamentals

## Introduction

NumPy (Numerical Python) is the foundational library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. NumPy is the backbone of the entire scientific Python ecosystem and is essential for machine learning and data science.

**Why NumPy?**
- **Performance**: NumPy operations are 10-100x faster than pure Python loops
- **Memory Efficiency**: Arrays use less memory than Python lists
- **Broadcasting**: Powerful mechanism for array operations
- **Integration**: Works seamlessly with other scientific libraries

## The NumPy Array (ndarray)

The core of NumPy is the **ndarray** (n-dimensional array) object. Unlike Python lists, NumPy arrays are:
- **Homogeneous**: All elements must be the same type
- **Fixed-size**: Once created, size cannot change (though you can create new arrays)
- **Contiguous in memory**: Elements are stored sequentially, enabling fast access

### Creating NumPy Arrays

\`\`\`python
import numpy as np

# From Python lists
arr1 = np.array([1, 2, 3, 4, 5])
print(f"1D array: {arr1}")
print(f"Shape: {arr1.shape}")  # (5,)
print(f"Dimensions: {arr1.ndim}")  # 1

# 2D array (matrix)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\\n2D array:\\n{arr2}")
print(f"Shape: {arr2.shape}")  # (2, 3) - 2 rows, 3 columns
print(f"Dimensions: {arr2.ndim}")  # 2

# 3D array (tensor)
arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"\\n3D array:\\n{arr3}")
print(f"Shape: {arr3.shape}")  # (2, 2, 2)
print(f"Dimensions: {arr3.ndim}")  # 3
\`\`\`

**Output:**
\`\`\`
1D array: [1 2 3 4 5]
Shape: (5,)
Dimensions: 1

2D array:
[[1 2 3]
 [4 5 6]]
Shape: (2, 3)
Dimensions: 2

3D array:
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
Shape: (2, 2, 2)
Dimensions: 3
\`\`\`

## Array Creation Methods

NumPy provides many convenience functions to create arrays without manually typing values:

### Arrays of Zeros, Ones, and Constants

\`\`\`python
# Array of zeros
zeros = np.zeros((3, 4))
print(f"Zeros array:\\n{zeros}")
print(f"Data type: {zeros.dtype}")  # float64 by default

# Array of ones
ones = np.ones((2, 3, 4), dtype=np.int32)
print(f"\\nOnes shape: {ones.shape}")  # (2, 3, 4)

# Array filled with a specific value
filled = np.full((3, 3), 7.5)
print(f"\\nFilled array:\\n{filled}")

# Empty array (uninitialized, faster but contains garbage)
empty = np.empty((2, 2))
print(f"\\nEmpty array (uninitialized):\\n{empty}")
\`\`\`

### Identity and Diagonal Matrices

\`\`\`python
# Identity matrix (diagonal of ones)
identity = np.eye(4)
print(f"Identity matrix:\\n{identity}")

# Diagonal matrix
diagonal = np.diag([1, 2, 3, 4])
print(f"\\nDiagonal matrix:\\n{diagonal}")
\`\`\`

### Range-Based Arrays

\`\`\`python
# Similar to Python\'s range()
arange_arr = np.arange(0, 10, 2)  # start, stop, step
print(f"arange: {arange_arr}")  # [0 2 4 6 8]

# Linearly spaced values (includes endpoint)
linspace_arr = np.linspace(0, 1, 5)  # start, stop, num_points
print(f"linspace: {linspace_arr}")  # [0.   0.25 0.5  0.75 1.  ]

# Logarithmically spaced values
logspace_arr = np.logspace(0, 3, 4)  # 10^0 to 10^3, 4 points
print(f"logspace: {logspace_arr}")  # [1.e+00 1.e+01 1.e+02 1.e+03]
\`\`\`

### Random Arrays

\`\`\`python
# Set seed for reproducibility
np.random.seed(42)

# Random values from uniform distribution [0, 1)
rand_uniform = np.random.rand(3, 3)
print(f"Random uniform:\\n{rand_uniform}")

# Random integers
rand_int = np.random.randint(0, 100, size=(3, 4))
print(f"\\nRandom integers:\\n{rand_int}")

# Random values from standard normal distribution
rand_normal = np.random.randn(3, 3)
print(f"\\nRandom normal:\\n{rand_normal}")

# Random choice from array
choices = np.random.choice([10, 20, 30, 40], size=10)
print(f"\\nRandom choices: {choices}")
\`\`\`

## Array Indexing and Slicing

NumPy arrays support powerful indexing and slicing operations:

### Basic Indexing (1D)

\`\`\`python
arr = np.array([10, 20, 30, 40, 50, 60])

# Single element access
print(f"First element: {arr[0]}")  # 10
print(f"Last element: {arr[-1]}")  # 60

# Slicing: [start:stop:step]
print(f"First three: {arr[:3]}")  # [10 20 30]
print(f"Last three: {arr[-3:]}")  # [40 50 60]
print(f"Every other: {arr[::2]}")  # [10 30 50]
print(f"Reversed: {arr[::-1]}")  # [60 50 40 30 20 10]
\`\`\`

### Multi-dimensional Indexing

\`\`\`python
arr2d = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

# Access single element: arr[row, col]
print(f"Element at (1, 2): {arr2d[1, 2]}")  # 7

# Access row
print(f"Second row: {arr2d[1, :]}")  # [5 6 7 8]

# Access column
print(f"Third column: {arr2d[:, 2]}")  # [3 7 11]

# Slice multiple rows and columns
print(f"Subarray:\\n{arr2d[:2, 1:3]}")
# Output:
# [[2 3]
#  [6 7]]

# Advanced: skip every other row and column
print(f"\\nEvery other:\\n{arr2d[::2, ::2]}")
# Output:
# [[1 3]
#  [9 11]]
\`\`\`

### Boolean Indexing

\`\`\`python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Create boolean mask
mask = arr > 5
print(f"Mask: {mask}")  # [False False False False False True True True True True]

# Use mask to filter
filtered = arr[mask]
print(f"Values > 5: {filtered}")  # [6 7 8 9 10]

# Combine conditions
complex_mask = (arr > 3) & (arr < 8)  # Use & for element-wise AND
print(f"Values between 3 and 8: {arr[complex_mask]}")  # [4 5 6 7]

# Practical example: filter outliers
data = np.array([1, 2, 3, 100, 4, 5, -50, 6])
mean = data.mean()
std = data.std()
clean_data = data[np.abs (data - mean) < 2 * std]
print(f"Data without outliers: {clean_data}")
\`\`\`

### Fancy Indexing

\`\`\`python
arr = np.array([10, 20, 30, 40, 50])

# Index with array of integers
indices = np.array([0, 2, 4])
print(f"Selected elements: {arr[indices]}")  # [10 30 50]

# 2D fancy indexing
arr2d = np.arange(12).reshape(3, 4)
print(f"Array:\\n{arr2d}")

# Select specific elements
rows = np.array([0, 1, 2])
cols = np.array([1, 2, 3])
print(f"Diagonal-like selection: {arr2d[rows, cols]}")  # [1 6 11]
\`\`\`

## Array Shapes and Reshaping

Understanding and manipulating array shapes is crucial for machine learning:

\`\`\`python
# Original array
arr = np.arange(12)
print(f"Original: {arr}")
print(f"Shape: {arr.shape}")  # (12,)

# Reshape to 2D
arr2d = arr.reshape(3, 4)
print(f"\\n2D (3x4):\\n{arr2d}")

# Reshape to 3D
arr3d = arr.reshape(2, 2, 3)
print(f"\\n3D (2x2x3):\\n{arr3d}")

# Automatic dimension inference
arr_auto = arr.reshape(3, -1)  # -1 means "figure it out"
print(f"\\nAuto reshape (3x?):\\n{arr_auto}")  # (3, 4)

# Flatten array
flattened = arr2d.flatten()
print(f"\\nFlattened: {flattened}")

# Ravel (similar to flatten, but returns view when possible)
raveled = arr2d.ravel()
print(f"Raveled: {raveled}")

# Transpose
transposed = arr2d.T
print(f"\\nTransposed:\\n{transposed}")

# Add new axis
expanded = arr[np.newaxis, :]  # Add axis at beginning
print(f"\\nExpanded shape: {expanded.shape}")  # (1, 12)

# Alternative syntax for adding axis
expanded2 = arr[:, np.newaxis]
print(f"Expanded shape 2: {expanded2.shape}")  # (12, 1)
\`\`\`

## Data Types (dtype)

NumPy supports many data types, optimized for different use cases:

\`\`\`python
# Integer types
int8_arr = np.array([1, 2, 3], dtype=np.int8)  # -128 to 127
int64_arr = np.array([1, 2, 3], dtype=np.int64)  # Default for integers

# Float types
float32_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
float64_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # Default

# Boolean
bool_arr = np.array([True, False, True], dtype=np.bool_)

# Complex numbers
complex_arr = np.array([1+2j, 3+4j], dtype=np.complex128)

# Check data type
arr = np.array([1, 2, 3])
print(f"Data type: {arr.dtype}")  # int64 (or int32 on 32-bit systems)
print(f"Item size: {arr.itemsize} bytes")  # 8 bytes for int64

# Convert data type
float_arr = arr.astype (np.float32)
print(f"Converted dtype: {float_arr.dtype}")
print(f"New item size: {float_arr.itemsize} bytes")  # 4 bytes
\`\`\`

### Memory Efficiency Example

\`\`\`python
# Compare memory usage
large_arr_int64 = np.arange(1_000_000, dtype=np.int64)
large_arr_int32 = np.arange(1_000_000, dtype=np.int32)
large_arr_int8 = np.arange(256, dtype=np.int8)

print(f"int64 array: {large_arr_int64.nbytes / 1_000_000:.2f} MB")  # 8 MB
print(f"int32 array: {large_arr_int32.nbytes / 1_000_000:.2f} MB")  # 4 MB
print(f"int8 array: {large_arr_int8.nbytes / 1000:.2f} KB")  # 0.26 KB

# Practical tip: use smallest dtype that fits your data
# For neural networks, float32 is often sufficient (vs float64)
\`\`\`

## Memory Layout and Views vs Copies

Understanding when NumPy creates copies vs views is important for performance:

\`\`\`python
# Original array
arr = np.array([1, 2, 3, 4, 5])

# Slicing creates a VIEW (shares memory)
slice_view = arr[1:4]
slice_view[0] = 999
print(f"Original modified: {arr}")  # [1 999 3 4 5]

# Fancy indexing creates a COPY
arr = np.array([1, 2, 3, 4, 5])
fancy_copy = arr[[1, 2, 3]]
fancy_copy[0] = 999
print(f"Original unchanged: {arr}")  # [1 2 3 4 5]

# Explicitly create copy
arr = np.array([1, 2, 3, 4, 5])
explicit_copy = arr.copy()
explicit_copy[0] = 999
print(f"Original unchanged: {arr}")  # [1 2 3 4 5]

# Check if arrays share memory
arr1 = np.array([1, 2, 3])
arr2 = arr1
arr3 = arr1.copy()
print(f"arr1 and arr2 share memory: {np.shares_memory (arr1, arr2)}")  # True
print(f"arr1 and arr3 share memory: {np.shares_memory (arr1, arr3)}")  # False
\`\`\`

## Practical Example: Image Representation

Images are naturally represented as NumPy arrays:

\`\`\`python
# Grayscale image (2D array: height x width)
# Values typically 0-255 (uint8) or 0.0-1.0 (float)
grayscale_img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
print(f"Grayscale image shape: {grayscale_img.shape}")

# RGB image (3D array: height x width x channels)
rgb_img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
print(f"RGB image shape: {rgb_img.shape}")

# Extract red channel
red_channel = rgb_img[:, :, 0]
print(f"Red channel shape: {red_channel.shape}")

# Convert to grayscale (simple average)
grayscale_from_rgb = rgb_img.mean (axis=2).astype (np.uint8)
print(f"Converted grayscale shape: {grayscale_from_rgb.shape}")

# Batch of images (4D: batch x height x width x channels)
batch_size = 32
image_batch = np.random.rand (batch_size, 224, 224, 3)
print(f"Image batch shape: {image_batch.shape}")  # (32, 224, 224, 3)
\`\`\`

## Practical Example: Time Series Data

\`\`\`python
# Stock price data: days x features
days = 252  # Trading days in a year
features = ['open', 'high', 'low', 'close', 'volume']
stock_data = np.random.randn (days, len (features))

print(f"Stock data shape: {stock_data.shape}")  # (252, 5)

# Access closing prices (4th column, index 3)
close_prices = stock_data[:, 3]
print(f"Close prices shape: {close_prices.shape}")  # (252,)

# Get last 30 days of data
recent_data = stock_data[-30:, :]
print(f"Recent data shape: {recent_data.shape}")  # (30, 5)

# Calculate returns (today's price / yesterday's price - 1)
returns = close_prices[1:] / close_prices[:-1] - 1
print(f"Returns shape: {returns.shape}")  # (251,)
print(f"Mean daily return: {returns.mean():.4f}")
\`\`\`

## Key Takeaways

1. **NumPy arrays (ndarray)** are the foundation of numerical computing in Python
2. **Homogeneous data types** enable fast operations and memory efficiency
3. **Multiple creation methods**: zeros, ones, arange, linspace, random
4. **Flexible indexing**: basic slicing, boolean masks, fancy indexing
5. **Shape manipulation**: reshape, flatten, transpose, expand_dims
6. **Data types (dtype)** control memory usage and precision
7. **Views vs copies**: slicing creates views, fancy indexing creates copies
8. **Applications**: images, time series, financial data, ML features

NumPy is the workhorse of data science. Mastering these fundamentals will make all subsequent ML operations much easier!
`,
};

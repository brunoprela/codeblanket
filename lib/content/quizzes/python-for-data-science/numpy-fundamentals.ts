import { QuizQuestion } from '../../../types';

export const numpyFundamentalsQuiz: QuizQuestion[] = [
  {
    id: 'numpy-fundamentals-dq-1',
    question:
      'Explain why NumPy arrays require all elements to be the same data type (homogeneous), and discuss the trade-offs of this design decision compared to Python lists.',
    sampleAnswer: `NumPy's requirement for homogeneous data types is a fundamental design choice that enables its exceptional performance:

**Why Homogeneous Types:**

1. **Memory Layout**: With all elements the same type, NumPy can store them contiguously in memory with predictable spacing. This enables fast element access via pointer arithmetic: element_address = base_address + (index * element_size).

2. **Vectorization**: Modern CPUs have SIMD (Single Instruction, Multiple Data) instructions that operate on multiple values simultaneously. These only work when all values are the same type and aligned in memory.

3. **Cache Efficiency**: Contiguous memory layout means better CPU cache utilization, leading to dramatic speedups for sequential operations.

4. **Compiled Operations**: NumPy operations are implemented in C/Fortran and can be highly optimized because the compiler knows the exact data types at compile time.

**Trade-offs:**

Advantages:
- 10-100x faster than Python lists for numerical operations
- 50-80% less memory usage per element
- Enables GPU acceleration for deep learning
- Predictable performance characteristics

Disadvantages:
- Cannot store mixed types (e.g., integers and strings together)
- Less flexible than Python lists for heterogeneous data
- Type conversion overhead when working with mixed data sources
- Must plan data types in advance

**Practical Implications:**

For numerical computing and ML, the performance benefits vastly outweigh the flexibility loss. When you need mixed types, use structured arrays or pandas DataFrames, which build on NumPy but add flexibility. The homogeneous constraint is what makes NumPy the foundation of scientific Python—speed matters when processing millions or billions of numbers.`,
    keyPoints: [
      'Homogeneous types enable contiguous memory storage and pointer arithmetic',
      'SIMD vectorization requires same-type aligned data',
      'Provides 10-100x speedup and 50-80% less memory usage vs Python lists',
      'Cannot store mixed types - use pandas for heterogeneous data',
      'Performance benefits outweigh flexibility loss for numerical computing',
    ],
  },
  {
    id: 'numpy-fundamentals-dq-2',
    question:
      'In machine learning pipelines, understanding views vs copies is critical for memory management. Describe a scenario where accidentally creating copies instead of views could cause memory issues, and provide strategies to avoid this.',
    sampleAnswer: `Understanding views vs copies is crucial in ML, especially when working with large datasets that might not fit in memory:

**Problematic Scenario:**

Imagine processing a large image dataset (1 million images, 224x224x3 pixels, uint8):

\`\`\`python
# Original data: ~150 GB
images = np.random.randint(0, 255, (1_000_000, 224, 224, 3), dtype=np.uint8)

# MISTAKE: This creates copies, doubling memory usage
normalized_images = images / 255.0  # Creates new array
augmented_images = normalized_images[:, ::-1, :, :]  # Another copy via fancy indexing
batch = augmented_images[[0, 1, 2, 3]]  # Yet another copy

# Memory usage: 150 GB (original) + 150 GB (normalized) + 150 GB (augmented) + small (batch)
# Total: ~450 GB! Might crash the system
\`\`\`

**Why This Happens:**

1. Arithmetic operations (/, *, +) create new arrays
2. Advanced indexing with lists creates copies
3. Boolean indexing creates copies
4. Some array operations trigger copies to ensure C-contiguous memory

**Memory-Efficient Strategies:**

1. **In-Place Operations:**
\`\`\`python
images = images.astype(np.float32)  # Convert type first
images /= 255.0  # In-place division (uses /=)
\`\`\`

2. **Process in Batches:**
\`\`\`python
batch_size = 32
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]  # View, not copy
    batch_normalized = batch / 255.0  # Only small batch copied
    process_batch(batch_normalized)
\`\`\`

3. **Use Views When Possible:**
\`\`\`python
# Slicing creates views
subset = images[:1000]  # View, shares memory
flipped = images[:, ::-1]  # View for some operations
\`\`\`

4. **Memory Mapping for Large Files:**
\`\`\`python
# Load data without loading all into RAM
images_mmap = np.load('images.npy', mmap_mode='r')  # Read-only memory map
\`\`\`

5. **Monitor Memory Usage:**
\`\`\`python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1e9:.2f} GB")
\`\`\`

**Best Practices:**

- Use \`np.shares_memory(arr1, arr2)\` to verify view relationships
- Prefer in-place operations (+=, -=, *=, /=) when possible
- Delete large intermediate arrays explicitly: \`del large_array\`
- Use generators and lazy evaluation for data pipelines
- Consider memory-mapped files for datasets larger than RAM
- Profile memory usage during development, not just in production

In production ML systems, running out of memory is a common failure mode. Understanding views vs copies can mean the difference between a pipeline that works and one that crashes.`,
    keyPoints: [
      'Arithmetic and fancy indexing operations create copies, potentially tripling memory usage',
      'Use in-place operations (+=, /=) to avoid creating copies',
      'Process large datasets in batches to keep memory usage bounded',
      'Basic slicing creates views that share memory with original',
      'Memory-map large files to avoid loading all data into RAM',
      'Monitor memory and use np.shares_memory() to verify view relationships',
    ],
  },
  {
    id: 'numpy-fundamentals-dq-3',
    question:
      'Compare np.linspace() and np.arange() for creating sequences of numbers. When would you use each, and what are the pitfalls of np.arange() with floating-point numbers?',
    sampleAnswer: `np.linspace() and np.arange() serve similar purposes but have important differences that affect their suitability for different tasks:

**np.arange(start, stop, step):**

- Works like Python's range() but for NumPy arrays
- Specifies the STEP size between elements
- Excludes the stop value
- Works well with integers

\`\`\`python
np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
np.arange(0, 1, 0.1)  # Problematic with floats!
\`\`\`

**np.linspace(start, stop, num):**

- Specifies the NUMBER of points wanted
- Includes both start and stop (by default)
- Always produces exact number of points
- Preferred for floating-point ranges

\`\`\`python
np.linspace(0, 10, 5)  # [0.0, 2.5, 5.0, 7.5, 10.0]
np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
\`\`\`

**Critical Pitfall: Floating-Point Errors in arange:**

\`\`\`python
# PROBLEM: Floating-point arithmetic is inexact
arr = np.arange(0, 1, 0.1)
print(len(arr))  # Might be 10 or 11 depending on rounding!
print(arr)
# [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
# Missing 1.0? Or includes 1.0? Unpredictable!

# SOLUTION: Use linspace
arr = np.linspace(0, 1, 11)  # Exactly 11 points including endpoints
print(len(arr))  # Always 11
\`\`\`

**When to Use Each:**

**Use np.arange() when:**
- Working with integers
- Step size is more natural than count (e.g., "every 5 units")
- You want exclusive stop behavior
- Indexing or enumeration tasks

\`\`\`python
# Good use cases
indices = np.arange(len(data))
time_steps = np.arange(0, 1000)  # 1000 time steps
bins = np.arange(0, 100, 10)  # Histogram bins
\`\`\`

**Use np.linspace() when:**
- Working with floating-point numbers
- You know how many points you need
- Creating coordinate grids for plotting
- Numerical integration or function evaluation

\`\`\`python
# Good use cases
x = np.linspace(0, 2*np.pi, 100)  # 100 points for plotting sine wave
y = np.sin(x)

# Create mesh grid for 3D plots
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)

# Temperature range for simulation
temperatures = np.linspace(273, 373, 101)  # 0°C to 100°C, 1° steps
\`\`\`

**Additional Considerations:**

1. **Performance**: arange() is slightly faster for integers, but negligible difference in practice

2. **Logarithmic Spacing**: Use np.logspace() for logarithmic scales
\`\`\`python
np.logspace(0, 3, 4)  # [1, 10, 100, 1000]
\`\`\`

3. **Geomspace**: Use np.geomspace() for geometric progressions
\`\`\`python
np.geomspace(1, 1000, 4)  # [1, 10, 100, 1000]
\`\`\`

**Rule of Thumb:**
- Integers? Use arange()
- Floats? Use linspace()
- When in doubt, linspace() is safer

This distinction matters in numerical computing where precision is critical—using the wrong function can introduce subtle bugs that are hard to track down.`,
    keyPoints: [
      'np.arange() specifies step size, excludes stop value - good for integers',
      'np.linspace() specifies number of points, includes endpoints - safe for floats',
      'Floating-point arithmetic in arange() can cause unpredictable array lengths',
      'Use arange() for indexing/enumeration, linspace() for plotting/numerical work',
      'linspace() guarantees exact number of points, avoiding floating-point errors',
    ],
  },
];

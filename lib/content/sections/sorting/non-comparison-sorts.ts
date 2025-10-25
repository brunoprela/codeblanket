/**
 * Non-Comparison Sorting Algorithms Section
 */

export const noncomparisonsortsSection = {
  id: 'non-comparison-sorts',
  title: 'Non-Comparison Sorting Algorithms',
  content: `**Non-comparison sorts** don't compare elements directly. They exploit properties of the data (like range of integers) to achieve **O(n)** time complexity!

**1. Counting Sort - O(n + k)**

Works for integers in a known range [0, k]. Counts occurrences of each value.

\`\`\`python
def counting_sort (arr):
    if not arr:
        return arr
    
    max_val = max (arr)
    min_val = min (arr)
    range_size = max_val - min_val + 1
    
    # Count occurrences
    count = [0] * range_size
    for num in arr:
        count[num - min_val] += 1
    
    # Reconstruct sorted array
    result = []
    for i in range (range_size):
        result.extend([i + min_val] * count[i])
    
    return result
\`\`\`

- **Time:** O(n + k) where k is the range of values
- **Space:** O(k) for count array
- **Stable:** Yes (with careful implementation)
- **Use case:** Small range of integers, when k = O(n)

**2. Radix Sort - O(d × n)**

Sorts integers digit by digit, from least to most significant.

\`\`\`python
def radix_sort (arr):
    if not arr:
        return arr
    
    max_val = max (arr)
    exp = 1  # Current digit position
    
    while max_val // exp > 0:
        counting_sort_by_digit (arr, exp)
        exp *= 10
    
    return arr

def counting_sort_by_digit (arr, exp):
    n = len (arr)
    output = [0] * n
    count = [0] * 10  # Digits 0-9
    
    # Count occurrences of digits
    for i in range (n):
        digit = (arr[i] // exp) % 10
        count[digit] += 1
    
    # Cumulative count
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # Build output array
    for i in range (n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
    
    # Copy back
    for i in range (n):
        arr[i] = output[i]
\`\`\`

- **Time:** O(d × n) where d is number of digits
- **Space:** O(n + k) auxiliary space
- **Stable:** Yes (relies on stable counting sort)
- **Use case:** Sorting integers or strings with fixed-length keys

**3. Bucket Sort - O(n + k)**

Distributes elements into buckets, sorts each bucket, then concatenates.

\`\`\`python
def bucket_sort (arr):
    if not arr:
        return arr
    
    # Create buckets
    bucket_count = len (arr)
    max_val = max (arr)
    min_val = min (arr)
    bucket_range = (max_val - min_val) / bucket_count + 1
    
    buckets = [[] for _ in range (bucket_count)]
    
    # Distribute into buckets
    for num in arr:
        index = int((num - min_val) / bucket_range)
        buckets[index].append (num)
    
    # Sort each bucket and concatenate
    result = []
    for bucket in buckets:
        result.extend (sorted (bucket))  # Use insertion sort for small buckets
    
    return result
\`\`\`

- **Time:** O(n + k) average, O(n²) worst if all elements in one bucket
- **Space:** O(n + k)
- **Stable:** Depends on bucket sort used
- **Use case:** Uniformly distributed data, floating-point numbers

**When to Use Non-Comparison Sorts:**

✅ **Use when:**
- Sorting integers in a known, small range (counting sort)
- Data has fixed-length keys or limited digits (radix sort)
- Data is uniformly distributed (bucket sort)
- Need O(n) time and can afford O(n) or O(k) space

❌ **Don't use when:**
- Sorting arbitrary objects with comparison function
- Range k >> n (counting sort wastes space)
- Data distribution is unknown (bucket sort degrades)
- Need truly general-purpose sorting

**Comparison:**

| Algorithm | Time | Space | Stable | Use Case |
|-----------|------|-------|--------|----------|
| Counting  | O(n+k) | O(k) | Yes | Small integer range |
| Radix     | O(d×n) | O(n) | Yes | Fixed-length keys |
| Bucket    | O(n+k) | O(n) | Depends | Uniform distribution |

**Real-World Applications:**
- **Counting Sort:** Sorting grades (0-100), sorting by age
- **Radix Sort:** Sorting IP addresses, sorting strings
- **Bucket Sort:** Sorting floating-point numbers, external sorting`,
};

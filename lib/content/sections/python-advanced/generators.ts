/**
 * Generators & Iterators Section
 */

export const generatorsSection = {
  id: 'generators',
  title: 'Generators & Iterators',
  content: `**What are Generators?**
Generators are functions that return an iterator that produces values lazily using yield.

**Basic Generator:**
\`\`\`python
def count_up_to(n):
    i = 1
    while i <= n:
        yield i
        i += 1

# Usage
for num in count_up_to(5):
    print(num)  # 1, 2, 3, 4, 5
\`\`\`

**Generator Expressions:**
\`\`\`python
# List comprehension (creates entire list)
squares_list = [x**2 for x in range(1000000)]

# Generator expression (lazy evaluation)
squares_gen = (x**2 for x in range(1000000))
\`\`\`

**Why Generators Matter:**
1. **Memory Efficiency:** Don't store all values in memory
2. **Lazy Evaluation:** Compute values only when needed
3. **Infinite Sequences:** Can represent unbounded sequences
4. **Pipeline Processing:** Chain operations efficiently

**Real-World Example - Processing Large Files:**
\`\`\`python
def read_large_file(filepath):
    """Memory-efficient file reading"""
    with open(filepath) as f:
        for line in f:
            yield line.strip()

def process_logs(filepath):
    """Process huge log files without loading into memory"""
    for line in read_large_file(filepath):
        if 'ERROR' in line:
            yield line

# Use it
for error_log in process_logs('huge_log.txt'):
    print(error_log)
\`\`\`

**Generator Pipeline:**
\`\`\`python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def take(n, iterable):
    """Take first n items"""
    for i, item in enumerate(iterable):
        if i >= n:
            break
        yield item

def is_even(n):
    return n % 2 == 0

# Chain generators
result = list(take(5, filter(is_even, fibonacci())))
print(result)  # [0, 2, 8, 34, 144]
\`\`\`

**send() and two-way communication:**
\`\`\`python
def running_average():
    total = 0
    count = 0
    average = None
    while True:
        value = yield average
        total += value
        count += 1
        average = total / count

avg = running_average()
next(avg)  # Prime the generator
print(avg.send(10))  # 10.0
print(avg.send(20))  # 15.0
print(avg.send(30))  # 20.0
\`\`\`

**Best Practices:**
- Use generators for large datasets
- Prefer generator expressions over list comprehensions when values are used once
- Chain generators for data pipelines
- Use itertools for advanced iterator patterns`,
};

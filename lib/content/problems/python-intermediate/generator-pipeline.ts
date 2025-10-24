/**
 * Data Processing Pipeline with Generators
 * Problem ID: intermediate-generator-pipeline
 * Order: 13
 */

import { Problem } from '../../../types';

export const intermediate_generator_pipelineProblem: Problem = {
  id: 'intermediate-generator-pipeline',
  title: 'Data Processing Pipeline with Generators',
  difficulty: 'Medium',
  description: `Create a memory-efficient data processing pipeline using generators.

**Pipeline Steps:**
1. Read lines from file (generator)
2. Filter lines matching pattern (generator)
3. Transform lines (generator)
4. Aggregate results

**Benefits:**
- Memory efficient (processes one item at a time)
- Lazy evaluation
- Composable operations

**Example:**
\`\`\`python
lines = read_lines('data.txt')
filtered = filter_lines(lines, pattern='ERROR')
transformed = transform_lines(filtered, str.upper)
result = list(transformed)
\`\`\``,
  examples: [
    {
      input: 'Pipeline processes large file',
      output: 'Memory-efficient streaming',
    },
  ],
  constraints: [
    'Use yield keyword',
    'Chain generators',
    'Process one item at a time',
  ],
  hints: [
    'yield returns values lazily',
    'Generators can be chained',
    'Use generator expressions',
  ],
  starterCode: `import re

def read_lines(filename):
    """
    Generator that yields lines from file.
    
    Args:
        filename: Path to file
        
    Yields:
        Individual lines from file
        
    Examples:
        >>> for line in read_lines('data.txt'):
        ...     print(line)
    """
    pass


def filter_lines(lines, pattern):
    """
    Generator that filters lines matching pattern.
    
    Args:
        lines: Iterator of lines
        pattern: Regex pattern to match
        
    Yields:
        Lines matching pattern
    """
    pass


def transform_lines(lines, transform_func):
    """
    Generator that transforms each line.
    
    Args:
        lines: Iterator of lines
        transform_func: Function to apply to each line
        
    Yields:
        Transformed lines
    """
    pass


def batch_lines(lines, batch_size):
    """
    Generator that groups lines into batches.
    
    Args:
        lines: Iterator of lines
        batch_size: Number of lines per batch
        
    Yields:
        Lists of lines (batches)
        
    Examples:
        >>> for batch in batch_lines(lines, 10):
        ...     process_batch(batch)
    """
    pass


# Test with virtual file
# Create test file
with open('test_data.txt', 'w') as f:
    f.write("""ERROR: Connection failed
INFO: System starting
ERROR: Database timeout
WARNING: Low memory
INFO: User logged in
ERROR: API error
""")

# Build pipeline
lines = read_lines('test_data.txt')
errors = filter_lines(lines, r'ERROR')
uppercase = transform_lines(errors, str.upper)

print("Filtered and transformed lines:")
for line in uppercase:
    print(line)

# Example with batching
lines2 = read_lines('test_data.txt')
batches = batch_lines(lines2, 2)
for i, batch in enumerate(batches, 1):
    print(f"\\nBatch {i}:")
    for line in batch:
        print(f"  {line}", end=')
`,
  testCases: [
    {
      input: ['test_data.txt', 'ERROR'],
      expected: 3,
      functionName: 'test_data_processing',
    },
  ],
  solution: `import re

# Create test file (for browser environment)
with open('test_data.txt', 'w') as f:
    f.write("""ERROR: Connection failed
INFO: System starting
ERROR: Database timeout
WARNING: Low memory
INFO: User logged in
ERROR: API error
""")


def read_lines(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line.rstrip('\\n')


def filter_lines(lines, pattern):
    regex = re.compile(pattern)
    for line in lines:
        if regex.search(line):
            yield line


def transform_lines(lines, transform_func):
    for line in lines:
        yield transform_func(line)


def batch_lines(lines, batch_size):
    batch = []
    for line in lines:
        batch.append(line)
        if len(batch) == batch_size:
            yield batch
            batch = []
    
    # Yield remaining items
    if batch:
        yield batch


# Test helper function (for automated testing)
def test_data_processing(filename, pattern):
    """Test function for data processing pipeline."""
    lines = read_lines(filename)
    filtered = filter_lines(lines, pattern)
    return len(list(filtered))


# Advanced: Generator with send()
def running_average():
    """Generator that calculates running average."""
    total = 0
    count = 0
    average = None
    
    while True:
        value = yield average
        total += value
        count += 1
        average = total / count


# Advanced: Generator expression examples
def process_large_file(filename):
    """Process file using generator expressions."""
    # Generator expression - memory efficient
    lines = (line.strip() for line in open(filename))
    errors = (line for line in lines if 'ERROR' in line)
    uppercase = (line.upper() for line in errors)
    
    return list(uppercase)`,
  timeComplexity: 'O(n) where n is number of lines',
  spaceComplexity: 'O(1) for generators, O(b) for batches',
  order: 13,
  topic: 'Python Intermediate',
};

/**
 * Python Intermediate - New Problems Batch 1 (22-31)
 * 10 problems
 */

import { Problem } from '../types';

export const pythonIntermediateBatch1: Problem[] = [
  {
    id: 'intermediate-list-comprehension-nested',
    title: 'Nested List Comprehension',
    difficulty: 'Medium',
    description: `Use nested list comprehensions to flatten and transform data.

**Example:**
\`\`\`python
matrix = [[1,2], [3,4]]
flat = [x for row in matrix for x in row]
# Result: [1, 2, 3, 4]
\`\`\`

This tests:
- List comprehension syntax
- Multiple loops in one line
- Conditional filtering`,
    examples: [
      {
        input: 'matrix = [[1,2,3], [4,5,6], [7,8,9]]',
        output: '[1, 2, 3, 4, 5, 6, 7, 8, 9]',
      },
    ],
    constraints: ['Use list comprehension', 'Single line preferred'],
    hints: [
      'Outer loop first, inner loop second',
      'Can add conditions with if',
      '[x for row in matrix for x in row]',
    ],
    starterCode: `def flatten_matrix(matrix):
    """
    Flatten 2D matrix using list comprehension.
    
    Args:
        matrix: 2D list
        
    Returns:
        Flattened list
        
    Examples:
        >>> flatten_matrix([[1,2], [3,4]])
        [1, 2, 3, 4]
    """
    pass


# Test
print(flatten_matrix([[1,2,3], [4,5,6], [7,8,9]]))
`,
    testCases: [
      {
        input: [
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        ],
        expected: [1, 2, 3, 4, 5, 6, 7, 8, 9],
      },
      {
        input: [
          [
            [1, 2],
            [3, 4],
          ],
        ],
        expected: [1, 2, 3, 4],
      },
    ],
    solution: `def flatten_matrix(matrix):
    return [x for row in matrix for x in row]


# With filtering (only even numbers)
def flatten_matrix_even(matrix):
    return [x for row in matrix for x in row if x % 2 == 0]`,
    timeComplexity: 'O(n * m)',
    spaceComplexity: 'O(n * m)',
    order: 22,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-zip-star',
    title: 'Transpose Matrix with zip(*matrix)',
    difficulty: 'Easy',
    description: `Use zip(*matrix) to transpose a matrix.

The * operator unpacks the matrix rows as arguments to zip.

**Example:**
\`\`\`python
matrix = [[1,2], [3,4], [5,6]]
transposed = list(zip(*matrix))
# Result: [(1,3,5), (2,4,6)]
\`\`\`

This tests:
- zip() function
- * unpacking operator
- Matrix operations`,
    examples: [
      {
        input: 'matrix = [[1,2,3], [4,5,6]]',
        output: '[(1,4), (2,5), (3,6)]',
      },
    ],
    constraints: ['Use zip()', 'Use * operator'],
    hints: [
      'zip(*matrix) unpacks rows',
      'Returns tuples',
      'Convert to list if needed',
    ],
    starterCode: `def transpose(matrix):
    """
    Transpose matrix using zip.
    
    Args:
        matrix: 2D list
        
    Returns:
        Transposed matrix as list of tuples
        
    Examples:
        >>> transpose([[1,2], [3,4]])
        [(1, 3), (2, 4)]
    """
    pass


# Test
print(transpose([[1,2,3], [4,5,6]]))
`,
    testCases: [
      {
        input: [
          [
            [1, 2, 3],
            [4, 5, 6],
          ],
        ],
        expected: [
          [1, 4],
          [2, 5],
          [3, 6],
        ],
      },
      {
        input: [
          [
            [1, 2],
            [3, 4],
          ],
        ],
        expected: [
          [1, 3],
          [2, 4],
        ],
      },
    ],
    solution: `def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


# Alternative
def transpose_alt(matrix):
    return list(map(list, zip(*matrix)))`,
    timeComplexity: 'O(n * m)',
    spaceComplexity: 'O(n * m)',
    order: 23,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-itertools-combinations',
    title: 'Combinations with itertools',
    difficulty: 'Easy',
    description: `Generate all combinations of elements using itertools.

**Example:**
\`\`\`python
from itertools import combinations
list(combinations([1,2,3], 2))
# Result: [(1,2), (1,3), (2,3)]
\`\`\`

This tests:
- itertools module
- Combinations vs permutations
- Iterator usage`,
    examples: [
      {
        input: 'items = [1,2,3,4], r = 2',
        output: '[(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]',
      },
    ],
    constraints: ['Use itertools.combinations', "Order doesn't matter"],
    hints: [
      'Import from itertools',
      'combinations(items, r)',
      'Returns iterator',
    ],
    starterCode: `from itertools import combinations

def get_combinations(items, r):
    """
    Get all r-length combinations.
    
    Args:
        items: List of items
        r: Length of combinations
        
    Returns:
        List of tuples
        
    Examples:
        >>> get_combinations([1,2,3], 2)
        [(1, 2), (1, 3), (2, 3)]
    """
    pass


# Test
print(get_combinations([1,2,3,4], 2))
`,
    testCases: [
      {
        input: [[1, 2, 3], 2],
        expected: [
          [1, 2],
          [1, 3],
          [2, 3],
        ],
      },
      {
        input: [[1, 2, 3, 4], 2],
        expected: [
          [1, 2],
          [1, 3],
          [1, 4],
          [2, 3],
          [2, 4],
          [3, 4],
        ],
      },
    ],
    solution: `from itertools import combinations

def get_combinations(items, r):
    return [list(combo) for combo in combinations(items, r)]


# For permutations (order matters)
from itertools import permutations

def get_permutations(items, r):
    return [list(perm) for perm in permutations(items, r)]`,
    timeComplexity: 'O(C(n,r))',
    spaceComplexity: 'O(C(n,r))',
    order: 24,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-collections-namedtuple',
    title: 'Named Tuple for Data',
    difficulty: 'Easy',
    description: `Use namedtuple to create lightweight data structures.

Named tuples are:
- Immutable
- More readable than regular tuples
- Memory efficient
- Can access by name or index

**Use Case:** Simple data classes, return values

This tests:
- collections.namedtuple
- Immutability
- Attribute access`,
    examples: [
      {
        input: 'Point(x=1, y=2)',
        output: 'Access as p.x and p.y',
      },
    ],
    constraints: ['Use namedtuple', 'Immutable'],
    hints: [
      'from collections import namedtuple',
      'Define fields',
      'Create like a class',
    ],
    starterCode: `from collections import namedtuple

# Define Point namedtuple
Point = namedtuple('Point', ['x', 'y'])

def calculate_distance(p1, p2):
    """
    Calculate distance between two points.
    
    Args:
        p1: Point namedtuple
        p2: Point namedtuple
        
    Returns:
        Distance as float
        
    Examples:
        >>> p1 = Point(0, 0)
        >>> p2 = Point(3, 4)
        >>> calculate_distance(p1, p2)
        5.0
    """
    pass


# Test
p1 = Point(0, 0)
p2 = Point(3, 4)
print(calculate_distance(p1, p2))
`,
    testCases: [
      {
        input: [
          [0, 0],
          [3, 4],
        ],
        expected: 5.0,
      },
      {
        input: [
          [0, 0],
          [5, 12],
        ],
        expected: 13.0,
      },
    ],
    solution: `from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

def calculate_distance(p1, p2):
    # Convert lists to Points if needed
    if isinstance(p1, list):
        p1 = Point(*p1)
    if isinstance(p2, list):
        p2 = Point(*p2)
    
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return (dx ** 2 + dy ** 2) ** 0.5`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 25,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-collections-chainmap',
    title: 'ChainMap for Nested Contexts',
    difficulty: 'Medium',
    description: `Use ChainMap to combine multiple dictionaries.

ChainMap features:
- Search multiple dicts
- Prioritizes first dict
- Fast lookups
- Used for nested scopes

**Use Case:** Configuration layers, template contexts

This tests:
- collections.ChainMap
- Dictionary chaining
- Scope resolution`,
    examples: [
      {
        input: 'ChainMap(local, global, default)',
        output: 'Searches in order',
      },
    ],
    constraints: ['Use ChainMap', 'Multiple dicts'],
    hints: [
      'from collections import ChainMap',
      'ChainMap(dict1, dict2, ...)',
      'Searches left to right',
    ],
    starterCode: `from collections import ChainMap

def get_config_value(key, local, global_config, defaults):
    """
    Get config value from layered dicts.
    
    Args:
        key: Config key
        local: Local overrides
        global_config: Global settings
        defaults: Default values
        
    Returns:
        Value from first dict containing key
        
    Examples:
        >>> local = {'color': 'red'}
        >>> global_config = {'color': 'blue', 'size': 10}
        >>> defaults = {'color': 'black', 'size': 8, 'style': 'solid'}
        >>> get_config_value('color', local, global_config, defaults)
        'red'
    """
    pass


# Test
print(get_config_value('size', {}, {'size': 10}, {'size': 8}))
`,
    testCases: [
      {
        input: ['size', {}, { size: 10 }, { size: 8 }],
        expected: 10,
      },
      {
        input: ['color', { color: 'red' }, { color: 'blue' }, {}],
        expected: 'red',
      },
    ],
    solution: `from collections import ChainMap

def get_config_value(key, local, global_config, defaults):
    config = ChainMap(local, global_config, defaults)
    return config.get(key)`,
    timeComplexity: 'O(n) where n is number of dicts',
    spaceComplexity: 'O(1)',
    order: 26,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-bisect-search',
    title: 'Binary Search with bisect',
    difficulty: 'Easy',
    description: `Use bisect module for efficient searching in sorted lists.

bisect functions:
- bisect_left: Leftmost insertion point
- bisect_right: Rightmost insertion point
- insort: Insert maintaining sort

**Use Case:** Maintaining sorted lists, range queries

This tests:
- bisect module
- Sorted list operations
- Binary search`,
    examples: [
      {
        input: 'bisect_left([1,3,5,7], 4)',
        output: '2 (insert position)',
      },
    ],
    constraints: ['Use bisect module', 'List must be sorted'],
    hints: [
      'import bisect',
      'bisect.bisect_left(list, value)',
      'O(log n) search',
    ],
    starterCode: `import bisect

def find_insert_position(sorted_list, value):
    """
    Find position to insert value to maintain sort.
    
    Args:
        sorted_list: Sorted list
        value: Value to insert
        
    Returns:
        Index where value should be inserted
        
    Examples:
        >>> find_insert_position([1,3,5,7], 4)
        2
    """
    pass


# Test
print(find_insert_position([1,3,5,7,9], 6))
`,
    testCases: [
      {
        input: [[1, 3, 5, 7], 4],
        expected: 2,
      },
      {
        input: [[1, 3, 5, 7, 9], 6],
        expected: 3,
      },
    ],
    solution: `import bisect

def find_insert_position(sorted_list, value):
    return bisect.bisect_left(sorted_list, value)


# Alternative: find if value exists
def binary_search(sorted_list, value):
    pos = bisect.bisect_left(sorted_list, value)
    if pos < len(sorted_list) and sorted_list[pos] == value:
        return pos
    return -1`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    order: 27,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-heapq-priority',
    title: 'Priority Queue with heapq',
    difficulty: 'Medium',
    description: `Use heapq to implement a priority queue.

heapq functions:
- heappush: Add item
- heappop: Remove smallest
- heapify: Convert list to heap

**Use Case:** Task scheduling, Dijkstra's algorithm

This tests:
- heapq module
- Min-heap operations
- Priority queues`,
    examples: [
      {
        input: 'Push tasks with priorities',
        output: 'Pop in priority order',
      },
    ],
    constraints: ['Use heapq', 'Min-heap (smallest first)'],
    hints: [
      'import heapq',
      'Use tuples (priority, item)',
      'heappush and heappop',
    ],
    starterCode: `import heapq

def process_tasks(tasks):
    """
    Process tasks by priority (lowest number = highest priority).
    
    Args:
        tasks: List of (priority, task_name) tuples
        
    Returns:
        List of task names in priority order
        
    Examples:
        >>> tasks = [(3, 'low'), (1, 'high'), (2, 'medium')]
        >>> process_tasks(tasks)
        ['high', 'medium', 'low']
    """
    pass


# Test
print(process_tasks([(3, 'task3'), (1, 'task1'), (2, 'task2')]))
`,
    testCases: [
      {
        input: [
          [
            [3, 'task3'],
            [1, 'task1'],
            [2, 'task2'],
          ],
        ],
        expected: ['task1', 'task2', 'task3'],
      },
      {
        input: [
          [
            [5, 'e'],
            [2, 'b'],
            [4, 'd'],
            [1, 'a'],
            [3, 'c'],
          ],
        ],
        expected: ['a', 'b', 'c', 'd', 'e'],
      },
    ],
    solution: `import heapq

def process_tasks(tasks):
    heap = []
    for priority, task_name in tasks:
        heapq.heappush(heap, (priority, task_name))
    
    result = []
    while heap:
        priority, task_name = heapq.heappop(heap)
        result.append(task_name)
    
    return result


# Alternative: heapify existing list
def process_tasks_heapify(tasks):
    heapq.heapify(tasks)
    return [task_name for priority, task_name in sorted(tasks)]`,
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(n)',
    order: 28,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-operator-itemgetter',
    title: 'Sorting with operator.itemgetter',
    difficulty: 'Easy',
    description: `Use operator.itemgetter for efficient sorting by specific fields.

itemgetter creates a callable that fetches items:
- Faster than lambda
- More readable
- Works with sort key

**Use Case:** Sorting complex data structures

This tests:
- operator module
- Sorting by key
- Function objects`,
    examples: [
      {
        input: 'Sort list of tuples by second element',
        output: 'Use itemgetter(1)',
      },
    ],
    constraints: ['Use operator.itemgetter', 'More efficient than lambda'],
    hints: [
      'from operator import itemgetter',
      'Use as sort key',
      'Can get multiple items',
    ],
    starterCode: `from operator import itemgetter

def sort_by_age(people):
    """
    Sort list of people by age.
    
    Args:
        people: List of (name, age) tuples
        
    Returns:
        Sorted list by age
        
    Examples:
        >>> sort_by_age([('Alice', 30), ('Bob', 25), ('Charlie', 35)])
        [('Bob', 25), ('Alice', 30), ('Charlie', 35)]
    """
    pass


# Test
print(sort_by_age([('Alice', 30), ('Bob', 25), ('Charlie', 35)]))
`,
    testCases: [
      {
        input: [
          [
            ['Alice', 30],
            ['Bob', 25],
            ['Charlie', 35],
          ],
        ],
        expected: [
          ['Bob', 25],
          ['Alice', 30],
          ['Charlie', 35],
        ],
      },
      {
        input: [
          [
            ['X', 5],
            ['Y', 2],
            ['Z', 8],
          ],
        ],
        expected: [
          ['Y', 2],
          ['X', 5],
          ['Z', 8],
        ],
      },
    ],
    solution: `from operator import itemgetter

def sort_by_age(people):
    return sorted(people, key=itemgetter(1))


# Sort by multiple fields
def sort_by_age_then_name(people):
    return sorted(people, key=itemgetter(1, 0))`,
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(n)',
    order: 29,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-partial-functions',
    title: 'Partial Functions with functools',
    difficulty: 'Medium',
    description: `Use functools.partial to create functions with pre-filled arguments.

Partial application:
- Fix some arguments
- Create specialized functions
- Useful for callbacks

**Example:**
\`\`\`python
from functools import partial
double = partial(multiply, 2)
double(5)  # Returns 10
\`\`\`

This tests:
- functools.partial
- Function composition
- Currying concept`,
    examples: [
      {
        input: 'Create add10 = partial(add, 10)',
        output: 'add10(5) returns 15',
      },
    ],
    constraints: ['Use functools.partial', 'Pre-fill arguments'],
    hints: [
      'from functools import partial',
      'partial(func, *args, **kwargs)',
      'Returns new function',
    ],
    starterCode: `from functools import partial

def power(base, exponent):
    """Calculate base ** exponent"""
    return base ** exponent


def test_partial():
    """Test partial functions"""
    # Create square function (exponent=2)
    square = partial(power, exponent=2)
    
    # Create cube function (exponent=3)
    cube = partial(power, exponent=3)
    
    # Test them
    result1 = square(5)  # 5^2 = 25
    result2 = cube(3)    # 3^3 = 27
    
    return result1 + result2
`,
    testCases: [
      {
        input: [],
        expected: 52,
        functionName: 'test_partial',
      },
    ],
    solution: `from functools import partial

def power(base, exponent):
    return base ** exponent


def test_partial():
    square = partial(power, exponent=2)
    cube = partial(power, exponent=3)
    
    result1 = square(5)
    result2 = cube(3)
    
    return result1 + result2`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 30,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-with-multiple-contexts',
    title: 'Multiple Context Managers',
    difficulty: 'Easy',
    description: `Use multiple context managers in a single with statement.

**Syntax:**
\`\`\`python
with open('file1') as f1, open('file2') as f2:
    # Both files open
\`\`\`

This tests:
- Multiple with items
- Context manager chaining
- Resource management`,
    examples: [
      {
        input: 'Open multiple files at once',
        output: 'All automatically closed',
      },
    ],
    constraints: ['Use single with statement', 'Multiple context managers'],
    hints: [
      'Separate with comma',
      'with cm1 as v1, cm2 as v2:',
      'All exit in reverse order',
    ],
    starterCode: `def test_multiple_contexts():
    """Test multiple context managers"""
    from io import StringIO
    
    # Create mock file objects
    file1 = StringIO("Hello")
    file2 = StringIO("World")
    
    # Use both in one with statement
    with file1 as f1, file2 as f2:
        content1 = f1.read()
        content2 = f2.read()
    
    return len(content1) + len(content2)
`,
    testCases: [
      {
        input: [],
        expected: 10,
        functionName: 'test_multiple_contexts',
      },
    ],
    solution: `def test_multiple_contexts():
    from io import StringIO
    
    file1 = StringIO("Hello")
    file2 = StringIO("World")
    
    with file1 as f1, file2 as f2:
        content1 = f1.read()
        content2 = f2.read()
    
    return len(content1) + len(content2)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 31,
    topic: 'Python Intermediate',
  },
];

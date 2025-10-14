/**
 * Python Intermediate - New Problems Batch 2 (32-41)
 * 10 problems
 */

import { Problem } from '../types';

export const pythonIntermediateBatch2: Problem[] = [
  {
    id: 'intermediate-exception-else',
    title: 'Try-Except-Else-Finally',
    difficulty: 'Medium',
    description: `Use all parts of exception handling: try, except, else, finally.

**Flow:**
- try: Code that might raise
- except: Handle exceptions
- else: Runs if no exception
- finally: Always runs

This tests:
- Exception handling flow
- else clause
- finally clause`,
    examples: [
      {
        input: 'Process with error handling',
        output: 'Proper cleanup in all cases',
      },
    ],
    constraints: ['Use all four parts', 'Handle cleanup properly'],
    hints: [
      'else runs when no exception',
      'finally always runs',
      'Order: try-except-else-finally',
    ],
    starterCode: `def divide_numbers(a, b):
    """
    Divide with proper exception handling.
    
    Returns:
        Result or error message
        
    Examples:
        >>> divide_numbers(10, 2)
        5.0
        >>> divide_numbers(10, 0)
        'Error: Division by zero'
    """
    try:
        result = a / b
    except ZeroDivisionError:
        return "Error: Division by zero"
    else:
        return result
    finally:
        pass  # Cleanup code


# Test
print(divide_numbers(10, 2))
`,
    testCases: [
      {
        input: [10, 2],
        expected: 5.0,
      },
      {
        input: [10, 0],
        expected: 'Error: Division by zero',
      },
    ],
    solution: `def divide_numbers(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        return "Error: Division by zero"
    else:
        return result
    finally:
        pass`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 32,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-custom-iterator-class',
    title: 'Custom Iterator Class',
    difficulty: 'Medium',
    description: `Create a class that implements the iterator protocol.

Iterator protocol requires:
- __iter__() returns self
- __next__() returns next value or raises StopIteration

**Use Case:** Custom sequences, data streaming

This tests:
- Iterator protocol
- __iter__ and __next__
- StopIteration`,
    examples: [
      {
        input: 'Counter(0, 5)',
        output: 'Yields 0, 1, 2, 3, 4',
      },
    ],
    constraints: [
      'Implement __iter__ and __next__',
      'Raise StopIteration when done',
    ],
    hints: [
      '__iter__ returns self',
      '__next__ returns next value',
      'Raise StopIteration at end',
    ],
    starterCode: `class Counter:
    """
    Iterator that counts from start to end.
    
    Examples:
        >>> for i in Counter(0, 3):
        ...     print(i)
        0
        1
        2
    """
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value


def test_iterator():
    """Test custom iterator"""
    counter = Counter(0, 5)
    result = list(counter)
    return sum(result)
`,
    testCases: [
      {
        input: [],
        expected: 10,
        functionName: 'test_iterator',
      },
    ],
    solution: `class Counter:
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value


def test_iterator():
    counter = Counter(0, 5)
    result = list(counter)
    return sum(result)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 33,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-class-method-static',
    title: 'Class Methods vs Static Methods',
    difficulty: 'Medium',
    description: `Understand the difference between @classmethod and @staticmethod.

**@classmethod:**
- Receives class as first argument (cls)
- Can access/modify class state
- Used for factory methods

**@staticmethod:**
- No special first argument
- Cannot access class or instance
- Utility functions

This tests:
- Decorator understanding
- Method types
- Use cases`,
    examples: [
      {
        input: 'Factory method with @classmethod',
        output: 'Creates instances from different inputs',
      },
    ],
    constraints: ['Use both decorators', 'Show difference'],
    hints: [
      '@classmethod gets cls',
      '@staticmethod gets no special arg',
      'Class methods can create instances',
    ],
    starterCode: `class Pizza:
    """Pizza with different methods"""
    def __init__(self, size, toppings):
        self.size = size
        self.toppings = toppings
    
    @classmethod
    def margherita(cls, size):
        """Factory method for margherita pizza"""
        return cls(size, ['cheese', 'tomato'])
    
    @classmethod
    def pepperoni(cls, size):
        """Factory method for pepperoni pizza"""
        return cls(size, ['cheese', 'tomato', 'pepperoni'])
    
    @staticmethod
    def is_valid_size(size):
        """Utility method to check size"""
        return size in ['small', 'medium', 'large']
    
    def topping_count(self):
        """Instance method"""
        return len(self.toppings)


def test_methods():
    """Test different method types"""
    # Use static method
    valid = Pizza.is_valid_size('medium')
    
    # Use class method
    pizza = Pizza.margherita('large')
    
    # Use instance method
    count = pizza.topping_count()
    
    return count
`,
    testCases: [
      {
        input: [],
        expected: 2,
        functionName: 'test_methods',
      },
    ],
    solution: `class Pizza:
    def __init__(self, size, toppings):
        self.size = size
        self.toppings = toppings
    
    @classmethod
    def margherita(cls, size):
        return cls(size, ['cheese', 'tomato'])
    
    @classmethod
    def pepperoni(cls, size):
        return cls(size, ['cheese', 'tomato', 'pepperoni'])
    
    @staticmethod
    def is_valid_size(size):
        return size in ['small', 'medium', 'large']
    
    def topping_count(self):
        return len(self.toppings)


def test_methods():
    valid = Pizza.is_valid_size('medium')
    pizza = Pizza.margherita('large')
    count = pizza.topping_count()
    return count`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 34,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-property-setter',
    title: 'Property with Getter and Setter',
    difficulty: 'Medium',
    description: `Use @property decorator with getter and setter.

Property benefits:
- Control attribute access
- Add validation
- Computed properties
- Backward compatibility

This tests:
- @property decorator
- @property.setter
- Encapsulation`,
    examples: [
      {
        input: 'Temperature with validation',
        output: 'Cannot set invalid values',
      },
    ],
    constraints: ['Use @property', 'Add validation'],
    hints: [
      '@property for getter',
      '@name.setter for setter',
      'Can validate in setter',
    ],
    starterCode: `class Temperature:
    """
    Temperature with Celsius/Fahrenheit conversion.
    """
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Get temperature in Celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Set temperature in Celsius with validation"""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        """Get temperature in Fahrenheit"""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Set temperature in Fahrenheit"""
        self.celsius = (value - 32) * 5/9


def test_property():
    """Test property getter/setter"""
    temp = Temperature(0)
    
    # Get Fahrenheit (should be 32)
    f = temp.fahrenheit
    
    # Set Celsius
    temp.celsius = 100
    
    # Get Fahrenheit (should be 212)
    f2 = temp.fahrenheit
    
    return int(f + f2)
`,
    testCases: [
      {
        input: [],
        expected: 244,
        functionName: 'test_property',
      },
    ],
    solution: `class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9


def test_property():
    temp = Temperature(0)
    f = temp.fahrenheit
    temp.celsius = 100
    f2 = temp.fahrenheit
    return int(f + f2)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 35,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-string-format-advanced',
    title: 'Advanced String Formatting',
    difficulty: 'Easy',
    description: `Use advanced f-string formatting options.

Format options:
- Alignment: {var:<10}, {var:>10}, {var:^10}
- Precision: {num:.2f}
- Padding: {var:0>5}
- Date: {date:%Y-%m-%d}

This tests:
- F-string features
- Format specifiers
- String alignment`,
    examples: [
      {
        input: 'Format numbers and strings',
        output: 'Aligned and formatted output',
      },
    ],
    constraints: ['Use f-strings', 'Use format specifiers'],
    hints: [
      'f"{var:format_spec}"',
      '< left, > right, ^ center',
      '.2f for 2 decimal places',
    ],
    starterCode: `def format_table(name, score, percentage):
    """
    Format data in table format.
    
    Args:
        name: String
        score: Integer
        percentage: Float
        
    Returns:
        Formatted string
        
    Examples:
        >>> format_table("Alice", 95, 0.95)
        'Alice     |  95 | 95.00%'
    """
    # Left-align name (10 chars), right-align score (4 chars), 
    # format percentage with 2 decimals
    result = f"{name:<10}| {score:>4} | {percentage:>6.2%}"
    return result


# Test
print(format_table("Bob", 87, 0.87))
`,
    testCases: [
      {
        input: ['Bob', 87, 0.87],
        expected: 'Bob       |   87 | 87.00%',
      },
      {
        input: ['Alice', 95, 0.95],
        expected: 'Alice     |   95 | 95.00%',
      },
    ],
    solution: `def format_table(name, score, percentage):
    return f"{name:<10}| {score:>4} | {percentage:>6.2%}"


# More examples
def format_currency(amount):
    # Comma separator, 2 decimals
    return f"${'$'}{amount:,.2f}"

def format_hex(number):
    # Hex with 0x prefix, padded
    return f"{number:#06x}"`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 36,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-defaultdict-grouping',
    title: 'Group Data with defaultdict',
    difficulty: 'Easy',
    description: `Use defaultdict to group data efficiently.

defaultdict features:
- No KeyError for missing keys
- Auto-creates default values
- Cleaner than dict.get()

**Use Case:** Grouping, counting, aggregation

This tests:
- collections.defaultdict
- Data grouping
- Default value types`,
    examples: [
      {
        input: 'Group students by grade',
        output: 'Dict of grade: [students]',
      },
    ],
    constraints: ['Use defaultdict', 'Group by key'],
    hints: [
      'from collections import defaultdict',
      'defaultdict(list) for grouping',
      'No need to check if key exists',
    ],
    starterCode: `from collections import defaultdict

def group_by_first_letter(words):
    """
    Group words by first letter.
    
    Args:
        words: List of strings
        
    Returns:
        Dict of letter: [words]
        
    Examples:
        >>> group_by_first_letter(['apple', 'ant', 'banana', 'bear'])
        {'a': ['apple', 'ant'], 'b': ['banana', 'bear']}
    """
    pass


# Test
result = group_by_first_letter(['cat', 'dog', 'cow', 'duck'])
print(result)
`,
    testCases: [
      {
        input: [['cat', 'dog', 'cow', 'duck']],
        expected: { c: ['cat', 'cow'], d: ['dog', 'duck'] },
      },
      {
        input: [['apple', 'ant', 'banana']],
        expected: { a: ['apple', 'ant'], b: ['banana'] },
      },
    ],
    solution: `from collections import defaultdict

def group_by_first_letter(words):
    groups = defaultdict(list)
    for word in words:
        groups[word[0]].append(word)
    return dict(groups)


# For counting
from collections import defaultdict

def count_occurrences(items):
    counts = defaultdict(int)
    for item in items:
        counts[item] += 1
    return dict(counts)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 37,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-dataclass-basic',
    title: 'Basic Dataclass',
    difficulty: 'Easy',
    description: `Use @dataclass decorator to create data classes.

Dataclass auto-generates:
- __init__
- __repr__
- __eq__
- And more

**Benefits:** Less boilerplate, type hints, immutability option

This tests:
- @dataclass decorator
- Type hints
- Auto-generated methods`,
    examples: [
      {
        input: 'Person dataclass',
        output: 'Auto __init__, __repr__, etc.',
      },
    ],
    constraints: ['Use @dataclass', 'Add type hints'],
    hints: [
      'from dataclasses import dataclass',
      'Add type annotations',
      'Methods auto-generated',
    ],
    starterCode: `from dataclasses import dataclass

@dataclass
class Person:
    """Person data class"""
    name: str
    age: int
    email: str
    
    def is_adult(self) -> bool:
        """Check if person is adult"""
        return self.age >= 18


def test_dataclass():
    """Test dataclass"""
    person = Person("Alice", 25, "alice@example.com")
    
    # Auto-generated __repr__
    repr_str = repr(person)
    
    # Check if adult
    adult = person.is_adult()
    
    return person.age
`,
    testCases: [
      {
        input: [],
        expected: 25,
        functionName: 'test_dataclass',
      },
    ],
    solution: `from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    email: str
    
    def is_adult(self) -> bool:
        return self.age >= 18


def test_dataclass():
    person = Person("Alice", 25, "alice@example.com")
    repr_str = repr(person)
    adult = person.is_adult()
    return person.age`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 38,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-enumerate-start',
    title: 'Enumerate with Custom Start',
    difficulty: 'Easy',
    description: `Use enumerate with custom start index.

**Syntax:**
\`\`\`python
for i, item in enumerate(items, start=1):
    print(f"{i}. {item}")
\`\`\`

This tests:
- enumerate function
- start parameter
- Iteration patterns`,
    examples: [
      {
        input: 'enumerate(items, start=1)',
        output: 'Index starts at 1',
      },
    ],
    constraints: ['Use enumerate', 'Custom start'],
    hints: [
      'enumerate(iterable, start=n)',
      'Default start is 0',
      'Useful for 1-based numbering',
    ],
    starterCode: `def create_numbered_list(items):
    """
    Create numbered list starting from 1.
    
    Args:
        items: List of strings
        
    Returns:
        List of "1. item", "2. item", etc.
        
    Examples:
        >>> create_numbered_list(['a', 'b', 'c'])
        ['1. a', '2. b', '3. c']
    """
    pass


# Test
print(create_numbered_list(['apple', 'banana', 'cherry']))
`,
    testCases: [
      {
        input: [['apple', 'banana', 'cherry']],
        expected: ['1. apple', '2. banana', '3. cherry'],
      },
      {
        input: [['x', 'y']],
        expected: ['1. x', '2. y'],
      },
    ],
    solution: `def create_numbered_list(items):
    return [f"{i}. {item}" for i, item in enumerate(items, start=1)]


# Alternative
def create_numbered_list_custom_start(items, start=1):
    return [f"{i}. {item}" for i, item in enumerate(items, start=start)]`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 39,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-dict-merge-operators',
    title: 'Dict Merge Operators (| and |=)',
    difficulty: 'Easy',
    description: `Use Python 3.9+ dict merge operators.

**Operators:**
- | : Merge (like union)
- |= : In-place merge

**Example:**
\`\`\`python
d1 = {'a': 1, 'b': 2}
d2 = {'b': 3, 'c': 4}
merged = d1 | d2  # {'a': 1, 'b': 3, 'c': 4}
\`\`\`

This tests:
- Dict operators
- Merge behavior
- Modern Python syntax`,
    examples: [
      {
        input: 'd1 | d2',
        output: 'Merged dict, d2 values win',
      },
    ],
    constraints: ['Use | operator', 'Right dict values take precedence'],
    hints: [
      'd1 | d2 creates new dict',
      'd1 |= d2 modifies d1',
      'Later values override',
    ],
    starterCode: `def merge_configs(defaults, overrides):
    """
    Merge configuration dicts.
    
    Args:
        defaults: Default config
        overrides: Override values
        
    Returns:
        Merged dict (overrides take precedence)
        
    Examples:
        >>> defaults = {'color': 'blue', 'size': 10}
        >>> overrides = {'size': 20}
        >>> merge_configs(defaults, overrides)
        {'color': 'blue', 'size': 20}
    """
    pass


# Test
print(merge_configs({'a': 1, 'b': 2}, {'b': 3, 'c': 4}))
`,
    testCases: [
      {
        input: [
          { a: 1, b: 2 },
          { b: 3, c: 4 },
        ],
        expected: { a: 1, b: 3, c: 4 },
      },
      {
        input: [{ x: 10 }, { y: 20 }],
        expected: { x: 10, y: 20 },
      },
    ],
    solution: `def merge_configs(defaults, overrides):
    return defaults | overrides


# In-place merge
def merge_configs_inplace(defaults, overrides):
    defaults |= overrides
    return defaults


# Pre-3.9 alternative
def merge_configs_old(defaults, overrides):
    return {**defaults, **overrides}`,
    timeComplexity: 'O(n + m)',
    spaceComplexity: 'O(n + m)',
    order: 40,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-any-all-functions',
    title: 'Using any() and all()',
    difficulty: 'Easy',
    description: `Use any() and all() for efficient boolean checks.

**any()**: Returns True if any element is truthy
**all()**: Returns True if all elements are truthy

Both short-circuit!

This tests:
- Built-in boolean functions
- Generator expressions
- Short-circuit evaluation`,
    examples: [
      {
        input: 'all([True, True, False])',
        output: 'False',
      },
    ],
    constraints: ['Use any() or all()', 'Can use with generator'],
    hints: ['any(iterable)', 'all(iterable)', 'Short-circuits (stops early)'],
    starterCode: `def has_even(numbers):
    """Check if any number is even"""
    return any(n % 2 == 0 for n in numbers)


def all_positive(numbers):
    """Check if all numbers are positive"""
    return all(n > 0 for n in numbers)


def test_any_all():
    """Test any and all"""
    # Test any
    result1 = has_even([1, 3, 5, 6, 7])  # True
    
    # Test all
    result2 = all_positive([1, 2, 3, 4])  # True
    result3 = all_positive([1, -2, 3])    # False
    
    # Count True results
    return sum([result1, result2, not result3])
`,
    testCases: [
      {
        input: [],
        expected: 3,
        functionName: 'test_any_all',
      },
    ],
    solution: `def has_even(numbers):
    return any(n % 2 == 0 for n in numbers)


def all_positive(numbers):
    return all(n > 0 for n in numbers)


def test_any_all():
    result1 = has_even([1, 3, 5, 6, 7])
    result2 = all_positive([1, 2, 3, 4])
    result3 = all_positive([1, -2, 3])
    
    return sum([result1, result2, not result3])`,
    timeComplexity: 'O(n) worst case, can be O(1) with short-circuit',
    spaceComplexity: 'O(1)',
    order: 41,
    topic: 'Python Intermediate',
  },
];

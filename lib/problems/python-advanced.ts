/**
 * Python Advanced problems - Testing advanced Python patterns and features
 */

import { Problem } from '../types';
import { pythonAdvancedNew } from './python-advanced-new';

export const pythonAdvancedProblems: Problem[] = [
  {
    id: 'decorator-retry',
    title: 'Retry Decorator',
    difficulty: 'Medium',
    description: `Create a decorator retry that automatically retries a function if it raises an exception.

The decorator should:
- Accept a parameter max_attempts (number of retry attempts)
- Retry the function if it raises an exception
- Raise the last exception if all attempts fail
- Print attempt numbers for debugging

**Example:**
python
@retry(max_attempts=3)
def flaky_function():
    import random
    if random.random() < 0.7:
        raise ValueError("Failed!")
    return "Success!"
`,
    examples: [
      {
        input:
          '@retry(max_attempts=3), function that fails twice then succeeds',
        output: '"Success!" after 3 attempts',
        explanation: 'Function retries until success or max attempts reached.',
      },
    ],
    constraints: [
      'max_attempts >= 1',
      'Preserve function metadata',
      'Must work with any callable',
    ],
    hints: [
      'Use a closure to capture max_attempts',
      'Use functools.wraps to preserve metadata',
      'Use a loop for retry logic',
    ],
    starterCode: `from functools import wraps

def retry(max_attempts):
    """
    Decorator that retries a function on exception.
    
    Args:
        max_attempts: Maximum number of attempts
        
    Returns:
        Decorated function
    """
    # Your code here
    pass


# Test code
attempt_count = 0

@retry(max_attempts=3)
def failing_function():
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ValueError(f"Attempt {attempt_count} failed")
    return "Success!"


def test_retry():
    """Test function that validates retry decorator"""
    import sys
    from io import StringIO
    
    # Reset attempt count
    global attempt_count
    attempt_count = 0
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        result = failing_function()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify function returned correct value
        if result != "Success!":
            return "FAIL: Wrong return value"
        
        # Verify retry happened (should have 3 attempts)
        if attempt_count != 3:
            return f"FAIL: Expected 3 attempts, got {attempt_count}"
        
        # Verify decorator printed attempt info
        if "attempt" not in output.lower():
            return "FAIL: Decorator should print attempt numbers"
        
        return "Success!"
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"
`,
    testCases: [
      {
        input: [],
        expected: 'Success!',
        functionName: 'test_retry',
      },
    ],
    solution: `from functools import wraps

def retry(max_attempts):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    print(f"Attempt {attempt}/{max_attempts}")
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts:
                        raise
            return None
        return wrapper
    return decorator


# Test code
attempt_count = 0

@retry(max_attempts=3)
def failing_function():
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ValueError(f"Attempt {attempt_count} failed")
    return "Success!"


def test_retry():
    """Test function that validates retry decorator"""
    import sys
    from io import StringIO
    
    # Reset attempt count
    global attempt_count
    attempt_count = 0
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        result = failing_function()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify function returned correct value
        if result != "Success!":
            return "FAIL: Wrong return value"
        
        # Verify retry happened (should have 3 attempts)
        if attempt_count != 3:
            return f"FAIL: Expected 3 attempts, got {attempt_count}"
        
        # Verify decorator printed attempt info
        if "attempt" not in output.lower():
            return "FAIL: Decorator should print attempt numbers"
        
        return "Success!"
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 1,
    topic: 'Python Advanced',
  },
  {
    id: 'decorator-cache',
    title: 'Custom Cache Decorator',
    difficulty: 'Medium',
    description: `Implement a caching decorator similar to functools.lru_cache but simpler.

The decorator should:
- Cache function results based on arguments
- Return cached result if arguments match
- Work with both args and kwargs
- Handle unhashable arguments gracefully

**Note:** This tests understanding of closures, dictionaries, and argument handling.`,
    examples: [
      {
        input: '@cache, fibonacci(5) called twice',
        output: 'Second call returns cached result instantly',
      },
    ],
    constraints: [
      'Cache must be a dictionary',
      'Handle both positional and keyword arguments',
      'Arguments must be hashable',
    ],
    hints: [
      'Store cache in closure',
      'Use tuple of args and frozenset of kwargs as key',
      'Check cache before calling function',
    ],
    starterCode: `from functools import wraps

def cache(func):
    """
    Simple caching decorator.
    
    Args:
        func: Function to cache
        
    Returns:
        Decorated function with caching
    """
    # Your code here
    pass


@cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


call_count = 0

@cache
def counting_function(x):
    """Function that counts how many times it's actually called"""
    global call_count
    call_count += 1
    return x * 2


def test_cache():
    """Test function that validates cache decorator"""
    global call_count
    
    # First, test fibonacci works
    result = fibonacci(10)
    if result != 55:
        return f"FAIL: Wrong fibonacci result: {result}"
    
    # Test caching by calling same function multiple times
    call_count = 0
    r1 = counting_function(5)
    r2 = counting_function(5)
    r3 = counting_function(5)
    
    # Verify result is correct
    if r1 != 10 or r2 != 10 or r3 != 10:
        return "FAIL: Wrong cached result"
    
    # Verify caching happened (should only call once)
    if call_count != 1:
        return f"FAIL: Cache not working, function called {call_count} times instead of 1"
    
    return 55
`,
    testCases: [
      {
        input: [],
        expected: 55,
        functionName: 'test_cache',
      },
    ],
    solution: `from functools import wraps

def cache(func):
    _cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from args and kwargs
        key = (args, tuple(sorted(kwargs.items())))
        
        if key not in _cache:
            _cache[key] = func(*args, **kwargs)
        return _cache[key]
    
    return wrapper


@cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


call_count = 0

@cache
def counting_function(x):
    """Function that counts how many times it's actually called"""
    global call_count
    call_count += 1
    return x * 2


def test_cache():
    """Test function that validates cache decorator"""
    global call_count
    
    # First, test fibonacci works
    result = fibonacci(10)
    if result != 55:
        return f"FAIL: Wrong fibonacci result: {result}"
    
    # Test caching by calling same function multiple times
    call_count = 0
    r1 = counting_function(5)
    r2 = counting_function(5)
    r3 = counting_function(5)
    
    # Verify result is correct
    if r1 != 10 or r2 != 10 or r3 != 10:
        return "FAIL: Wrong cached result"
    
    # Verify caching happened (should only call once)
    if call_count != 1:
        return f"FAIL: Cache not working, function called {call_count} times instead of 1"
    
    return 55`,
    timeComplexity: 'O(1) for cached calls',
    spaceComplexity: 'O(n) for n unique calls',
    order: 2,
    topic: 'Python Advanced',
  },
  {
    id: 'decorator-timer',
    title: 'Function Timer Decorator',
    difficulty: 'Easy',
    description: `Create a decorator that measures and prints the execution time of a function.

The decorator should:
- Measure time before and after function execution
- Print the elapsed time in seconds
- Return the function result unchanged
- Preserve function metadata

**Use Case:** Performance profiling and optimization.`,
    examples: [
      {
        input: 'Function that sleeps for 1 second',
        output: 'Prints "Execution time: 1.00s"',
      },
    ],
    constraints: [
      'Use time.time() for measurement',
      'Print with 2 decimal places',
      'Must work with any function',
    ],
    hints: [
      'Import time module',
      'Record time before and after function call',
      'Use f-string for formatting',
    ],
    starterCode: `import time
from functools import wraps

def timer(func):
    """
    Decorator that times function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Decorated function
    """
    # Your code here
    pass


@timer
def slow_function():
    time.sleep(1)
    return "Done"


# Test helper function (for automated testing)
def test_timer():
    """Test function that verifies decorator works correctly"""
    import sys
    from io import StringIO
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Call the decorated function
        result = slow_function()
        
        # Get the printed output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify the function returned the correct value
        if result != "Done":
            return "FAIL: Wrong return value"
        
        # Verify decorator printed timing information
        if not output or "time" not in output.lower():
            return "FAIL: Decorator didn't print timing info"
        
        # Verify the output contains a number (the timing)
        import re
        if not re.search(r'\\d+\\.?\\d*', output):
            return "FAIL: No timing value found in output"
        
        return "Done"
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"
`,
    testCases: [
      {
        input: [],
        expected: 'Done',
        functionName: 'test_timer',
      },
    ],
    solution: `import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper


@timer
def slow_function():
    time.sleep(1)
    return "Done"


# Test helper function (for automated testing)
def test_timer():
    """Test function that verifies decorator works correctly"""
    import sys
    from io import StringIO
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Call the decorated function
        result = slow_function()
        
        # Get the printed output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify the function returned the correct value
        if result != "Done":
            return "FAIL: Wrong return value"
        
        # Verify decorator printed timing information
        if not output or "time" not in output.lower():
            return "FAIL: Decorator didn't print timing info"
        
        # Verify the output contains a number (the timing)
        import re
        if not re.search(r'\\d+\\.?\\d*', output):
            return "FAIL: No timing value found in output"
        
        return "Done"
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"`,
    timeComplexity: 'O(1) overhead',
    spaceComplexity: 'O(1)',
    order: 3,
    topic: 'Python Advanced',
  },
  {
    id: 'generator-fibonacci',
    title: 'Fibonacci Generator',
    difficulty: 'Easy',
    description: `Implement a generator that yields Fibonacci numbers infinitely.

The generator should:
- Yield Fibonacci numbers one at a time
- Start with 0, 1
- Never terminate (infinite sequence)
- Use O(1) space (only store last two numbers)

**Why Generator:** Fibonacci sequence can be infinite, and we often only need the first N numbers.`,
    examples: [
      {
        input: 'First 5 numbers',
        output: '[0, 1, 1, 2, 3]',
      },
    ],
    constraints: [
      'Must use yield',
      'O(1) space complexity',
      'Must be infinite',
    ],
    hints: [
      'Use two variables to track previous numbers',
      'Yield values in infinite loop',
      'Update variables after each yield',
    ],
    starterCode: `def fibonacci():
    """
    Generator that yields Fibonacci numbers infinitely.
    
    Yields:
        Next Fibonacci number
    """
    # Your code here
    pass


# Get first 10 Fibonacci numbers
import itertools
result = list(itertools.islice(fibonacci(), 10))
`,
    testCases: [
      {
        input: [],
        expected: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
      },
    ],
    solution: `def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


# Get first 10 Fibonacci numbers
import itertools
result = list(itertools.islice(fibonacci(), 10))`,
    timeComplexity: 'O(1) per number',
    spaceComplexity: 'O(1)',
    order: 4,
    topic: 'Python Advanced',
  },
  {
    id: 'generator-file-reader',
    title: 'Memory-Efficient File Reader',
    difficulty: 'Medium',
    description: `Create a generator that reads a large file line by line and filters lines containing a keyword.

The generator should:
- Read file lazily (one line at a time)
- Filter lines containing the keyword
- Strip whitespace from each line
- Work with files of any size without loading into memory

**Use Case:** Processing huge log files efficiently.`,
    examples: [
      {
        input: 'File with "ERROR" keyword',
        output: 'Yields only lines containing "ERROR"',
      },
    ],
    constraints: [
      'Must use generator (yield)',
      'Cannot load entire file into memory',
      'Case-sensitive search',
    ],
    hints: [
      'Use with open() for proper file handling',
      'Check if keyword in line',
      'Yield matching lines',
    ],
    starterCode: `def read_matching_lines(filepath, keyword):
    """
    Generator that yields lines containing keyword.
    
    Args:
        filepath: Path to file
        keyword: Keyword to search for
        
    Yields:
        Lines containing the keyword
    """
    # Your code here
    pass


# Test with simulated file lines (for testing without actual file)
def test_read_matching(keyword, lines):
    """Test helper that simulates file reading"""
    # Mock file by iterating over lines
    def mock_generator():
        for line in lines:
            if keyword in line:
                yield line.strip()
    return list(mock_generator())

# Test
result = test_read_matching('ERROR', ['INFO: Starting', 'ERROR: Failed', 'INFO: Done'])
`,
    testCases: [
      {
        input: [],
        expected: ['ERROR: Failed'],
      },
    ],
    solution: `def read_matching_lines(filepath, keyword):
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if keyword in line:
                yield line


# Test with simulated file lines (for testing without actual file)
def test_read_matching(keyword, lines):
    """Test helper that simulates file reading"""
    def mock_generator():
        for line in lines:
            if keyword in line:
                yield line.strip()
    return list(mock_generator())

result = test_read_matching('ERROR', ['INFO: Starting', 'ERROR: Failed', 'INFO: Done'])`,
    timeComplexity: 'O(n) where n is number of lines',
    spaceComplexity: 'O(1)',
    order: 5,
    topic: 'Python Advanced',
  },
  {
    id: 'context-manager-timer',
    title: 'Timer Context Manager',
    difficulty: 'Easy',
    description: `Implement a context manager that times code execution in a with block.

The context manager should:
- Record start time on entry
- Calculate and print elapsed time on exit
- Work even if exception occurs
- Print time with 3 decimal places

**Pattern:**
python
with Timer():
    # code to time
    time.sleep(1)
# Prints: "Elapsed time: 1.000s"
`,
    examples: [
      {
        input: 'Code block that takes 0.5 seconds',
        output: 'Prints "Elapsed time: 0.500s"',
      },
    ],
    constraints: [
      'Must implement __enter__ and __exit__',
      'Print even if exception occurs',
      'Use time.time() for measurement',
    ],
    hints: [
      'Store start time in __enter__',
      'Calculate elapsed in __exit__',
      '__exit__ receives exception info',
    ],
    starterCode: `import time

class Timer:
    """
    Context manager that times code execution.
    """
    
    def __init__(self):
        """Initialize timer."""
        # TODO: Initialize start and end times
        self.start = None
        self.end = None
    
    def __enter__(self):
        # TODO: Record start time
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: Calculate and print elapsed time
        pass


with Timer():
    time.sleep(0.5)


# Test helper function (for automated testing)
def test_timer(sleep_duration):
    """Test function for Timer - implement the class methods above first!"""
    try:
        with Timer():
            time.sleep(sleep_duration)
        return 'Elapsed'  # If it completes without error
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: [0.5], // sleep duration
        expected: 'Elapsed',
        functionName: 'test_timer',
      },
    ],
    solution: `import time

class Timer:
    def __init__(self):
        self.start = None
        self.end = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        elapsed = self.end - self.start
        print(f"Elapsed time: {elapsed:.3f}s")
        return False  # Don't suppress exceptions


# Test helper function (for automated testing)
def test_timer(sleep_duration):
    """Test function for Timer."""
    with Timer():
        time.sleep(sleep_duration)
    return 'Elapsed'`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 6,
    topic: 'Python Advanced',
  },
  {
    id: 'context-manager-database',
    title: 'Database Transaction Context Manager',
    difficulty: 'Medium',
    description: `Create a context manager that simulates database transaction management.

The context manager should:
- Begin transaction on entry
- Commit if no exception occurs
- Rollback if exception occurs
- Close connection in both cases

**Pattern:**
python
with Transaction(db):
    db.execute("INSERT ...")
    db.execute("UPDATE ...")
# Commits if successful, rolls back if error
`,
    examples: [
      {
        input: 'Successful operations',
        output: 'Commit called',
      },
      {
        input: 'Operation raises exception',
        output: 'Rollback called',
      },
    ],
    constraints: [
      'Commit only if no exception',
      'Always rollback on exception',
      'Connection must close regardless',
    ],
    hints: [
      'Check exc_type in __exit__',
      'exc_type is None if no exception',
      'Use try/finally for cleanup',
    ],
    starterCode: `class Transaction:
    """
    Context manager for database transactions.
    """
    
    def __init__(self, connection):
        self.conn = connection
    
    def __enter__(self):
        # Your code here
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Your code here
        pass


# Mock database connection
class MockDB:
    def begin(self):
        print("Transaction started")
    
    def commit(self):
        print("Transaction committed")
    
    def rollback(self):
        print("Transaction rolled back")
    
    def close(self):
        print("Connection closed")

db = MockDB()
with Transaction(db):
    print("Doing work...")


# Test helper function (for automated testing)
def test_transaction(should_succeed):
    """Test function for Transaction - implement the class methods above first!"""
    try:
        mock_db = MockDB()
        status = []
        
        # Override methods to capture what was called
        original_commit = mock_db.commit
        original_rollback = mock_db.rollback
        
        def capture_commit():
            status.append('committed')
            original_commit()
        
        def capture_rollback():
            status.append('rolled back')
            original_rollback()
        
        mock_db.commit = capture_commit
        mock_db.rollback = capture_rollback
        
        try:
            with Transaction(mock_db):
                if not should_succeed:
                    raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected for failure case
        
        return status[0] if status else None
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: [true], // success case
        expected: 'committed',
        functionName: 'test_transaction',
      },
      {
        input: [false], // error case
        expected: 'rolled back',
        functionName: 'test_transaction',
      },
    ],
    solution: `class Transaction:
    def __init__(self, connection):
        self.conn = connection
    
    def __enter__(self):
        self.conn.begin()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.conn.close()
        return False  # Don't suppress exceptions


class MockDB:
    def begin(self):
        print("Transaction started")
    
    def commit(self):
        print("Transaction committed")
    
    def rollback(self):
        print("Transaction rolled back")
    
    def close(self):
        print("Connection closed")


# Test helper function (for automated testing)
def test_transaction(should_succeed):
    """Test function for Transaction."""
    mock_db = MockDB()
    status = []
    
    original_commit = mock_db.commit
    original_rollback = mock_db.rollback
    
    def capture_commit():
        status.append('committed')
        original_commit()
    
    def capture_rollback():
        status.append('rolled back')
        original_rollback()
    
    mock_db.commit = capture_commit
    mock_db.rollback = capture_rollback
    
    try:
        with Transaction(mock_db):
            if not should_succeed:
                raise ValueError("Simulated error")
    except ValueError:
        pass
    
    return status[0] if status else None`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 7,
    topic: 'Python Advanced',
  },
  {
    id: 'metaclass-singleton',
    title: 'Singleton Metaclass',
    difficulty: 'Hard',
    description: `Implement a metaclass that ensures a class can only have one instance (Singleton pattern).

The metaclass should:
- Store instances in a class-level dictionary
- Return existing instance if one exists
- Create new instance only if needed
- Work with any class that uses it

**Use Case:** Database connections, configuration objects, logging.`,
    examples: [
      {
        input: 'Database() called twice',
        output: 'Returns same instance both times',
      },
    ],
    constraints: [
      'Must be a metaclass',
      'Thread-safety not required',
      'Support class arguments',
    ],
    hints: [
      'Override __call__ method',
      'Store instances in _instances dict',
      'Check if class in dict before creating',
    ],
    starterCode: `class Singleton(type):
    """
    Metaclass that creates singleton classes.
    """
    
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        # Your code here
        pass


class Database(metaclass=Singleton):
    def __init__(self, name="default"):
        self.name = name


db1 = Database("prod")
db2 = Database("dev")
print(db1 is db2)  # Should be True
print(db1.name)    # Should be "prod" (first call wins)


# Test helper function (for automated testing)
def test_singleton(names):
    """Test function for Singleton - implement the metaclass above first!"""
    try:
        class TestDB(metaclass=Singleton):
            def __init__(self, name="default"):
                self.name = name
        
        db1 = TestDB(names[0])
        db2 = TestDB(names[1])
        return db1 is db2  # Should be True for singleton
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: [['prod', 'dev']],
        expected: true, // db1 is db2
        functionName: 'test_singleton',
      },
    ],
    solution: `class Singleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# Test helper function (for automated testing)
def test_singleton(names):
    """Test function for Singleton."""
    class TestDB(metaclass=Singleton):
        def __init__(self, name="default"):
            self.name = name
    
    db1 = TestDB(names[0])
    db2 = TestDB(names[1])
    return db1 is db2`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1) per class',
    order: 8,
    topic: 'Python Advanced',
  },
  {
    id: 'property-descriptor',
    title: 'Validated Property Descriptor',
    difficulty: 'Hard',
    description: `Create a descriptor that validates values before setting them.

The descriptor should:
- Accept a validation function
- Validate value in __set__
- Raise ValueError if validation fails
- Store value in instance dictionary

**Example:**
python
class Person:
    age = ValidatedProperty(lambda x: 0 <= x <= 150)

p = Person()
p.age = 25  # OK
p.age = -5  # Raises ValueError
`,
    examples: [
      {
        input: 'age = 25',
        output: 'Value stored successfully',
      },
      {
        input: 'age = -5',
        output: 'ValueError raised',
      },
    ],
    constraints: [
      'Must be a descriptor',
      'Implement __get__ and __set__',
      'Store data in instance __dict__',
    ],
    hints: [
      'Use instance.__dict__ for storage',
      'Call validation function before setting',
      'Use unique attribute name to avoid recursion',
    ],
    starterCode: `class ValidatedProperty:
    """
    Descriptor that validates values.
    """
    
    def __init__(self, validator):
        self.validator = validator
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        # Your code here
        pass
    
    def __set__(self, instance, value):
        # Your code here
        pass


class Person:
    age = ValidatedProperty(lambda x: 0 <= x <= 150)

p = Person()
p.age = 25
print(p.age)


# Test helper function (for automated testing)
def test_validated_property(value):
    """Test function for ValidatedProperty - implement the descriptor above first!"""
    try:
        class TestPerson:
            age = ValidatedProperty(lambda x: 0 <= x <= 150)
        
        p = TestPerson()
        p.age = value
        return p.age
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: [25],
        expected: 25,
        functionName: 'test_validated_property',
      },
    ],
    solution: `class ValidatedProperty:
    def __init__(self, validator):
        self.validator = validator
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = f"_{name}"
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if not self.validator(value):
            raise ValueError(f"Invalid value: {value}")
        instance.__dict__[self.name] = value


# Test helper function (for automated testing)
def test_validated_property(value):
    """Test function for ValidatedProperty."""
    class TestPerson:
        age = ValidatedProperty(lambda x: 0 <= x <= 150)
    
    p = TestPerson()
    p.age = value
    return p.age`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 9,
    topic: 'Python Advanced',
  },
  {
    id: 'iterator-custom',
    title: 'Custom Range Iterator',
    difficulty: 'Medium',
    description: `Implement a custom iterator class similar to Python's range().

The iterator should:
- Support start, stop, and step parameters
- Implement __iter__ and __next__
- Raise StopIteration when done
- Support both forward and backward iteration

**Pattern:**
python
for i in CustomRange(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8
`,
    examples: [
      {
        input: 'CustomRange(0, 10, 2)',
        output: '[0, 2, 4, 6, 8]',
      },
      {
        input: 'CustomRange(10, 0, -2)',
        output: '[10, 8, 6, 4, 2]',
      },
    ],
    constraints: [
      'Must implement iterator protocol',
      'Support positive and negative step',
      'Handle edge cases',
    ],
    hints: [
      '__iter__ should return self',
      'Track current value',
      'Check stopping condition in __next__',
    ],
    starterCode: `class CustomRange:
    """
    Custom range iterator.
    """
    
    def __init__(self, start, stop, step=1):
        # TODO: Store start, stop, step and initialize current
        self.start = start
        self.stop = stop
        self.step = step
        self.current = start
    
    def __iter__(self):
        # TODO: Return self and reset current
        pass
    
    def __next__(self):
        # TODO: Check if done, return current and increment
        pass


# Test
result = list(CustomRange(0, 10, 2))
print(result)


# Test helper function (for automated testing)
def test_custom_range(start, stop, step):
    """Test function for CustomRange - implement the class methods above first!"""
    try:
        return list(CustomRange(start, stop, step))
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: [0, 10, 2],
        expected: [0, 2, 4, 6, 8],
        functionName: 'test_custom_range',
      },
      {
        input: [10, 0, -2],
        expected: [10, 8, 6, 4, 2],
        functionName: 'test_custom_range',
      },
    ],
    solution: `class CustomRange:
    def __init__(self, start, stop, step=1):
        self.current = start
        self.stop = stop
        self.step = step
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if (self.step > 0 and self.current >= self.stop) or \\
           (self.step < 0 and self.current <= self.stop):
            raise StopIteration
        
        value = self.current
        self.current += self.step
        return value


# Test helper function (for automated testing)
def test_custom_range(start, stop, step):
    """Test function for CustomRange."""
    return list(CustomRange(start, stop, step))`,
    timeComplexity: 'O(1) per iteration',
    spaceComplexity: 'O(1)',
    order: 10,
    topic: 'Python Advanced',
  },
  {
    id: 'coroutine-pipeline',
    title: 'Data Processing Pipeline with Coroutines',
    difficulty: 'Hard',
    description: `Create a data processing pipeline using coroutines (generators with send()).

Build a pipeline that:
- Accepts data via send()
- Processes data through multiple stages
- Each stage is a coroutine
- Data flows: source -> processor -> sink

**Pattern:**
python
pipeline = source() | process() | sink()
for item in data:
    pipeline.send(item)
`,
    examples: [
      {
        input: 'Numbers 1-5',
        output: 'Processes through pipeline stages',
      },
    ],
    constraints: [
      'Use coroutines (yield with send)',
      'Chain coroutines together',
      'Prime coroutines with next()',
    ],
    hints: [
      'Each coroutine yields then receives',
      'Target coroutine in each stage',
      'Prime with next() before sending',
    ],
    starterCode: `def producer(target):
    """
    Producer coroutine that sends data to target.
    """
    # Your code here
    pass

def processor(target, transform):
    """
    Processor coroutine that transforms and forwards data.
    """
    # Your code here
    pass

def consumer():
    """
    Consumer coroutine that receives and prints data.
    """
    # Your code here
    pass


# Build pipeline: double numbers then print
sink = consumer()
proc = processor(sink, lambda x: x * 2)
source = producer(proc)

for i in range(5):
    source.send(i)
`,
    testCases: [
      {
        input: [[1, 2, 3]],
        expected: [2, 4, 6],
      },
    ],
    solution: `def producer(target):
    while True:
        item = yield
        target.send(item)

def processor(target, transform):
    while True:
        item = yield
        result = transform(item)
        target.send(result)

def consumer():
    while True:
        item = yield
        print(item)

# Prime coroutines
def prime(coro):
    next(coro)
    return coro`,
    timeComplexity: 'O(n) where n is items',
    spaceComplexity: 'O(1)',
    order: 11,
    topic: 'Python Advanced',
  },
  {
    id: 'async-context-manager',
    title: 'Async Context Manager',
    difficulty: 'Hard',
    description: `Implement an async context manager for async resource management.

The context manager should:
- Implement __aenter__ and __aexit__
- Work with async with statement
- Handle async setup and cleanup
- Properly handle exceptions

**Use Case:** Async database connections, async file I/O.`,
    examples: [
      {
        input: 'Async resource access',
        output: 'Async setup and cleanup',
      },
    ],
    constraints: [
      'Must be async context manager',
      'Use async/await',
      'Handle exceptions properly',
    ],
    hints: [
      'Implement __aenter__ and __aexit__',
      'Both methods are async',
      'Use await for async operations',
    ],
    starterCode: `import asyncio

class AsyncResource:
    """
    Async context manager for resource.
    """
    
    async def __aenter__(self):
        # Your code here
        pass
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Your code here
        pass


async def main():
    async with AsyncResource() as resource:
        print("Using resource")

asyncio.run(main())
`,
    testCases: [
      {
        input: [true],
        expected: 'Success',
      },
    ],
    solution: `import asyncio

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        await asyncio.sleep(0.1)  # Simulate async setup
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource")
        await asyncio.sleep(0.1)  # Simulate async cleanup
        return False`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 12,
    topic: 'Python Advanced',
  },
  {
    id: 'decorator-params',
    title: 'Parameterized Decorator with State',
    difficulty: 'Hard',
    description: `Create a decorator that counts function calls and limits execution.

The decorator should:
- Accept max_calls parameter
- Count how many times function is called
- Raise exception after max_calls
- Provide a reset() method

**Example:**
python
@limit_calls(max_calls=3)
def api_call():
    return "Success"

api_call()  # OK
api_call()  # OK
api_call()  # OK
api_call()  # Raises RuntimeError
`,
    examples: [
      {
        input: 'max_calls=3, called 4 times',
        output: 'Fourth call raises RuntimeError',
      },
    ],
    constraints: [
      'Decorator takes parameters',
      'Must track state across calls',
      'Provide reset mechanism',
    ],
    hints: [
      'Three levels of functions needed',
      'Store count in closure',
      'Add reset as wrapper attribute',
    ],
    starterCode: `from functools import wraps

def limit_calls(max_calls):
    """
    Decorator that limits function calls.
    
    Args:
        max_calls: Maximum number of calls allowed
    """
    # Your code here
    pass


@limit_calls(max_calls=3)
def api_call():
    return "Success"

results = []
for i in range(4):
    try:
        results.append(api_call())
    except RuntimeError as e:
        results.append(f"Error: {e}")
`,
    testCases: [
      {
        input: [],
        expected: [
          'Success',
          'Success',
          'Success',
          'Error: api_call called more than 3 times',
        ],
      },
    ],
    solution: `from functools import wraps

def limit_calls(max_calls):
    def decorator(func):
        count = [0]  # Use list for mutability in closure
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            if count[0] >= max_calls:
                raise RuntimeError(
                    f"{func.__name__} called more than {max_calls} times"
                )
            count[0] += 1
            return func(*args, **kwargs)
        
        def reset():
            count[0] = 0
        
        wrapper.reset = reset
        return wrapper
    return decorator


@limit_calls(max_calls=3)
def api_call():
    return "Success"

results = []
for i in range(4):
    try:
        results.append(api_call())
    except RuntimeError as e:
        results.append(f"Error: {e}")`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 13,
    topic: 'Python Advanced',
  },
  {
    id: 'generator-send',
    title: 'Generator with Send - Running Average',
    difficulty: 'Medium',
    description: `Implement a generator that calculates a running average using send().

The generator should:
- Accept values via send()
- Maintain running total and count
- Yield current average after each value
- Handle first next() call (prime)

**Use Case:** Real-time statistics, streaming data analysis.`,
    examples: [
      {
        input: 'send(10), send(20), send(30)',
        output: 'Yields 10.0, 15.0, 20.0',
      },
    ],
    constraints: [
      'Use generator with send()',
      'Calculate average correctly',
      'Handle initialization',
    ],
    hints: [
      'First yield returns None (for priming)',
      'Receive value with yield',
      'Update total and count',
    ],
    starterCode: `def running_average():
    """
    Generator that calculates running average.
    
    Yields:
        Current average
    """
    # Your code here
    pass


avg = running_average()
next(avg)  # Prime the generator

print(avg.send(10))  # 10.0
print(avg.send(20))  # 15.0
print(avg.send(30))  # 20.0


# Test helper function
def test_running_average(values):
    """Test running average with list of values"""
    avg = running_average()
    next(avg)  # Prime the generator
    results = []
    for val in values:
        results.append(avg.send(val))
    return results

result = test_running_average([10, 20, 30])
`,
    testCases: [
      {
        input: [],
        expected: [10.0, 15.0, 20.0],
      },
    ],
    solution: `def running_average():
    total = 0
    count = 0
    average = None
    
    while True:
        value = yield average
        if value is not None:
            total += value
            count += 1
            average = total / count


# Test helper function
def test_running_average(values):
    """Test running average with list of values"""
    avg = running_average()
    next(avg)  # Prime the generator
    results = []
    for val in values:
        results.append(avg.send(val))
    return results

result = test_running_average([10, 20, 30])`,
    timeComplexity: 'O(1) per value',
    spaceComplexity: 'O(1)',
    order: 14,
    topic: 'Python Advanced',
  },
  {
    id: 'metaclass-registry',
    title: 'Auto-Registration Metaclass',
    difficulty: 'Hard',
    description: `Create a metaclass that automatically registers classes in a registry.

The metaclass should:
- Maintain a class-level registry
- Auto-register classes on creation
- Provide lookup by name
- Skip abstract base classes

**Use Case:** Plugin systems, command registries, API endpoints.`,
    examples: [
      {
        input: 'class UserCommand(Command)',
        output: 'Automatically registered as "user"',
      },
    ],
    constraints: [
      'Must be a metaclass',
      'Auto-register on class creation',
      'Skip classes without name attribute',
    ],
    hints: [
      'Override __new__ method',
      'Check for name attribute',
      'Store in class-level dictionary',
    ],
    starterCode: `class Registry(type):
    """
    Metaclass that auto-registers classes.
    """
    
    _registry = {}
    
    def __new__(mcs, name, bases, attrs):
        # Your code here
        pass
    
    @classmethod
    def get(mcs, name):
        """Get class by registered name."""
        # Your code here
        pass


class Command(metaclass=Registry):
    name = None  # Subclasses should set this


class UserCommand(Command):
    name = "user"
    
    def execute(self):
        return "User command executed"


# Test
cmd_class = Registry.get("user")
cmd = cmd_class()
print(cmd.execute())


# Test helper function (for automated testing)
def test_registry(name):
    """Test function for Registry - implement the metaclass above first!"""
    try:
        class TestRegistry(type):
            _registry = {}
            
            def __new__(mcs, cls_name, bases, attrs):
                cls = super().__new__(mcs, cls_name, bases, attrs)
                if 'name' in attrs and attrs['name'] is not None:
                    mcs._registry[attrs['name']] = cls
                return cls
            
            @classmethod
            def get(mcs, name):
                return mcs._registry.get(name)
        
        class TestCommand(metaclass=TestRegistry):
            name = None
        
        class TestUserCommand(TestCommand):
            name = "user"
            def execute(self):
                return "User command executed"
        
        cmd_class = TestRegistry.get(name)
        if cmd_class:
            cmd = cmd_class()
            return cmd.execute()
        return None
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: ['user'],
        expected: 'User command executed',
        functionName: 'test_registry',
      },
    ],
    solution: `class Registry(type):
    _registry = {}
    
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        
        # Register if has name attribute
        if 'name' in attrs and attrs['name'] is not None:
            mcs._registry[attrs['name']] = cls
        
        return cls
    
    @classmethod
    def get(mcs, name):
        return mcs._registry.get(name)


# Test helper function (for automated testing)
def test_registry(name):
    """Test function for Registry."""
    cmd_class = Registry.get(name)
    if cmd_class:
        cmd = cmd_class()
        return cmd.execute()
    return None


class Command(metaclass=Registry):
    name = None


class UserCommand(Command):
    name = "user"
    
    def execute(self):
        return "User command executed"`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(n) for n classes',
    order: 15,
    topic: 'Python Advanced',
  },
  {
    id: 'descriptor-validation',
    title: 'Type-Validated Descriptor',
    difficulty: 'Hard',
    description: `Create a descriptor that enforces type checking on attribute assignment.

The descriptor should:
- Accept expected type(s) in __init__
- Validate type in __set__
- Raise TypeError for wrong types
- Support multiple types (Union)

**Example:**
python
class Person:
    age = TypedProperty(int)
    name = TypedProperty(str, type(None))

p = Person()
p.age = 25  # OK
p.age = "25"  # TypeError
p.name = None  # OK (allows None)
`,
    examples: [
      {
        input: 'age = 25 (int)',
        output: 'Accepted',
      },
      {
        input: 'age = "25" (str)',
        output: 'TypeError raised',
      },
    ],
    constraints: [
      'Must be a descriptor',
      'Support multiple types',
      'Clear error messages',
    ],
    hints: [
      'Use isinstance for type checking',
      'Store allowed types',
      'Use __set_name__ for attribute name',
    ],
    starterCode: `class TypedProperty:
    """
    Descriptor that enforces type checking.
    """
    
    def __init__(self, *expected_types):
        # Your code here
        pass
    
    def __set_name__(self, owner, name):
        # Your code here
        pass
    
    def __get__(self, instance, owner):
        # Your code here
        pass
    
    def __set__(self, instance, value):
        # Your code here
        pass


class Person:
    age = TypedProperty(int)
    name = TypedProperty(str, type(None))

p = Person()
p.age = 25
print(p.age)


def test_typed_property(age_value):
    """Test function for TypedProperty."""
    class Person:
        age = TypedProperty(int)
    
    p = Person()
    p.age = age_value
    return p.age
`,
    testCases: [
      {
        input: [25],
        expected: 25,
        functionName: 'test_typed_property',
      },
    ],
    solution: `class TypedProperty:
    def __init__(self, *expected_types):
        self.expected_types = expected_types
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = f"_{name}"
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if not isinstance(value, self.expected_types):
            raise TypeError(
                f"{self.name[1:]} must be {self.expected_types}, "
                f"got {type(value)}"
            )
        instance.__dict__[self.name] = value


def test_typed_property(age_value):
    """Test function for TypedProperty."""
    class Person:
        age = TypedProperty(int)
    
    p = Person()
    p.age = age_value
    return p.age`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 16,
    topic: 'Python Advanced',
  },
  {
    id: 'generator-pipeline',
    title: 'Chained Generator Pipeline',
    difficulty: 'Medium',
    description: `Build a data pipeline using chained generators for memory-efficient processing.

Create generators that:
- read_numbers: yields numbers from a list
- filter_even: filters even numbers
- square: squares each number
- Chain them together

**Key Concept:** Generators enable lazy evaluation - no intermediate lists created.`,
    examples: [
      {
        input: '[1, 2, 3, 4, 5, 6]',
        output: '[4, 16, 36] (even numbers squared)',
      },
    ],
    constraints: [
      'Each stage must be a generator',
      'No intermediate lists',
      'Chain with function composition',
    ],
    hints: [
      'Each generator takes previous as input',
      'Use yield in loops',
      'Composition: square(filter_even(read_numbers()))',
    ],
    starterCode: `def read_numbers(numbers):
    """Yield numbers from list."""
    # Your code here
    pass

def filter_even(numbers):
    """Yield only even numbers."""
    # Your code here
    pass

def square(numbers):
    """Yield squared numbers."""
    # Your code here
    pass


# Build pipeline
data = [1, 2, 3, 4, 5, 6]
pipeline = square(filter_even(read_numbers(data)))
result = list(pipeline)
`,
    testCases: [
      {
        input: [],
        expected: [4, 16, 36],
      },
    ],
    solution: `def read_numbers(numbers):
    for num in numbers:
        yield num

def filter_even(numbers):
    for num in numbers:
        if num % 2 == 0:
            yield num

def square(numbers):
    for num in numbers:
        yield num ** 2


# Build pipeline
data = [1, 2, 3, 4, 5, 6]
pipeline = square(filter_even(read_numbers(data)))
result = list(pipeline)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 17,
    topic: 'Python Advanced',
  },
  {
    id: 'context-manager-suppress',
    title: 'Exception Suppressing Context Manager',
    difficulty: 'Medium',
    description: `Create a context manager that suppresses specific exceptions.

The context manager should:
- Accept exception types to suppress
- Suppress only those exceptions
- Let other exceptions propagate
- Log suppressed exceptions

**Use Case:** Gracefully handling expected errors.`,
    examples: [
      {
        input: 'with suppress(ValueError): int("abc")',
        output: 'ValueError suppressed',
      },
    ],
    constraints: [
      'Accept multiple exception types',
      'Only suppress specified types',
      'Return True to suppress in __exit__',
    ],
    hints: [
      'Store exception types in __init__',
      'Check exc_type in __exit__',
      'Use isinstance for checking',
    ],
    starterCode: `class suppress:
    """
    Context manager that suppresses exceptions.
    """
    
    def __init__(self, *exceptions):
        # Your code here
        pass
    
    def __enter__(self):
        # Your code here
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Your code here
        pass


# Test
with suppress(ValueError, TypeError):
    int("not a number")  # Suppressed
    print("This runs")

with suppress(ValueError):
    1 / 0  # Not suppressed (ZeroDivisionError)


def test_suppress(suppress_types, error_type):
    """Test function for suppress context manager."""
    # Map string names to exception classes
    exception_map = {
        'ValueError': ValueError,
        'TypeError': TypeError,
        'ZeroDivisionError': ZeroDivisionError
    }
    
    # Get exception classes from strings
    exceptions_to_suppress = tuple(exception_map[name] for name in suppress_types)
    error_to_raise = exception_map[error_type]
    
    try:
        with suppress(*exceptions_to_suppress):
            raise error_to_raise("test error")
        return 'suppressed'
    except:
        return 'not suppressed'
`,
    testCases: [
      {
        input: [['ValueError'], 'ValueError'],
        expected: 'suppressed',
        functionName: 'test_suppress',
      },
      {
        input: [['ValueError'], 'TypeError'],
        expected: 'not suppressed',
        functionName: 'test_suppress',
      },
    ],
    solution: `class suppress:
    def __init__(self, *exceptions):
        self.exceptions = exceptions
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False
        
        # Suppress if exception type matches
        if issubclass(exc_type, self.exceptions):
            print(f"Suppressed {exc_type.__name__}")
            return True  # Suppress exception
        
        return False  # Let exception propagate


def test_suppress(suppress_types, error_type):
    """Test function for suppress context manager."""
    # Map string names to exception classes
    exception_map = {
        'ValueError': ValueError,
        'TypeError': TypeError,
        'ZeroDivisionError': ZeroDivisionError
    }
    
    # Get exception classes from strings
    exceptions_to_suppress = tuple(exception_map[name] for name in suppress_types)
    error_to_raise = exception_map[error_type]
    
    try:
        with suppress(*exceptions_to_suppress):
            raise error_to_raise("test error")
        return 'suppressed'
    except:
        return 'not suppressed'`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 18,
    topic: 'Python Advanced',
  },
  {
    id: 'decorator-class',
    title: 'Class-Based Decorator',
    difficulty: 'Medium',
    description: `Implement a decorator using a class instead of a function.

The class decorator should:
- Implement __init__ and __call__
- Store the wrapped function
- Count calls to the function
- Provide a get_count() method

**Pattern:** Class decorators use __call__ to make instances callable.`,
    examples: [
      {
        input: '@CountCalls, function called 3 times',
        output: 'get_count() returns 3',
      },
    ],
    constraints: ['Must be a class', 'Implement __call__', 'Track call count'],
    hints: [
      '__init__ receives the function',
      '__call__ makes instance callable',
      'Use instance variable for count',
    ],
    starterCode: `from functools import wraps

class CountCalls:
    """
    Class-based decorator that counts calls.
    """
    
    def __init__(self, func):
        # Your code here
        pass
    
    def __call__(self, *args, **kwargs):
        # Your code here
        pass
    
    def get_count(self):
        # Your code here
        pass


@CountCalls
def greet(name):
    return f"Hello, {name}!"

greet("Alice")
greet("Bob")
greet("Charlie")

result = greet.get_count()
`,
    testCases: [
      {
        input: [],
        expected: 3,
      },
    ],
    solution: `from functools import wraps, update_wrapper

class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0
        update_wrapper(self, func)
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)
    
    def get_count(self):
        return self.count


@CountCalls
def greet(name):
    return f"Hello, {name}!"

greet("Alice")
greet("Bob")
greet("Charlie")

result = greet.get_count()`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 19,
    topic: 'Python Advanced',
  },
  {
    id: 'functools-compose',
    title: 'Function Composition',
    difficulty: 'Medium',
    description: `Implement a compose function that combines multiple functions into one.

The compose function should:
- Accept multiple functions as arguments
- Return a new function that applies them right-to-left
- Handle any number of functions
- Work like mathematical composition: (f ∘ g)(x) = f(g(x))

**Use Case:** Functional programming, data transformations.`,
    examples: [
      {
        input: 'compose(str, lambda x: x*2, lambda x: x+1)(5)',
        output: '"12" ((5+1)*2 converted to string)',
      },
    ],
    constraints: [
      'Apply functions right-to-left',
      'Work with any number of functions',
      'Handle any argument types',
    ],
    hints: [
      'Use reduce or loop through functions',
      'Apply functions from right to left',
      'Return a new function that does composition',
    ],
    starterCode: `from functools import reduce

def compose(*functions):
    """
    Compose multiple functions into one.
    
    Args:
        *functions: Functions to compose (applied right-to-left)
        
    Returns:
        Composed function
    """
    # Your code here
    pass


# Test
add_one = lambda x: x + 1
double = lambda x: x * 2
to_string = str

combined = compose(to_string, double, add_one)
print(combined(5))  # "12"
`,
    testCases: [
      {
        input: [5],
        expected: '12',
      },
    ],
    solution: `from functools import reduce

def compose(*functions):
    def composed(arg):
        return reduce(
            lambda result, func: func(result),
            reversed(functions),
            arg
        )
    return composed

# Alternative using foldr pattern
def compose_alt(*functions):
    if not functions:
        return lambda x: x
    
    if len(functions) == 1:
        return functions[0]
    
    def inner(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    
    return inner`,
    timeComplexity: 'O(n) where n is number of functions',
    spaceComplexity: 'O(1)',
    order: 20,
    topic: 'Python Advanced',
  },
  {
    id: 'advanced-collections-counter',
    title: 'Counter for Frequency Analysis',
    difficulty: 'Easy',
    description: `Use collections.Counter for efficient frequency counting and operations.

Implement functions using Counter:
- Find the k most common elements
- Find elements that appear more than n times
- Perform counter arithmetic (addition, subtraction)
- Find missing elements between two collections

**Library:** collections.Counter provides dict subclass for counting hashable objects.`,
    examples: [
      {
        input: 'most_common([1,1,1,2,2,3], k=2)',
        output: '[(1, 3), (2, 2)]',
      },
    ],
    constraints: [
      'Use Counter methods',
      'Handle edge cases',
      'O(n) time complexity',
    ],
    hints: [
      'Counter has most_common() method',
      'Counters support arithmetic operations',
      'Subtract to find differences',
    ],
    starterCode: `from collections import Counter

def most_common_elements(items, k):
    """Find k most common elements.
    
    Args:
        items: List of items
        k: Number of most common to return
        
    Returns:
        List of (item, count) tuples
    """
    pass


def elements_above_threshold(items, threshold):
    """Find elements appearing more than threshold times.
    
    Args:
        items: List of items
        threshold: Minimum count
        
    Returns:
        List of items above threshold
    """
    pass


def counter_difference(list1, list2):
    """Find elements in list1 but not in list2 or with fewer occurrences.
    
    Args:
        list1: First list
        list2: Second list
        
    Returns:
        Counter of differences
    """
    pass


# Test
print(most_common_elements([1,1,1,2,2,3,3,3,3], 2))
print(elements_above_threshold(['a','a','b','b','b','c'], 2))
print(counter_difference([1,1,2,2,3], [1,2,2,2]))
,`,
    testCases: [
      {
        input: [[1, 1, 1, 2, 2, 3], 2],
        expected: [
          [1, 3],
          [2, 2],
        ],
      },
    ],
    solution: `from collections import Counter

def most_common_elements(items, k):
    return Counter(items).most_common(k)


def elements_above_threshold(items, threshold):
    counter = Counter(items)
    return [item for item, count in counter.items() if count > threshold]


def counter_difference(list1, list2):
    c1 = Counter(list1)
    c2 = Counter(list2)
    return c1 - c2  # Keeps only positive counts,`,
    timeComplexity: 'O(n) for counting, O(n log k) for most_common',
    spaceComplexity: 'O(n)',
    order: 21,
    topic: 'Python Advanced',
  },
  {
    id: 'advanced-collections-deque',
    title: 'Deque for Efficient Queue Operations',
    difficulty: 'Medium',
    description: `Use collections.deque for O(1) append and pop from both ends.

Implement:
- Sliding window maximum using deque
- LRU cache with bounded deque
- Palindrome checker
- Rotate operations

**Advantage:** O(1) operations on both ends vs O(n) for list.`,
    examples: [
      {
        input: 'sliding_window_max([1,3,-1,-3,5,3,6,7], k=3)',
        output: '[3,3,5,5,6,7]',
      },
    ],
    constraints: [
      'Use deque operations',
      'Maintain O(1) or O(n) time',
      'Handle edge cases',
    ],
    hints: [
      'appendleft/popleft for O(1) operations',
      'rotate() for rotation',
      'maxlen parameter for bounded deque',
    ],
    starterCode: `from collections import deque

def sliding_window_max(nums, k):
    """Find maximum in each sliding window of size k.
    
    Args:
        nums: List of numbers
        k: Window size
        
    Returns:
        List of maximums
    """
    pass


def rotate_list(items, k):
    """Rotate list k positions to the right.
    
    Args:
        items: List to rotate
        k: Positions to rotate
        
    Returns:
        Rotated list
    """
    pass


# Test
print(sliding_window_max([1,3,-1,-3,5,3,6,7], 3))
print(rotate_list([1,2,3,4,5], 2))
`,
    testCases: [
      {
        input: [[1, 3, -1, -3, 5, 3, 6, 7], 3],
        expected: [3, 3, 5, 5, 6, 7],
      },
    ],
    solution: `from collections import deque

def sliding_window_max(nums, k):
    if not nums or k == 0:
        return []
    
    dq = deque()
    result = []
    
    for i, num in enumerate(nums):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements (they won't be max)
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


def rotate_list(items, k):
    d = deque(items)
    d.rotate(k)
    return list(d)`,
    timeComplexity: 'O(n) for sliding window',
    spaceComplexity: 'O(k)',
    order: 22,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-itertools-combinations',
    title: 'Itertools for Combinations and Permutations',
    difficulty: 'Medium',
    description: `Use itertools for efficient iteration over combinations, permutations, and products.

Implement using itertools:
- Generate all subsets of size k
- Find all permutations
- Cartesian product of multiple lists
- Combinations with replacement

**Library:** itertools provides memory-efficient iterator tools.`,
    examples: [
      {
        input: 'all_subsets([1,2,3], k=2)',
        output: '[(1,2), (1,3), (2,3)]',
      },
    ],
    constraints: [
      'Use itertools functions',
      'Return iterators or lists',
      'Handle empty inputs',
    ],
    hints: [
      'combinations() for subsets',
      'permutations() for arrangements',
      'product() for cartesian product',
    ],
    starterCode: `from itertools import combinations, permutations, product, combinations_with_replacement

def all_subsets(items, k):
    """Generate all subsets of size k.
    
    Args:
        items: List of items
        k: Subset size
        
    Returns:
        List of k-sized tuples
    """
    pass


def all_permutations(items):
    """Generate all permutations of items.
    
    Args:
        items: List of items
        
    Returns:
        List of permutation tuples
    """
    pass


def cartesian_product(*lists):
    """Compute cartesian product of multiple lists.
    
    Args:
        *lists: Variable number of lists
        
    Returns:
        List of product tuples
    """
    pass


# Test
print(all_subsets([1,2,3], 2))
print(all_permutations([1,2,3]))
print(cartesian_product([1,2], ['a','b']))
`,
    testCases: [
      {
        input: [[1, 2, 3], 2],
        expected: '[(1,2), (1,3), (2,3)]',
      },
    ],
    solution: `from itertools import combinations, permutations, product, combinations_with_replacement

def all_subsets(items, k):
    return list(combinations(items, k))


def all_permutations(items):
    return list(permutations(items))


def cartesian_product(*lists):
    return list(product(*lists))`,
    timeComplexity: 'O(n choose k) for combinations, O(n!) for permutations',
    spaceComplexity: 'O(output size)',
    order: 23,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-itertools-groupby',
    title: 'GroupBy for Consecutive Grouping',
    difficulty: 'Medium',
    description: `Use itertools.groupby to group consecutive elements by a key function.

Tasks:
- Group consecutive identical elements
- Run-length encoding
- Group by custom key
- Find consecutive runs

**Note:** groupby only groups consecutive elements, so sort first if needed.`,
    examples: [
      {
        input: 'run_length_encode("aaabbccca")',
        output: '[("a",3), ("b",2), ("c",3), ("a",1)]',
      },
    ],
    constraints: [
      'Use itertools.groupby',
      'Handle consecutive grouping',
      'Sort if grouping all occurrences',
    ],
    hints: [
      'groupby(iterable, key=func)',
      'Returns (key, group_iterator) pairs',
      'Sort before groupby if needed',
    ],
    starterCode: `from itertools import groupby

def run_length_encode(s):
    """Encode string using run-length encoding.
    
    Args:
        s: Input string
        
    Returns:
        List of (char, count) tuples
    """
    pass


def group_consecutive(nums):
    """Group consecutive numbers.
    
    Args:
        nums: List of numbers
        
    Returns:
        List of lists of consecutive numbers
    """
    pass


# Test
print(run_length_encode("aaabbccca"))
print(group_consecutive([1,2,3,5,6,8,9,10]))
`,
    testCases: [
      {
        input: ['aaabbccca'],
        expected: '[("a",3), ("b",2), ("c",3), ("a",1)]',
      },
    ],
    solution: `from itertools import groupby

def run_length_encode(s):
    return [(char, len(list(group))) for char, group in groupby(s)]


def group_consecutive(nums):
    result = []
    for k, g in groupby(enumerate(nums), lambda x: x[1] - x[0]):
        result.append([x[1] for x in g])
    return result`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 24,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-functools-reduce',
    title: 'Reduce for Aggregation',
    difficulty: 'Medium',
    description: `Use functools.reduce to aggregate values using a binary function.

Implement using reduce:
- Product of all numbers
- Flatten nested lists
- Find GCD of multiple numbers
- Compose multiple functions

**Pattern:** Reduce applies function cumulatively to items from left to right.`,
    examples: [
      {
        input: 'product([1,2,3,4,5])',
        output: '120',
      },
    ],
    constraints: [
      'Use functools.reduce',
      'Handle empty sequences',
      'Provide initial values when needed',
    ],
    hints: [
      'reduce(function, sequence, initial)',
      'Function takes two arguments',
      'Use operator module for common operations',
    ],
    starterCode: `from functools import reduce
import operator
import math

def product(numbers):
    """Calculate product of all numbers.
    
    Args:
        numbers: List of numbers
        
    Returns:
        Product of all numbers
    """
    pass


def flatten_list(nested_list):
    """Flatten a list of lists.
    
    Args:
        nested_list: List of lists
        
    Returns:
        Flattened list
    """
    pass


def gcd_multiple(numbers):
    """Find GCD of multiple numbers.
    
    Args:
        numbers: List of integers
        
    Returns:
        GCD of all numbers
    """
    pass


# Test
result = product([1,2,3,4,5])
`,
    testCases: [
      {
        input: [],
        expected: 120,
      },
    ],
    solution: `from functools import reduce
import operator
import math

def product(numbers):
    return reduce(operator.mul, numbers, 1)


def flatten_list(nested_list):
    return reduce(operator.add, nested_list, [])


def gcd_multiple(numbers):
    return reduce(math.gcd, numbers)


# Test
result = product([1,2,3,4,5])`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) for product/gcd, O(n) for flatten',
    order: 25,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-functools-partial',
    title: 'Partial Function Application',
    difficulty: 'Easy',
    description: `Use functools.partial to create new functions with some arguments pre-filled.

Create partial functions:
- Power functions (square, cube)
- Specialized filters
- Callback functions with fixed parameters
- Custom sorting functions

**Use Case:** Simplify function calls by fixing some arguments.`,
    examples: [
      {
        input: 'square = partial(pow, exp=2); square(5)',
        output: '25',
      },
    ],
    constraints: [
      'Use functools.partial',
      'Pre-fill specific arguments',
      'Create reusable functions',
    ],
    hints: [
      'partial(func, *args, **kwargs)',
      'Returns new callable',
      'Can partially apply positional and keyword args',
    ],
    starterCode: `from functools import partial

def create_power_functions():
    """Create square and cube functions using partial.
    
    Returns:
        Tuple of (square_func, cube_func)
    """
    pass


def create_filter_functions():
    """Create specialized filter functions.
    
    Returns:
        Tuple of (filter_even, filter_positive)
    """
    # Use partial to create filter(predicate, iterable) variants
    pass


# Test
square, cube = create_power_functions()
print(square(5))
print(cube(3))

filter_even, filter_positive = create_filter_functions()
print(list(filter_even([1,2,3,4,5,6])))
print(list(filter_positive([-2,-1,0,1,2])))
`,
    testCases: [
      {
        input: [5],
        expected: 25,
      },
    ],
    solution: `from functools import partial

def create_power_functions():
    square = partial(pow, exp=2)
    cube = partial(pow, exp=3)
    return square, cube


def create_filter_functions():
    is_even = lambda x: x % 2 == 0
    is_positive = lambda x: x > 0
    filter_even = partial(filter, is_even)
    filter_positive = partial(filter, is_positive)
    return filter_even, filter_positive`,
    timeComplexity: 'O(1) to create partial functions',
    spaceComplexity: 'O(1)',
    order: 26,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-functools-lru-cache',
    title: 'LRU Cache for Memoization',
    difficulty: 'Medium',
    description: `Use @functools.lru_cache decorator for automatic memoization with LRU eviction.

Apply LRU cache to:
- Fibonacci calculation
- Expensive recursive functions
- API calls simulation
- Dynamic programming problems

**Benefit:** Automatic caching with bounded memory using LRU policy.`,
    examples: [
      {
        input: 'fibonacci(100)',
        output: 'Fast result with memoization',
      },
    ],
    constraints: [
      'Use @lru_cache decorator',
      'Specify maxsize if needed',
      'Arguments must be hashable',
    ],
    hints: [
      '@lru_cache(maxsize=128)',
      'maxsize=None for unlimited cache',
      'Use cache_info() to see stats',
    ],
    starterCode: `from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    """Calculate nth Fibonacci number with memoization.
    
    Args:
        n: Position in sequence
        
    Returns:
        nth Fibonacci number
    """
    pass


@lru_cache(maxsize=128)
def count_ways_to_climb(n, steps=(1, 2)):
    """Count ways to climb n stairs with given step sizes.
    
    Args:
        n: Number of stairs
        steps: Tuple of allowed step sizes
        
    Returns:
        Number of ways to climb
    """
    pass


# Test
print(fibonacci(100))
print(count_ways_to_climb(10))
print(fibonacci.cache_info())  # View cache statistics
`,
    testCases: [
      {
        input: [10],
        expected: 55,
      },
    ],
    solution: `from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


@lru_cache(maxsize=128)
def count_ways_to_climb(n, steps=(1, 2)):
    if n == 0:
        return 1
    if n < 0:
        return 0
    return sum(count_ways_to_climb(n - step, steps) for step in steps)`,
    timeComplexity: 'O(n) with memoization vs O(2^n) without',
    spaceComplexity: 'O(n) for cache',
    order: 27,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-heapq-operations',
    title: 'Heap Operations with Heapq',
    difficulty: 'Medium',
    description: `Use heapq module for efficient priority queue operations.

Implement:
- Find k largest/smallest elements
- Merge sorted iterables
- Heap-based priority queue
- Running median using two heaps

**Library:** heapq provides min-heap implementation.`,
    examples: [
      {
        input: 'nlargest([1,4,2,8,5,3], 3)',
        output: '[8,5,4]',
      },
    ],
    constraints: [
      'Use heapq functions',
      'Maintain heap invariant',
      'O(n log k) for k largest',
    ],
    hints: [
      'heappush/heappop for basic ops',
      'nlargest/nsmallest for top k',
      'Use negative values for max heap',
    ],
    starterCode: `import heapq

def find_k_largest(nums, k):
    """Find k largest elements.
    
    Args:
        nums: List of numbers
        k: Number of largest to find
        
    Returns:
        List of k largest elements
    """
    pass


def merge_sorted_lists(*lists):
    """Merge multiple sorted lists into one sorted list.
    
    Args:
        *lists: Variable number of sorted lists
        
    Returns:
        Single sorted list
    """
    pass


class PriorityQueue:
    """Priority queue using heapq."""
    
    def __init__(self):
        self.heap = []
        self.counter = 0
    
    def push(self, item, priority):
        """Add item with priority."""
        pass
    
    def pop(self):
        """Remove and return highest priority item."""
        pass


# Test
print(find_k_largest([1,4,2,8,5,3], 3))
print(merge_sorted_lists([1,3,5], [2,4,6], [0,7,8]))
`,
    testCases: [
      {
        input: [[1, 4, 2, 8, 5, 3], 3],
        expected: [8, 5, 4],
      },
    ],
    solution: `import heapq

def find_k_largest(nums, k):
    return heapq.nlargest(k, nums)


def merge_sorted_lists(*lists):
    return list(heapq.merge(*lists))


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0
    
    def push(self, item, priority):
        # Use counter for stability (FIFO for same priority)
        heapq.heappush(self.heap, (priority, self.counter, item))
        self.counter += 1
    
    def pop(self):
        if not self.heap:
            raise IndexError("pop from empty priority queue")
        return heapq.heappop(self.heap)[2]`,
    timeComplexity: 'O(n log k) for k largest, O(n log n) for merge',
    spaceComplexity: 'O(k) for k largest, O(n) for merge',
    order: 28,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-bisect-binary-search',
    title: 'Binary Search with Bisect',
    difficulty: 'Easy',
    description: `Use bisect module for binary search on sorted sequences.

Implement:
- Find insertion point for value
- Count elements in range
- Find left and right boundaries
- Maintain sorted list with insertions

**Library:** bisect provides binary search functions.`,
    examples: [
      {
        input: 'bisect_left([1,2,4,4,5], 4)',
        output: '2',
      },
    ],
    constraints: [
      'Use bisect functions',
      'List must be sorted',
      'O(log n) search time',
    ],
    hints: [
      'bisect_left finds leftmost position',
      'bisect_right finds rightmost position',
      'insort maintains sorted order',
    ],
    starterCode: `import bisect

def count_in_range(sorted_list, low, high):
    """Count elements in range [low, high].
    
    Args:
        sorted_list: Sorted list of numbers
        low: Lower bound (inclusive)
        high: Upper bound (inclusive)
        
    Returns:
        Count of elements in range
    """
    pass


def find_closest(sorted_list, target):
    """Find element closest to target.
    
    Args:
        sorted_list: Sorted list
        target: Target value
        
    Returns:
        Closest element
    """
    pass


class SortedList:
    """Maintain sorted list with efficient insertions."""
    
    def __init__(self):
        self.items = []
    
    def insert(self, value):
        """Insert value maintaining sorted order."""
        pass
    
    def remove(self, value):
        """Remove value if exists."""
        pass


# Test
print(count_in_range([1,2,4,4,4,5,7,8], 4, 6))
print(find_closest([1,3,5,7,9], 6))
`,
    testCases: [
      {
        input: [[1, 2, 4, 4, 4, 5, 7, 8], 4, 6],
        expected: 4,
      },
    ],
    solution: `import bisect

def count_in_range(sorted_list, low, high):
    left = bisect.bisect_left(sorted_list, low)
    right = bisect.bisect_right(sorted_list, high)
    return right - left


def find_closest(sorted_list, target):
    pos = bisect.bisect_left(sorted_list, target)
    if pos == 0:
        return sorted_list[0]
    if pos == len(sorted_list):
        return sorted_list[-1]
    before = sorted_list[pos - 1]
    after = sorted_list[pos]
    return after if (after - target) < (target - before) else before


class SortedList:
    def __init__(self):
        self.items = []
    
    def insert(self, value):
        bisect.insort(self.items, value)
    
    def remove(self, value):
        pos = bisect.bisect_left(self.items, value)
        if pos < len(self.items) and self.items[pos] == value:
            self.items.pop(pos)`,
    timeComplexity: 'O(log n) for search, O(n) for insertion (list shift)',
    spaceComplexity: 'O(n)',
    order: 29,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-operator-functions',
    title: 'Operator Module for Functional Programming',
    difficulty: 'Easy',
    description: `Use operator module for function versions of operators.

Use operator functions for:
- Sorting by attributes
- Reduce operations
- itemgetter for extracting values
- attrgetter for object attributes

**Benefit:** Avoid lambda functions for simple operations.`,
    examples: [
      {
        input: 'sorted(users, key=operator.attrgetter("age"))',
        output: 'Users sorted by age',
      },
    ],
    constraints: [
      'Use operator module',
      'Prefer operator over lambda when possible',
      'Understand operator.itemgetter/attrgetter',
    ],
    hints: [
      'itemgetter(1) extracts index 1',
      'attrgetter("name") gets .name attribute',
      'methodcaller("upper") calls .upper()',
    ],
    starterCode: `import operator
from collections import namedtuple

User = namedtuple('User', ['name', 'age', 'score'])

def sort_by_multiple_keys(users):
    """Sort users by age, then score.
    
    Args:
        users: List of User namedtuples
        
    Returns:
        Sorted list
    """
    # Use operator.attrgetter
    pass


def extract_column(data, col_index):
    """Extract column from list of lists.
    
    Args:
        data: List of lists
        col_index: Column index to extract
        
    Returns:
        List of column values
    """
    # Use operator.itemgetter
    pass


def apply_operation(a, b, op_name):
    """Apply named operation to two numbers.
    
    Args:
        a, b: Numbers
        op_name: Operation name ('add', 'mul', 'sub', 'truediv')
        
    Returns:
        Result of operation
    """
    # Use operator functions
    pass


# Test
users = [
    User('Alice', 30, 95),
    User('Bob', 25, 85),
    User('Charlie', 30, 90)
]
print(sort_by_multiple_keys(users))

data = [[1,2,3], [4,5,6], [7,8,9]]
print(extract_column(data, 1))

print(apply_operation(10, 5, 'add'))
`,
    testCases: [
      {
        input: [10, 5, 'add'],
        expected: 15,
      },
    ],
    solution: `import operator
from collections import namedtuple

User = namedtuple('User', ['name', 'age', 'score'])

def sort_by_multiple_keys(users):
    return sorted(users, key=operator.attrgetter('age', 'score'))


def extract_column(data, col_index):
    getter = operator.itemgetter(col_index)
    return [getter(row) for row in data]


def apply_operation(a, b, op_name):
    ops = {
        'add': operator.add,
        'mul': operator.mul,
        'sub': operator.sub,
        'truediv': operator.truediv,
    }
    return ops[op_name](a, b)`,
    timeComplexity: 'O(n log n) for sorting, O(n) for extraction',
    spaceComplexity: 'O(n)',
    order: 30,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-typing-annotations',
    title: 'Advanced Type Annotations',
    difficulty: 'Medium',
    description: `Use advanced typing features for better type safety and documentation.

Implement with type hints:
- Generic functions
- TypeVar for constraints
- Union and Optional types
- Callable types
- Literal types

**Benefit:** Better IDE support, documentation, and runtime type checking with tools.`,
    examples: [
      {
        input: 'def get_first(items: List[T]) -> Optional[T]',
        output: 'Generic function with type variable',
      },
    ],
    constraints: [
      'Use typing module',
      'Add comprehensive type hints',
      'Support generics where appropriate',
    ],
    hints: [
      'TypeVar("T") for generics',
      'Optional[X] = Union[X, None]',
      'Use Callable[[Args], Return]',
    ],
    starterCode: `from typing import TypeVar, List, Optional, Callable, Union, Literal, Dict, Any

T = TypeVar('T')
Number = TypeVar('Number', int, float)

def get_first(items: List[T]) -> Optional[T]:
    """Get first element or None if empty.
    
    Args:
        items: List of items
        
    Returns:
        First item or None
    """
    pass


def apply_twice(func: Callable[[T], T], value: T) -> T:
    """Apply function twice to value.
    
    Args:
        func: Function to apply
        value: Input value
        
    Returns:
        Result after applying func twice
    """
    pass


def safe_divide(a: Number, b: Number) -> Union[Number, Literal["error"]]:
    """Divide a by b, return "error" on division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        
    Returns:
        Result or "error"
    """
    pass


# Test with type checking
result = get_first([1,2,3])
`,
    testCases: [
      {
        input: [],
        expected: 1,
      },
    ],
    solution: `from typing import TypeVar, List, Optional, Callable, Union, Literal, Dict, Any

T = TypeVar('T')
Number = TypeVar('Number', int, float)

def get_first(items: List[T]) -> Optional[T]:
    return items[0] if items else None


def apply_twice(func: Callable[[T], T], value: T) -> T:
    return func(func(value))


def safe_divide(a: Number, b: Number) -> Union[Number, Literal["error"]]:
    if b == 0:
        return "error"
    return a / b


# Test with type checking
result = get_first([1,2,3])`,
    timeComplexity: 'O(1) for all functions',
    spaceComplexity: 'O(1)',
    order: 31,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-comprehensions-nested',
    title: 'Advanced List/Dict/Set Comprehensions',
    difficulty: 'Medium',
    description: `Master complex comprehensions with nesting, conditionals, and multiple iterations.

Create comprehensions for:
- Nested list flattening
- Matrix operations
- Dictionary inversions
- Set operations with filtering

**Pattern:** Comprehensions are more Pythonic than loops for data transformations.`,
    examples: [
      {
        input: 'flatten([[1,2], [3,4]])',
        output: '[1,2,3,4]',
      },
    ],
    constraints: [
      'Use comprehensions (not loops)',
      'Handle nested structures',
      'Combine with conditionals',
    ],
    hints: [
      'Nested comprehensions: [x for list in lists for x in list]',
      'Conditional: [x for x in items if condition]',
      'Dict comprehension: {k: v for k, v in items}',
    ],
    starterCode: `def flatten(nested_list):
    """Flatten nested list using comprehension.
    
    Args:
        nested_list: List of lists
        
    Returns:
        Flattened list
    """
    pass


def transpose_matrix(matrix):
    """Transpose matrix using comprehension.
    
    Args:
        matrix: 2D list
        
    Returns:
        Transposed matrix
    """
    pass


def invert_dict(d):
    """Invert dictionary (swap keys and values).
    
    Args:
        d: Dictionary with unique values
        
    Returns:
        Inverted dictionary
    """
    pass


def word_lengths(text):
    """Create dict of word -> length for words > 3 chars.
    
    Args:
        text: String of words
        
    Returns:
        Dict of word -> length
    """
    pass


# Test
print(flatten([[1,2], [3,4], [5]]))
print(transpose_matrix([[1,2,3], [4,5,6]]))
print(invert_dict({'a': 1, 'b': 2, 'c': 3}))
print(word_lengths("the quick brown fox"))
`,
    testCases: [
      {
        input: [[[1, 2], [3, 4], [5]]],
        expected: [1, 2, 3, 4, 5],
      },
    ],
    solution: `def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def transpose_matrix(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


def invert_dict(d):
    return {v: k for k, v in d.items()}


def word_lengths(text):
    return {word: len(word) for word in text.split() if len(word) > 3}`,
    timeComplexity: 'O(n) where n is total elements',
    spaceComplexity: 'O(n)',
    order: 32,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-walrus-operator',
    title: 'Walrus Operator (:=) for Assignment Expressions',
    difficulty: 'Easy',
    description: `Use the walrus operator (:=) for inline assignments in expressions.

Apply walrus operator in:
- While loop conditions
- List comprehensions with reuse
- If statements with assignments
- Complex conditionals

**Benefit:** Avoid duplicate evaluations and reduce code verbosity.`,
    examples: [
      {
        input: 'if (n := len(items)) > 10: print(f"Too many: {n}")',
        output: 'Assigns and uses n in one line',
      },
    ],
    constraints: [
      'Use := operator',
      'Valid in Python 3.8+',
      'Understand expression vs statement',
    ],
    hints: [
      'Syntax: (var := expression)',
      'Returns value of expression',
      'Useful in comprehensions and conditionals',
    ],
    starterCode: `def process_items(items):
    """Process items using walrus operator.
    
    Args:
        items: List of items
        
    Returns:
        List of processed lengths
    """
    # Use walrus operator in comprehension
    # Only include items where (length := len(item)) > 3
    pass


def read_until_stop(get_input):
    """Read input until "stop" using walrus operator.
    
    Args:
        get_input: Function that returns next input
        
    Returns:
        List of inputs (excluding "stop")
    """
    # Use walrus operator in while condition
    pass


def categorize_number(n):
    """Categorize number using walrus operator.
    
    Args:
        n: Number to categorize
        
    Returns:
        Category string
    """
    # Use walrus operator in if statements
    # if (abs_n := abs(n)) > 100: return "large"
    # elif abs_n > 10: return "medium"
    # else: return "small"
    pass


# Test
result = process_items(["hi", "hello", "hey", "goodbye"])
`,
    testCases: [
      {
        input: [],
        expected: [5, 7],
      },
    ],
    solution: `def process_items(items):
    return [length for item in items if (length := len(item)) > 3]


def read_until_stop(get_input):
    results = []
    while (value := get_input()) != "stop":
        results.append(value)
    return results


def categorize_number(n):
    if (abs_n := abs(n)) > 100:
        return "large"
    elif abs_n > 10:
        return "medium"
    else:
        return "small"


# Test
result = process_items(["hi", "hello", "hey", "goodbye"])`,
    timeComplexity:
      'O(n) for process_items and read_until_stop, O(1) for categorize',
    spaceComplexity: 'O(n)',
    order: 33,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-match-case',
    title: 'Structural Pattern Matching (Match-Case)',
    difficulty: 'Medium',
    description: `Use Python 3.10+ match-case for structural pattern matching.

Implement match-case for:
- Type-based dispatch
- Destructuring sequences
- Matching with guards
- Handling complex data structures

**Pattern:** More powerful and readable than if-elif chains.`,
    examples: [
      {
        input: 'match_shape(("circle", 5))',
        output: 'Circle with radius 5',
      },
    ],
    constraints: [
      'Use match-case statements',
      'Python 3.10+ required',
      'Handle all cases with default',
    ],
    hints: [
      'case pattern if guard:',
      'Use | for OR patterns',
      'Destructure with case (a, b):',
    ],
    starterCode: `def match_command(command):
    """Match command patterns.
    
    Args:
        command: Tuple of (action, *args)
        
    Returns:
        String describing action
    """
    # match command:
    #     case ("quit",):
    #         return "Quitting"
    #     case ("move", x, y):
    #         return f"Moving to {x}, {y}"
    #     case ("draw", shape, *params):
    #         return f"Drawing {shape} with {params}"
    #     case _:
    #         return "Unknown command"
    pass


def classify_point(point):
    """Classify point location.
    
    Args:
        point: Tuple of (x, y)
        
    Returns:
        Location description
    """
    # Use match with guards
    # case (0, 0): origin
    # case (0, y): on y-axis
    # case (x, 0): on x-axis
    # case (x, y) if x == y: on diagonal
    # case (x, y) if x > 0 and y > 0: quadrant 1
    pass


# Test
result = match_command(("move", 10, 20))
`,
    testCases: [
      {
        input: [],
        expected: 'Moving to 10, 20',
      },
    ],
    solution: `def match_command(command):
    match command:
        case ("quit",):
            return "Quitting"
        case ("move", x, y):
            return f"Moving to {x}, {y}"
        case ("draw", shape, *params):
            return f"Drawing {shape} with {params}"
        case _:
            return "Unknown command"


def classify_point(point):
    match point:
        case (0, 0):
            return "origin"
        case (0, y):
            return f"on y-axis at {y}"
        case (x, 0):
            return f"on x-axis at {x}"
        case (x, y) if x == y:
            return f"on diagonal at ({x}, {y})"
        case (x, y) if x > 0 and y > 0:
            return f"quadrant 1: ({x}, {y})"
        case (x, y) if x < 0 and y > 0:
            return f"quadrant 2: ({x}, {y})"
        case (x, y) if x < 0 and y < 0:
            return f"quadrant 3: ({x}, {y})"
        case (x, y):
            return f"quadrant 4: ({x}, {y})"


# Test
result = match_command(("move", 10, 20))`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 34,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-chainmap',
    title: 'ChainMap for Layered Dictionaries',
    difficulty: 'Medium',
    description: `Use collections.ChainMap to manage multiple dictionaries as a single view.

Use ChainMap for:
- Configuration layers (defaults, user, environment)
- Scope chains (local, global)
- Template contexts
- Fallback lookups

**Benefit:** Efficient layered lookups without copying dictionaries.`,
    examples: [
      {
        input: 'ChainMap(user_config, default_config)',
        output: 'User config with defaults as fallback',
      },
    ],
    constraints: [
      'Use collections.ChainMap',
      'Understand lookup order',
      'First dict has priority',
    ],
    hints: [
      'ChainMap(*dicts) creates layered view',
      'Lookups search from first to last',
      'Updates only affect first dict',
    ],
    starterCode: `from collections import ChainMap

def create_config_system(defaults, user_config, env_config):
    """Create layered configuration system.
    
    Args:
        defaults: Default config dict
        user_config: User overrides dict
        env_config: Environment overrides dict
        
    Returns:
        ChainMap with proper priority
    """
    # Priority: env > user > defaults
    pass


def simulate_scope_chain():
    """Simulate variable scope chain.
    
    Returns:
        ChainMap representing local -> global scope
    """
    global_scope = {'x': 1, 'y': 2, 'z': 3}
    local_scope = {'x': 10, 'y': 20}
    
    # Create ChainMap for scope lookup
    pass


# Test
defaults = {'host': 'localhost', 'port': 8080, 'debug': False}
user = {'port': 3000, 'debug': True}
env = {'host': '0.0.0.0'}

config = create_config_system(defaults, user, env)
print(dict(config))  # Should show env > user > defaults priority
`,
    testCases: [
      {
        input: [{ a: 1 }, { a: 2, b: 3 }],
        expected: '{"a": 1, "b": 3}',
      },
    ],
    solution: `from collections import ChainMap

def create_config_system(defaults, user_config, env_config):
    # First dict in ChainMap has highest priority
    return ChainMap(env_config, user_config, defaults)


def simulate_scope_chain():
    global_scope = {'x': 1, 'y': 2, 'z': 3}
    local_scope = {'x': 10, 'y': 20}
    return ChainMap(local_scope, global_scope)`,
    timeComplexity: 'O(k) where k is number of maps (usually small)',
    spaceComplexity: 'O(1) - no copying, just references',
    order: 35,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-named-tuple',
    title: 'NamedTuple for Structured Data',
    difficulty: 'Easy',
    description: `Use collections.namedtuple or typing.NamedTuple for lightweight data structures.

Create namedtuples for:
- Function return values
- CSV row representation
- Immutable records
- Type-hinted data classes (typing.NamedTuple)

**Benefit:** Memory-efficient, immutable, with named fields.`,
    examples: [
      {
        input: 'Point = namedtuple("Point", ["x", "y"])',
        output: 'Tuple with named fields',
      },
    ],
    constraints: [
      'Use namedtuple or typing.NamedTuple',
      'Fields are immutable',
      'Support all tuple operations',
    ],
    hints: [
      'collections.namedtuple for runtime',
      'typing.NamedTuple for type hints',
      'Access by name or index',
    ],
    starterCode: `from collections import namedtuple
from typing import NamedTuple

# Method 1: collections.namedtuple
Point = namedtuple('Point', ['x', 'y'])

# Method 2: typing.NamedTuple (preferred)
class Person(NamedTuple):
    name: str
    age: int
    email: str
    
    def is_adult(self):
        """Add methods to NamedTuple."""
        pass


def parse_csv_row(row_string):
    """Parse CSV row into namedtuple.
    
    Args:
        row_string: Comma-separated values
        
    Returns:
        CSVRow namedtuple
    """
    # Create namedtuple type and instance
    pass


def calculate_distance(p1: Point, p2: Point) -> float:
    """Calculate distance between two points.
    
    Args:
        p1, p2: Point namedtuples
        
    Returns:
        Euclidean distance
    """
    pass


# Test
p1 = Point(0, 0)
p2 = Point(3, 4)
print(calculate_distance(p1, p2))

person = Person("Alice", 30, "alice@example.com")
print(person.is_adult())
`,
    testCases: [
      {
        input: [0, 0, 3, 4],
        expected: 5.0,
      },
    ],
    solution: `from collections import namedtuple
from typing import NamedTuple
import math

Point = namedtuple('Point', ['x', 'y'])

class Person(NamedTuple):
    name: str
    age: int
    email: str
    
    def is_adult(self):
        return self.age >= 18


def parse_csv_row(row_string):
    fields = row_string.split(',')
    CSVRow = namedtuple('CSVRow', ['field' + str(i) for i in range(len(fields))])
    return CSVRow(*fields)


def calculate_distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)`,
    timeComplexity: 'O(1) for field access',
    spaceComplexity: 'O(n) where n is number of fields',
    order: 36,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-slots',
    title: '__slots__ for Memory Optimization',
    difficulty: 'Medium',
    description: `Use __slots__ to reduce memory usage and improve attribute access speed.

Implement classes with __slots__:
- Memory-efficient data classes
- Fixed attribute classes
- Performance-critical objects
- Compare memory usage with/without slots

**Benefit:** 40-50% memory reduction, faster attribute access, prevents dynamic attributes.`,
    examples: [
      {
        input: 'class Point: __slots__ = ["x", "y"]',
        output: 'Memory-efficient Point class',
      },
    ],
    constraints: [
      'Use __slots__ attribute',
      'Cannot add dynamic attributes',
      'Incompatible with __dict__',
    ],
    hints: [
      '__slots__ = ["attr1", "attr2"]',
      'Defined at class level',
      'All instances share same slots',
    ],
    starterCode: `import sys

class RegularPoint:
    """Regular class with __dict__."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y


class SlottedPoint:
    """Memory-efficient class with __slots__."""
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y


def compare_memory_usage():
    """Compare memory usage of regular vs slotted classes.
    
    Returns:
        Tuple of (regular_size, slotted_size, savings_percent)
    """
    pass


class Vector:
    """3D vector with __slots__."""
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def magnitude(self):
        """Calculate vector magnitude."""
        pass
    
    def __add__(self, other):
        """Add two vectors."""
        pass


# Test
regular = RegularPoint(1, 2)
slotted = SlottedPoint(1, 2)

print(f"Regular size: {sys.getsizeof(regular.__dict__)}")
print(f"Slotted size: {sys.getsizeof(slotted)}")

v1 = Vector(1, 2, 3)
v2 = Vector(4, 5, 6)
print((v1 + v2).__dict__)  # This will fail - no __dict__!
`,
    testCases: [
      {
        input: [1, 2, 3],
        expected: 'Vector(1,2,3)',
      },
    ],
    solution: `import sys
import math

class RegularPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class SlottedPoint:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y


def compare_memory_usage():
    regular = RegularPoint(1, 2)
    slotted = SlottedPoint(1, 2)
    
    # Regular has __dict__ overhead
    regular_size = sys.getsizeof(regular.__dict__)
    slotted_size = sys.getsizeof(slotted)
    
    savings = (1 - slotted_size / regular_size) * 100
    return regular_size, slotted_size, savings


class Vector:
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y}, {self.z})"`,
    timeComplexity: 'O(1) for attribute access (faster than __dict__)',
    spaceComplexity: 'O(1) per instance (40-50% less than regular class)',
    order: 37,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-weakref',
    title: 'Weak References for Cache Management',
    difficulty: 'Hard',
    description: `Use weakref module to create references that don't prevent garbage collection.

Implement with weakref:
- Cache that doesn't prevent cleanup
- Observer pattern without memory leaks
- Object tracking without ownership
- WeakValueDictionary for caches

**Use Case:** Caching, callbacks, and avoiding circular references.`,
    examples: [
      {
        input: 'WeakValueDictionary for object cache',
        output: 'Cache that auto-cleans when objects deleted',
      },
    ],
    constraints: [
      'Use weakref module',
      'Understand when objects are collected',
      'Handle when weak references become invalid',
    ],
    hints: [
      'weakref.ref(obj) creates weak reference',
      'WeakValueDictionary for weak values',
      'Weak references return None when object deleted',
    ],
    starterCode: `import weakref

class ObjectCache:
    """Cache using weak references."""
    
    def __init__(self):
        self.cache = weakref.WeakValueDictionary()
    
    def get(self, key):
        """Get object from cache.
        
        Returns None if not in cache or was garbage collected.
        """
        pass
    
    def put(self, key, obj):
        """Add object to cache with weak reference."""
        pass


class Observable:
    """Subject in observer pattern using weak references."""
    
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        """Attach observer with weak reference."""
        # Use weakref.ref to avoid preventing garbage collection
        pass
    
    def notify(self, message):
        """Notify all live observers."""
        # Check if weak references are still valid
        pass


# Test
cache = ObjectCache()

class MyObject:
    def __init__(self, value):
        self.value = value

obj = MyObject(42)
cache.put('key1', obj)
print(cache.get('key1'))  # Should work

del obj  # Delete strong reference
print(cache.get('key1'))  # Should return None (object was collected)
`,
    testCases: [
      {
        input: ['test_key', 'test_value'],
        expected: 'cached then None after del',
      },
    ],
    solution: `import weakref

class ObjectCache:
    def __init__(self):
        self.cache = weakref.WeakValueDictionary()
    
    def get(self, key):
        return self.cache.get(key)
    
    def put(self, key, obj):
        self.cache[key] = obj


class Observable:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        weak_observer = weakref.ref(observer)
        self._observers.append(weak_observer)
    
    def notify(self, message):
        # Clean up dead references and notify live ones
        live_observers = []
        for weak_observer in self._observers:
            observer = weak_observer()
            if observer is not None:
                observer.update(message)
                live_observers.append(weak_observer)
        self._observers = live_observers`,
    timeComplexity: 'O(1) for cache operations, O(n) for notify',
    spaceComplexity: 'O(n) but allows garbage collection',
    order: 38,
    topic: 'Python Advanced',
  },

  {
    id: 'advanced-contextlib-utilities',
    title: 'Contextlib Utilities for Context Managers',
    difficulty: 'Medium',
    description: `Use contextlib module utilities to create context managers easily.

Use contextlib for:
- @contextmanager decorator for generators
- ExitStack for dynamic context managers
- suppress() to ignore exceptions
- redirect_stdout/redirect_stderr

**Benefit:** Create context managers without defining __enter__/__exit__.`,
    examples: [
      {
        input: '@contextmanager def timer(): ...',
        output: 'Simple timer context manager',
      },
    ],
    constraints: [
      'Use contextlib utilities',
      'Understand generator-based context managers',
      'Handle cleanup properly',
    ],
    hints: [
      '@contextmanager with yield',
      'Code before yield is __enter__',
      'Code after yield is __exit__',
    ],
    starterCode: `from contextlib import contextmanager, ExitStack, suppress, redirect_stdout
import time
import io

@contextmanager
def timer(name):
    """Context manager that times code execution.
    
    Args:
        name: Name of timed section
    """
    # Implement using @contextmanager
    # Start timer before yield
    # Stop and print time after yield
    pass


def open_multiple_files(filenames):
    """Open multiple files using ExitStack.
    
    Args:
        filenames: List of filenames to open
        
    Returns:
        List of file objects (all closed automatically)
    """
    # Use ExitStack to manage multiple context managers
    pass


def safe_int_convert(value):
    """Convert to int, return None on error.
    
    Args:
        value: Value to convert
        
    Returns:
        Int value or None
    """
    # Use suppress(ValueError) to ignore conversion errors
    pass


def capture_print_output(func):
    """Capture stdout from function.
    
    Args:
        func: Function to call
        
    Returns:
        Captured output as string
    """
    # Use redirect_stdout
    pass


# Test
with timer("test operation"):
    time.sleep(0.1)

print(safe_int_convert("123"))
print(safe_int_convert("not a number"))
`,
    testCases: [
      {
        input: ['test'],
        expected: 'timed execution',
      },
    ],
    solution: `from contextlib import contextmanager, ExitStack, suppress, redirect_stdout
import time
import io

@contextmanager
def timer(name):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{name} took {end - start:.4f} seconds")


def open_multiple_files(filenames):
    files = []
    with ExitStack() as stack:
        for filename in filenames:
            files.append(stack.enter_context(open(filename)))
        return files  # All will be closed when ExitStack exits


def safe_int_convert(value):
    with suppress(ValueError, TypeError):
        return int(value)
    return None


def capture_print_output(func):
    f = io.StringIO()
    with redirect_stdout(f):
        func()
    return f.getvalue()`,
    timeComplexity: 'O(1) for context manager operations',
    spaceComplexity: 'O(1) or O(n) for ExitStack',
    order: 39,
    topic: 'Python Advanced',
  },
  // New problems (40-50)
  ...pythonAdvancedNew,
];

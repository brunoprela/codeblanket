/**
 * Python Advanced module content - Master advanced Python patterns and techniques
 */

import { Module } from '@/lib/types';

export const pythonAdvancedModule: Module = {
  id: 'python-advanced',
  title: 'Python Advanced',
  description:
    'Master advanced Python features including decorators, generators, context managers, and metaclasses.',
  icon: '🐍',
  sections: [
    {
      id: 'introduction',
      title: 'Advanced Python: Beyond the Basics',
      content: `Advanced Python features allow you to write more elegant, efficient, and Pythonic code. These patterns are used extensively in production Python applications and frameworks.

**Why These Topics Matter:**
- **Decorators:** Modify function behavior without changing their code
- **Generators:** Memory-efficient iteration over large datasets
- **Context Managers:** Proper resource management and cleanup
- **Metaclasses:** Control class creation and behavior

**Real-World Applications:**
- **Decorators:** Authentication, logging, caching, rate limiting
- **Generators:** Processing large files, data pipelines, infinite sequences
- **Context Managers:** File handling, database connections, locks
- **Metaclasses:** ORMs (like Django models), singletons, API clients

**Key Insight:**
These advanced features are not just syntactic sugar—they fundamentally change how you structure and think about your code, leading to more maintainable and performant applications.`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain what decorators are and why they are useful. Give a concrete example where decorators solve a real problem.',
          sampleAnswer:
            'Decorators are functions that modify or enhance other functions without changing their source code. They use the @ syntax and are a form of metaprogramming. For example, in a web API, I might have dozens of endpoints that need authentication. Instead of adding auth checking code to each function, I can create an @require_auth decorator that wraps functions and checks authentication before execution. This follows the DRY principle, makes the code cleaner, and centralizes authentication logic. If I need to change how auth works, I update one decorator instead of 50 functions.',
          keyPoints: [
            'Functions that modify other functions',
            'Applied with @ syntax',
            'Example: @require_auth for authentication',
            'Follows DRY principle',
            'Centralizes cross-cutting concerns',
          ],
        },
        {
          id: 'q2',
          question:
            'What are generators and how do they differ from regular functions? When should you use them?',
          sampleAnswer:
            'Generators are functions that use yield instead of return, creating iterators that produce values lazily on-demand. Unlike regular functions that compute all values at once, generators produce one value at a time and maintain their state between calls. Use generators when: (1) processing large datasets that would not fit in memory, (2) creating infinite sequences, or (3) building data pipelines. For example, reading a 10GB log file line by line with a generator uses constant memory, while loading it all at once would use 10GB.',
          keyPoints: [
            'Use yield instead of return',
            'Produce values lazily on-demand',
            'Maintain state between calls',
            'Memory efficient for large data',
            'Example: reading huge files line by line',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain the with statement and context managers. Why are they important for resource management?',
          sampleAnswer:
            'Context managers handle setup and cleanup of resources automatically using the with statement. They implement __enter__ and __exit__ methods. This is crucial because it guarantees cleanup happens even if errors occur. For example: "with open(file) as f:" ensures the file is closed even if an exception is raised while reading. Without context managers, you need try/finally blocks everywhere, which is error-prone. Context managers are used for files, database connections, locks, and any resource that needs cleanup.',
          keyPoints: [
            'Automatic resource setup and cleanup',
            '__enter__ and __exit__ methods',
            'Guarantees cleanup even with errors',
            'Example: with open() ensures file closes',
            'Used for files, databases, locks',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What does the yield keyword do in Python?',
          options: [
            'It returns a value and exits the function',
            'It produces a value and pauses the function, maintaining its state',
            'It creates a new thread',
            'It raises an exception',
          ],
          correctAnswer: 1,
          explanation:
            'yield produces a value and pauses the generator function, maintaining its state (local variables, instruction pointer) until the next value is requested. This allows for lazy evaluation and memory-efficient iteration.',
        },
        {
          id: 'mc2',
          question: 'What is the main advantage of using decorators?',
          options: [
            'Faster execution speed',
            'Reduced memory usage',
            'Code reusability and separation of concerns',
            'Automatic error handling',
          ],
          correctAnswer: 2,
          explanation:
            'Decorators allow you to reuse functionality across multiple functions and separate cross-cutting concerns (like logging, authentication) from business logic, following the DRY principle.',
        },
        {
          id: 'mc3',
          question:
            'Which methods must a context manager implement to work with the "with" statement?',
          options: [
            '__init__ and __del__',
            '__enter__ and __exit__',
            '__start__ and __end__',
            '__open__ and __close__',
          ],
          correctAnswer: 1,
          explanation:
            'Context managers must implement __enter__ (called when entering the with block) and __exit__ (called when leaving, even if an exception occurred).',
        },
        {
          id: 'mc4',
          question:
            'When should you use a generator instead of returning a list?',
          options: [
            'When you need random access to elements',
            'When processing large datasets that might not fit in memory',
            'When you need to sort the results',
            'When you need to access elements multiple times',
          ],
          correctAnswer: 1,
          explanation:
            'Generators are ideal for large datasets because they produce values on-demand (lazy evaluation) rather than creating the entire list in memory at once. This makes them memory-efficient.',
        },
        {
          id: 'mc5',
          question: 'What is a metaclass in Python?',
          options: [
            'A class that inherits from multiple parents',
            'A class of classes that controls class creation',
            'A class with only class methods',
            'An abstract base class',
          ],
          correctAnswer: 1,
          explanation:
            'A metaclass is a class of classes - it controls how classes are created and behave, similar to how classes control how instances are created.',
        },
      ],
    },
    {
      id: 'decorators',
      title: 'Decorators & Function Wrapping',
      content: `**What are Decorators?**
Decorators are a powerful way to modify or enhance functions and classes without changing their source code.

**Basic Decorator Pattern:**
\`\`\`python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        # Do something before
        result = func(*args, **kwargs)
        # Do something after
        return result
    return wrapper

@my_decorator
def my_function():
    pass
\`\`\`

**Common Decorator Use Cases:**

1. **Timing/Profiling:**
\`\`\`python
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.2f}s")
        return result
    return wrapper
\`\`\`

2. **Caching/Memoization:**
\`\`\`python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
\`\`\`

3. **Authentication/Authorization:**
\`\`\`python
def require_auth(func):
    def wrapper(user, *args, **kwargs):
        if not user.is_authenticated:
            raise PermissionError("Not authenticated")
        return func(user, *args, **kwargs)
    return wrapper
\`\`\`

**Decorators with Arguments:**
\`\`\`python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet():
    print("Hello!")
\`\`\`

**Class Decorators:**
\`\`\`python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    pass
\`\`\`

**Best Practices:**
- Use functools.wraps to preserve function metadata
- Keep decorators simple and focused
- Chain decorators carefully (order matters)
- Consider using parameterized decorators for flexibility`,
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through how @lru_cache improves performance. What trade-offs does it make?',
          sampleAnswer:
            '@lru_cache memoizes function results in a dictionary, keyed by the function arguments. When the function is called again with the same arguments, it returns the cached result instead of recomputing. This trades memory for speed. For recursive functions like fibonacci, it turns O(2^n) into O(n) by eliminating redundant calculations. The trade-off is memory usage—the cache stores up to maxsize results. It only works for functions with hashable arguments and can consume lots of memory if results are large or if there are many unique argument combinations.',
          keyPoints: [
            'Stores results in a dictionary cache',
            'Returns cached result for same arguments',
            'Trades memory for speed',
            'Example: fibonacci O(2^n) to O(n)',
            'Requires hashable arguments',
          ],
        },
        {
          id: 'q2',
          question:
            'Why do we need functools.wraps when creating decorators? What problem does it solve?',
          sampleAnswer:
            'Without functools.wraps, the decorated function loses its original metadata like __name__, __doc__, and __module__. This breaks introspection and makes debugging harder. For example, if I decorate my_function, its __name__ would become "wrapper" instead of "my_function", and help(my_function) would show the wrapper docs, not the original docs. functools.wraps copies the metadata from the original function to the wrapper, preserving the function identity. This is critical for debugging, documentation generation, and any tools that rely on function introspection.',
          keyPoints: [
            'Preserves original function metadata',
            '__name__, __doc__, __module__ preserved',
            'Without it, all functions named "wrapper"',
            'Critical for debugging and introspection',
            'Copies metadata from original to wrapper',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain how decorator chaining works and why order matters. What happens when you stack multiple decorators?',
          sampleAnswer:
            'When you stack decorators like @decorator1 @decorator2 @decorator3 def func(), they execute from bottom to top during decoration, meaning func is first wrapped by decorator3, that result is wrapped by decorator2, and finally by decorator1. However, at runtime, they execute top to bottom - decorator1 runs first, then decorator2, then decorator3, and finally func. Order matters because each decorator modifies what the next one sees. For example, if you have @auth @cache, auth runs first (good - no caching unauthorized requests). But @cache @auth would cache before auth checking (bad - security risk). Always consider the logical flow.',
          keyPoints: [
            'Decoration: bottom to top (innermost first)',
            'Execution: top to bottom (outermost first)',
            'Each decorator wraps the previous result',
            'Order affects behavior and can cause bugs',
            'Example: @auth @cache vs @cache @auth',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What does functools.wraps do?',
          options: [
            'Makes functions run faster',
            'Preserves the original function metadata in the wrapper',
            'Adds error handling to functions',
            'Converts functions to generators',
          ],
          correctAnswer: 1,
          explanation:
            'functools.wraps copies metadata like __name__, __doc__, and __module__ from the original function to the wrapper, preserving the function identity for debugging and introspection.',
        },
        {
          id: 'mc2',
          question:
            'What is the time complexity improvement of using @lru_cache on fibonacci?',
          options: [
            'O(n) to O(1)',
            'O(2^n) to O(n)',
            'O(n^2) to O(n)',
            'O(n!) to O(n^2)',
          ],
          correctAnswer: 1,
          explanation:
            'Without memoization, recursive fibonacci is O(2^n) due to redundant calculations. With @lru_cache, each fibonacci(n) is calculated only once, reducing time complexity to O(n).',
        },
        {
          id: 'mc3',
          question:
            'When stacking decorators like @a @b @c def func(), what is the execution order at runtime?',
          options: [
            'a, b, c, func',
            'c, b, a, func',
            'func, a, b, c',
            'func, c, b, a',
          ],
          correctAnswer: 0,
          explanation:
            'Decorators execute from top to bottom at runtime: a runs first, then b, then c, and finally the original function func.',
        },
        {
          id: 'mc4',
          question:
            'What requirement must function arguments meet to use @lru_cache?',
          options: [
            'Must be strings',
            'Must be hashable (immutable)',
            'Must be integers',
            'No requirements',
          ],
          correctAnswer: 1,
          explanation:
            '@lru_cache stores results in a dictionary keyed by arguments, so arguments must be hashable (immutable types like int, str, tuple).',
        },
        {
          id: 'mc5',
          question: 'What is a common use case for decorators?',
          options: [
            'Adding authentication checks to functions',
            'Sorting lists',
            'Creating loops',
            'Defining variables',
          ],
          correctAnswer: 0,
          explanation:
            'Decorators are commonly used for cross-cutting concerns like authentication, logging, caching, and timing - functionality that applies to multiple functions.',
        },
      ],
    },
    {
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the difference between a generator and a list comprehension. When would you use each?',
          sampleAnswer:
            'A list comprehension creates and stores the entire list in memory immediately. A generator expression looks similar but uses parentheses and produces values lazily on-demand. Use list comprehensions when you need: (1) the entire dataset in memory, (2) random access to elements, (3) to iterate multiple times, or (4) the dataset is small. Use generators when: (1) processing large datasets, (2) values are used only once, (3) building data pipelines, or (4) working with infinite sequences. For example, [x**2 for x in range(1000000)] creates a million-element list in memory. (x**2 for x in range(1000000)) creates an iterator that computes each square on-demand.',
          keyPoints: [
            'List comp: stores entire list in memory',
            'Generator: produces values on-demand',
            'Use list comp: need random access, multiple iterations',
            'Use generator: large data, one-time use, pipelines',
            'Memory: O(n) vs O(1)',
          ],
        },
        {
          id: 'q2',
          question:
            'How do generators enable processing of datasets that do not fit in memory? Give a concrete example.',
          sampleAnswer:
            'Generators process one item at a time without storing the entire dataset. For example, processing a 50GB log file: with a list, you would read all 50GB into memory and crash. With a generator, you read and process one line at a time—memory usage stays constant regardless of file size. The key is that generators maintain state between yields but only hold the current item. This allows processing datasets larger than RAM. It is how tools like grep process terabyte files: stream processing, not batch loading.',
          keyPoints: [
            'Process one item at a time',
            'Constant memory usage',
            'Example: 50GB file read line by line',
            'Maintains state, not full dataset',
            'Enables stream processing',
          ],
        },
        {
          id: 'q3',
          question:
            'What is the advantage of chaining generators in a pipeline versus processing data in steps? How does it affect memory usage?',
          sampleAnswer:
            'Chaining generators creates a lazy pipeline where each stage processes one item at a time before passing it to the next stage. This keeps memory usage constant (O(1)) regardless of dataset size. In contrast, processing in steps requires storing intermediate results. For example: list → filter → map → list requires three full lists in memory. But generator → generator → generator processes one item through all stages before moving to the next, using minimal memory. This is how Unix pipes work (cat file | grep pattern | sort) - streaming data through transformations without materializing intermediate results.',
          keyPoints: [
            'Lazy pipeline: one item through all stages',
            'Constant O(1) memory usage',
            'No intermediate result storage',
            'Similar to Unix pipes',
            'Efficient for large datasets',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the memory complexity of a generator that yields n items?',
          options: ['O(1)', 'O(log n)', 'O(n)', 'O(n^2)'],
          correctAnswer: 0,
          explanation:
            'Generators have O(1) space complexity because they produce values one at a time without storing all values in memory, regardless of how many items they yield.',
        },
        {
          id: 'mc2',
          question: 'Which syntax creates a generator expression?',
          options: [
            '[x**2 for x in range(10)]',
            '{x**2 for x in range(10)}',
            '(x**2 for x in range(10))',
            'x**2 for x in range(10)',
          ],
          correctAnswer: 2,
          explanation:
            'Generator expressions use parentheses (). Square brackets create lists, curly braces create sets, and the last option is invalid syntax.',
        },
        {
          id: 'mc3',
          question: 'What happens when a generator function is called?',
          options: [
            'The function executes immediately',
            'It returns a generator object without executing the function body',
            'It raises an error',
            'It returns None',
          ],
          correctAnswer: 1,
          explanation:
            'Calling a generator function returns a generator object without executing the function body. The code runs only when you iterate over the generator or call next().',
        },
        {
          id: 'mc4',
          question: 'Can generators represent infinite sequences?',
          options: [
            'No, they must be finite',
            'Yes, because they produce values lazily',
            'Only with special syntax',
            'Yes, but they crash after 1 million items',
          ],
          correctAnswer: 1,
          explanation:
            'Generators can represent infinite sequences because they produce values lazily on-demand, never storing the entire sequence. Example: while True: yield value',
        },
        {
          id: 'mc5',
          question: 'What is the main benefit of using generator pipelines?',
          options: [
            'Faster execution',
            'Memory efficiency through lazy evaluation',
            'Automatic parallelization',
            'Better error handling',
          ],
          correctAnswer: 1,
          explanation:
            'Generator pipelines keep memory usage constant by processing one item through all stages before moving to the next, avoiding intermediate result storage.',
        },
      ],
    },
    {
      id: 'context-managers',
      title: 'Context Managers & Resource Management',
      content: `**What are Context Managers?**
Context managers handle resource setup and cleanup automatically using the with statement.

**Basic Pattern:**
\`\`\`python
class MyContextManager:
    def __enter__(self):
        # Setup code
        print("Entering context")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup code (always runs)
        print("Exiting context")
        return False  # Don't suppress exceptions

with MyContextManager() as manager:
    print("Inside context")
\`\`\`

**Common Use Cases:**

1. **File Handling:**
\`\`\`python
# Without context manager (error-prone)
f = open('file.txt')
try:
    data = f.read()
finally:
    f.close()  # Must remember to close

# With context manager (automatic cleanup)
with open('file.txt') as f:
    data = f.read()
# File automatically closed, even if exception occurs
\`\`\`

2. **Database Connections:**
\`\`\`python
class DatabaseConnection:
    def __enter__(self):
        self.conn = database.connect()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()

with DatabaseConnection() as conn:
    conn.execute("INSERT ...")
\`\`\`

3. **Locks and Threading:**
\`\`\`python
import threading

lock = threading.Lock()

with lock:
    # Critical section
    # Lock automatically released
    pass
\`\`\`

**Using contextlib:**
\`\`\`python
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"Time: {end - start:.2f}s")

with timer():
    # Code to time
    time.sleep(1)
\`\`\`

**Exception Handling:**
\`\`\`python
class FileHandler:
    def __enter__(self):
        self.file = open('data.txt', 'w')
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if exc_type is IOError:
            print("IO Error occurred")
            return True  # Suppress the exception
        return False  # Don't suppress
\`\`\`

**Multiple Context Managers:**
\`\`\`python
# Python 3.10+
with (
    open('input.txt') as infile,
    open('output.txt', 'w') as outfile
):
    outfile.write(infile.read())
\`\`\`

**Best Practices:**
- Always use context managers for resources that need cleanup
- __exit__ is always called, even with exceptions
- Return True from __exit__ to suppress exceptions (use carefully)
- Use contextlib for simpler context managers`,
      quiz: [
        {
          id: 'q1',
          question:
            'Why are context managers critical for resource management? What problem do they solve?',
          sampleAnswer:
            'Context managers guarantee cleanup happens even when errors occur. Without them, if an exception is raised while using a resource like a file or database connection, you might forget to clean up, causing resource leaks. try/finally blocks work but are verbose and error-prone—developers forget them. Context managers centralize the cleanup logic and make it impossible to forget. For example, "with open(file)" ensures the file closes even if an exception occurs while reading. This prevents file descriptor exhaustion, database connection pool exhaustion, and other resource leak issues.',
          keyPoints: [
            'Guarantee cleanup even with exceptions',
            'Prevent resource leaks',
            'Centralize cleanup logic',
            'Example: file always closes',
            'Better than try/finally (less error-prone)',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the __exit__ method. What are its parameters and when should you return True vs False?',
          sampleAnswer:
            '__exit__ receives three parameters: exc_type (exception class or None), exc_val (exception instance), and exc_tb (traceback). It is called when exiting the with block, even if an exception occurred. Return False (default) to let exceptions propagate normally. Return True to suppress the exception—use this carefully only when you can properly handle the error. For example, a database context manager might rollback on exception and return False so the caller knows something failed. Only return True if the exception is expected and fully handled within __exit__.',
          keyPoints: [
            'Parameters: exc_type, exc_val, exc_tb',
            'Always called, even with exceptions',
            'Return False: let exception propagate (default)',
            'Return True: suppress exception (use carefully)',
            'Example: rollback on error, return False',
          ],
        },
        {
          id: 'q3',
          question:
            'Compare the @contextmanager decorator approach versus implementing __enter__/__exit__ directly. When would you use each?',
          sampleAnswer:
            "The @contextmanager decorator (from contextlib) lets you create context managers with a simple generator function: code before yield is __enter__, yield provides the value, code after yield is __exit__. This is much simpler for straightforward cases. Use it when: 1) cleanup logic is simple, 2) you don't need complex exception handling. Implement __enter__/__exit__ directly when: 1) you need fine-grained control over exception handling, 2) the context manager is a class with state and methods, 3) you want to reuse the same object multiple times. For example, a simple timer uses @contextmanager; a database connection pool with state uses __enter__/__exit__.",
          keyPoints: [
            '@contextmanager: simple, generator-based',
            'Direct implementation: more control, stateful',
            'Use decorator for simple cleanup',
            'Use class for complex state/exception handling',
            'Example: timer vs connection pool',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'When is __exit__ called in a context manager?',
          options: [
            'Only when the with block completes successfully',
            'Only when an exception occurs',
            'Always, whether an exception occurs or not',
            'Never, it must be called manually',
          ],
          correctAnswer: 2,
          explanation:
            '__exit__ is always called when leaving the with block, regardless of whether an exception occurred. This guarantees cleanup happens.',
        },
        {
          id: 'mc2',
          question: 'What does returning True from __exit__ do?',
          options: [
            'Indicates the context manager completed successfully',
            'Suppresses any exception that occurred in the with block',
            'Raises a new exception',
            'Forces the context manager to re-enter',
          ],
          correctAnswer: 1,
          explanation:
            'Returning True from __exit__ suppresses any exception that occurred in the with block. Return False (default) to let exceptions propagate normally.',
        },
        {
          id: 'mc3',
          question: 'Which module provides the @contextmanager decorator?',
          options: ['contextlib', 'functools', 'itertools', 'contextmanager'],
          correctAnswer: 0,
          explanation:
            'The contextlib module provides the @contextmanager decorator for creating simple context managers using generator functions.',
        },
        {
          id: 'mc4',
          question:
            'What is the main advantage of context managers over try/finally blocks?',
          options: [
            'Faster execution',
            'Less verbose and impossible to forget cleanup',
            'Automatic error recovery',
            'Parallel execution',
          ],
          correctAnswer: 1,
          explanation:
            'Context managers make cleanup code less verbose and guarantee it runs, making it impossible to forget cleanup. try/finally works but is verbose and error-prone.',
        },
        {
          id: 'mc5',
          question:
            'Can you use multiple context managers in a single with statement?',
          options: [
            'No, only one at a time',
            'Yes, separated by commas',
            'Only with special syntax',
            'Yes, but deprecated',
          ],
          correctAnswer: 1,
          explanation:
            'You can use multiple context managers in a single with statement, separated by commas: with open(f1) as a, open(f2) as b:',
        },
      ],
    },
    {
      id: 'metaclasses',
      title: 'Metaclasses & Class Creation',
      content: `**What are Metaclasses?**
Metaclasses are "classes of classes" that control how classes are created.

**Basic Concept:**
\`\`\`python
# type is the default metaclass
class MyClass:
    pass

# Equivalent to:
MyClass = type('MyClass', (), {})

# Everything is an object
isinstance(MyClass, type)  # True
isinstance(5, int)  # True
isinstance(int, type)  # True
\`\`\`

**Custom Metaclass:**
\`\`\`python
class Meta(type):
    def __new__(mcs, name, bases, attrs):
        # Modify class before creation
        attrs['created_by'] = 'Meta'
        return super().__new__(mcs, name, bases, attrs)

class MyClass(metaclass=Meta):
    pass

print(MyClass.created_by)  # 'Meta'
\`\`\`

**Real-World Use Cases:**

1. **ORM (like Django):**
\`\`\`python
class ModelMeta(type):
    def __new__(mcs, name, bases, attrs):
        # Auto-create database fields
        fields = {}
        for key, value in attrs.items():
            if isinstance(value, Field):
                fields[key] = value
        attrs['_fields'] = fields
        return super().__new__(mcs, name, bases, attrs)

class Model(metaclass=ModelMeta):
    pass

class User(Model):
    name = CharField()
    email = EmailField()
\`\`\`

2. **Singleton Pattern:**
\`\`\`python
class Singleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=Singleton):
    pass

db1 = Database()
db2 = Database()
assert db1 is db2  # Same instance
\`\`\`

3. **API Client Registration:**
\`\`\`python
class APIRegistry(type):
    _apis = {}
    
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if 'endpoint' in attrs:
            mcs._apis[attrs['endpoint']] = cls
        return cls

class API(metaclass=APIRegistry):
    pass

class UsersAPI(API):
    endpoint = '/users'
\`\`\`

**__init__ vs __new__:**
\`\`\`python
class Meta(type):
    def __new__(mcs, name, bases, attrs):
        # Called before class is created
        # Can modify class definition
        return super().__new__(mcs, name, bases, attrs)
    
    def __init__(cls, name, bases, attrs):
        # Called after class is created
        # Can initialize class attributes
        super().__init__(name, bases, attrs)
\`\`\`

**When to Use Metaclasses:**
- Framework and library development
- Enforcing coding standards at class level
- Automatic registration patterns
- ORM implementations
- Plugin systems

**Alternatives to Consider:**
- Class decorators (simpler, often sufficient)
- __init_subclass__ (Python 3.6+, cleaner)
- Descriptors for attribute access control

**Best Practices:**
- Metaclasses are powerful but complex—use sparingly
- Consider simpler alternatives first
- Document metaclass behavior clearly
- Used mainly in frameworks, not application code`,
      quiz: [
        {
          id: 'q1',
          question:
            'What are metaclasses and when should you use them? Why are they considered advanced?',
          sampleAnswer:
            'Metaclasses are classes whose instances are classes. They control how classes are created, just like classes control how objects are created. Use metaclasses for: (1) ORMs like Django models where you need to transform class definitions into database schemas, (2) enforcing class-level constraints or patterns, (3) automatic registration systems, or (4) plugin architectures. They are considered advanced because: they are meta-programming (code that writes code), they are rarely needed in application code, there are usually simpler alternatives, and misuse can make code hard to understand. The Python mantra is "metaclasses are deeper magic than 99% of users should ever worry about."',
          keyPoints: [
            'Classes whose instances are classes',
            'Control how classes are created',
            'Use cases: ORMs, registration, constraints',
            'Rarely needed in application code',
            'Often simpler alternatives exist',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the difference between __new__ and __init__ in a metaclass.',
          sampleAnswer:
            'In a metaclass, __new__ is called before the class is created and receives the class name, bases, and attributes dict. It can modify these before the class is constructed and must return the class object. __init__ is called after the class is created to initialize it. The key difference: __new__ can prevent class creation or modify the class definition, while __init__ can only set attributes on an already-created class. Use __new__ when you need to modify the class structure (add/remove methods, change bases), and __init__ for simple initialization like registering the class or setting metadata.',
          keyPoints: [
            '__new__: called before class creation',
            '__new__: can modify class definition',
            '__init__: called after class created',
            '__init__: initializes the class',
            '__new__ for structure, __init__ for initialization',
          ],
        },
        {
          id: 'q3',
          question:
            'What are some alternatives to metaclasses that are simpler but solve similar problems? When would you use each?',
          sampleAnswer:
            'Modern Python offers simpler alternatives: 1) **Class decorators**: Apply @decorator to a class to modify it after creation. Use for adding methods, wrapping methods, or registering classes. Simpler than metaclasses. 2) **__init_subclass__** (Python 3.6+): A class method called when a class is subclassed. Use for validation, registration, or modifying subclasses. Cleaner than metaclasses for inheritance patterns. 3) **Descriptors**: Control attribute access at the instance level. Use for validation, computed properties. Only use metaclasses when you need to: control how ALL subclasses are created, modify class structure deeply, or implement frameworks like ORMs. Rule of thumb: try decorators first, then __init_subclass__, metaclasses last.',
          keyPoints: [
            'Class decorators: simpler, for post-creation modification',
            '__init_subclass__: cleaner for inheritance patterns',
            'Descriptors: for attribute-level control',
            'Use metaclasses only when alternatives insufficient',
            'Order: decorators → __init_subclass__ → metaclasses',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the default metaclass in Python?',
          options: ['object', 'type', 'class', 'meta'],
          correctAnswer: 1,
          explanation:
            'type is the default metaclass in Python. All classes (unless specified otherwise) are instances of type.',
        },
        {
          id: 'mc2',
          question: 'When should you prefer class decorators over metaclasses?',
          options: [
            'When you need to modify the class structure',
            'When you need simpler, more readable class modification',
            'When you need to control all subclasses',
            'When implementing an ORM',
          ],
          correctAnswer: 1,
          explanation:
            'Class decorators are simpler and more readable than metaclasses for most use cases. Use metaclasses only when you need to control subclass creation or modify the class at a structural level.',
        },
        {
          id: 'mc3',
          question:
            'Which method in a metaclass is called first during class creation?',
          options: ['__init__', '__new__', '__call__', '__create__'],
          correctAnswer: 1,
          explanation:
            '__new__ is called first in a metaclass to create the class object before __init__ initializes it.',
        },
        {
          id: 'mc4',
          question: 'What is a common use case for metaclasses?',
          options: [
            'Sorting lists',
            'ORM implementations like Django models',
            'File I/O',
            'String manipulation',
          ],
          correctAnswer: 1,
          explanation:
            'ORMs like Django use metaclasses to transform class definitions into database schemas, automatically creating fields and methods.',
        },
        {
          id: 'mc5',
          question: 'What is __init_subclass__ used for?',
          options: [
            'Initializing object instances',
            'A simpler alternative to metaclasses for controlling subclass creation',
            'Defining class variables',
            'Creating abstract methods',
          ],
          correctAnswer: 1,
          explanation:
            '__init_subclass__ (Python 3.6+) is a simpler alternative to metaclasses for customizing subclass creation, without needing a metaclass.',
        },
      ],
    },
  ],
  keyTakeaways: [
    'Decorators modify functions without changing their code—use @functools.wraps to preserve metadata',
    'Generators provide memory-efficient lazy evaluation using yield—ideal for large datasets',
    'Context managers guarantee cleanup with __enter__ and __exit__—always use for resources',
    'Metaclasses control class creation—powerful but rarely needed, consider simpler alternatives first',
    'Advanced features enable elegant solutions—master them for production Python development',
  ],
  relatedProblems: [
    'decorator-retry',
    'decorator-cache',
    'decorator-timer',
    'generator-fibonacci',
    'generator-file-reader',
    'context-manager-timer',
    'context-manager-database',
    'metaclass-singleton',
    'property-descriptor',
    'iterator-custom',
    'coroutine-pipeline',
    'async-context-manager',
    'decorator-params',
    'generator-send',
    'metaclass-registry',
    'descriptor-validation',
    'generator-pipeline',
    'context-manager-suppress',
    'decorator-class',
    'functools-compose',
  ],
};

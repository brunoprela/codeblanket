/**
 * Contextlib Utilities for Context Managers
 * Problem ID: advanced-contextlib-utilities
 * Order: 39
 */

import { Problem } from '../../../types';

export const contextlib_utilitiesProblem: Problem = {
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
};

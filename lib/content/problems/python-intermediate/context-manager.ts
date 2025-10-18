/**
 * Custom Context Manager
 * Problem ID: intermediate-context-manager
 * Order: 12
 */

import { Problem } from '../../../types';

export const intermediate_context_managerProblem: Problem = {
  id: 'intermediate-context-manager',
  title: 'Custom Context Manager',
  difficulty: 'Medium',
  description: `Create a context manager that temporarily changes directory and ensures cleanup.

**Requirements:**
- Change to specified directory
- Automatically return to original directory
- Handle errors gracefully
- Support both class-based and function-based implementations

**Usage:**
\`\`\`python
with ChangeDirectory('/tmp'):
    # Working in /tmp
    print(os.getcwd())
# Automatically back to original directory
\`\`\``,
  examples: [
    {
      input: 'with ChangeDirectory("/tmp"): pass',
      output: 'Changes dir and returns automatically',
    },
  ],
  constraints: [
    'Implement __enter__ and __exit__',
    'Restore original directory even on error',
    'Support with statement',
  ],
  hints: [
    'Save os.getcwd() before changing',
    'Use try/finally in __exit__',
    'Or use @contextmanager decorator',
  ],
  starterCode: `import os
from contextlib import contextmanager

class ChangeDirectory:
    """
    Context manager to temporarily change directory.
    
    Examples:
        >>> with ChangeDirectory('/tmp'):
        ...     print(os.getcwd())
        '/tmp'
    """
    
    def __init__(self, path):
        """
        Initialize with target directory.
        
        Args:
            path: Directory to change to
        """
        # TODO: Store the path and initialize original_dir to None
        self.path = path
        self.original_dir = None
    
    def __enter__(self):
        """Enter context - change directory."""
        # TODO: Save current directory and change to new path
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context - restore directory.
        
        Args:
            exc_type: Exception type if any
            exc_val: Exception value if any
            exc_tb: Exception traceback if any
            
        Returns:
            False to propagate exceptions
        """
        # TODO: Restore original directory
        pass


@contextmanager
def change_directory(path):
    """
    Function-based context manager using decorator.
    
    Args:
        path: Directory to change to
        
    Yields:
        None
        
    Examples:
        >>> with change_directory('/tmp'):
        ...     pass
    """
    # TODO: Implement using yield
    pass


# Test
print(f"Original directory: {os.getcwd()}")

try:
    with ChangeDirectory('/tmp'):
        print(f"Inside context: {os.getcwd()}")
        # Could raise exception here
finally:
    print(f"After context: {os.getcwd()}")


# Test helper function (for automated testing)
def test_change_directory(target_path):
    """Test function for ChangeDirectory - implement the class methods above first!"""
    try:
        original = os.getcwd()
        with ChangeDirectory(target_path):
            changed = os.getcwd()
        restored = os.getcwd()
        # Return True if we successfully changed and restored
        return changed == target_path and restored == original
    except:
        return None  # Return None if methods not yet implemented
`,
  testCases: [
    {
      input: ['/tmp'],
      expected: true,
      functionName: 'test_change_directory',
    },
  ],
  solution: `import os
from contextlib import contextmanager

class ChangeDirectory:
    def __init__(self, path):
        self.path = path
        self.original_dir = None
    
    def __enter__(self):
        self.original_dir = os.getcwd()
        os.chdir(self.path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always restore original directory
        os.chdir(self.original_dir)
        # Return False to propagate exceptions
        return False


@contextmanager
def change_directory(path):
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)


# Test helper function (for automated testing)
def test_change_directory(target_path):
    """Test function for ChangeDirectory."""
    original = os.getcwd()
    with ChangeDirectory(target_path):
        changed = os.getcwd()
    restored = os.getcwd()
    return changed == target_path and restored == original


# More advanced: File opener with automatic cleanup
class FileOpener:
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 12,
  topic: 'Python Intermediate',
};

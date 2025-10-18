/**
 * Context Manager Class
 * Problem ID: oop-context-manager-class
 * Order: 13
 */

import { Problem } from '../../../types';

export const context_manager_classProblem: Problem = {
  id: 'oop-context-manager-class',
  title: 'Context Manager Class',
  difficulty: 'Medium',
  description: `Create a class that can be used with 'with' statement.

**Protocol:**
- __enter__ called when entering
- __exit__ called when exiting
- Handle exceptions in __exit__

This tests:
- Context manager protocol
- Resource management
- Exception handling`,
  examples: [
    {
      input: 'with Manager() as m:',
      output: 'Automatic setup and cleanup',
    },
  ],
  constraints: ['Implement __enter__ and __exit__', 'Handle cleanup'],
  hints: [
    '__enter__ returns self or resource',
    '__exit__ gets exception info',
    'Return True to suppress exception',
  ],
  starterCode: `class FileManager:
    """Context manager for file operations"""
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        """Enter context - open file"""
        from io import StringIO
        # For testing, use StringIO
        self.file = StringIO()
        self.file.write("test content")
        self.file.seek(0)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - close file"""
        if self.file:
            self.file.close()
        # Return False to propagate exceptions
        return False


def test_context_manager():
    """Test context manager"""
    with FileManager("test.txt", "r") as f:
        content = f.read()
    
    return len(content)
`,
  testCases: [
    {
      input: [],
      expected: 12,
      functionName: 'test_context_manager',
    },
  ],
  solution: `class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        from io import StringIO
        self.file = StringIO()
        self.file.write("test content")
        self.file.seek(0)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False


def test_context_manager():
    with FileManager("test.txt", "r") as f:
        content = f.read()
    
    return len(content)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 13,
  topic: 'Python Object-Oriented Programming',
};

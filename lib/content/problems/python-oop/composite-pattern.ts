/**
 * Composite Pattern for File System
 * Problem ID: oop-composite-pattern
 * Order: 8
 */

import { Problem } from '../../../types';

export const composite_patternProblem: Problem = {
  id: 'oop-composite-pattern',
  title: 'Composite Pattern for File System',
  difficulty: 'Hard',
  description: `Implement the Composite pattern to represent a file system hierarchy.

Create:
- \`FileSystemItem\` abstract base class with \`get_size()\` method
- \`File\` class (leaf) with size attribute
- \`Directory\` class (composite) that can contain files and directories
- Methods: \`add(item)\`, \`remove(item)\`, \`get_size()\` (sum of all contents)

**Pattern:** Composite lets clients treat individual objects and compositions uniformly.`,
  examples: [
    {
      input: 'dir.add(File(100)); dir.add(File(200)); dir.get_size()',
      output: '300',
    },
  ],
  constraints: [
    'Both File and Directory inherit from FileSystemItem',
    'Directory can contain files and other directories',
    'get_size() recursively calculates total size',
  ],
  hints: [
    'Directory stores children in a list',
    'File returns its size directly',
    "Directory sums children's sizes",
  ],
  starterCode: `from abc import ABC, abstractmethod

class FileSystemItem(ABC):
    """Abstract base for files and directories."""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def get_size(self):
        """Get size in bytes."""
        pass


class File(FileSystemItem):
    """File with fixed size."""
    
    def __init__(self, name, size):
        super().__init__(name)
        self.size = size
    
    def get_size(self):
        # Return file size
        pass


class Directory(FileSystemItem):
    """Directory that can contain files and directories."""
    
    def __init__(self, name):
        super().__init__(name)
        self.children = []
    
    def add(self, item):
        """Add a file or directory."""
        pass
    
    def remove(self, item):
        """Remove a file or directory."""
        pass
    
    def get_size(self):
        """Get total size of all contents."""
        pass


# Test
root = Directory("root")
docs = Directory("documents")
pics = Directory("pictures")

docs.add(File("resume.pdf", 1024))
docs.add(File("cover_letter.pdf", 512))
pics.add(File("photo1.jpg", 2048))
pics.add(File("photo2.jpg", 1536))

root.add(docs)
root.add(pics)
root.add(File("readme.txt", 256))

print(f"Total size: {root.get_size()} bytes")


def test_composite_pattern(*file_sizes):
    """Test function for Composite pattern."""
    directory = Directory("test")
    for i, size in enumerate(file_sizes):
        directory.add(File(f"file{i}.txt", size))
    return directory.get_size()
`,
  testCases: [
    {
      input: [1024, 512, 256], // file sizes
      expected: 1792, // sum
      functionName: 'test_composite_pattern',
    },
  ],
  solution: `from abc import ABC, abstractmethod

class FileSystemItem(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def get_size(self):
        pass


class File(FileSystemItem):
    def __init__(self, name, size):
        super().__init__(name)
        self.size = size
    
    def get_size(self):
        return self.size


class Directory(FileSystemItem):
    def __init__(self, name):
        super().__init__(name)
        self.children = []
    
    def add(self, item):
        if not isinstance(item, FileSystemItem):
            raise TypeError("Can only add FileSystemItem")
        self.children.append(item)
    
    def remove(self, item):
        if item in self.children:
            self.children.remove(item)
    
    def get_size(self):
        return sum(child.get_size() for child in self.children)


def test_composite_pattern(*file_sizes):
    """Test function for Composite pattern."""
    directory = Directory("test")
    for i, size in enumerate(file_sizes):
        directory.add(File(f"file{i}.txt", size))
    return directory.get_size()`,
  timeComplexity: 'O(n) where n is total number of items',
  spaceComplexity: 'O(d) where d is depth of directory tree',
  order: 8,
  topic: 'Python Object-Oriented Programming',
};

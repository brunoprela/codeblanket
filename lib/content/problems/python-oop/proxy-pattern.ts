/**
 * Proxy Pattern
 * Problem ID: oop-proxy-pattern
 * Order: 40
 */

import { Problem } from '../../../types';

export const proxy_patternProblem: Problem = {
  id: 'oop-proxy-pattern',
  title: 'Proxy Pattern',
  difficulty: 'Medium',
  description: `Implement proxy pattern to control access to an object.

**Pattern:**
- Surrogate/placeholder
- Controls access
- Add functionality (caching, logging, access control)
- Same interface as real object

This tests:
- Proxy pattern
- Access control
- Lazy loading`,
  examples: [
    {
      input: 'Proxy controls access to real object',
      output: 'Can add caching, logging, etc.',
    },
  ],
  constraints: ['Same interface as real object', 'Control access'],
  hints: [
    'Proxy wraps real object',
    'Delegates requests',
    'Can add checks/caching',
  ],
  starterCode: `class Image:
    """Real object"""
    def __init__(self, filename):
        self.filename = filename
        self._load_from_disk()
    
    def _load_from_disk(self):
        self.data = f"Image data from {self.filename}"
    
    def display(self):
        return f"Displaying {self.filename}"


class ImageProxy:
    """Proxy for lazy loading"""
    def __init__(self, filename):
        self.filename = filename
        self._image = None
    
    def display(self):
        """Lazy load on first access"""
        if self._image is None:
            self._image = Image(self.filename)
        return self._image.display()


def test_proxy():
    """Test proxy pattern"""
    # Proxy doesn't load until needed
    proxy = ImageProxy("photo.jpg")
    
    # First display loads image
    result1 = proxy.display()
    
    # Second display uses loaded image
    result2 = proxy.display()
    
    return len(result1 + result2)
`,
  testCases: [
    {
      input: [],
      expected: 42,
      functionName: 'test_proxy',
    },
  ],
  solution: `class Image:
    def __init__(self, filename):
        self.filename = filename
        self._load_from_disk()
    
    def _load_from_disk(self):
        self.data = f"Image data from {self.filename}"
    
    def display(self):
        return f"Displaying {self.filename}"


class ImageProxy:
    def __init__(self, filename):
        self.filename = filename
        self._image = None
    
    def display(self):
        if self._image is None:
            self._image = Image(self.filename)
        return self._image.display()


def test_proxy():
    proxy = ImageProxy("photo.jpg")
    result1 = proxy.display()
    result2 = proxy.display()
    return len(result1 + result2)`,
  timeComplexity: 'O(1) after first access',
  spaceComplexity: 'O(1)',
  order: 40,
  topic: 'Python Object-Oriented Programming',
};

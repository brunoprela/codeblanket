/**
 * advanced-weak-references
 * Order: 44
 */

import { Problem } from '../../../types';

export const weak_referencesProblem: Problem = {
  id: 'advanced-weak-references',
  title: 'WeakRef for Cache',
  difficulty: 'Hard',
  description: `Use weak references to create a cache that doesn't prevent garbage collection.

WeakRef allows objects to be collected:
- Normal references keep objects alive
- Weak references don't prevent collection
- Useful for caches, callbacks

**Use Case:** Large object caching without memory leaks

This tests:
- Memory management
- Garbage collection
- WeakValueDictionary`,
  examples: [
    {
      input: 'Weak reference cache',
      output: 'Objects can be garbage collected',
    },
  ],
  constraints: ['Use weakref module', 'Objects can be collected'],
  hints: [
    'Use WeakValueDictionary',
    'Values can disappear',
    'Check if key exists before access',
  ],
  starterCode: `import weakref

class WeakCache:
    """
    Cache using weak references.
    """
    def __init__(self):
        self.cache = weakref.WeakValueDictionary()
    
    def set(self, key, value):
        """Store value with weak reference"""
        self.cache[key] = value
    
    def get(self, key):
        """Get value if still alive"""
        return self.cache.get(key)
    
    def contains(self, key):
        """Check if key exists"""
        return key in self.cache


def test_weak_cache():
    """Test weak reference cache"""
    cache = WeakCache()
    
    # Store object
    obj = [1, 2, 3]
    cache.set('data', obj)
    
    # Object is alive, should retrieve it
    result = cache.get('data')
    
    if result != [1, 2, 3]:
        return "FAIL: Should retrieve object"
    
    return result[0]
`,
  testCases: [
    {
      input: [],
      expected: 1,
      functionName: 'test_weak_cache',
    },
  ],
  solution: `import weakref

class WeakCache:
    def __init__(self):
        self.cache = weakref.WeakValueDictionary()
    
    def set(self, key, value):
        self.cache[key] = value
    
    def get(self, key):
        return self.cache.get(key)
    
    def contains(self, key):
        return key in self.cache


def test_weak_cache():
    cache = WeakCache()
    
    obj = [1, 2, 3]
    cache.set('data', obj)
    
    result = cache.get('data')
    
    if result != [1, 2, 3]:
        return "FAIL: Should retrieve object"
    
    return result[0]`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(n)',
  order: 44,
  topic: 'Python Advanced',
};

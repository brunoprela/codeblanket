/**
 * Weak References for Cache Management
 * Problem ID: advanced-weakref
 * Order: 38
 */

import { Problem } from '../../../types';

export const weakrefProblem: Problem = {
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
};

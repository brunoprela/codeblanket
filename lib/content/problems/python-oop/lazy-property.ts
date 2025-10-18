/**
 * Lazy Property Evaluation
 * Problem ID: oop-lazy-property
 * Order: 29
 */

import { Problem } from '../../../types';

export const lazy_propertyProblem: Problem = {
  id: 'oop-lazy-property',
  title: 'Lazy Property Evaluation',
  difficulty: 'Medium',
  description: `Create a property that computes value only once (lazy evaluation).

**Pattern:**
- Compute value on first access
- Cache result
- Don't recompute

This tests:
- Lazy evaluation
- Property caching
- Performance optimization`,
  examples: [
    {
      input: 'Expensive computation cached',
      output: 'Only computed once',
    },
  ],
  constraints: ['Compute on first access', 'Cache result'],
  hints: [
    'Use property decorator',
    'Store in _cached attribute',
    'Check if already computed',
  ],
  starterCode: `class LazyProperty:
    """Descriptor for lazy property"""
    def __init__(self, function):
        self.function = function
        self.name = function.__name__
    
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        
        # Check if value already cached
        attr_name = f'_lazy_{self.name}'
        if not hasattr(obj, attr_name):
            # Compute and cache
            setattr(obj, attr_name, self.function(obj))
        
        return getattr(obj, attr_name)


class DataAnalyzer:
    """Analyze data with lazy properties"""
    def __init__(self, data):
        self.data = data
    
    @LazyProperty
    def average(self):
        """Expensive computation"""
        print("Computing average...")
        return sum(self.data) / len(self.data)
    
    @LazyProperty
    def total(self):
        """Another expensive computation"""
        print("Computing total...")
        return sum(self.data)


def test_lazy():
    """Test lazy property"""
    analyzer = DataAnalyzer([1, 2, 3, 4, 5])
    
    # First access computes
    avg1 = analyzer.average
    
    # Second access uses cache
    avg2 = analyzer.average
    
    return int(avg1 + avg2)
`,
  testCases: [
    {
      input: [],
      expected: 6,
      functionName: 'test_lazy',
    },
  ],
  solution: `class LazyProperty:
    def __init__(self, function):
        self.function = function
        self.name = function.__name__
    
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        
        attr_name = f'_lazy_{self.name}'
        if not hasattr(obj, attr_name):
            setattr(obj, attr_name, self.function(obj))
        
        return getattr(obj, attr_name)


class DataAnalyzer:
    def __init__(self, data):
        self.data = data
    
    @LazyProperty
    def average(self):
        return sum(self.data) / len(self.data)
    
    @LazyProperty
    def total(self):
        return sum(self.data)


def test_lazy():
    analyzer = DataAnalyzer([1, 2, 3, 4, 5])
    avg1 = analyzer.average
    avg2 = analyzer.average
    return int(avg1 + avg2)`,
  timeComplexity: 'O(1) after first access',
  spaceComplexity: 'O(1)',
  order: 29,
  topic: 'Python Object-Oriented Programming',
};

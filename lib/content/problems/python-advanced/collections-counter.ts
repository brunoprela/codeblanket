/**
 * Counter for Frequency Analysis
 * Problem ID: advanced-collections-counter
 * Order: 21
 */

import { Problem } from '../../../types';

export const collections_counterProblem: Problem = {
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
};

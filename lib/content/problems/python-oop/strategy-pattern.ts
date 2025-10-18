/**
 * Strategy Pattern for Sorting
 * Problem ID: oop-strategy-pattern
 * Order: 7
 */

import { Problem } from '../../../types';

export const strategy_patternProblem: Problem = {
  id: 'oop-strategy-pattern',
  title: 'Strategy Pattern for Sorting',
  difficulty: 'Medium',
  description: `Implement the Strategy pattern to allow dynamic selection of sorting algorithms.

Create:
- \`SortStrategy\` interface with \`sort(data)\` method
- Concrete strategies: \`BubbleSort\`, \`QuickSort\`, \`MergeSort\`
- \`DataProcessor\` class that accepts and uses a strategy
- Ability to change strategy at runtime

**Pattern:** Strategy encapsulates interchangeable algorithms.`,
  examples: [
    {
      input: 'processor.set_strategy(QuickSort()); processor.sort([3,1,2])',
      output: '[1, 2, 3] using QuickSort',
    },
  ],
  constraints: [
    'Strategies implement common interface',
    'Strategy can be changed at runtime',
    'DataProcessor delegates to strategy',
  ],
  hints: [
    'Store strategy as instance variable',
    'Call strategy.sort() from processor',
    'Each strategy implements differently',
  ],
  starterCode: `from abc import ABC, abstractmethod

class SortStrategy(ABC):
    """Strategy interface for sorting."""
    
    @abstractmethod
    def sort(self, data):
        """Sort the data using this strategy."""
        pass


class BubbleSort(SortStrategy):
    """Bubble sort strategy."""
    
    def sort(self, data):
        # Implement bubble sort
        pass


class QuickSort(SortStrategy):
    """Quick sort strategy."""
    
    def sort(self, data):
        # Implement quick sort (simplified)
        pass


class MergeSort(SortStrategy):
    """Merge sort strategy."""
    
    def sort(self, data):
        # Implement merge sort (simplified)
        pass


class DataProcessor:
    """Processor that uses a sorting strategy."""
    
    def __init__(self, strategy=None):
        self._strategy = strategy
    
    def set_strategy(self, strategy):
        """Change the sorting strategy."""
        pass
    
    def sort(self, data):
        """Sort data using current strategy."""
        pass


# Test
processor = DataProcessor()
processor.set_strategy(QuickSort())
result = processor.sort([3, 1, 4, 1, 5, 9, 2, 6])
print(result)

processor.set_strategy(BubbleSort())
result = processor.sort([3, 1, 4, 1, 5, 9, 2, 6])
print(result)


def test_strategy_pattern(data, strategy_name):
    """Test function for Strategy pattern."""
    processor = DataProcessor()
    if strategy_name == 'QuickSort':
        processor.set_strategy(QuickSort())
    elif strategy_name == 'BubbleSort':
        processor.set_strategy(BubbleSort())
    elif strategy_name == 'MergeSort':
        processor.set_strategy(MergeSort())
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return processor.sort(data)
`,
  testCases: [
    {
      input: [[3, 1, 2], 'QuickSort'],
      expected: [1, 2, 3],
      functionName: 'test_strategy_pattern',
    },
  ],
  solution: `from abc import ABC, abstractmethod

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass


class BubbleSort(SortStrategy):
    def sort(self, data):
        arr = data.copy()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr


class QuickSort(SortStrategy):
    def sort(self, data):
        return sorted(data)  # Simplified using built-in


class MergeSort(SortStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return data.copy()
        
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result


class DataProcessor:
    def __init__(self, strategy=None):
        self._strategy = strategy
    
    def set_strategy(self, strategy):
        self._strategy = strategy
    
    def sort(self, data):
        if self._strategy is None:
            raise ValueError("No strategy set")
        return self._strategy.sort(data)


def test_strategy_pattern(data, strategy_name):
    """Test function for Strategy pattern."""
    processor = DataProcessor()
    if strategy_name == 'QuickSort':
        processor.set_strategy(QuickSort())
    elif strategy_name == 'BubbleSort':
        processor.set_strategy(BubbleSort())
    elif strategy_name == 'MergeSort':
        processor.set_strategy(MergeSort())
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return processor.sort(data)`,
  timeComplexity:
    'Depends on strategy (O(n²) for bubble, O(n log n) for others)',
  spaceComplexity: 'O(n)',
  order: 7,
  topic: 'Python Object-Oriented Programming',
};

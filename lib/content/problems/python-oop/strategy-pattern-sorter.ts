/**
 * Strategy Pattern
 * Problem ID: oop-strategy-pattern-sorter
 * Order: 32
 */

import { Problem } from '../../../types';

export const strategy_pattern_sorterProblem: Problem = {
  id: 'oop-strategy-pattern-sorter',
  title: 'Strategy Pattern',
  difficulty: 'Medium',
  description: `Implement strategy pattern to swap algorithms dynamically.

**Pattern:**
- Define family of algorithms
- Encapsulate each one
- Make them interchangeable
- Client chooses strategy

This tests:
- Strategy pattern
- Algorithm swapping
- Polymorphism`,
  examples: [
    {
      input: 'context.set_strategy(new_strategy)',
      output: 'Changes behavior dynamically',
    },
  ],
  constraints: ['Define strategy interface', 'Swap strategies'],
  hints: [
    'Strategy base class/interface',
    'Context holds strategy',
    'Delegate to strategy',
  ],
  starterCode: `class SortStrategy:
    """Base strategy"""
    def sort(self, data):
        raise NotImplementedError


class QuickSort(SortStrategy):
    """Quick sort strategy"""
    def sort(self, data):
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)


class BubbleSort(SortStrategy):
    """Bubble sort strategy"""
    def sort(self, data):
        data = list(data)
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        return data


class Sorter:
    """Context that uses strategy"""
    def __init__(self, strategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy):
        """Change strategy"""
        self.strategy = strategy
    
    def sort(self, data):
        """Delegate to strategy"""
        return self.strategy.sort(data)


def test_strategy():
    """Test strategy pattern"""
    data = [5, 2, 8, 1, 9]
    
    # Use quick sort
    sorter = Sorter(QuickSort())
    result1 = sorter.sort(data)
    
    # Switch to bubble sort
    sorter.set_strategy(BubbleSort())
    result2 = sorter.sort(data)
    
    return result1[0] + result2[-1]
`,
  testCases: [
    {
      input: [],
      expected: 10,
      functionName: 'test_strategy',
    },
  ],
  solution: `class SortStrategy:
    def sort(self, data):
        raise NotImplementedError


class QuickSort(SortStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)


class BubbleSort(SortStrategy):
    def sort(self, data):
        data = list(data)
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        return data


class Sorter:
    def __init__(self, strategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy):
        self.strategy = strategy
    
    def sort(self, data):
        return self.strategy.sort(data)


def test_strategy():
    data = [5, 2, 8, 1, 9]
    sorter = Sorter(QuickSort())
    result1 = sorter.sort(data)
    sorter.set_strategy(BubbleSort())
    result2 = sorter.sort(data)
    return result1[0] + result2[-1]`,
  timeComplexity: 'Depends on strategy',
  spaceComplexity: 'Depends on strategy',
  order: 32,
  topic: 'Python Object-Oriented Programming',
};

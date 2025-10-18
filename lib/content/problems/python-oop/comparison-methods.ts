/**
 * Comparison Methods (__lt__, __le__, etc.)
 * Problem ID: oop-comparison-methods
 * Order: 19
 */

import { Problem } from '../../../types';

export const comparison_methodsProblem: Problem = {
  id: 'oop-comparison-methods',
  title: 'Comparison Methods (__lt__, __le__, etc.)',
  difficulty: 'Medium',
  description: `Implement comparison methods for custom ordering.

**Comparison methods:**
- __lt__ for <
- __le__ for <=
- __gt__ for >
- __ge__ for >=
- __eq__ for ==
- __ne__ for !=

Or use @total_ordering with just __eq__ and one other.

This tests:
- Comparison protocol
- Sorting support
- Ordering logic`,
  examples: [
    {
      input: 'person1 < person2',
      output: 'Custom comparison',
    },
  ],
  constraints: ['Implement comparison methods', 'Enable sorting'],
  hints: [
    'Implement __lt__ and __eq__',
    'Can use @total_ordering',
    'Enable sorted(), min(), max()',
  ],
  starterCode: `from functools import total_ordering

@total_ordering
class Student:
    """Student with grade comparison"""
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other):
        """Equal if same grade"""
        return self.grade == other.grade
    
    def __lt__(self, other):
        """Less than if lower grade"""
        return self.grade < other.grade
    
    def __repr__(self):
        return f"Student({self.name}, {self.grade})"


def test_comparisons():
    """Test comparison methods"""
    students = [
        Student("Alice", 85),
        Student("Bob", 92),
        Student("Charlie", 78),
    ]
    
    # Sort by grade
    sorted_students = sorted(students)
    
    # Get highest grade
    best = max(students)
    
    return best.grade
`,
  testCases: [
    {
      input: [],
      expected: 92,
      functionName: 'test_comparisons',
    },
  ],
  solution: `from functools import total_ordering

@total_ordering
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other):
        return self.grade == other.grade
    
    def __lt__(self, other):
        return self.grade < other.grade
    
    def __repr__(self):
        return f"Student({self.name}, {self.grade})"


def test_comparisons():
    students = [
        Student("Alice", 85),
        Student("Bob", 92),
        Student("Charlie", 78),
    ]
    
    sorted_students = sorted(students)
    best = max(students)
    
    return best.grade`,
  timeComplexity: 'O(n log n) for sorting',
  spaceComplexity: 'O(n)',
  order: 19,
  topic: 'Python Object-Oriented Programming',
};

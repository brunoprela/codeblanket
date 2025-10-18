/**
 * Mixin Classes
 * Problem ID: oop-mixin-classes
 * Order: 24
 */

import { Problem } from '../../../types';

export const mixin_classesProblem: Problem = {
  id: 'oop-mixin-classes',
  title: 'Mixin Classes',
  difficulty: 'Medium',
  description: `Use mixin classes to add functionality without main inheritance.

**Mixin:**
- Provides specific functionality
- Not meant to stand alone
- Combined with main class
- Multiple mixins possible

This tests:
- Mixin pattern
- Multiple inheritance
- Modular functionality`,
  examples: [
    {
      input: 'class MyClass(Mixin1, Mixin2, Base)',
      output: 'Combines functionality',
    },
  ],
  constraints: ['Create mixin classes', 'Combine with base class'],
  hints: [
    'Mixins provide specific methods',
    'Order in inheritance list',
    'Use multiple mixins',
  ],
  starterCode: `class JSONMixin:
    """Mixin for JSON serialization"""
    def to_json(self):
        import json
        return json.dumps(self.__dict__)


class ReprMixin:
    """Mixin for nice repr"""
    def __repr__(self):
        attrs = ', '.join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


class User(JSONMixin, ReprMixin):
    """User with mixin functionality"""
    def __init__(self, name, age):
        self.name = name
        self.age = age


def test_mixins():
    """Test mixin classes"""
    user = User("Alice", 30)
    
    # Use JSONMixin method
    json_str = user.to_json()
    
    # Use ReprMixin method
    repr_str = repr(user)
    
    return len(json_str)
`,
  testCases: [
    {
      input: [],
      expected: 28,
      functionName: 'test_mixins',
    },
  ],
  solution: `class JSONMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)


class ReprMixin:
    def __repr__(self):
        attrs = ', '.join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


class User(JSONMixin, ReprMixin):
    def __init__(self, name, age):
        self.name = name
        self.age = age


def test_mixins():
    user = User("Alice", 30)
    json_str = user.to_json()
    repr_str = repr(user)
    return len(json_str)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 24,
  topic: 'Python Object-Oriented Programming',
};

/**
 * Singleton Metaclass
 * Problem ID: metaclass-singleton
 * Order: 8
 */

import { Problem } from '../../../types';

export const metaclass_singletonProblem: Problem = {
  id: 'metaclass-singleton',
  title: 'Singleton Metaclass',
  difficulty: 'Hard',
  description: `Implement a metaclass that ensures a class can only have one instance (Singleton pattern).

The metaclass should:
- Store instances in a class-level dictionary
- Return existing instance if one exists
- Create new instance only if needed
- Work with any class that uses it

**Use Case:** Database connections, configuration objects, logging.`,
  examples: [
    {
      input: 'Database() called twice',
      output: 'Returns same instance both times',
    },
  ],
  constraints: [
    'Must be a metaclass',
    'Thread-safety not required',
    'Support class arguments',
  ],
  hints: [
    'Override __call__ method',
    'Store instances in _instances dict',
    'Check if class in dict before creating',
  ],
  starterCode: `class Singleton(type):
    """
    Metaclass that creates singleton classes.
    """
    
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        # Your code here
        pass


class Database(metaclass=Singleton):
    def __init__(self, name="default"):
        self.name = name


db1 = Database("prod")
db2 = Database("dev")
print(db1 is db2)  # Should be True
print(db1.name)    # Should be "prod" (first call wins)


# Test helper function (for automated testing)
def test_singleton(names):
    """Test function for Singleton - implement the metaclass above first!"""
    try:
        class TestDB(metaclass=Singleton):
            def __init__(self, name="default"):
                self.name = name
        
        db1 = TestDB(names[0])
        db2 = TestDB(names[1])
        return db1 is db2  # Should be True for singleton
    except:
        return None  # Return None if methods not yet implemented
`,
  testCases: [
    {
      input: [['prod', 'dev']],
      expected: true, // db1 is db2
      functionName: 'test_singleton',
    },
  ],
  solution: `class Singleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# Test helper function (for automated testing)
def test_singleton(names):
    """Test function for Singleton."""
    class TestDB(metaclass=Singleton):
        def __init__(self, name="default"):
            self.name = name
    
    db1 = TestDB(names[0])
    db2 = TestDB(names[1])
    return db1 is db2`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1) per class',
  order: 8,
  topic: 'Python Advanced',
};

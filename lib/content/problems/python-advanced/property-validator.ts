/**
 * advanced-property-validator
 * Order: 40
 */

import { Problem } from '../../../types';

export const property_validatorProblem: Problem = {
  id: 'advanced-property-validator',
  title: 'Property with Validation',
  difficulty: 'Medium',
  description: `Create a property descriptor that validates values before setting.

The descriptor should:
- Validate value before assignment
- Raise ValueError for invalid values
- Support custom validation functions
- Work with any class

**Use Case:** Input validation, type checking, range constraints.`,
  examples: [
    {
      input: 'Age property that only accepts 0-150',
      output: 'Raises ValueError for invalid ages',
    },
  ],
  constraints: ['Must use descriptor protocol', 'Support custom validators'],
  hints: [
    'Implement __set__ method',
    'Call validator function',
    'Use WeakKeyDictionary for storage',
  ],
  starterCode: `class ValidatedProperty:
    """
    Descriptor with validation.
    
    Args:
        validator: Function that validates value
        
    Examples:
        >>> def is_positive(x):
        ...     if x <= 0:
        ...         raise ValueError("Must be positive")
        >>> class Product:
        ...     price = ValidatedProperty(is_positive)
    """
    def __init__(self, validator=None):
        # Your code here
        pass
    
    def __get__(self, obj, objtype=None):
        # Your code here
        pass
    
    def __set__(self, obj, value):
        # Your code here
        pass


def test_validator():
    """Test validated property"""
    def is_positive(x):
        if x <= 0:
            raise ValueError("Must be positive")
    
    class Product:
        price = ValidatedProperty(is_positive)
        
        def __init__(self, price):
            self.price = price
    
    try:
        p = Product(10)
        result = p.price
        p.price = -5  # Should raise
        return "FAIL: Should have raised"
    except ValueError:
        return result
`,
  testCases: [
    {
      input: [],
      expected: 10,
      functionName: 'test_validator',
    },
  ],
  solution: `from weakref import WeakKeyDictionary

class ValidatedProperty:
    def __init__(self, validator=None):
        self.validator = validator
        self.data = WeakKeyDictionary()
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.data.get(obj)
    
    def __set__(self, obj, value):
        if self.validator:
            self.validator(value)
        self.data[obj] = value


def test_validator():
    def is_positive(x):
        if x <= 0:
            raise ValueError("Must be positive")
    
    class Product:
        price = ValidatedProperty(is_positive)
        
        def __init__(self, price):
            self.price = price
    
    try:
        p = Product(10)
        result = p.price
        p.price = -5
        return "FAIL: Should have raised"
    except ValueError:
        return result`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(n) for n instances',
  order: 40,
  topic: 'Python Advanced',
};

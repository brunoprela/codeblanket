/**
 * advanced-custom-exception-hierarchy
 * Order: 50
 */

import { Problem } from '../../../types';

export const custom_exception_hierarchyProblem: Problem = {
  id: 'advanced-custom-exception-hierarchy',
  title: 'Custom Exception Hierarchy',
  difficulty: 'Medium',
  description: `Create a custom exception hierarchy for better error handling.

Exception hierarchy allows:
- Catch by type
- Common base exception
- Specific error information
- Better error handling

**Example:**
\`\`\`python
try:
    raise InvalidInputError()
except ValidationError:  # Base class
    handle_validation()
\`\`\`

This tests:
- Exception inheritance
- Custom exceptions
- Error hierarchies`,
  examples: [
    {
      input: 'Custom validation exceptions',
      output: 'Hierarchy of related errors',
    },
  ],
  constraints: ['Inherit from Exception', 'Create hierarchy'],
  hints: [
    'Base exception for category',
    'Specific exceptions inherit from base',
    'Can catch by base or specific type',
  ],
  starterCode: `class ValidationError(Exception):
    """Base class for validation errors"""
    pass


class InvalidEmailError(ValidationError):
    """Raised when email is invalid"""
    pass


class InvalidAgeError(ValidationError):
    """Raised when age is invalid"""
    pass


def validate_user(email: str, age: int):
    """Validate user data"""
    if '@' not in email:
        raise InvalidEmailError(f"Invalid email: {email}")
    
    if age < 0 or age > 150:
        raise InvalidAgeError(f"Invalid age: {age}")
    
    return "Valid"


def test_exceptions():
    """Test custom exceptions"""
    # Test valid
    result1 = validate_user("test@example.com", 25)
    
    # Test invalid email
    try:
        validate_user("notanemail", 25)
        return "FAIL: Should raise InvalidEmailError"
    except ValidationError:
        pass
    
    # Test invalid age
    try:
        validate_user("test@example.com", 200)
        return "FAIL: Should raise InvalidAgeError"
    except ValidationError:
        pass
    
    return len(result1)
`,
  testCases: [
    {
      input: [],
      expected: 5,
      functionName: 'test_exceptions',
    },
  ],
  solution: `class ValidationError(Exception):
    pass


class InvalidEmailError(ValidationError):
    pass


class InvalidAgeError(ValidationError):
    pass


def validate_user(email: str, age: int):
    if '@' not in email:
        raise InvalidEmailError(f"Invalid email: {email}")
    
    if age < 0 or age > 150:
        raise InvalidAgeError(f"Invalid age: {age}")
    
    return "Valid"


def test_exceptions():
    result1 = validate_user("test@example.com", 25)
    
    try:
        validate_user("notanemail", 25)
        return "FAIL: Should raise InvalidEmailError"
    except ValidationError:
        pass
    
    try:
        validate_user("test@example.com", 200)
        return "FAIL: Should raise InvalidAgeError"
    except ValidationError:
        pass
    
    return len(result1)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 50,
  topic: 'Python Advanced',
};

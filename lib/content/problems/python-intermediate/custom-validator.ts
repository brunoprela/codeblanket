/**
 * Data Validator with Custom Exceptions
 * Problem ID: intermediate-custom-validator
 * Order: 2
 */

import { Problem } from '../../../types';

export const intermediate_custom_validatorProblem: Problem = {
  id: 'intermediate-custom-validator',
  title: 'Data Validator with Custom Exceptions',
  difficulty: 'Medium',
  description: `Create a data validator that validates user input and raises custom exceptions.

**Custom Exceptions:**
- \`InvalidEmailError\` - for invalid email format
- \`InvalidAgeError\` - for ages outside 0-150 range
- \`InvalidPhoneError\` - for invalid phone format

**Validation Rules:**
- Email: must contain @ and .
- Age: integer between 0 and 150
- Phone: format XXX-XXX-XXXX (X = digit)

Create a \`validate_user_data\` function that checks all fields.`,
  examples: [
    {
      input: 'validate_user_data("test@email.com", 25, "555-123-4567")',
      output: 'Returns True if all valid',
    },
    {
      input: 'validate_user_data("invalid", 25, "555-123-4567")',
      output: 'Raises InvalidEmailError',
    },
  ],
  constraints: [
    'Create custom exception classes',
    'Validate all three fields',
    'Provide descriptive error messages',
  ],
  hints: [
    'Inherit from Exception class',
    'Use regex for phone validation',
    'Check email contains @ and .',
  ],
  starterCode: `import re

class InvalidEmailError(Exception):
    """Raised when email format is invalid."""
    pass


class InvalidAgeError(Exception):
    """Raised when age is out of valid range."""
    pass


class InvalidPhoneError(Exception):
    """Raised when phone format is invalid."""
    pass


def validate_user_data(email, age, phone):
    """
    Validate user data.
    
    Args:
        email: Email address string
        age: Age integer
        phone: Phone string in XXX-XXX-XXXX format
        
    Returns:
        True if all validations pass
        
    Raises:
        InvalidEmailError: If email is invalid
        InvalidAgeError: If age is out of range
        InvalidPhoneError: If phone format is wrong
        
    Examples:
        >>> validate_user_data("test@example.com", 25, "555-123-4567")
        True
    """
    pass


# Test
try:
    validate_user_data("test@example.com", 25, "555-123-4567")
    print("Valid data")
except (InvalidEmailError, InvalidAgeError, InvalidPhoneError) as e:
    print(f"Validation error: {e}")
`,
  testCases: [
    {
      input: ['test@example.com', 25, '555-123-4567'],
      expected: true,
    },
    {
      input: ['invalid', 25, '555-123-4567'],
      expected: 'InvalidEmailError',
    },
  ],
  solution: `import re

class InvalidEmailError(Exception):
    """Raised when email format is invalid."""
    pass


class InvalidAgeError(Exception):
    """Raised when age is out of valid range."""
    pass


class InvalidPhoneError(Exception):
    """Raised when phone format is invalid."""
    pass


def validate_user_data(email, age, phone):
    # Validate email
    if '@' not in email or '.' not in email:
        raise InvalidEmailError(f"Invalid email format: {email}")
    
    # Validate age
    if not isinstance(age, int) or age < 0 or age > 150:
        raise InvalidAgeError(f"Age must be between 0 and 150, got: {age}")
    
    # Validate phone
    phone_pattern = r'^\\d{3}-\\d{3}-\\d{4}$'
    if not re.match(phone_pattern, phone):
        raise InvalidPhoneError(f"Phone must be XXX-XXX-XXXX format, got: {phone}")
    
    return True`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 2,
  topic: 'Python Intermediate',
};

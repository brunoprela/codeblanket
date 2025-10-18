/**
 * Multiple Inheritance with Mixins
 * Problem ID: oop-multiple-inheritance-mixin
 * Order: 10
 */

import { Problem } from '../../../types';

export const multiple_inheritance_mixinProblem: Problem = {
  id: 'oop-multiple-inheritance-mixin',
  title: 'Multiple Inheritance with Mixins',
  difficulty: 'Hard',
  description: `Create a flexible class system using mixins for shared functionality.

Implement:
- \`SerializableMixin\` with \`to_dict()\` and \`from_dict()\` methods
- \`TimestampMixin\` that adds created_at and updated_at timestamps
- \`User\` class that inherits from both mixins
- Proper MRO (Method Resolution Order) handling

**Pattern:** Mixins provide reusable functionality without deep inheritance hierarchies.`,
  examples: [
    {
      input: 'user = User("alice"); user.to_dict()',
      output: 'Dictionary with user data and timestamp',
    },
  ],
  constraints: [
    'Mixins should be independent',
    'User class combines both mixins',
    'Demonstrate proper MRO',
  ],
  hints: [
    "Mixins typically don't have __init__",
    'Use super() for cooperative inheritance',
    'Check class.mro() for resolution order',
  ],
  starterCode: `from datetime import datetime
import json

class SerializableMixin:
    """Mixin for JSON serialization."""
    
    def to_dict(self):
        """Convert object to dictionary."""
        pass
    
    @classmethod
    def from_dict(cls, data):
        """Create object from dictionary."""
        pass


class TimestampMixin:
    """Mixin for automatic timestamps."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def touch(self):
        """Update the updated_at timestamp."""
        pass


class User(SerializableMixin, TimestampMixin):
    """User class with serialization and timestamps."""
    
    def __init__(self, username, email=None):
        super().__init__()
        self.username = username
        self.email = email


# Test
user = User("alice", "alice@example.com")
print(user.to_dict())

user.touch()
print(user.updated_at)

# Test MRO
print(User.mro())


def test_mixin(username, email):
    """Test function for Mixin pattern."""
    user = User(username, email)
    result = user.to_dict()
    return type(result).__name__
`,
  testCases: [
    {
      input: ['alice', 'alice@example.com'],
      expected: 'dict',
      functionName: 'test_mixin',
    },
  ],
  solution: `from datetime import datetime
import json

class SerializableMixin:
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data):
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        for key, value in data.items():
            # Try to parse datetime strings
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value)
                except (ValueError, AttributeError):
                    pass
            setattr(instance, key, value)
        return instance
    
    def to_json(self):
        return json.dumps(self.to_dict())


class TimestampMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def touch(self):
        self.updated_at = datetime.now()


class User(SerializableMixin, TimestampMixin):
    def __init__(self, username, email=None):
        super().__init__()
        self.username = username
        self.email = email


def test_mixin(username, email):
    """Test function for Mixin pattern."""
    user = User(username, email)
    result = user.to_dict()
    return type(result).__name__`,
  timeComplexity: 'O(n) where n is number of attributes',
  spaceComplexity: 'O(n)',
  order: 10,
  topic: 'Python Object-Oriented Programming',
};

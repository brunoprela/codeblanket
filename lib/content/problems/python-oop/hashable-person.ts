/**
 * Hashable Person Class
 * Problem ID: hashable-person
 * Order: 18
 */

import { Problem } from '../../../types';

export const hashable_personProblem: Problem = {
  id: 'hashable-person',
  title: 'Hashable Person Class',
  difficulty: 'Medium',
  category: 'python-oop',
  description: `Create a \`Person\` class that can be used in sets and as dictionary keys.

Implement:
- \`__init__(name, age, email)\`: Initialize person
- \`__eq__(other)\`: Two people are equal if same email
- \`__hash__()\`: Hash based on email (immutable identifier)
- \`__repr__()\`: Return "Person(name, age, email)"

**Why this matters:** For objects to work in sets/dicts, they must be hashable. If you implement \`__eq__\`, you must implement \`__hash__\` such that equal objects have equal hashes.

**Examples:**
\`\`\`python
p1 = Person("Alice", 30, "alice@example.com")
p2 = Person("Alice Smith", 30, "alice@example.com")  # Same email
p3 = Person("Bob", 25, "bob@example.com")

print(p1 == p2)        # True (same email)
print(p1 is p2)        # False (different objects)

people = {p1, p2, p3}  # Set treats p1 and p2 as same
print(len(people))     # 2

lookup = {p1: "Manager", p3: "Engineer"}
print(lookup[p2])      # "Manager" (p2 treated same as p1)
\`\`\``,
  starterCode: `class Person:
    def __init__(self, name, age, email):
        """Initialize person."""
        pass
    
    def __eq__(self, other):
        """Check equality based on email."""
        pass
    
    def __hash__(self):
        """Return hash based on email."""
        pass
    
    def __repr__(self):
        """Return string representation."""
        pass`,
  testCases: [
    {
      input: [
        ['Person', 'Alice', 30, 'alice@example.com'],
        ['Person', 'Alice Smith', 30, 'alice@example.com'],
        ['equals'],
      ],
      expected: true,
    },
    {
      input: [
        ['Person', 'Alice', 30, 'alice@example.com'],
        ['Person', 'Alice', 30, 'alice@example.com'],
        ['set_len'],
      ],
      expected: 1,
    },
    {
      input: [
        ['Person', 'Alice', 30, 'alice@example.com'],
        ['hash_equals_self'],
      ],
      expected: true,
    },
  ],
  hints: [
    'Email is the unique identifier (like SSN)',
    '__hash__ should return hash(self.email)',
    '__eq__ should check if emails are equal',
    'Always check isinstance(other, Person) in __eq__',
  ],
  solution: `class Person:
    def __init__(self, name, age, email):
        """Initialize person."""
        self.name = name
        self.age = age
        self.email = email
    
    def __eq__(self, other):
        """Check equality based on email."""
        if not isinstance(other, Person):
            return False
        return self.email == other.email
    
    def __hash__(self):
        """Return hash based on email (immutable identifier)."""
        return hash(self.email)
    
    def __repr__(self):
        """Return string representation."""
        return f"Person('{self.name}', {self.age}, '{self.email}')"


# Test
p1 = Person("Alice", 30, "alice@example.com")
p2 = Person("Alice Smith", 30, "alice@example.com")  # Same email!
p3 = Person("Bob", 25, "bob@example.com")

print(p1 == p2)        # True
print(hash(p1) == hash(p2))  # True

# Use in set
people = {p1, p2, p3}
print(len(people))     # 2 (p1 and p2 treated as one)

# Use as dict key
lookup = {p1: "Manager", p3: "Engineer"}
print(lookup[p2])      # "Manager" (p2 same as p1)`,
  timeComplexity: 'O(1) for all operations',
  spaceComplexity: 'O(1)',
  order: 18,
  topic: 'Python Object-Oriented Programming',
};

/**
 * Dataclasses for Structured Data Section
 */

export const dataclassesSection = {
  id: 'dataclasses',
  title: 'Dataclasses for Structured Data',
  content: `**What are Dataclasses?**
Dataclasses (Python 3.7+) are a cleaner, more concise way to create classes that primarily store data, automatically generating \`__init__\`, \`__repr__\`, \`__eq__\`, and other methods.

**Basic Dataclass:**
\`\`\`python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    email: str

# Automatically generated __init__
person = Person("Alice", 30, "alice@example.com")

# Automatically generated __repr__
print(person)  # Person(name='Alice', age=30, email='alice@example.com')

# Automatically generated __eq__
person2 = Person("Alice", 30, "alice@example.com")
print(person == person2)  # True
\`\`\`

**Default Values:**
\`\`\`python
from dataclasses import dataclass, field

@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0  # Default value
    tags: list = field(default_factory=list)  # Mutable default

# Use default values
product = Product("Widget", 19.99)
print(product.quantity)  # 0

# IMPORTANT: Use field(default_factory) for mutable defaults!
# WRONG: tags: list = []  # Shared between instances!
# CORRECT: tags: list = field(default_factory=list)
\`\`\`

**Post-Init Processing:**
\`\`\`python
@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)  # Not in __init__
    
    def __post_init__(self):
        """Called after __init__"""
        self.area = self.width * self.height
        if self.width < 0 or self.height < 0:
            raise ValueError("Dimensions must be positive")

rect = Rectangle(5, 10)
print(rect.area)  # 50
\`\`\`

**Immutable Dataclasses:**
\`\`\`python
@dataclass(frozen=True)
class Point:
    x: float
    y: float

point = Point(10, 20)
# point.x = 30  # FrozenInstanceError!

# Frozen dataclasses are hashable
points_set = {Point(0, 0), Point(1, 1)}
\`\`\`

**Ordering:**
\`\`\`python
@dataclass(order=True)
class Score:
    value: int
    player: str = field(compare=False)  # Don't use in comparisons

scores = [Score(90, "Alice"), Score(85, "Bob"), Score(95, "Charlie")]
print(sorted(scores))  # Sorted by value
# [Score(value=85, player='Bob'), Score(value=90, player='Alice'), ...]
\`\`\`

**Field Options:**
\`\`\`python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Student:
    name: str
    id: int = field(repr=False)  # Don't show in repr
    grades: List[int] = field(default_factory=list)
    gpa: float = field(init=False, repr=False)  # Calculated field
    
    def __post_init__(self):
        if self.grades:
            self.gpa = sum(self.grades) / len(self.grades)
        else:
            self.gpa = 0.0
\`\`\`

**Inheritance with Dataclasses:**
\`\`\`python
@dataclass
class Person:
    name: str
    age: int

@dataclass
class Employee(Person):
    employee_id: int
    department: str

# Inherits fields from Person
emp = Employee("Alice", 30, 12345, "Engineering")
print(emp)  # Employee(name='Alice', age=30, employee_id=12345, ...)
\`\`\`

**Conversion Methods:**
\`\`\`python
from dataclasses import dataclass, asdict, astuple

@dataclass
class Config:
    host: str
    port: int
    debug: bool

config = Config("localhost", 8080, True)

# Convert to dict
print(asdict(config))
# {'host': 'localhost', 'port': 8080, 'debug': True}

# Convert to tuple
print(astuple(config))
# ('localhost', 8080, True)
\`\`\`

**Real-World Example - API Response:**
\`\`\`python
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

@dataclass
class User:
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True
    roles: List[str] = field(default_factory=list)
    
    @classmethod
    def from_json(cls, data: dict):
        """Create from JSON response"""
        return cls(
            id=data['id'],
            username=data['username'],
            email=data['email'],
            created_at=datetime.fromisoformat(data['created_at']),
            is_active=data.get('is_active', True),
            roles=data.get('roles', [])
        )
    
    def has_role(self, role: str) -> bool:
        return role in self.roles

# Usage
user_data = {
    'id': 1,
    'username': 'alice',
    'email': 'alice@example.com',
    'created_at': '2024-01-01T00:00:00',
    'roles': ['admin', 'user']
}
user = User.from_json(user_data)
print(user.has_role('admin'))  # True
\`\`\`

**Dataclass vs Regular Class:**

| Feature | Regular Class | Dataclass |
|---------|--------------|-----------|
| \`__init__\` | Manual | Auto-generated |
| \`__repr__\` | Manual | Auto-generated |
| \`__eq__\` | Manual | Auto-generated |
| Type Hints | Optional | Required |
| Boilerplate | Lots | Minimal |
| Flexibility | Maximum | Good |

**When to Use Dataclasses:**
- ✅ Classes that primarily store data
- ✅ API models, config objects, DTOs
- ✅ When you want immutability (frozen=True)
- ✅ Need auto-generated comparison methods
- ❌ Complex business logic classes (use regular classes)
- ❌ Need dynamic attributes (use regular class or dict)

**Best Practices:**
- Always use \`field(default_factory)\` for mutable defaults
- Use type hints for all fields
- Use \`frozen=True\` for immutable data
- Add \`__post_init__\` for validation or computed fields
- Use \`@classmethod\` for alternative constructors`,
};

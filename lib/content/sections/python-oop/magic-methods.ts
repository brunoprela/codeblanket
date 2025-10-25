/**
 * Magic Methods (Dunder Methods) Section
 */

export const magicmethodsSection = {
  id: 'magic-methods',
  title: 'Magic Methods (Dunder Methods)',
  content: `# Magic Methods (Dunder Methods)

Magic methods (also called **dunder methods** for "double underscore") allow you to define custom behavior for Python\'s built-in operations. They make your objects feel like native Python types.

## Why Magic Methods Matter

Magic methods let you:
- Use operators like \`+\`, \`<\`, \`==\` with your objects
- Make objects printable, iterable, callable
- Support indexing (\`obj[key]\`) and slicing
- Integrate with built-in functions like \`len()\`, \`str()\`

**They're essential for creating Pythonic, professional code!**

---

## String Representation

### \`__str__\` and \`__repr__\`

\`\`\`python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        """Human-readable string (for print, str())"""
        return f"Point at ({self.x}, {self.y})"
    
    def __repr__(self):
        """Developer-friendly representation (for debugging)"""
        return f"Point({self.x}, {self.y})"

p = Point(3, 4)
print(str (p))   # Point at (3, 4)  - uses __str__
print(repr (p))  # Point(3, 4)      - uses __repr__
print(p)        # Point at (3, 4)  - print uses __str__ if available
print([p])      # [Point(3, 4)]    - containers use __repr__
\`\`\`

**Rule of thumb:**
- \`__str__\`: User-friendly output
- \`__repr__\`: Unambiguous, ideally \`eval (repr (obj)) == obj\`
- Always implement \`__repr__\`; \`__str__\` is optional

---

## Comparison Magic Methods

Make objects comparable with \`<\`, \`>\`, \`==\`, etc.

\`\`\`python
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other):
        """Check equality (==)"""
        if not isinstance (other, Student):
            return False
        return self.name == other.name and self.grade == other.grade
    
    def __lt__(self, other):
        """Less than (<)"""
        return self.grade < other.grade
    
    def __le__(self, other):
        """Less than or equal (<=)"""
        return self.grade <= other.grade
    
    def __gt__(self, other):
        """Greater than (>)"""
        return self.grade > other.grade
    
    def __ge__(self, other):
        """Greater than or equal (>=)"""
        return self.grade >= other.grade
    
    def __repr__(self):
        return f"Student('{self.name}', {self.grade})"

alice = Student("Alice", 95)
bob = Student("Bob", 87)

print(alice > bob)         # True
print(alice == alice)      # True
print(sorted([alice, bob]))  # [Student('Bob', 87), Student('Alice', 95)]
\`\`\`

**Pro tip:** Use \`@functools.total_ordering\` decorator:
\`\`\`python
from functools import total_ordering

@total_ordering
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other):
        return self.grade == other.grade
    
    def __lt__(self, other):
        return self.grade < other.grade
    
    # Python auto-generates <=, >, >= from __eq__ and __lt__!
\`\`\`

---

## Arithmetic Magic Methods

Make objects work with \`+\`, \`-\`, \`*\`, etc.

\`\`\`python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Addition (+)"""
        return Vector (self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Subtraction (-)"""
        return Vector (self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Multiplication (*)"""
        return Vector (self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        """Division (/)"""
        return Vector (self.x / scalar, self.y / scalar)
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)    # Vector(4, 6)
print(v1 - v2)    # Vector(2, 2)
print(v1 * 2)     # Vector(6, 8)
print(v1 / 2)     # Vector(1.5, 2.0)
\`\`\`

**In-place variants:**
\`\`\`python
class Vector:
    # ... previous code ...
    
    def __iadd__(self, other):
        """In-place addition (+=)"""
        self.x += other.x
        self.y += other.y
        return self
    
    def __imul__(self, scalar):
        """In-place multiplication (*=)"""
        self.x *= scalar
        self.y *= scalar
        return self

v = Vector(3, 4)
v += Vector(1, 1)  # Uses __iadd__
print(v)  # Vector(4, 5)
\`\`\`

---

## Container Magic Methods

Make objects behave like lists/dicts.

\`\`\`python
class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def __len__(self):
        """Support len()"""
        return len (self.items)
    
    def __getitem__(self, index):
        """Support indexing: cart[0]"""
        return self.items[index]
    
    def __setitem__(self, index, value):
        """Support assignment: cart[0] = 'apple'"""
        self.items[index] = value
    
    def __delitem__(self, index):
        """Support deletion: del cart[0]"""
        del self.items[index]
    
    def __contains__(self, item):
        """Support 'in' operator"""
        return item in self.items
    
    def __iter__(self):
        """Make iterable (for loops)"""
        return iter (self.items)
    
    def add (self, item):
        self.items.append (item)

cart = ShoppingCart()
cart.add("apple")
cart.add("banana")

print(len (cart))         # 2 - uses __len__
print(cart[0])           # 'apple' - uses __getitem__
print("apple" in cart)   # True - uses __contains__

for item in cart:        # Uses __iter__
    print(item)
\`\`\`

---

## Callable Objects

Make instances callable like functions.

\`\`\`python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        """Make object callable"""
        return x * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))   # 10 - calls __call__(5)
print(triple(5))   # 15

# Common use: function decorators, closures, functors
\`\`\`

---

## Context Managers

Implement \`with\` statement.

\`\`\`python
class Timer:
    def __enter__(self):
        """Called when entering 'with' block"""
        import time
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting 'with' block"""
        import time
        elapsed = time.time() - self.start
        print(f"Elapsed: {elapsed:.4f} seconds")
        return False  # Don't suppress exceptions

with Timer():
    # Code to time
    sum (range(1000000))
# Output: Elapsed: 0.0234 seconds

# Classic use: file handling, database connections, locks
\`\`\`

---

## Hash and Equality

Make objects hashable (for sets, dict keys).

\`\`\`python
class Person:
    def __init__(self, name, ssn):
        self.name = name
        self.ssn = ssn  # Immutable identifier
    
    def __eq__(self, other):
        """Objects are equal if SSNs match"""
        if not isinstance (other, Person):
            return False
        return self.ssn == other.ssn
    
    def __hash__(self):
        """Hash based on immutable attribute"""
        return hash (self.ssn)
    
    def __repr__(self):
        return f"Person('{self.name}', '{self.ssn}')"

p1 = Person("Alice", "123-45-6789")
p2 = Person("Alice", "123-45-6789")
p3 = Person("Bob", "987-65-4321")

print(p1 == p2)  # True - same SSN
print(p1 is p2)  # False - different objects

# Can use in sets/dicts
people = {p1, p2, p3}
print(len (people))  # 2 - p1 and p2 treated as same

lookup = {p1: "Manager", p3: "Engineer"}
print(lookup[p2])  # "Manager" - p2 treated as same key as p1
\`\`\`

**Important:** If \`__eq__\` is defined, \`__hash__\` must also be defined for hashable objects.

---

## Attribute Access

Control attribute access/assignment.

\`\`\`python
class DynamicAttributes:
    def __init__(self):
        self._data = {}
    
    def __getattr__(self, name):
        """Called when attribute not found"""
        print(f"Getting {name}")
        return self._data.get (name, f"No attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Called on all attribute assignment"""
        print(f"Setting {name} = {value}")
        if name == '_data':
            super().__setattr__(name, value)  # Avoid recursion
        else:
            self._data[name] = value
    
    def __delattr__(self, name):
        """Called when deleting attribute"""
        print(f"Deleting {name}")
        del self._data[name]

obj = DynamicAttributes()
obj.x = 10     # Setting x = 10
print(obj.x)   # Getting x \n 10
del obj.x      # Deleting x
\`\`\`

---

## Quick Reference Table

| Method | Operator/Function | Description |
|--------|-------------------|-------------|
| \`__init__\` | Constructor | Initialize object |
| \`__str__\` | \`str()\`, \`print()\` | User-friendly string |
| \`__repr__\` | \`repr()\` | Developer string |
| \`__eq__\` | \`==\` | Equality |
| \`__lt__\` | \`<\` | Less than |
| \`__le__\` | \`<=\` | Less or equal |
| \`__gt__\` | \`>\` | Greater than |
| \`__ge__\` | \`>=\` | Greater or equal |
| \`__add__\` | \`+\` | Addition |
| \`__sub__\` | \`-\` | Subtraction |
| \`__mul__\` | \`*\` | Multiplication |
| \`__truediv__\` | \`/\` | Division |
| \`__len__\` | \`len()\` | Length |
| \`__getitem__\` | \`obj[key]\` | Index access |
| \`__setitem__\` | \`obj[key]=val\` | Index assignment |
| \`__contains__\` | \`in\` | Membership test |
| \`__iter__\` | \`for x in obj\` | Iteration |
| \`__call__\` | \`obj()\` | Make callable |
| \`__enter__\` | \`with obj:\` | Enter context |
| \`__exit__\` | \`with obj:\` | Exit context |
| \`__hash__\` | \`hash()\` | Hash value |

---

## Best Practices

1. **Always implement \`__repr__\`** - critical for debugging
2. **\`__eq__\` + \`__hash__\`** - if you define one, define both (for hashable objects)
3. **Use \`@total_ordering\`** - reduces comparison method boilerplate
4. **Check types** - use \`isinstance()\` in comparison/arithmetic methods
5. **Return \`NotImplemented\`** - when operation not supported with given type
\`\`\`python
def __add__(self, other):
    if not isinstance (other, Vector):
        return NotImplemented  # Let Python try other.__radd__(self)
    return Vector (self.x + other.x, self.y + other.y)
\`\`\`
6. **Immutable hash** - \`__hash__\` must be based on immutable attributes

---

## Real-World Examples

**Custom Exception:**
\`\`\`python
class ValidationError(Exception):
    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")
\`\`\`

**Singleton Pattern:**
\`\`\`python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
\`\`\`

**Lazy Property:**
\`\`\`python
class Dataset:
    def __init__(self, path):
        self.path = path
        self._data = None
    
    def __len__(self):
        if self._data is None:
            self._data = load_data (self.path)  # Load on first access
        return len (self._data)
\`\`\`

---

## Interview Relevance

Magic methods appear in:
- **Custom data structures** - implement list/dict-like objects
- **Algorithm problems** - comparable objects for sorting/heaps
- **Design problems** - context managers, decorators, iterators
- **LeetCode** - \`__lt__\` for custom priority queues

**Example:** LeetCode's "Merge K Sorted Lists" is easier with:
\`\`\`python
class ListNode:
    def __lt__(self, other):
        return self.val < other.val

# Now can use with heapq!
import heapq
heap = [node1, node2, node3]
heapq.heapify (heap)
\`\`\``,
};

/**
 * Python OOP - New Problems Batch 1 (11-20)
 * 10 problems
 */

import { Problem } from '../types';

export const pythonOOPBatch1: Problem[] = [
  {
    id: 'oop-multiple-inheritance',
    title: 'Multiple Inheritance',
    difficulty: 'Medium',
    description: `Create a class that inherits from multiple parent classes.

**MRO (Method Resolution Order):**
- Python uses C3 linearization
- Check with Class.__mro__
- super() follows MRO

This tests:
- Multiple inheritance
- Method resolution
- Diamond problem awareness`,
    examples: [
      {
        input: 'class Child(Parent1, Parent2)',
        output: 'Inherits from both parents',
      },
    ],
    constraints: ['Inherit from multiple classes', 'Handle method conflicts'],
    hints: [
      'List parents in class definition',
      'Order matters for MRO',
      'Use super() carefully',
    ],
    starterCode: `class Flyer:
    """Can fly"""
    def move(self):
        return "flying"
    
    def fly(self):
        return "soaring through the air"


class Swimmer:
    """Can swim"""
    def move(self):
        return "swimming"
    
    def swim(self):
        return "diving in water"


class Duck(Flyer, Swimmer):
    """Duck can both fly and swim"""
    def __init__(self, name):
        self.name = name
    
    def move(self):
        # Calls Flyer's move (first in MRO)
        return super().move()


def test_multiple_inheritance():
    """Test multiple inheritance"""
    duck = Duck("Donald")
    
    # Can fly (from Flyer)
    fly_result = duck.fly()
    
    # Can swim (from Swimmer)
    swim_result = duck.swim()
    
    # move() uses Flyer's version (MRO)
    move_result = duck.move()
    
    return len(fly_result) + len(swim_result)
`,
    testCases: [
      {
        input: [],
        expected: 38,
        functionName: 'test_multiple_inheritance',
      },
    ],
    solution: `class Flyer:
    def move(self):
        return "flying"
    
    def fly(self):
        return "soaring through the air"


class Swimmer:
    def move(self):
        return "swimming"
    
    def swim(self):
        return "diving in water"


class Duck(Flyer, Swimmer):
    def __init__(self, name):
        self.name = name
    
    def move(self):
        return super().move()


def test_multiple_inheritance():
    duck = Duck("Donald")
    fly_result = duck.fly()
    swim_result = duck.swim()
    move_result = duck.move()
    
    return len(fly_result) + len(swim_result)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 11,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-operator-overloading',
    title: 'Operator Overloading',
    difficulty: 'Medium',
    description: `Overload operators like +, -, *, ==, etc.

**Magic methods:**
- __add__ for +
- __sub__ for -
- __mul__ for *
- __eq__ for ==
- __lt__ for <

This tests:
- Operator overloading
- Magic methods
- Custom behavior`,
    examples: [
      {
        input: 'Vector(1,2) + Vector(3,4)',
        output: 'Vector(4,6)',
      },
    ],
    constraints: ['Implement magic methods', 'Support operators'],
    hints: [
      '__add__ for addition',
      '__eq__ for equality',
      'Return new instance',
    ],
    starterCode: `class Vector:
    """2D Vector with operator overloading"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Overload + operator"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Overload - operator"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Overload * operator for scalar multiplication"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        """Overload == operator"""
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"


def test_operators():
    """Test operator overloading"""
    v1 = Vector(1, 2)
    v2 = Vector(3, 4)
    
    # Addition
    v3 = v1 + v2  # Vector(4, 6)
    
    # Multiplication
    v4 = v1 * 2  # Vector(2, 4)
    
    return v3.x + v4.y
`,
    testCases: [
      {
        input: [],
        expected: 8,
        functionName: 'test_operators',
      },
    ],
    solution: `class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"


def test_operators():
    v1 = Vector(1, 2)
    v2 = Vector(3, 4)
    v3 = v1 + v2
    v4 = v1 * 2
    return v3.x + v4.y`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 12,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-context-manager-class',
    title: 'Context Manager Class',
    difficulty: 'Medium',
    description: `Create a class that can be used with 'with' statement.

**Protocol:**
- __enter__ called when entering
- __exit__ called when exiting
- Handle exceptions in __exit__

This tests:
- Context manager protocol
- Resource management
- Exception handling`,
    examples: [
      {
        input: 'with Manager() as m:',
        output: 'Automatic setup and cleanup',
      },
    ],
    constraints: ['Implement __enter__ and __exit__', 'Handle cleanup'],
    hints: [
      '__enter__ returns self or resource',
      '__exit__ gets exception info',
      'Return True to suppress exception',
    ],
    starterCode: `class FileManager:
    """Context manager for file operations"""
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        """Enter context - open file"""
        from io import StringIO
        # For testing, use StringIO
        self.file = StringIO()
        self.file.write("test content")
        self.file.seek(0)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - close file"""
        if self.file:
            self.file.close()
        # Return False to propagate exceptions
        return False


def test_context_manager():
    """Test context manager"""
    with FileManager("test.txt", "r") as f:
        content = f.read()
    
    return len(content)
`,
    testCases: [
      {
        input: [],
        expected: 12,
        functionName: 'test_context_manager',
      },
    ],
    solution: `class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        from io import StringIO
        self.file = StringIO()
        self.file.write("test content")
        self.file.seek(0)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False


def test_context_manager():
    with FileManager("test.txt", "r") as f:
        content = f.read()
    
    return len(content)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 13,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-singleton-pattern',
    title: 'Singleton Pattern',
    difficulty: 'Medium',
    description: `Implement singleton pattern ensuring only one instance exists.

**Implementation:**
- Override __new__ method
- Store instance in class variable
- Return same instance always

This tests:
- Design patterns
- __new__ method
- Class-level state`,
    examples: [
      {
        input: 'Config() == Config()',
        output: 'Same instance',
      },
    ],
    constraints: ['Only one instance allowed', 'Use __new__'],
    hints: [
      'Override __new__',
      'Store instance as class variable',
      'Check if instance exists',
    ],
    starterCode: `class Singleton:
    """Singleton pattern implementation"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not hasattr(self, 'initialized'):
            self.value = 0
            self.initialized = True


def test_singleton():
    """Test singleton pattern"""
    # Create first instance
    s1 = Singleton()
    s1.value = 42
    
    # Create second instance (should be same)
    s2 = Singleton()
    
    # Both should reference same object
    if s1 is not s2:
        return "FAIL: Not same instance"
    
    # s2 should have s1's value
    return s2.value
`,
    testCases: [
      {
        input: [],
        expected: 42,
        functionName: 'test_singleton',
      },
    ],
    solution: `class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.value = 0
            self.initialized = True


def test_singleton():
    s1 = Singleton()
    s1.value = 42
    s2 = Singleton()
    
    if s1 is not s2:
        return "FAIL: Not same instance"
    
    return s2.value`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 14,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-factory-pattern',
    title: 'Factory Pattern',
    difficulty: 'Medium',
    description: `Implement factory pattern to create objects without specifying exact class.

**Pattern:**
- Factory method returns objects
- Decides which class to instantiate
- Encapsulates object creation

This tests:
- Factory pattern
- Class methods
- Polymorphism`,
    examples: [
      {
        input: 'ShapeFactory.create("circle")',
        output: 'Returns Circle instance',
      },
    ],
    constraints: ['Use factory method', 'Return appropriate class'],
    hints: [
      'Factory method chooses class',
      'Use @classmethod',
      'Return subclass instances',
    ],
    starterCode: `class Shape:
    """Base shape class"""
    def area(self):
        raise NotImplementedError


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height


class ShapeFactory:
    """Factory for creating shapes"""
    @staticmethod
    def create_shape(shape_type, *args):
        if shape_type == "circle":
            return Circle(*args)
        elif shape_type == "rectangle":
            return Rectangle(*args)
        else:
            raise ValueError(f"Unknown shape: {shape_type}")


def test_factory():
    """Test factory pattern"""
    # Create circle
    circle = ShapeFactory.create_shape("circle", 5)
    area1 = circle.area()
    
    # Create rectangle
    rect = ShapeFactory.create_shape("rectangle", 4, 5)
    area2 = rect.area()
    
    return int(area1 + area2)
`,
    testCases: [
      {
        input: [],
        expected: 98,
        functionName: 'test_factory',
      },
    ],
    solution: `class Shape:
    def area(self):
        raise NotImplementedError


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height


class ShapeFactory:
    @staticmethod
    def create_shape(shape_type, *args):
        if shape_type == "circle":
            return Circle(*args)
        elif shape_type == "rectangle":
            return Rectangle(*args)
        else:
            raise ValueError(f"Unknown shape: {shape_type}")


def test_factory():
    circle = ShapeFactory.create_shape("circle", 5)
    area1 = circle.area()
    rect = ShapeFactory.create_shape("rectangle", 4, 5)
    area2 = rect.area()
    return int(area1 + area2)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 15,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-observer-pattern-subject',
    title: 'Observer Pattern',
    difficulty: 'Hard',
    description: `Implement observer pattern for event notification.

**Pattern:**
- Subject maintains list of observers
- Notifies observers of changes
- Observers update themselves

**Use Case:** Event systems, MVC

This tests:
- Observer pattern
- Event notification
- Loose coupling`,
    examples: [
      {
        input: 'subject.attach(observer)',
        output: 'Observer gets notified of changes',
      },
    ],
    constraints: [
      'Subject notifies observers',
      'Observers register themselves',
    ],
    hints: [
      'Maintain observer list',
      'notify() calls update on each',
      'Observers implement update()',
    ],
    starterCode: `class Subject:
    """Subject being observed"""
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        """Add observer"""
        self._observers.append(observer)
    
    def detach(self, observer):
        """Remove observer"""
        self._observers.remove(observer)
    
    def notify(self):
        """Notify all observers"""
        for observer in self._observers:
            observer.update(self._state)
    
    def set_state(self, state):
        """Change state and notify"""
        self._state = state
        self.notify()


class Observer:
    """Observer base class"""
    def __init__(self, name):
        self.name = name
        self.state = None
    
    def update(self, state):
        """Receive update from subject"""
        self.state = state


def test_observer():
    """Test observer pattern"""
    subject = Subject()
    
    # Create observers
    obs1 = Observer("Observer1")
    obs2 = Observer("Observer2")
    
    # Attach observers
    subject.attach(obs1)
    subject.attach(obs2)
    
    # Change state
    subject.set_state(42)
    
    # Both observers should have new state
    return obs1.state + obs2.state
`,
    testCases: [
      {
        input: [],
        expected: 84,
        functionName: 'test_observer',
      },
    ],
    solution: `class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self._state)
    
    def set_state(self, state):
        self._state = state
        self.notify()


class Observer:
    def __init__(self, name):
        self.name = name
        self.state = None
    
    def update(self, state):
        self.state = state


def test_observer():
    subject = Subject()
    obs1 = Observer("Observer1")
    obs2 = Observer("Observer2")
    
    subject.attach(obs1)
    subject.attach(obs2)
    subject.set_state(42)
    
    return obs1.state + obs2.state`,
    timeComplexity: 'O(n) for notify',
    spaceComplexity: 'O(n) for observers',
    order: 16,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-descriptor-protocol',
    title: 'Descriptor Protocol',
    difficulty: 'Hard',
    description: `Implement descriptor protocol with __get__, __set__, __delete__.

**Protocol:**
- __get__(self, obj, type=None)
- __set__(self, obj, value)
- __delete__(self, obj)

**Use Case:** Properties, validators, lazy loading

This tests:
- Descriptor protocol
- Attribute access control
- Advanced OOP`,
    examples: [
      {
        input: 'obj.attr accesses descriptor',
        output: '__get__ is called',
      },
    ],
    constraints: ['Implement descriptor methods', 'Control attribute access'],
    hints: [
      '__get__ for reading',
      '__set__ for writing',
      'Store data elsewhere',
    ],
    starterCode: `class TypedProperty:
    """Descriptor with type checking"""
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.name} must be {self.expected_type}")
        obj.__dict__[self.name] = value
    
    def __delete__(self, obj):
        del obj.__dict__[self.name]


class Person:
    """Person with typed properties"""
    name = TypedProperty("name", str)
    age = TypedProperty("age", int)
    
    def __init__(self, name, age):
        self.name = name
        self.age = age


def test_descriptor():
    """Test descriptor protocol"""
    person = Person("Alice", 30)
    
    # Get values
    name_len = len(person.name)
    age_value = person.age
    
    # Try invalid type (should raise)
    try:
        person.age = "thirty"
        return "FAIL: Should raise TypeError"
    except TypeError:
        pass
    
    return name_len + age_value
`,
    testCases: [
      {
        input: [],
        expected: 35,
        functionName: 'test_descriptor',
      },
    ],
    solution: `class TypedProperty:
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.name} must be {self.expected_type}")
        obj.__dict__[self.name] = value
    
    def __delete__(self, obj):
        del obj.__dict__[self.name]


class Person:
    name = TypedProperty("name", str)
    age = TypedProperty("age", int)
    
    def __init__(self, name, age):
        self.name = name
        self.age = age


def test_descriptor():
    person = Person("Alice", 30)
    name_len = len(person.name)
    age_value = person.age
    
    try:
        person.age = "thirty"
        return "FAIL: Should raise TypeError"
    except TypeError:
        pass
    
    return name_len + age_value`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 17,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-callable-class',
    title: 'Callable Class (__call__)',
    difficulty: 'Medium',
    description: `Make a class instance callable like a function.

**__call__ method:**
- obj() calls obj.__call__()
- Makes instance behave like function
- Useful for stateful functions

This tests:
- __call__ method
- Callable protocol
- Stateful behavior`,
    examples: [
      {
        input: 'counter() increments and returns',
        output: 'Instance acts like function',
      },
    ],
    constraints: ['Implement __call__', 'Make instance callable'],
    hints: [
      'Define __call__ method',
      'Can maintain state',
      'Call with instance()',
    ],
    starterCode: `class Counter:
    """Callable counter class"""
    def __init__(self):
        self.count = 0
    
    def __call__(self):
        """Make instance callable"""
        self.count += 1
        return self.count


class Multiplier:
    """Callable multiplier"""
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        """Multiply x by factor"""
        return x * self.factor


def test_callable():
    """Test callable classes"""
    # Counter
    counter = Counter()
    result1 = counter()  # 1
    result2 = counter()  # 2
    result3 = counter()  # 3
    
    # Multiplier
    double = Multiplier(2)
    result4 = double(5)  # 10
    
    return result1 + result2 + result3 + result4
`,
    testCases: [
      {
        input: [],
        expected: 16,
        functionName: 'test_callable',
      },
    ],
    solution: `class Counter:
    def __init__(self):
        self.count = 0
    
    def __call__(self):
        self.count += 1
        return self.count


class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        return x * self.factor


def test_callable():
    counter = Counter()
    result1 = counter()
    result2 = counter()
    result3 = counter()
    
    double = Multiplier(2)
    result4 = double(5)
    
    return result1 + result2 + result3 + result4`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 18,
    topic: 'Python Object-Oriented Programming',
  },
  {
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
  },
  {
    id: 'oop-container-methods',
    title: 'Container Methods (__len__, __getitem__, etc.)',
    difficulty: 'Medium',
    description: `Implement container protocol to act like list/dict.

**Container protocol:**
- __len__ for len()
- __getitem__ for []
- __setitem__ for [] =
- __delitem__ for del []
- __contains__ for in
- __iter__ for iteration

This tests:
- Container protocol
- Custom collections
- Indexing/iteration`,
    examples: [
      {
        input: 'obj[0], len(obj), item in obj',
        output: 'Custom container behavior',
      },
    ],
    constraints: ['Implement container methods', 'Act like built-in container'],
    hints: [
      '__getitem__ for indexing',
      '__len__ for length',
      '__iter__ for for-loops',
    ],
    starterCode: `class CustomList:
    """Custom list-like container"""
    def __init__(self):
        self._items = []
    
    def __len__(self):
        """Return length"""
        return len(self._items)
    
    def __getitem__(self, index):
        """Get item by index"""
        return self._items[index]
    
    def __setitem__(self, index, value):
        """Set item by index"""
        self._items[index] = value
    
    def __contains__(self, item):
        """Check if item in container"""
        return item in self._items
    
    def __iter__(self):
        """Return iterator"""
        return iter(self._items)
    
    def append(self, item):
        """Add item"""
        self._items.append(item)


def test_container():
    """Test container methods"""
    container = CustomList()
    
    # Add items
    container.append(10)
    container.append(20)
    container.append(30)
    
    # Use len()
    length = len(container)
    
    # Use indexing
    first = container[0]
    
    # Use in
    has_20 = 20 in container
    
    # Use iteration
    total = sum(container)
    
    return total
`,
    testCases: [
      {
        input: [],
        expected: 60,
        functionName: 'test_container',
      },
    ],
    solution: `class CustomList:
    def __init__(self):
        self._items = []
    
    def __len__(self):
        return len(self._items)
    
    def __getitem__(self, index):
        return self._items[index]
    
    def __setitem__(self, index, value):
        self._items[index] = value
    
    def __contains__(self, item):
        return item in self._items
    
    def __iter__(self):
        return iter(self._items)
    
    def append(self, item):
        self._items.append(item)


def test_container():
    container = CustomList()
    container.append(10)
    container.append(20)
    container.append(30)
    
    length = len(container)
    first = container[0]
    has_20 = 20 in container
    total = sum(container)
    
    return total`,
    timeComplexity: 'O(1) for most operations',
    spaceComplexity: 'O(n)',
    order: 20,
    topic: 'Python Object-Oriented Programming',
  },
];

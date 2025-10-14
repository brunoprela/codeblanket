/**
 * Python OOP - New Problems Batch 2 (21-30)
 * 10 problems
 */

import { Problem } from '../types';

export const pythonOOPBatch2: Problem[] = [
  {
    id: 'oop-meta class',
    title: 'Metaclass Basics',
    difficulty: 'Hard',
    description: `Create a custom metaclass to control class creation.

**Metaclass:**
- Class of a class
- type is the default metaclass
- Controls class instantiation
- Can modify class attributes

This tests:
- Metaclass concept
- Class creation control
- Advanced OOP`,
    examples: [
      {
        input: 'class MyClass(metaclass=MyMeta)',
        output: 'MyMeta controls creation',
      },
    ],
    constraints: ['Create metaclass', 'Inherit from type'],
    hints: [
      'Inherit from type',
      '__new__ creates class',
      'Use metaclass= parameter',
    ],
    starterCode: `class UpperAttrMetaclass(type):
    """Metaclass that uppercases all attribute names"""
    def __new__(cls, name, bases, dct):
        # Convert all attribute names to uppercase
        uppercase_attr = {}
        for attr_name, attr_value in dct.items():
            if not attr_name.startswith('__'):
                uppercase_attr[attr_name.upper()] = attr_value
            else:
                uppercase_attr[attr_name] = attr_value
        
        return super().__new__(cls, name, bases, uppercase_attr)


class MyClass(metaclass=UpperAttrMetaclass):
    """Class with uppercase attributes"""
    x = 10
    y = 20


def test_metaclass():
    """Test metaclass"""
    obj = MyClass()
    
    # Attributes are uppercase
    return obj.X + obj.Y
`,
    testCases: [
      {
        input: [],
        expected: 30,
        functionName: 'test_metaclass',
      },
    ],
    solution: `class UpperAttrMetaclass(type):
    def __new__(cls, name, bases, dct):
        uppercase_attr = {}
        for attr_name, attr_value in dct.items():
            if not attr_name.startswith('__'):
                uppercase_attr[attr_name.upper()] = attr_value
            else:
                uppercase_attr[attr_name] = attr_value
        
        return super().__new__(cls, name, bases, uppercase_attr)


class MyClass(metaclass=UpperAttrMetaclass):
    x = 10
    y = 20


def test_metaclass():
    obj = MyClass()
    return obj.X + obj.Y`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 21,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-method-chaining',
    title: 'Method Chaining (Fluent Interface)',
    difficulty: 'Easy',
    description: `Implement method chaining by returning self.

**Pattern:**
\`\`\`python
obj.method1().method2().method3()
\`\`\`

Each method returns self to enable chaining.

This tests:
- Fluent interface
- Method design
- Return self pattern`,
    examples: [
      {
        input: 'builder.set_x(1).set_y(2).build()',
        output: 'Chained method calls',
      },
    ],
    constraints: ['Return self from methods', 'Enable chaining'],
    hints: [
      'Return self from each method',
      'Allows chaining',
      'Common in builders',
    ],
    starterCode: `class QueryBuilder:
    """SQL query builder with chaining"""
    def __init__(self):
        self._select = []
        self._where = []
        self._limit = None
    
    def select(self, *fields):
        """Add fields to SELECT"""
        self._select.extend(fields)
        return self
    
    def where(self, condition):
        """Add WHERE condition"""
        self._where.append(condition)
        return self
    
    def limit(self, n):
        """Set LIMIT"""
        self._limit = n
        return self
    
    def build(self):
        """Build query string"""
        query = f"SELECT {', '.join(self._select)}"
        if self._where:
            query += f" WHERE {' AND '.join(self._where)}"
        if self._limit:
            query += f" LIMIT {self._limit}"
        return query


def test_chaining():
    """Test method chaining"""
    query = (QueryBuilder()
             .select('name', 'age')
             .where('age > 18')
             .where('city = "NYC"')
             .limit(10)
             .build())
    
    return len(query)
`,
    testCases: [
      {
        input: [],
        expected: 67,
        functionName: 'test_chaining',
      },
    ],
    solution: `class QueryBuilder:
    def __init__(self):
        self._select = []
        self._where = []
        self._limit = None
    
    def select(self, *fields):
        self._select.extend(fields)
        return self
    
    def where(self, condition):
        self._where.append(condition)
        return self
    
    def limit(self, n):
        self._limit = n
        return self
    
    def build(self):
        query = f"SELECT {', '.join(self._select)}"
        if self._where:
            query += f" WHERE {' AND '.join(self._where)}"
        if self._limit:
            query += f" LIMIT {self._limit}"
        return query


def test_chaining():
    query = (QueryBuilder()
             .select('name', 'age')
             .where('age > 18')
             .where('city = "NYC"')
             .limit(10)
             .build())
    
    return len(query)`,
    timeComplexity: 'O(1) per method',
    spaceComplexity: 'O(n) for query parts',
    order: 22,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-composition-over-inheritance',
    title: 'Composition Over Inheritance',
    difficulty: 'Medium',
    description: `Use composition instead of inheritance for flexibility.

**Composition:**
- Has-a relationship
- More flexible than is-a
- Easier to modify
- Avoids deep hierarchies

This tests:
- Composition pattern
- Delegation
- Design principles`,
    examples: [
      {
        input: 'Car has Engine (composition)',
        output: 'vs Car is Vehicle (inheritance)',
      },
    ],
    constraints: ['Use composition', 'Delegate to components'],
    hints: [
      'Store component as attribute',
      'Delegate method calls',
      'More flexible',
    ],
    starterCode: `class Engine:
    """Engine component"""
    def __init__(self, horsepower):
        self.horsepower = horsepower
        self.running = False
    
    def start(self):
        self.running = True
        return "Engine started"
    
    def stop(self):
        self.running = False
        return "Engine stopped"


class Wheels:
    """Wheels component"""
    def __init__(self, count):
        self.count = count
    
    def rotate(self):
        return f"{self.count} wheels rotating"


class Car:
    """Car using composition"""
    def __init__(self, horsepower, wheel_count):
        self.engine = Engine(horsepower)
        self.wheels = Wheels(wheel_count)
    
    def start(self):
        """Delegate to engine"""
        return self.engine.start()
    
    def drive(self):
        """Use multiple components"""
        if self.engine.running:
            return self.wheels.rotate()
        return "Engine not running"


def test_composition():
    """Test composition pattern"""
    car = Car(200, 4)
    
    # Start engine
    car.start()
    
    # Drive
    result = car.drive()
    
    return len(result)
`,
    testCases: [
      {
        input: [],
        expected: 18,
        functionName: 'test_composition',
      },
    ],
    solution: `class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower
        self.running = False
    
    def start(self):
        self.running = True
        return "Engine started"
    
    def stop(self):
        self.running = False
        return "Engine stopped"


class Wheels:
    def __init__(self, count):
        self.count = count
    
    def rotate(self):
        return f"{self.count} wheels rotating"


class Car:
    def __init__(self, horsepower, wheel_count):
        self.engine = Engine(horsepower)
        self.wheels = Wheels(wheel_count)
    
    def start(self):
        return self.engine.start()
    
    def drive(self):
        if self.engine.running:
            return self.wheels.rotate()
        return "Engine not running"


def test_composition():
    car = Car(200, 4)
    car.start()
    result = car.drive()
    return len(result)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 23,
    topic: 'Python Object-Oriented Programming',
  },
  {
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
  },
  {
    id: 'oop-private-attributes',
    title: 'Private Attributes (Name Mangling)',
    difficulty: 'Easy',
    description: `Use name mangling for pseudo-private attributes.

**Name mangling:**
- __attribute becomes _ClassName__attribute
- Prevents accidental access
- Not truly private
- Convention: _ prefix for internal

This tests:
- Name mangling
- Encapsulation
- Private attributes`,
    examples: [
      {
        input: '__private_var',
        output: 'Name mangled to _Class__private_var',
      },
    ],
    constraints: ['Use double underscore prefix', 'Understand mangling'],
    hints: [
      '__var for name mangling',
      '_var for internal use',
      'Not truly private',
    ],
    starterCode: `class BankAccount:
    """Bank account with private balance"""
    def __init__(self, initial_balance):
        self.__balance = initial_balance  # Name mangled
    
    def deposit(self, amount):
        """Public method to deposit"""
        if amount > 0:
            self.__balance += amount
    
    def withdraw(self, amount):
        """Public method to withdraw"""
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return True
        return False
    
    def get_balance(self):
        """Public method to get balance"""
        return self.__balance


def test_private():
    """Test private attributes"""
    account = BankAccount(100)
    
    # Use public methods
    account.deposit(50)
    account.withdraw(30)
    
    # Get balance
    balance = account.get_balance()
    
    # Try direct access (would fail in real code)
    # account.__balance  # AttributeError
    
    return balance
`,
    testCases: [
      {
        input: [],
        expected: 120,
        functionName: 'test_private',
      },
    ],
    solution: `class BankAccount:
    def __init__(self, initial_balance):
        self.__balance = initial_balance
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return True
        return False
    
    def get_balance(self):
        return self.__balance


def test_private():
    account = BankAccount(100)
    account.deposit(50)
    account.withdraw(30)
    balance = account.get_balance()
    return balance`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 25,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-class-vs-static-methods',
    title: 'Class Method vs Static Method',
    difficulty: 'Easy',
    description: `Understand difference between @classmethod and @staticmethod.

**@classmethod:**
- First arg is cls (class)
- Can access/modify class state
- Used for factory methods

**@staticmethod:**
- No special first arg
- Cannot access class/instance
- Utility functions

This tests:
- Method types
- Use cases
- Decorators`,
    examples: [
      {
        input: 'Factory method vs utility',
        output: '@classmethod vs @staticmethod',
      },
    ],
    constraints: ['Use both decorators', 'Show difference'],
    hints: [
      '@classmethod gets cls',
      '@staticmethod gets nothing special',
      'Different use cases',
    ],
    starterCode: `class Date:
    """Date class with different method types"""
    def __init__(self, day, month, year):
        self.day = day
        self.month = month
        self.year = year
    
    @classmethod
    def from_string(cls, date_string):
        """Factory method - creates Date from string"""
        day, month, year = map(int, date_string.split('-'))
        return cls(day, month, year)
    
    @staticmethod
    def is_valid_date(day, month, year):
        """Utility method - validates date"""
        return 1 <= day <= 31 and 1 <= month <= 12 and year > 0
    
    def __repr__(self):
        return f"Date({self.day}, {self.month}, {self.year})"


def test_methods():
    """Test class and static methods"""
    # Use static method
    valid = Date.is_valid_date(15, 6, 2024)
    
    # Use class method
    date = Date.from_string("15-6-2024")
    
    return date.day + date.month
`,
    testCases: [
      {
        input: [],
        expected: 21,
        functionName: 'test_methods',
      },
    ],
    solution: `class Date:
    def __init__(self, day, month, year):
        self.day = day
        self.month = month
        self.year = year
    
    @classmethod
    def from_string(cls, date_string):
        day, month, year = map(int, date_string.split('-'))
        return cls(day, month, year)
    
    @staticmethod
    def is_valid_date(day, month, year):
        return 1 <= day <= 31 and 1 <= month <= 12 and year > 0
    
    def __repr__(self):
        return f"Date({self.day}, {self.month}, {self.year})"


def test_methods():
    valid = Date.is_valid_date(15, 6, 2024)
    date = Date.from_string("15-6-2024")
    return date.day + date.month`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 26,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-abstract-properties',
    title: 'Abstract Properties',
    difficulty: 'Medium',
    description: `Combine @property with @abstractmethod.

**Abstract properties:**
- Force subclasses to implement
- Define interface for properties
- Use @property and @abstractmethod together

This tests:
- Abstract properties
- ABC with properties
- Interface design`,
    examples: [
      {
        input: 'Abstract property area',
        output: 'Subclass must implement',
      },
    ],
    constraints: [
      'Use @abstractmethod with @property',
      'Subclass must implement',
    ],
    hints: [
      'Stack decorators',
      '@property @abstractmethod',
      'Subclass implements property',
    ],
    starterCode: `from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract shape with abstract property"""
    @property
    @abstractmethod
    def area(self):
        """Area must be implemented by subclass"""
        pass
    
    @property
    @abstractmethod
    def perimeter(self):
        """Perimeter must be implemented by subclass"""
        pass


class Square(Shape):
    """Concrete shape"""
    def __init__(self, side):
        self._side = side
    
    @property
    def area(self):
        return self._side ** 2
    
    @property
    def perimeter(self):
        return 4 * self._side


def test_abstract_properties():
    """Test abstract properties"""
    square = Square(5)
    
    # Access properties
    area = square.area
    perimeter = square.perimeter
    
    return area + perimeter
`,
    testCases: [
      {
        input: [],
        expected: 45,
        functionName: 'test_abstract_properties',
      },
    ],
    solution: `from abc import ABC, abstractmethod

class Shape(ABC):
    @property
    @abstractmethod
    def area(self):
        pass
    
    @property
    @abstractmethod
    def perimeter(self):
        pass


class Square(Shape):
    def __init__(self, side):
        self._side = side
    
    @property
    def area(self):
        return self._side ** 2
    
    @property
    def perimeter(self):
        return 4 * self._side


def test_abstract_properties():
    square = Square(5)
    area = square.area
    perimeter = square.perimeter
    return area + perimeter`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 27,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-immutable-class',
    title: 'Immutable Class',
    difficulty: 'Medium',
    description: `Create an immutable class where attributes cannot be changed after creation.

**Techniques:**
- Use __slots__
- Override __setattr__
- Raise exception on modification

This tests:
- Immutability
- __setattr__ override
- Data protection`,
    examples: [
      {
        input: 'Cannot modify after init',
        output: 'Raises error on attempt',
      },
    ],
    constraints: ['Prevent attribute modification', 'Raise exception'],
    hints: [
      'Override __setattr__',
      'Set attributes differently in __init__',
      'Raise AttributeError',
    ],
    starterCode: `class ImmutablePoint:
    """Immutable 2D point"""
    def __init__(self, x, y):
        # Use object.__setattr__ to bypass our override
        object.__setattr__(self, 'x', x)
        object.__setattr__(self, 'y', y)
        object.__setattr__(self, '_initialized', True)
    
    def __setattr__(self, name, value):
        """Prevent attribute modification after init"""
        if hasattr(self, '_initialized'):
            raise AttributeError("Cannot modify immutable object")
        object.__setattr__(self, name, value)
    
    def __repr__(self):
        return f"ImmutablePoint({self.x}, {self.y})"


def test_immutable():
    """Test immutable class"""
    point = ImmutablePoint(10, 20)
    
    # Can access
    x = point.x
    y = point.y
    
    # Try to modify (should fail)
    try:
        point.x = 30
        return "FAIL: Should not allow modification"
    except AttributeError:
        pass
    
    return x + y
`,
    testCases: [
      {
        input: [],
        expected: 30,
        functionName: 'test_immutable',
      },
    ],
    solution: `class ImmutablePoint:
    def __init__(self, x, y):
        object.__setattr__(self, 'x', x)
        object.__setattr__(self, 'y', y)
        object.__setattr__(self, '_initialized', True)
    
    def __setattr__(self, name, value):
        if hasattr(self, '_initialized'):
            raise AttributeError("Cannot modify immutable object")
        object.__setattr__(self, name, value)
    
    def __repr__(self):
        return f"ImmutablePoint({self.x}, {self.y})"


def test_immutable():
    point = ImmutablePoint(10, 20)
    x = point.x
    y = point.y
    
    try:
        point.x = 30
        return "FAIL: Should not allow modification"
    except AttributeError:
        pass
    
    return x + y`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 28,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-lazy-property',
    title: 'Lazy Property Evaluation',
    difficulty: 'Medium',
    description: `Create a property that computes value only once (lazy evaluation).

**Pattern:**
- Compute value on first access
- Cache result
- Don't recompute

This tests:
- Lazy evaluation
- Property caching
- Performance optimization`,
    examples: [
      {
        input: 'Expensive computation cached',
        output: 'Only computed once',
      },
    ],
    constraints: ['Compute on first access', 'Cache result'],
    hints: [
      'Use property decorator',
      'Store in _cached attribute',
      'Check if already computed',
    ],
    starterCode: `class LazyProperty:
    """Descriptor for lazy property"""
    def __init__(self, function):
        self.function = function
        self.name = function.__name__
    
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        
        # Check if value already cached
        attr_name = f'_lazy_{self.name}'
        if not hasattr(obj, attr_name):
            # Compute and cache
            setattr(obj, attr_name, self.function(obj))
        
        return getattr(obj, attr_name)


class DataAnalyzer:
    """Analyze data with lazy properties"""
    def __init__(self, data):
        self.data = data
    
    @LazyProperty
    def average(self):
        """Expensive computation"""
        print("Computing average...")
        return sum(self.data) / len(self.data)
    
    @LazyProperty
    def total(self):
        """Another expensive computation"""
        print("Computing total...")
        return sum(self.data)


def test_lazy():
    """Test lazy property"""
    analyzer = DataAnalyzer([1, 2, 3, 4, 5])
    
    # First access computes
    avg1 = analyzer.average
    
    # Second access uses cache
    avg2 = analyzer.average
    
    return int(avg1 + avg2)
`,
    testCases: [
      {
        input: [],
        expected: 6,
        functionName: 'test_lazy',
      },
    ],
    solution: `class LazyProperty:
    def __init__(self, function):
        self.function = function
        self.name = function.__name__
    
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        
        attr_name = f'_lazy_{self.name}'
        if not hasattr(obj, attr_name):
            setattr(obj, attr_name, self.function(obj))
        
        return getattr(obj, attr_name)


class DataAnalyzer:
    def __init__(self, data):
        self.data = data
    
    @LazyProperty
    def average(self):
        return sum(self.data) / len(self.data)
    
    @LazyProperty
    def total(self):
        return sum(self.data)


def test_lazy():
    analyzer = DataAnalyzer([1, 2, 3, 4, 5])
    avg1 = analyzer.average
    avg2 = analyzer.average
    return int(avg1 + avg2)`,
    timeComplexity: 'O(1) after first access',
    spaceComplexity: 'O(1)',
    order: 29,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-copy-deepcopy',
    title: 'Copy vs Deepcopy',
    difficulty: 'Medium',
    description: `Understand difference between shallow copy and deep copy.

**copy.copy():**
- Shallow copy
- Copies object but not nested objects
- Nested objects are references

**copy.deepcopy():**
- Deep copy
- Recursively copies all objects
- Completely independent

This tests:
- Copy module
- Reference vs value
- Nested objects`,
    examples: [
      {
        input: 'Shallow copy shares nested objects',
        output: 'Deep copy duplicates everything',
      },
    ],
    constraints: ['Use copy module', 'Show difference'],
    hints: [
      'import copy',
      'copy.copy() for shallow',
      'copy.deepcopy() for deep',
    ],
    starterCode: `import copy

class Person:
    """Person with nested address"""
    def __init__(self, name, address):
        self.name = name
        self.address = address  # Nested object


class Address:
    """Address class"""
    def __init__(self, city):
        self.city = city


def test_copy():
    """Test copy vs deepcopy"""
    # Original
    addr = Address("NYC")
    person1 = Person("Alice", addr)
    
    # Shallow copy
    person2 = copy.copy(person1)
    
    # Deep copy
    person3 = copy.deepcopy(person1)
    
    # Modify original address
    addr.city = "LA"
    
    # person2 shares address (shallow)
    # person3 has independent address (deep)
    
    return len(person2.address.city) + len(person3.address.city)
`,
    testCases: [
      {
        input: [],
        expected: 5,
        functionName: 'test_copy',
      },
    ],
    solution: `import copy

class Person:
    def __init__(self, name, address):
        self.name = name
        self.address = address


class Address:
    def __init__(self, city):
        self.city = city


def test_copy():
    addr = Address("NYC")
    person1 = Person("Alice", addr)
    
    person2 = copy.copy(person1)
    person3 = copy.deepcopy(person1)
    
    addr.city = "LA"
    
    return len(person2.address.city) + len(person3.address.city)`,
    timeComplexity: 'O(n) for deepcopy',
    spaceComplexity: 'O(n) for deepcopy',
    order: 30,
    topic: 'Python Object-Oriented Programming',
  },
];

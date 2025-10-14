/**
 * Python Advanced - New Problems (40-50)
 * 11 new problems to reach 50 total
 */

import { Problem } from '../types';

export const pythonAdvancedNew: Problem[] = [
  {
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
  },
  {
    id: 'advanced-chained-decorators',
    title: 'Chaining Multiple Decorators',
    difficulty: 'Medium',
    description: `Create a function with multiple stacked decorators.

Understand execution order:
- Decorators apply from bottom to top
- Inner decorator wraps function first
- Outer decorator wraps the result

**Example:** @auth @log @cache def func()
Order: cache(log(auth(func)))

This tests:
- Decorator composition
- Execution order
- Wrapper functions`,
    examples: [
      {
        input: 'Multiple decorators on one function',
        output: 'Decorators execute in correct order',
      },
    ],
    constraints: ['Decorators must preserve metadata', 'Order matters'],
    hints: [
      'Bottom decorator wraps function first',
      'Each decorator wraps previous result',
      'Use @wraps to preserve metadata',
    ],
    starterCode: `from functools import wraps

def uppercase(func):
    """Decorator that uppercases string result"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper() if isinstance(result, str) else result
    return wrapper

def exclaim(func):
    """Decorator that adds exclamation"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result + '!' if isinstance(result, str) else result
    return wrapper

@uppercase
@exclaim
def greet(name):
    return f"hello {name}"


def test_chained():
    """Test chained decorators"""
    result = greet("world")
    # exclaim runs first: "hello world!"
    # uppercase runs second: "HELLO WORLD!"
    return result
`,
    testCases: [
      {
        input: [],
        expected: 'HELLO WORLD!',
        functionName: 'test_chained',
      },
    ],
    solution: `from functools import wraps

def uppercase(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper() if isinstance(result, str) else result
    return wrapper

def exclaim(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result + '!' if isinstance(result, str) else result
    return wrapper

@uppercase
@exclaim
def greet(name):
    return f"hello {name}"


def test_chained():
    result = greet("world")
    return result`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 41,
    topic: 'Python Advanced',
  },
  {
    id: 'advanced-async-generator',
    title: 'Async Generator',
    difficulty: 'Hard',
    description: `Create an async generator that yields values asynchronously.

Async generators combine:
- Generator protocol (yield)
- Async protocol (await)
- Used with async for

**Example:** 
\`\`\`python
async for item in async_range(5):
    print(item)
\`\`\`

This tests:
- Async/await syntax
- Generator protocol
- Async iteration`,
    examples: [
      {
        input: 'Async range generator',
        output: 'Yields 0,1,2,3,4 asynchronously',
      },
    ],
    constraints: ['Must be async generator', 'Use yield not return'],
    hints: ['Use async def', 'Use yield for values', 'Can use await inside'],
    starterCode: `import asyncio

async def async_range(n):
    """
    Async generator that yields numbers.
    
    Args:
        n: Upper limit
        
    Examples:
        >>> async for i in async_range(3):
        ...     print(i)
        0
        1
        2
    """
    # Your code here
    pass


async def test_async_gen():
    """Test async generator"""
    results = []
    async for i in async_range(5):
        results.append(i)
    return results[2]  # Return middle value


# For testing, we need to run async function
def test_runner():
    """Synchronous test runner"""
    try:
        return asyncio.run(test_async_gen())
    except:
        return None
`,
    testCases: [
      {
        input: [],
        expected: 2,
        functionName: 'test_runner',
      },
    ],
    solution: `import asyncio

async def async_range(n):
    for i in range(n):
        await asyncio.sleep(0)  # Yield control
        yield i


async def test_async_gen():
    results = []
    async for i in async_range(5):
        results.append(i)
    return results[2]


def test_runner():
    try:
        return asyncio.run(test_async_gen())
    except:
        return None`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 42,
    topic: 'Python Advanced',
  },
  {
    id: 'advanced-slots-point',
    title: '__slots__ for Memory Optimization',
    difficulty: 'Medium',
    description: `Use __slots__ to reduce memory usage of class instances.

__slots__ prevents __dict__ creation:
- Faster attribute access
- Reduced memory per instance
- No dynamic attribute creation

**Trade-off:** Can't add attributes dynamically

This tests:
- Memory optimization
- Class design
- Understanding __dict__`,
    examples: [
      {
        input: 'Class with __slots__',
        output: 'Reduced memory usage',
      },
    ],
    constraints: ['Must define __slots__', 'Cannot add dynamic attributes'],
    hints: [
      'Define __slots__ as tuple/list',
      'Include all attributes',
      'No __dict__ created',
    ],
    starterCode: `class Point:
    """
    2D point with __slots__.
    
    Attributes:
        x: X coordinate
        y: Y coordinate
    """
    __slots__ = ('x', 'y')
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance_from_origin(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


def test_slots():
    """Test __slots__ class"""
    p = Point(3, 4)
    result = p.distance_from_origin()
    
    # Try to add dynamic attribute (should fail)
    try:
        p.z = 5
        return "FAIL: Should not allow dynamic attributes"
    except AttributeError:
        return result
`,
    testCases: [
      {
        input: [],
        expected: 5.0,
        functionName: 'test_slots',
      },
    ],
    solution: `class Point:
    __slots__ = ('x', 'y')
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance_from_origin(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


def test_slots():
    p = Point(3, 4)
    result = p.distance_from_origin()
    
    try:
        p.z = 5
        return "FAIL: Should not allow dynamic attributes"
    except AttributeError:
        return result`,
    timeComplexity: 'O(1) attribute access',
    spaceComplexity: 'O(1) per instance',
    order: 43,
    topic: 'Python Advanced',
  },
  {
    id: 'advanced-weak-references',
    title: 'WeakRef for Cache',
    difficulty: 'Hard',
    description: `Use weak references to create a cache that doesn't prevent garbage collection.

WeakRef allows objects to be collected:
- Normal references keep objects alive
- Weak references don't prevent collection
- Useful for caches, callbacks

**Use Case:** Large object caching without memory leaks

This tests:
- Memory management
- Garbage collection
- WeakValueDictionary`,
    examples: [
      {
        input: 'Weak reference cache',
        output: 'Objects can be garbage collected',
      },
    ],
    constraints: ['Use weakref module', 'Objects can be collected'],
    hints: [
      'Use WeakValueDictionary',
      'Values can disappear',
      'Check if key exists before access',
    ],
    starterCode: `import weakref

class WeakCache:
    """
    Cache using weak references.
    """
    def __init__(self):
        self.cache = weakref.WeakValueDictionary()
    
    def set(self, key, value):
        """Store value with weak reference"""
        self.cache[key] = value
    
    def get(self, key):
        """Get value if still alive"""
        return self.cache.get(key)
    
    def contains(self, key):
        """Check if key exists"""
        return key in self.cache


def test_weak_cache():
    """Test weak reference cache"""
    cache = WeakCache()
    
    # Store object
    obj = [1, 2, 3]
    cache.set('data', obj)
    
    # Object is alive, should retrieve it
    result = cache.get('data')
    
    if result != [1, 2, 3]:
        return "FAIL: Should retrieve object"
    
    return result[0]
`,
    testCases: [
      {
        input: [],
        expected: 1,
        functionName: 'test_weak_cache',
      },
    ],
    solution: `import weakref

class WeakCache:
    def __init__(self):
        self.cache = weakref.WeakValueDictionary()
    
    def set(self, key, value):
        self.cache[key] = value
    
    def get(self, key):
        return self.cache.get(key)
    
    def contains(self, key):
        return key in self.cache


def test_weak_cache():
    cache = WeakCache()
    
    obj = [1, 2, 3]
    cache.set('data', obj)
    
    result = cache.get('data')
    
    if result != [1, 2, 3]:
        return "FAIL: Should retrieve object"
    
    return result[0]`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(n)',
    order: 44,
    topic: 'Python Advanced',
  },
  {
    id: 'advanced-abstract-base-class',
    title: 'Abstract Base Class',
    difficulty: 'Medium',
    description: `Create an abstract base class using ABC module.

ABC (Abstract Base Class) features:
- Define interface contracts
- Force subclasses to implement methods
- Cannot instantiate abstract class
- Use @abstractmethod decorator

**Use Case:** Plugin systems, frameworks, interface design

This tests:
- ABC module
- Abstract methods
- Inheritance contracts`,
    examples: [
      {
        input: 'Abstract Shape class',
        output: 'Subclasses must implement area()',
      },
    ],
    constraints: ['Use ABC module', 'Mark methods as abstract'],
    hints: [
      'Inherit from ABC',
      'Use @abstractmethod',
      'Subclasses must implement',
    ],
    starterCode: `from abc import ABC, abstractmethod

class Shape(ABC):
    """
    Abstract base class for shapes.
    """
    @abstractmethod
    def area(self):
        """Calculate area - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate perimeter - must be implemented by subclasses"""
        pass


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius


def test_abc():
    """Test abstract base class"""
    # Try to instantiate abstract class (should fail)
    try:
        s = Shape()
        return "FAIL: Should not instantiate abstract class"
    except TypeError:
        pass
    
    # Concrete class should work
    c = Circle(5)
    area = c.area()
    
    return int(area)
`,
    testCases: [
      {
        input: [],
        expected: 78,
        functionName: 'test_abc',
      },
    ],
    solution: `from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius


def test_abc():
    try:
        s = Shape()
        return "FAIL: Should not instantiate abstract class"
    except TypeError:
        pass
    
    c = Circle(5)
    area = c.area()
    
    return int(area)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 45,
    topic: 'Python Advanced',
  },
  {
    id: 'advanced-protocol-duck-typing',
    title: 'Protocol and Duck Typing',
    difficulty: 'Medium',
    description: `Use Protocol for structural subtyping (duck typing with type hints).

Protocol features (Python 3.8+):
- Define interface without inheritance
- Structural typing (duck typing)
- Type checker friendly
- No runtime enforcement

**Example:** Anything with .read() is file-like

This tests:
- Protocol definition
- Structural typing
- Type annotations`,
    examples: [
      {
        input: 'File-like protocol',
        output: 'Any object with read() method',
      },
    ],
    constraints: ['Use typing.Protocol', 'Define method signatures'],
    hints: [
      'Inherit from Protocol',
      'Define method signatures',
      'No implementation needed',
    ],
    starterCode: `from typing import Protocol

class Readable(Protocol):
    """Protocol for readable objects"""
    def read(self) -> str:
        ...


class FileWrapper:
    """File-like object"""
    def __init__(self, content):
        self.content = content
    
    def read(self) -> str:
        return self.content


class StringWrapper:
    """Another file-like object"""
    def __init__(self, text):
        self.text = text
    
    def read(self) -> str:
        return self.text


def process_readable(obj: Readable) -> int:
    """Process any readable object"""
    content = obj.read()
    return len(content)


def test_protocol():
    """Test protocol"""
    f = FileWrapper("hello world")
    s = StringWrapper("test")
    
    result1 = process_readable(f)
    result2 = process_readable(s)
    
    return result1 + result2
`,
    testCases: [
      {
        input: [],
        expected: 15,
        functionName: 'test_protocol',
      },
    ],
    solution: `from typing import Protocol

class Readable(Protocol):
    def read(self) -> str:
        ...


class FileWrapper:
    def __init__(self, content):
        self.content = content
    
    def read(self) -> str:
        return self.content


class StringWrapper:
    def __init__(self, text):
        self.text = text
    
    def read(self) -> str:
        return self.text


def process_readable(obj: Readable) -> int:
    content = obj.read()
    return len(content)


def test_protocol():
    f = FileWrapper("hello world")
    s = StringWrapper("test")
    
    result1 = process_readable(f)
    result2 = process_readable(s)
    
    return result1 + result2`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 46,
    topic: 'Python Advanced',
  },
  {
    id: 'advanced-enum-auto',
    title: 'Enum with Auto Values',
    difficulty: 'Easy',
    description: `Create an Enum with automatically assigned values.

Enum features:
- Named constants
- Auto-generated values
- Iteration support
- Type safety

**Use Case:** Status codes, colors, states

This tests:
- Enum class
- auto() function
- Enum iteration`,
    examples: [
      {
        input: 'Color enum',
        output: 'RED, GREEN, BLUE with auto values',
      },
    ],
    constraints: ['Use Enum class', 'Use auto() for values'],
    hints: [
      'Import Enum and auto',
      'Values auto-increment',
      'Access by name or value',
    ],
    starterCode: `from enum import Enum, auto

class Status(Enum):
    """Status enumeration"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


def test_enum():
    """Test enum"""
    # Get value
    pending_value = Status.PENDING.value
    
    # Count members
    count = len(Status)
    
    # Get by value
    status = Status(2)
    
    return status.value
`,
    testCases: [
      {
        input: [],
        expected: 2,
        functionName: 'test_enum',
      },
    ],
    solution: `from enum import Enum, auto

class Status(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


def test_enum():
    pending_value = Status.PENDING.value
    count = len(Status)
    status = Status(2)
    
    return status.value`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 47,
    topic: 'Python Advanced',
  },
  {
    id: 'advanced-dataclass-frozen',
    title: 'Frozen Dataclass (Immutable)',
    difficulty: 'Easy',
    description: `Create an immutable dataclass using frozen=True.

Frozen dataclass features:
- Immutable after creation
- Hashable (can use as dict key)
- Thread-safe
- Cannot modify attributes

**Use Case:** Value objects, configuration, cache keys

This tests:
- Dataclass decorator
- Immutability
- Hashability`,
    examples: [
      {
        input: 'Frozen Point(1, 2)',
        output: 'Cannot modify x or y',
      },
    ],
    constraints: ['Use @dataclass(frozen=True)', 'Cannot modify after init'],
    hints: [
      'Set frozen=True',
      'Attributes cannot be changed',
      'Instance is hashable',
    ],
    starterCode: `from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    """Immutable point"""
    x: int
    y: int
    
    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5


def test_frozen():
    """Test frozen dataclass"""
    p = Point(3, 4)
    distance = p.distance_from_origin()
    
    # Try to modify (should fail)
    try:
        p.x = 10
        return "FAIL: Should not allow modification"
    except:
        pass
    
    # Should be hashable
    points = {p: "origin"}
    
    return int(distance)
`,
    testCases: [
      {
        input: [],
        expected: 5,
        functionName: 'test_frozen',
      },
    ],
    solution: `from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: int
    y: int
    
    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5


def test_frozen():
    p = Point(3, 4)
    distance = p.distance_from_origin()
    
    try:
        p.x = 10
        return "FAIL: Should not allow modification"
    except:
        pass
    
    points = {p: "origin"}
    
    return int(distance)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 48,
    topic: 'Python Advanced',
  },
  {
    id: 'advanced-contextvar',
    title: 'Context Variables',
    difficulty: 'Hard',
    description: `Use contextvars for context-local state (better than thread-local).

Context variables features:
- Async-safe (unlike threading.local)
- Context-specific values
- Inherited by child tasks
- Used in async frameworks

**Use Case:** Request IDs, user context, logging

This tests:
- contextvars module
- Context isolation
- Async compatibility`,
    examples: [
      {
        input: 'Request ID per context',
        output: 'Different IDs in different contexts',
      },
    ],
    constraints: ['Use contextvars.ContextVar', 'Values isolated per context'],
    hints: [
      'Create ContextVar instance',
      'Use .set() and .get()',
      'Each context has own value',
    ],
    starterCode: `from contextvars import ContextVar

# Create context variable
request_id: ContextVar[str] = ContextVar('request_id', default='none')


def process_request(req_id: str) -> str:
    """Process request with context-specific ID"""
    # Set context variable
    request_id.set(req_id)
    
    # Get context variable
    current_id = request_id.get()
    
    return current_id


def test_contextvar():
    """Test context variables"""
    result1 = process_request('req-123')
    result2 = process_request('req-456')
    
    # Each call has its own context
    return len(result1) + len(result2)
`,
    testCases: [
      {
        input: [],
        expected: 14,
        functionName: 'test_contextvar',
      },
    ],
    solution: `from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar('request_id', default='none')


def process_request(req_id: str) -> str:
    request_id.set(req_id)
    current_id = request_id.get()
    return current_id


def test_contextvar():
    result1 = process_request('req-123')
    result2 = process_request('req-456')
    
    return len(result1) + len(result2)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1) per context',
    order: 49,
    topic: 'Python Advanced',
  },
  {
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
  },
];

/**
 * Python OOP - New Problems Batch 4 (41-50)
 * 10 problems to reach 50 total
 */

import { Problem } from '../types';

export const pythonOOPBatch4: Problem[] = [
  {
    id: 'oop-chain-of-responsibility',
    title: 'Chain of Responsibility Pattern',
    difficulty: 'Hard',
    description: `Implement chain of responsibility pattern.

**Pattern:**
- Chain of handlers
- Request passes along chain
- Handler processes or passes to next
- Decouples sender from receiver

This tests:
- Chain of responsibility
- Request handling
- Handler chaining`,
    examples: [
      {
        input: 'Request → Handler1 → Handler2 → Handler3',
        output: 'First capable handler processes',
      },
    ],
    constraints: ['Chain handlers', 'Pass request along chain'],
    hints: [
      'Each handler has reference to next',
      'Process or delegate',
      'Set up chain in advance',
    ],
    starterCode: `class Handler:
    """Base handler"""
    def __init__(self):
        self.next_handler = None
    
    def set_next(self, handler):
        """Set next handler in chain"""
        self.next_handler = handler
        return handler
    
    def handle(self, request):
        """Handle or pass to next"""
        if self.next_handler:
            return self.next_handler.handle(request)
        return None


class LowPriorityHandler(Handler):
    """Handles low priority (< 10)"""
    def handle(self, request):
        if request < 10:
            return f"Low priority handler: {request}"
        return super().handle(request)


class MediumPriorityHandler(Handler):
    """Handles medium priority (10-50)"""
    def handle(self, request):
        if 10 <= request < 50:
            return f"Medium priority handler: {request}"
        return super().handle(request)


class HighPriorityHandler(Handler):
    """Handles high priority (>= 50)"""
    def handle(self, request):
        if request >= 50:
            return f"High priority handler: {request}"
        return super().handle(request)


def test_chain():
    """Test chain of responsibility"""
    # Build chain
    low = LowPriorityHandler()
    medium = MediumPriorityHandler()
    high = HighPriorityHandler()
    
    low.set_next(medium).set_next(high)
    
    # Send requests
    result1 = low.handle(5)   # Low
    result2 = low.handle(25)  # Medium
    result3 = low.handle(75)  # High
    
    return len(result1) + len(result2) + len(result3)
`,
    testCases: [
      {
        input: [],
        expected: 85,
        functionName: 'test_chain',
      },
    ],
    solution: `class Handler:
    def __init__(self):
        self.next_handler = None
    
    def set_next(self, handler):
        self.next_handler = handler
        return handler
    
    def handle(self, request):
        if self.next_handler:
            return self.next_handler.handle(request)
        return None


class LowPriorityHandler(Handler):
    def handle(self, request):
        if request < 10:
            return f"Low priority handler: {request}"
        return super().handle(request)


class MediumPriorityHandler(Handler):
    def handle(self, request):
        if 10 <= request < 50:
            return f"Medium priority handler: {request}"
        return super().handle(request)


class HighPriorityHandler(Handler):
    def handle(self, request):
        if request >= 50:
            return f"High priority handler: {request}"
        return super().handle(request)


def test_chain():
    low = LowPriorityHandler()
    medium = MediumPriorityHandler()
    high = HighPriorityHandler()
    
    low.set_next(medium).set_next(high)
    
    result1 = low.handle(5)
    result2 = low.handle(25)
    result3 = low.handle(75)
    
    return len(result1) + len(result2) + len(result3)`,
    timeComplexity: 'O(n) where n is chain length',
    spaceComplexity: 'O(1)',
    order: 41,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-iterator-pattern',
    title: 'Iterator Pattern',
    difficulty: 'Medium',
    description: `Implement iterator pattern for custom collection.

**Pattern:**
- Traverse collection without exposing structure
- __iter__ and __next__
- Raise StopIteration when done

This tests:
- Iterator protocol
- Custom iteration
- Collection traversal`,
    examples: [
      {
        input: 'for item in collection',
        output: 'Custom iteration logic',
      },
    ],
    constraints: ['Implement __iter__ and __next__', 'Hide internal structure'],
    hints: [
      '__iter__ returns iterator',
      '__next__ returns next item',
      'Raise StopIteration at end',
    ],
    starterCode: `class ReverseIterator:
    """Iterator that goes backward"""
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]


class ReverseCollection:
    """Collection with reverse iteration"""
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        """Return iterator"""
        return ReverseIterator(self.data)


def test_iterator():
    """Test iterator pattern"""
    collection = ReverseCollection([1, 2, 3, 4, 5])
    
    # Iterate in reverse
    result = list(collection)
    
    # Should be [5, 4, 3, 2, 1]
    return sum(result)
`,
    testCases: [
      {
        input: [],
        expected: 15,
        functionName: 'test_iterator',
      },
    ],
    solution: `class ReverseIterator:
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]


class ReverseCollection:
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        return ReverseIterator(self.data)


def test_iterator():
    collection = ReverseCollection([1, 2, 3, 4, 5])
    result = list(collection)
    return sum(result)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) for iterator',
    order: 42,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-memento-pattern',
    title: 'Memento Pattern',
    difficulty: 'Hard',
    description: `Implement memento pattern for undo functionality.

**Pattern:**
- Save and restore object state
- Memento stores state
- Originator creates/restores from memento
- Caretaker manages mementos

This tests:
- Memento pattern
- State saving
- Undo/redo`,
    examples: [
      {
        input: 'Save state, modify, restore',
        output: 'Object returns to saved state',
      },
    ],
    constraints: [
      'Save state without exposing internals',
      'Restore from memento',
    ],
    hints: [
      'Memento holds state',
      'Originator creates memento',
      'Caretaker stores mementos',
    ],
    starterCode: `class Memento:
    """Stores state"""
    def __init__(self, state):
        self._state = state
    
    def get_state(self):
        return self._state


class TextEditor:
    """Originator"""
    def __init__(self):
        self._content = ""
    
    def write(self, text):
        """Modify state"""
        self._content += text
    
    def get_content(self):
        return self._content
    
    def save(self):
        """Create memento"""
        return Memento(self._content)
    
    def restore(self, memento):
        """Restore from memento"""
        self._content = memento.get_state()


class History:
    """Caretaker"""
    def __init__(self):
        self._mementos = []
    
    def save(self, memento):
        self._mementos.append(memento)
    
    def undo(self):
        if self._mementos:
            return self._mementos.pop()
        return None


def test_memento():
    """Test memento pattern"""
    editor = TextEditor()
    history = History()
    
    # Write and save
    editor.write("Hello ")
    history.save(editor.save())
    
    editor.write("World")
    history.save(editor.save())
    
    editor.write("!")
    
    # Current: "Hello World!"
    current_len = len(editor.get_content())
    
    # Undo to "Hello World"
    memento = history.undo()
    editor.restore(memento)
    
    after_undo_len = len(editor.get_content())
    
    return current_len + after_undo_len
`,
    testCases: [
      {
        input: [],
        expected: 23,
        functionName: 'test_memento',
      },
    ],
    solution: `class Memento:
    def __init__(self, state):
        self._state = state
    
    def get_state(self):
        return self._state


class TextEditor:
    def __init__(self):
        self._content = ""
    
    def write(self, text):
        self._content += text
    
    def get_content(self):
        return self._content
    
    def save(self):
        return Memento(self._content)
    
    def restore(self, memento):
        self._content = memento.get_state()


class History:
    def __init__(self):
        self._mementos = []
    
    def save(self, memento):
        self._mementos.append(memento)
    
    def undo(self):
        if self._mementos:
            return self._mementos.pop()
        return None


def test_memento():
    editor = TextEditor()
    history = History()
    
    editor.write("Hello ")
    history.save(editor.save())
    
    editor.write("World")
    history.save(editor.save())
    
    editor.write("!")
    
    current_len = len(editor.get_content())
    
    memento = history.undo()
    editor.restore(memento)
    
    after_undo_len = len(editor.get_content())
    
    return current_len + after_undo_len`,
    timeComplexity: 'O(1) for save/restore',
    spaceComplexity: 'O(n) for n saves',
    order: 43,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-visitor-pattern',
    title: 'Visitor Pattern',
    difficulty: 'Hard',
    description: `Implement visitor pattern to add operations without modifying classes.

**Pattern:**
- Separate algorithm from object structure
- Add new operations easily
- Double dispatch
- Visit different types

This tests:
- Visitor pattern
- Double dispatch
- Extensibility`,
    examples: [
      {
        input: 'Visitor visits different element types',
        output: 'Operation without modifying elements',
      },
    ],
    constraints: [
      'Implement accept() and visit()',
      'Support multiple element types',
    ],
    hints: [
      'Elements accept visitors',
      'Visitors implement visit methods',
      'Double dispatch pattern',
    ],
    starterCode: `class Visitor:
    """Base visitor"""
    def visit_circle(self, circle):
        pass
    
    def visit_rectangle(self, rectangle):
        pass


class AreaVisitor(Visitor):
    """Visitor that calculates area"""
    def visit_circle(self, circle):
        return 3.14159 * circle.radius ** 2
    
    def visit_rectangle(self, rectangle):
        return rectangle.width * rectangle.height


class Circle:
    """Element"""
    def __init__(self, radius):
        self.radius = radius
    
    def accept(self, visitor):
        """Accept visitor"""
        return visitor.visit_circle(self)


class Rectangle:
    """Element"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def accept(self, visitor):
        """Accept visitor"""
        return visitor.visit_rectangle(self)


def test_visitor():
    """Test visitor pattern"""
    shapes = [
        Circle(5),
        Rectangle(4, 6),
        Circle(3),
    ]
    
    # Apply visitor to all shapes
    area_visitor = AreaVisitor()
    total_area = sum(shape.accept(area_visitor) for shape in shapes)
    
    return int(total_area)
`,
    testCases: [
      {
        input: [],
        expected: 130,
        functionName: 'test_visitor',
      },
    ],
    solution: `class Visitor:
    def visit_circle(self, circle):
        pass
    
    def visit_rectangle(self, rectangle):
        pass


class AreaVisitor(Visitor):
    def visit_circle(self, circle):
        return 3.14159 * circle.radius ** 2
    
    def visit_rectangle(self, rectangle):
        return rectangle.width * rectangle.height


class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def accept(self, visitor):
        return visitor.visit_circle(self)


class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def accept(self, visitor):
        return visitor.visit_rectangle(self)


def test_visitor():
    shapes = [
        Circle(5),
        Rectangle(4, 6),
        Circle(3),
    ]
    
    area_visitor = AreaVisitor()
    total_area = sum(shape.accept(area_visitor) for shape in shapes)
    
    return int(total_area)`,
    timeComplexity: 'O(n) for n elements',
    spaceComplexity: 'O(1)',
    order: 44,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-type-hints-generics',
    title: 'Type Hints with Generics',
    difficulty: 'Medium',
    description: `Use generic type hints for flexible type checking.

**Generics:**
- TypeVar for type variables
- Generic[T] base class
- Flexible type-safe code
- Used in collections, containers

This tests:
- Generic types
- Type variables
- Type safety`,
    examples: [
      {
        input: 'class Stack(Generic[T])',
        output: 'Works with any type',
      },
    ],
    constraints: ['Use TypeVar and Generic', 'Type-safe container'],
    hints: [
      'from typing import TypeVar, Generic',
      'TypeVar("T")',
      'class Stack(Generic[T])',
    ],
    starterCode: `from typing import TypeVar, Generic, List

T = TypeVar('T')


class Stack(Generic[T]):
    """Generic stack"""
    def __init__(self):
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        """Push item"""
        self._items.append(item)
    
    def pop(self) -> T:
        """Pop item"""
        return self._items.pop()
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get size"""
        return len(self._items)


def test_generics():
    """Test generic types"""
    # Int stack
    int_stack: Stack[int] = Stack()
    int_stack.push(1)
    int_stack.push(2)
    int_stack.push(3)
    
    # String stack
    str_stack: Stack[str] = Stack()
    str_stack.push("hello")
    str_stack.push("world")
    
    return int_stack.size() + str_stack.size()
`,
    testCases: [
      {
        input: [],
        expected: 5,
        functionName: 'test_generics',
      },
    ],
    solution: `from typing import TypeVar, Generic, List

T = TypeVar('T')


class Stack(Generic[T]):
    def __init__(self):
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()
    
    def is_empty(self) -> bool:
        return len(self._items) == 0
    
    def size(self) -> int:
        return len(self._items)


def test_generics():
    int_stack: Stack[int] = Stack()
    int_stack.push(1)
    int_stack.push(2)
    int_stack.push(3)
    
    str_stack: Stack[str] = Stack()
    str_stack.push("hello")
    str_stack.push("world")
    
    return int_stack.size() + str_stack.size()`,
    timeComplexity: 'O(1) for operations',
    spaceComplexity: 'O(n)',
    order: 45,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-abstract-class-template',
    title: 'Abstract Class with Template Methods',
    difficulty: 'Medium',
    description: `Combine abstract base class with template method pattern.

**Pattern:**
- Abstract methods force implementation
- Template method defines workflow
- Best of both patterns

This tests:
- ABC with template method
- Abstract + concrete methods
- Design patterns combination`,
    examples: [
      {
        input: 'Abstract methods + template workflow',
        output: 'Enforced implementation with structure',
      },
    ],
    constraints: ['Use ABC', 'Template method calls abstract methods'],
    hints: [
      'ABC for abstract methods',
      'Template method for workflow',
      'Subclass implements abstracts',
    ],
    starterCode: `from abc import ABC, abstractmethod

class DataImporter(ABC):
    """Abstract importer with template"""
    def import_data(self, source):
        """Template method"""
        # Step 1
        data = self.read_source(source)
        
        # Step 2
        validated = self.validate(data)
        
        # Step 3
        transformed = self.transform(validated)
        
        # Step 4
        self.save(transformed)
        
        return len(transformed)
    
    @abstractmethod
    def read_source(self, source):
        """Must implement: read data"""
        pass
    
    def validate(self, data):
        """Optional hook: validate data"""
        return data
    
    @abstractmethod
    def transform(self, data):
        """Must implement: transform data"""
        pass
    
    def save(self, data):
        """Optional hook: save data"""
        pass


class CSVImporter(DataImporter):
    """Concrete importer"""
    def read_source(self, source):
        """Read CSV"""
        return source.split(',')
    
    def transform(self, data):
        """Transform: uppercase"""
        return [item.upper() for item in data]


def test_abstract_template():
    """Test abstract template pattern"""
    importer = CSVImporter()
    result = importer.import_data("apple,banana,cherry")
    
    return result
`,
    testCases: [
      {
        input: [],
        expected: 3,
        functionName: 'test_abstract_template',
      },
    ],
    solution: `from abc import ABC, abstractmethod

class DataImporter(ABC):
    def import_data(self, source):
        data = self.read_source(source)
        validated = self.validate(data)
        transformed = self.transform(validated)
        self.save(transformed)
        return len(transformed)
    
    @abstractmethod
    def read_source(self, source):
        pass
    
    def validate(self, data):
        return data
    
    @abstractmethod
    def transform(self, data):
        pass
    
    def save(self, data):
        pass


class CSVImporter(DataImporter):
    def read_source(self, source):
        return source.split(',')
    
    def transform(self, data):
        return [item.upper() for item in data]


def test_abstract_template():
    importer = CSVImporter()
    result = importer.import_data("apple,banana,cherry")
    return result`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 46,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-property-decorators',
    title: 'Advanced Property Decorators',
    difficulty: 'Medium',
    description: `Use property decorators with getters, setters, and deleters.

**Decorators:**
- @property for getter
- @name.setter for setter
- @name.deleter for deleter

This tests:
- Property decorators
- Attribute management
- Encapsulation`,
    examples: [
      {
        input: '@property, @x.setter, @x.deleter',
        output: 'Full attribute control',
      },
    ],
    constraints: [
      'Use all three decorators',
      'Control access/modification/deletion',
    ],
    hints: [
      '@property first',
      '@name.setter for modification',
      '@name.deleter for deletion',
    ],
    starterCode: `class Temperature:
    """Temperature with full property control"""
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Get Celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Set Celsius with validation"""
        if value < -273.15:
            raise ValueError("Below absolute zero")
        self._celsius = value
    
    @celsius.deleter
    def celsius(self):
        """Delete Celsius"""
        del self._celsius
    
    @property
    def fahrenheit(self):
        """Get Fahrenheit (computed)"""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Set Fahrenheit (converts to Celsius)"""
        self.celsius = (value - 32) * 5/9


def test_property_decorators():
    """Test property decorators"""
    temp = Temperature(25)
    
    # Get Celsius
    c = temp.celsius
    
    # Set via Fahrenheit
    temp.fahrenheit = 86
    
    # Get new Celsius (should be 30)
    new_c = temp.celsius
    
    return int(c + new_c)
`,
    testCases: [
      {
        input: [],
        expected: 55,
        functionName: 'test_property_decorators',
      },
    ],
    solution: `class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Below absolute zero")
        self._celsius = value
    
    @celsius.deleter
    def celsius(self):
        del self._celsius
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9


def test_property_decorators():
    temp = Temperature(25)
    c = temp.celsius
    temp.fahrenheit = 86
    new_c = temp.celsius
    return int(c + new_c)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 47,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-class-decorators',
    title: 'Class Decorators',
    difficulty: 'Hard',
    description: `Create and use class decorators to modify classes.

**Class decorators:**
- Decorate entire class
- Add/modify class attributes/methods
- Return modified or new class

This tests:
- Class decorators
- Meta-programming
- Dynamic class modification`,
    examples: [
      {
        input: '@add_methods decorator',
        output: 'Adds methods to class',
      },
    ],
    constraints: ['Create class decorator', 'Modify class'],
    hints: [
      'Decorator receives class',
      'Modify class attributes',
      'Return class',
    ],
    starterCode: `def add_str_method(cls):
    """Class decorator that adds __str__ method"""
    def __str__(self):
        return f"{cls.__name__} instance"
    
    cls.__str__ = __str__
    return cls


def add_id(cls):
    """Class decorator that adds class ID"""
    cls.class_id = id(cls)
    return cls


@add_str_method
@add_id
class Person:
    """Person with decorators"""
    def __init__(self, name):
        self.name = name


def test_class_decorators():
    """Test class decorators"""
    person = Person("Alice")
    
    # Has __str__ from decorator
    str_repr = str(person)
    
    # Has class_id from decorator
    has_id = hasattr(Person, 'class_id')
    
    return len(str_repr) + (10 if has_id else 0)
`,
    testCases: [
      {
        input: [],
        expected: 25,
        functionName: 'test_class_decorators',
      },
    ],
    solution: `def add_str_method(cls):
    def __str__(self):
        return f"{cls.__name__} instance"
    
    cls.__str__ = __str__
    return cls


def add_id(cls):
    cls.class_id = id(cls)
    return cls


@add_str_method
@add_id
class Person:
    def __init__(self, name):
        self.name = name


def test_class_decorators():
    person = Person("Alice")
    str_repr = str(person)
    has_id = hasattr(Person, 'class_id')
    return len(str_repr) + (10 if has_id else 0)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 48,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-enum-with-methods',
    title: 'Enum with Methods',
    difficulty: 'Easy',
    description: `Create enum with custom methods and properties.

**Enum features:**
- Named constants
- Can have methods
- Can have properties
- Iteration support

This tests:
- Enum with behavior
- Methods in enum
- Custom enum functionality`,
    examples: [
      {
        input: 'Color.RED.is_warm()',
        output: 'Enum members with methods',
      },
    ],
    constraints: ['Use Enum', 'Add custom methods'],
    hints: [
      'Inherit from Enum',
      'Add methods like normal class',
      'Methods can use self.value',
    ],
    starterCode: `from enum import Enum

class Color(Enum):
    """Color enum with methods"""
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4
    
    def is_primary(self):
        """Check if primary color"""
        return self in (Color.RED, Color.GREEN, Color.BLUE)
    
    def is_warm(self):
        """Check if warm color"""
        return self in (Color.RED, Color.YELLOW)
    
    @classmethod
    def get_warm_colors(cls):
        """Get all warm colors"""
        return [color for color in cls if color.is_warm()]


def test_enum_methods():
    """Test enum with methods"""
    red = Color.RED
    blue = Color.BLUE
    
    # Check primary
    red_primary = red.is_primary()
    
    # Check warm
    red_warm = red.is_warm()
    blue_warm = blue.is_warm()
    
    # Get warm colors
    warm_colors = Color.get_warm_colors()
    
    return len(warm_colors) + (1 if red_primary else 0)
`,
    testCases: [
      {
        input: [],
        expected: 3,
        functionName: 'test_enum_methods',
      },
    ],
    solution: `from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4
    
    def is_primary(self):
        return self in (Color.RED, Color.GREEN, Color.BLUE)
    
    def is_warm(self):
        return self in (Color.RED, Color.YELLOW)
    
    @classmethod
    def get_warm_colors(cls):
        return [color for color in cls if color.is_warm()]


def test_enum_methods():
    red = Color.RED
    blue = Color.BLUE
    
    red_primary = red.is_primary()
    red_warm = red.is_warm()
    blue_warm = blue.is_warm()
    warm_colors = Color.get_warm_colors()
    
    return len(warm_colors) + (1 if red_primary else 0)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 49,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-dataclass-inheritance',
    title: 'Dataclass with Inheritance',
    difficulty: 'Medium',
    description: `Use dataclasses with inheritance.

**Dataclass inheritance:**
- Subclass inherits fields
- Can add new fields
- Field order matters
- Use post_init for complex logic

This tests:
- Dataclass inheritance
- Field ordering
- post_init method`,
    examples: [
      {
        input: 'Base dataclass + derived',
        output: 'Inherited and new fields',
      },
    ],
    constraints: ['Use @dataclass', 'Proper inheritance'],
    hints: [
      'Parent fields first',
      'Child adds new fields',
      '__post_init__ for logic',
    ],
    starterCode: `from dataclasses import dataclass

@dataclass
class Person:
    """Base person dataclass"""
    name: str
    age: int
    
    def is_adult(self) -> bool:
        return self.age >= 18


@dataclass
class Employee(Person):
    """Employee extends Person"""
    employee_id: str
    salary: float
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.salary < 0:
            raise ValueError("Salary must be positive")
    
    def annual_bonus(self) -> float:
        """Calculate bonus"""
        return self.salary * 0.1


def test_dataclass_inheritance():
    """Test dataclass inheritance"""
    emp = Employee(
        name="Alice",
        age=30,
        employee_id="E001",
        salary=50000
    )
    
    # Use parent method
    adult = emp.is_adult()
    
    # Use child method
    bonus = emp.annual_bonus()
    
    return int(bonus / 1000)
`,
    testCases: [
      {
        input: [],
        expected: 5,
        functionName: 'test_dataclass_inheritance',
      },
    ],
    solution: `from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    
    def is_adult(self) -> bool:
        return self.age >= 18


@dataclass
class Employee(Person):
    employee_id: str
    salary: float
    
    def __post_init__(self):
        if self.salary < 0:
            raise ValueError("Salary must be positive")
    
    def annual_bonus(self) -> float:
        return self.salary * 0.1


def test_dataclass_inheritance():
    emp = Employee(
        name="Alice",
        age=30,
        employee_id="E001",
        salary=50000
    )
    
    adult = emp.is_adult()
    bonus = emp.annual_bonus()
    
    return int(bonus / 1000)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 50,
    topic: 'Python Object-Oriented Programming',
  },
];

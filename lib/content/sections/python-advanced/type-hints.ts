/**
 * Type Hints & Static Type Checking Section
 */

export const typehintsSection = {
  id: 'type-hints',
  title: 'Type Hints & Static Type Checking',
  content: `**What are Type Hints?**
Type hints (PEP 484) let you annotate variables, function parameters, and return values with type information, enabling better IDE support, documentation, and static type checking with tools like mypy.

**Basic Type Hints:**
\`\`\`python
def greet(name: str) -> str:
    """Function with type hints"""
    return f"Hello, {name}!"

age: int = 30
price: float = 19.99
is_valid: bool = True
names: list[str] = ["Alice", "Bob"]  # Python 3.9+
\`\`\`

**Generic Types:**
\`\`\`python
from typing import List, Dict, Set, Tuple, Optional, Union

# Collections (pre-Python 3.9)
numbers: List[int] = [1, 2, 3]
user_data: Dict[str, int] = {"age": 30, "score": 95}
unique_items: Set[str] = {"apple", "banana"}
coordinates: Tuple[float, float] = (10.5, 20.3)

# Python 3.9+ - use built-in types directly
numbers: list[int] = [1, 2, 3]
user_data: dict[str, int] = {"age": 30}
\`\`\`

**Optional and Union:**
\`\`\`python
from typing import Optional, Union

# Optional[T] = Union[T, None]
def find_user(user_id: int) -> Optional[str]:
    """Returns username or None if not found"""
    if user_id == 1:
        return "Alice"
    return None

# Union for multiple possible types
def process_id(id: Union[int, str]) -> str:
    """Accepts int or str"""
    return str(id)

# Python 3.10+ - use | operator
def process_id(id: int | str) -> str:
    return str(id)
\`\`\`

**Generic Functions with TypeVar:**
\`\`\`python
from typing import TypeVar, List

T = TypeVar('T')  # Generic type variable

def first_element(items: List[T]) -> Optional[T]:
    """Get first element, preserving type"""
    return items[0] if items else None

# Usage preserves types
nums: List[int] = [1, 2, 3]
first_num: Optional[int] = first_element(nums)  # Type: Optional[int]

names: List[str] = ["Alice", "Bob"]
first_name: Optional[str] = first_element(names)  # Type: Optional[str]
\`\`\`

**Constrained TypeVars:**
\`\`\`python
from typing import TypeVar

# Constrain to specific types
Number = TypeVar('Number', int, float)

def add(a: Number, b: Number) -> Number:
    """Works with int or float, but both must be same type"""
    return a + b

result1: int = add(1, 2)        # OK: both int
result2: float = add(1.5, 2.5)  # OK: both float
# result3 = add(1, 2.5)         # Error: mixing types
\`\`\`

**Callable Types:**
\`\`\`python
from typing import Callable

def apply_twice(func: Callable[[int], int], value: int) -> int:
    """Apply function twice to value"""
    return func(func(value))

def double(x: int) -> int:
    return x * 2

result: int = apply_twice(double, 5)  # 20
\`\`\`

**Literal Types:**
\`\`\`python
from typing import Literal

def set_log_level(level: Literal["debug", "info", "warning", "error"]) -> None:
    """Only accepts these exact string values"""
    print(f"Log level set to {level}")

set_log_level("debug")    # OK
# set_log_level("trace")  # Error: not in Literal values
\`\`\`

**Type Aliases:**
\`\`\`python
from typing import List, Dict, Tuple

# Create readable type aliases
UserId = int
Username = str
UserData = Dict[UserId, Username]
Coordinates = Tuple[float, float]
Point = Tuple[float, float, float]

users: UserData = {1: "Alice", 2: "Bob"}
location: Coordinates = (10.5, 20.3)
\`\`\`

**Protocol (Structural Typing):**
\`\`\`python
from typing import Protocol

class Drawable(Protocol):
    """Anything with a draw() method"""
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Square:
    def draw(self) -> None:
        print("Drawing square")

def render(shape: Drawable) -> None:
    """Works with any object that has draw()"""
    shape.draw()

# Both work without explicit inheritance!
render(Circle())
render(Square())
\`\`\`

**TypedDict for Structured Dictionaries:**
\`\`\`python
from typing import TypedDict

class User(TypedDict):
    name: str
    age: int
    email: str

def create_user(name: str, age: int, email: str) -> User:
    return {"name": name, "age": age, "email": email}

user: User = create_user("Alice", 30, "alice@example.com")
print(user["name"])  # IDE knows this exists and is str
\`\`\`

**Static Type Checking with mypy:**
\`\`\`python
# Run: mypy script.py

def add_numbers(a: int, b: int) -> int:
    return a + b

# mypy will catch this error:
result: str = add_numbers(1, 2)  # Error: incompatible types
\`\`\`

**Best Practices:**
- Start with function signatures (parameters and return types)
- Use Optional for values that can be None
- Prefer Protocol over ABC for flexibility
- Use mypy in CI/CD pipeline
- Type hints are optional—add where they help most
- Use \`# type: ignore\` for exceptions

**Benefits:**
- Better IDE autocomplete and error detection
- Self-documenting code
- Catch bugs before runtime
- Easier refactoring
- No runtime overhead (hints are ignored at runtime)`,
};

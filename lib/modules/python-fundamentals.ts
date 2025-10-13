/**
 * Python Fundamentals - Core Python concepts for beginners
 */

import { Module } from '../types';

export const pythonFundamentalsModule: Module = {
  id: 'python-fundamentals',
  title: 'Python Fundamentals',
  description:
    'Master the core concepts of Python programming, from basic syntax to essential data structures and control flow.',
  category: 'Python',
  difficulty: 'Beginner',
  estimatedTime: '8 hours',
  prerequisites: [],
  icon: '🐍',
  keyTakeaways: [
    'Write Python code with proper syntax and style',
    'Use variables, data types, and operators effectively',
    'Control program flow with conditionals and loops',
    'Work with lists, dictionaries, tuples, and sets',
    'Create and call functions with parameters',
    'Manipulate strings and perform common operations',
    'Handle basic errors with try-except',
  ],
  learningObjectives: [
    'Understand Python syntax and basic data types',
    'Work with lists, tuples, dictionaries, and sets',
    'Master control flow with loops and conditionals',
    'Write and use functions effectively',
    'Handle strings and perform common operations',
    'Understand basic exception handling',
    'Use list comprehensions for concise code',
    'Work with files for input and output',
  ],
  sections: [
    {
      id: 'variables-types',
      title: 'Variables and Data Types',
      content: `# Variables and Data Types

## Variables

Python uses dynamic typing - you don't need to declare variable types explicitly.

\`\`\`python
# Variable assignment
name = "Alice"
age = 30
height = 5.7
is_student = True

# Multiple assignment
x, y, z = 1, 2, 3

# Swapping variables
a, b = 5, 10
a, b = b, a  # Now a=10, b=5
\`\`\`

## Basic Data Types

### 1. Numbers
- **int**: Whole numbers (unlimited precision)
- **float**: Decimal numbers
- **complex**: Complex numbers

\`\`\`python
# Integer
count = 42
big_number = 999_999_999  # Underscores for readability

# Float
price = 19.99
scientific = 1.5e-4  # 0.00015

# Complex
z = 3 + 4j
\`\`\`

### 2. Strings
- Immutable sequences of characters
- Single or double quotes

\`\`\`python
single = 'Hello'
double = "World"
multiline = """This is
a multiline
string"""

# String operations
greeting = "Hello" + " " + "World"  # Concatenation
repeated = "Ha" * 3  # "HaHaHa"
length = len("Python")  # 6
\`\`\`

### 3. Boolean
- True or False (capitalized!)

\`\`\`python
is_valid = True
has_errors = False

# Boolean operations
result = True and False  # False
result = True or False   # True
result = not True        # False
\`\`\`

## Type Conversion

\`\`\`python
# String to number
num = int("42")        # 42
decimal = float("3.14")  # 3.14

# Number to string
text = str(100)        # "100"

# To boolean
bool(0)     # False
bool(1)     # True
bool("")    # False
bool("Hi")  # True
\`\`\`

## Key Concepts

1. **Dynamic Typing**: Variables can change types
2. **Strong Typing**: Operations between incompatible types raise errors
3. **Duck Typing**: "If it walks like a duck and quacks like a duck, it's a duck"

## Best Practices

- Use descriptive variable names: \`user_count\` not \`uc\`
- Follow naming conventions: \`snake_case\` for variables and functions
- Constants in UPPERCASE: \`MAX_SIZE = 100\`
- Avoid reserved keywords: \`class\`, \`if\`, \`for\`, etc.`,
      videoUrl: 'https://www.youtube.com/watch?v=Z1Yd7upQsXY',
      quiz: [
        {
          id: 'pf-variables-q-1',
          question: 'What is the result of: 5 + 3 * 2?',
          options: ['16', '11', '10', '13'],
          correctAnswer: 1,
          explanation:
            'Multiplication has higher precedence than addition, so 3 * 2 = 6, then 5 + 6 = 11.',
        },
        {
          id: 'pf-variables-q-2',
          question: 'Which of these is NOT a valid variable name in Python?',
          options: ['my_var', '_private', '2fast', 'myVar'],
          correctAnswer: 2,
          explanation:
            'Variable names cannot start with a number. "2fast" is invalid.',
        },
        {
          id: 'pf-variables-q-3',
          question: 'What is the type of: x = 3.0?',
          options: ['int', 'float', 'str', 'number'],
          correctAnswer: 1,
          explanation:
            'Numbers with decimal points are float type, even if the decimal part is zero.',
        },
      ],
      discussion: [
        {
          question: 'When should you use int vs float?',
          answer:
            'Use int for whole numbers (counting, indexing) and float for measurements or calculations requiring precision. Be aware of floating-point precision issues.',
        },
        {
          question: 'Why is dynamic typing useful?',
          answer:
            'Dynamic typing makes Python flexible and easy to write. You can reassign variables to different types, which is convenient for prototyping. However, it can lead to runtime errors if not careful.',
        },
      ],
      multipleChoice: [
        {
          id: 'pf-variables-mc-1',
          question:
            'What does the following code output?\n\nx = "5"\ny = 2\nprint(x * y)',
          options: ['10', '7', '"55"', '"52"'],
          correctAnswer: 2,
          explanation:
            'String multiplication repeats the string. "5" * 2 = "55".',
        },
        {
          id: 'pf-variables-mc-2',
          question:
            'Which statement correctly converts the string "123" to an integer?',
          options: [
            'integer("123")',
            'int("123")',
            'toInt("123")',
            'Number("123")',
          ],
          correctAnswer: 1,
          explanation:
            'The int() function converts strings to integers in Python.',
        },
      ],
    },
    {
      id: 'control-flow',
      title: 'Control Flow',
      content: `# Control Flow

## If Statements

\`\`\`python
age = 18

if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

# Inline if (ternary operator)
status = "Adult" if age >= 18 else "Minor"
\`\`\`

## Comparison Operators

\`\`\`python
==  # Equal to
!=  # Not equal to
>   # Greater than
<   # Less than
>=  # Greater than or equal to
<=  # Less than or equal to

# Chaining comparisons
if 0 < x < 10:
    print("x is between 0 and 10")
\`\`\`

## Logical Operators

\`\`\`python
and  # Both conditions must be True
or   # At least one condition must be True
not  # Inverts the boolean value

# Examples
if age >= 18 and has_license:
    print("Can drive")

if is_weekend or is_holiday:
    print("No work!")
\`\`\`

## For Loops

\`\`\`python
# Iterate over sequence
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Using range()
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 6):  # 1, 2, 3, 4, 5
    print(i)

for i in range(0, 10, 2):  # 0, 2, 4, 6, 8
    print(i)

# Enumerate for index and value
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
\`\`\`

## While Loops

\`\`\`python
count = 0
while count < 5:
    print(count)
    count += 1

# Infinite loop with break
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input == 'quit':
        break
    print(f"You entered: {user_input}")
\`\`\`

## Loop Control

\`\`\`python
# break - exit loop completely
for i in range(10):
    if i == 5:
        break  # Stop at 5
    print(i)  # Prints 0, 1, 2, 3, 4

# continue - skip to next iteration
for i in range(5):
    if i == 2:
        continue  # Skip 2
    print(i)  # Prints 0, 1, 3, 4

# else clause - executes if loop completes without break
for i in range(5):
    if i == 10:
        break
else:
    print("Loop completed")  # This will print
\`\`\`

## Match-Case (Python 3.10+)

\`\`\`python
def http_status(status):
    match status:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case 500:
            return "Server Error"
        case _:
            return "Unknown"
\`\`\``,
      videoUrl: 'https://www.youtube.com/watch?v=Zp5MuPOtsSY',
      quiz: [
        {
          id: 'pf-control-q-1',
          question: 'What does range(5) produce?',
          options: [
            '1, 2, 3, 4, 5',
            '0, 1, 2, 3, 4',
            '0, 1, 2, 3, 4, 5',
            '1, 2, 3, 4',
          ],
          correctAnswer: 1,
          explanation:
            'range(5) generates numbers from 0 up to (but not including) 5.',
        },
        {
          id: 'pf-control-q-2',
          question: 'What happens when "break" is used in a loop?',
          options: [
            'Skips current iteration',
            'Exits the loop completely',
            'Pauses the loop',
            'Restarts the loop',
          ],
          correctAnswer: 1,
          explanation:
            'break exits the loop immediately, skipping any remaining iterations.',
        },
      ],
      discussion: [
        {
          question: 'When should you use for vs while loops?',
          answer:
            'Use for loops when you know the number of iterations (iterating over a sequence). Use while loops when the number of iterations depends on a condition.',
        },
      ],
      multipleChoice: [
        {
          id: 'pf-control-mc-1',
          question:
            'What is the output?\n\nfor i in range(3):\n    if i == 1:\n        continue\n    print(i)',
          options: ['0 1 2', '0 2', '1 2', '0 1'],
          correctAnswer: 1,
          explanation:
            'continue skips the rest of the current iteration, so 1 is not printed.',
        },
      ],
    },
    {
      id: 'data-structures',
      title: 'Data Structures',
      content: `# Python Data Structures

## Lists

Ordered, mutable sequences.

\`\`\`python
# Creating lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# Accessing elements
first = numbers[0]      # 1
last = numbers[-1]      # 5
slice = numbers[1:3]    # [2, 3]

# Modifying lists
numbers.append(6)       # Add to end
numbers.insert(0, 0)    # Insert at position
numbers.remove(3)       # Remove first occurrence
popped = numbers.pop()  # Remove and return last
numbers[0] = 10        # Change element

# List operations
len(numbers)           # Length
3 in numbers          # Membership test
numbers.sort()        # Sort in place
numbers.reverse()     # Reverse in place
numbers.count(2)      # Count occurrences
\`\`\`

## Tuples

Ordered, immutable sequences.

\`\`\`python
# Creating tuples
coords = (10, 20)
single = (42,)  # Note the comma!
point = 1, 2, 3  # Parentheses optional

# Unpacking
x, y = coords
a, b, c = point

# Tuples are immutable
# coords[0] = 5  # TypeError!

# Use cases
def get_min_max(numbers):
    return min(numbers), max(numbers)

min_val, max_val = get_min_max([1, 2, 3])
\`\`\`

## Dictionaries

Unordered key-value pairs.

\`\`\`python
# Creating dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

# Accessing values
name = person["name"]           # KeyError if not found
age = person.get("age", 0)     # Default value if not found

# Modifying
person["age"] = 31             # Update
person["email"] = "a@ex.com"   # Add new key
del person["city"]             # Delete key

# Dictionary operations
keys = person.keys()           # Get all keys
values = person.values()       # Get all values
items = person.items()         # Get key-value pairs

# Iteration
for key in person:
    print(key, person[key])

for key, value in person.items():
    print(f"{key}: {value}")
\`\`\`

## Sets

Unordered collections of unique elements.

\`\`\`python
# Creating sets
fruits = {"apple", "banana", "cherry"}
empty_set = set()  # {} creates empty dict!

# Set operations
fruits.add("orange")           # Add element
fruits.remove("banana")        # Remove (KeyError if not found)
fruits.discard("grape")        # Remove (no error)

# Set math
a = {1, 2, 3}
b = {3, 4, 5}

a | b  # Union: {1, 2, 3, 4, 5}
a & b  # Intersection: {3}
a - b  # Difference: {1, 2}
a ^ b  # Symmetric difference: {1, 2, 4, 5}
\`\`\`

## List Comprehensions

Concise way to create lists.

\`\`\`python
# Basic syntax: [expression for item in iterable]
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(10) if x % 2 == 0]

# Nested
matrix = [[i*j for j in range(3)] for i in range(3)]

# Dictionary comprehension
word_lengths = {word: len(word) for word in ["hi", "hello", "hey"]}

# Set comprehension
unique_lengths = {len(word) for word in ["hi", "hello", "hey"]}
\`\`\``,
      videoUrl: 'https://www.youtube.com/watch?v=W8KRzm-HUcc',
      quiz: [
        {
          id: 'pf-datastructures-q-1',
          question: 'What is the main difference between lists and tuples?',
          options: [
            'Lists are faster',
            'Tuples are immutable',
            'Lists can only store numbers',
            'Tuples can be nested',
          ],
          correctAnswer: 1,
          explanation:
            'Tuples are immutable (cannot be changed after creation), while lists are mutable.',
        },
        {
          id: 'pf-datastructures-q-2',
          question: 'How do you create an empty set?',
          options: ['{}', 'set()', '[]', 'Set()'],
          correctAnswer: 1,
          explanation:
            '{} creates an empty dictionary. Use set() for an empty set.',
        },
      ],
      discussion: [
        {
          question: 'When should you use a tuple instead of a list?',
          answer:
            'Use tuples for fixed collections of heterogeneous data (like coordinates, function returns) or when you want immutability. Use lists for homogeneous, mutable sequences.',
        },
      ],
      multipleChoice: [
        {
          id: 'pf-datastructures-mc-1',
          question:
            'What does this list comprehension create?\n\n[x*2 for x in range(5) if x > 2]',
          options: ['[0, 2, 4, 6, 8]', '[6, 8]', '[3, 4]', '[6, 8, 10]'],
          correctAnswer: 1,
          explanation: 'Filters x > 2 (3, 4), then multiplies by 2: [6, 8].',
        },
      ],
    },
    {
      id: 'functions',
      title: 'Functions',
      content: `# Functions

## Defining Functions

\`\`\`python
def greet(name):
    """Function with one parameter."""
    return f"Hello, {name}!"

# Calling the function
message = greet("Alice")
print(message)  # Hello, Alice!
\`\`\`

## Parameters and Arguments

\`\`\`python
# Default parameters
def power(base, exponent=2):
    return base ** exponent

power(3)      # 9 (uses default exponent=2)
power(3, 3)   # 27

# Keyword arguments
def describe_pet(animal, name):
    print(f"I have a {animal} named {name}")

describe_pet(animal="dog", name="Max")
describe_pet(name="Max", animal="dog")  # Order doesn't matter

# *args - variable number of positional arguments
def sum_all(*numbers):
    return sum(numbers)

sum_all(1, 2, 3, 4)  # 10

# **kwargs - variable number of keyword arguments
def print_info(**info):
    for key, value in info.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=30, city="NYC")
\`\`\`

## Return Values

\`\`\`python
# Single return value
def square(x):
    return x ** 2

# Multiple return values (tuple)
def get_min_max(numbers):
    return min(numbers), max(numbers)

minimum, maximum = get_min_max([1, 2, 3, 4, 5])

# No return (returns None)
def print_greeting(name):
    print(f"Hello, {name}")
\`\`\`

## Lambda Functions

Anonymous, one-line functions.

\`\`\`python
# Syntax: lambda arguments: expression
square = lambda x: x ** 2
add = lambda x, y: x + y

# Common use with map, filter, sorted
numbers = [1, 2, 3, 4, 5]

squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

# Sorting with key
people = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
sorted_by_age = sorted(people, key=lambda person: person[1])
\`\`\`

## Scope

\`\`\`python
# Global scope
global_var = "I'm global"

def my_function():
    # Local scope
    local_var = "I'm local"
    print(global_var)  # Can access global
    
    # Modify global variable
    global global_var
    global_var = "Modified"

# global_var is accessible here
# local_var is not accessible here
\`\`\`

## Docstrings

\`\`\`python
def calculate_area(length, width):
    """
    Calculate the area of a rectangle.
    
    Args:
        length (float): The length of the rectangle
        width (float): The width of the rectangle
        
    Returns:
        float: The area of the rectangle
        
    Examples:
        >>> calculate_area(5, 3)
        15
    """
    return length * width

# Access docstring
print(calculate_area.__doc__)
\`\`\`

## Best Practices

1. **Single Responsibility**: One function, one purpose
2. **Descriptive Names**: \`calculate_total\` not \`calc\`
3. **Keep it Short**: If it's too long, split it up
4. **Use Docstrings**: Document what the function does
5. **Avoid Side Effects**: Don't modify global state`,
      videoUrl: 'https://www.youtube.com/watch?v=9Os0o3wzS_I',
      quiz: [
        {
          id: 'pf-functions-q-1',
          question: 'What does *args allow you to do?',
          options: [
            'Pass a variable number of keyword arguments',
            'Pass a variable number of positional arguments',
            'Make an argument optional',
            'Pass arguments by reference',
          ],
          correctAnswer: 1,
          explanation:
            '*args collects variable numbers of positional arguments into a tuple.',
        },
        {
          id: 'pf-functions-q-2',
          question:
            'What does a function return if no return statement is used?',
          options: ['0', 'Empty string', 'None', 'Error'],
          correctAnswer: 2,
          explanation:
            'Functions without a return statement implicitly return None.',
        },
      ],
      discussion: [
        {
          question: 'When should you use lambda functions?',
          answer:
            'Use lambda for simple, one-line functions often passed as arguments (e.g., in map, filter, sorted). For complex logic, use regular def functions for better readability.',
        },
      ],
      multipleChoice: [
        {
          id: 'pf-functions-mc-1',
          question:
            'What is the output?\n\ndef multiply(a, b=2):\n    return a * b\n\nprint(multiply(5))',
          options: ['10', '5', '7', 'Error'],
          correctAnswer: 0,
          explanation: 'b defaults to 2, so 5 * 2 = 10.',
        },
      ],
    },
    {
      id: 'strings',
      title: 'String Operations',
      content: `# String Operations

## String Basics

\`\`\`python
# Creating strings
single = 'Hello'
double = "World"
triple = '''Multi
line
string'''

# Escape characters
quote = "He said \\"Hello\\""
newline = "Line 1\\nLine 2"
tab = "Column1\\tColumn2"

# Raw strings (ignore escapes)
path = r"C:\\Users\\Name"
\`\`\`

## String Methods

\`\`\`python
text = "Hello World"

# Case conversion
text.upper()        # "HELLO WORLD"
text.lower()        # "hello world"
text.capitalize()   # "Hello world"
text.title()        # "Hello World"

# Searching
text.find("World")      # 6 (index)
text.index("World")     # 6 (raises error if not found)
text.startswith("Hello")  # True
text.endswith("World")    # True

# Checking content
"123".isdigit()      # True
"abc".isalpha()      # True
"abc123".isalnum()   # True

# Trimming whitespace
"  hello  ".strip()   # "hello"
"  hello  ".lstrip()  # "hello  "
"  hello  ".rstrip()  # "  hello"

# Replacing
text.replace("World", "Python")  # "Hello Python"

# Splitting and joining
words = "a,b,c".split(",")     # ["a", "b", "c"]
joined = "-".join(["a", "b"])  # "a-b"
\`\`\`

## String Formatting

\`\`\`python
name = "Alice"
age = 30

# f-strings (Python 3.6+) - Preferred!
message = f"My name is {name} and I'm {age} years old"
formatted = f"Pi is approximately {3.14159:.2f}"  # 3.14

# Format method
message = "Name: {}, Age: {}".format(name, age)
message = "Name: {n}, Age: {a}".format(n=name, a=age)

# Old style (avoid)
message = "Name: %s, Age: %d" % (name, age)
\`\`\`

## String Slicing

\`\`\`python
text = "Python"

# Basic slicing [start:end:step]
text[0]      # "P"
text[-1]     # "n"
text[0:3]    # "Pyt"
text[2:]     # "thon"
text[:4]     # "Pyth"
text[::2]    # "Pto" (every 2nd char)
text[::-1]   # "nohtyP" (reverse)
\`\`\`

## String Iteration

\`\`\`python
# Iterate through characters
for char in "Python":
    print(char)

# With enumerate
for index, char in enumerate("Python"):
    print(f"{index}: {char}")
\`\`\`

## Common Patterns

\`\`\`python
# Check if substring exists
if "Python" in text:
    print("Found!")

# Count occurrences
count = "banana".count("a")  # 3

# Palindrome check
word = "racecar"
is_palindrome = word == word[::-1]

# Remove characters
text = "Hello, World!"
no_punctuation = text.replace(",", "").replace("!", "")

# Center and align
"Python".center(10)      # "  Python  "
"Python".ljust(10, "-")  # "Python----"
"Python".rjust(10, "-")  # "----Python"
\`\`\`

## String Immutability

\`\`\`python
# Strings cannot be modified
text = "Hello"
# text[0] = "h"  # TypeError!

# Instead, create new strings
text = "h" + text[1:]  # "hello"
text = text.replace("H", "h")  # "hello"
\`\`\``,
      videoUrl: 'https://www.youtube.com/watch?v=k9TUPpGqYTo',
      quiz: [
        {
          id: 'pf-strings-q-1',
          question: 'What does "Python"[::-1] return?',
          options: ['"Python"', '"nohtyP"', '"Pytho"', 'Error'],
          correctAnswer: 1,
          explanation: '[::-1] reverses the string using negative step.',
        },
        {
          id: 'pf-strings-q-2',
          question: 'Which method checks if a string contains only digits?',
          options: ['isnum()', 'isdigit()', 'isnumber()', 'isint()'],
          correctAnswer: 1,
          explanation: 'isdigit() returns True if all characters are digits.',
        },
      ],
      discussion: [
        {
          question: 'Why are strings immutable in Python?',
          answer:
            'Immutability allows strings to be used as dictionary keys, enables string interning for memory efficiency, and makes them thread-safe. When you "modify" a string, Python creates a new string object.',
        },
      ],
      multipleChoice: [
        {
          id: 'pf-strings-mc-1',
          question:
            'What is the output?\n\ntext = "hello world"\nprint(text.title())',
          options: [
            '"Hello World"',
            '"HELLO WORLD"',
            '"hello world"',
            '"Hello world"',
          ],
          correctAnswer: 0,
          explanation: 'title() capitalizes the first letter of each word.',
        },
      ],
    },
  ],
};

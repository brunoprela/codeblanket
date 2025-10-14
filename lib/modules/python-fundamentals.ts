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
          question:
            'Explain the difference between mutable and immutable types in Python. Why does it matter when assigning variables?',
          hint: 'Think about what happens when you modify a list vs. a string, and what happens with variable assignment.',
          sampleAnswer:
            'In Python, immutable types (like int, float, str, tuple) cannot be changed after creation. When you "modify" them, you actually create a new object. Mutable types (like list, dict, set) can be changed in place. This matters for variable assignment because multiple variables can point to the same mutable object, so changes through one variable affect all references. For example: a = [1, 2]; b = a; b.append(3) will modify the list that both a and b reference.',
          keyPoints: [
            'Immutable types: int, float, str, tuple, frozenset',
            'Mutable types: list, dict, set',
            'Assignment creates references, not copies',
            'Use .copy() or copy.deepcopy() for actual copies',
          ],
        },
        {
          id: 'pf-variables-q-2',
          question:
            'When would you choose to use a float instead of an integer in Python? What are the potential pitfalls of using floats?',
          hint: 'Consider precision requirements, mathematical operations, and how computers represent decimal numbers.',
          sampleAnswer:
            "Use floats when you need decimal precision, like measurements, scientific calculations, or division operations. However, floats have precision limitations due to binary representation. For example, 0.1 + 0.2 doesn't exactly equal 0.3 in Python due to floating-point arithmetic. For financial calculations requiring exact decimal precision, use the Decimal class instead. Use integers when working with counts, indices, or when exact values are critical.",
          keyPoints: [
            'Floats are for decimal/fractional values',
            'Binary representation causes precision issues',
            'Use Decimal class for exact decimal arithmetic',
            'Integers are exact and should be preferred when possible',
          ],
        },
        {
          id: 'pf-variables-q-3',
          question:
            "Explain Python's dynamic typing. What are the advantages and disadvantages compared to statically-typed languages?",
          hint: 'Think about flexibility, runtime behavior, debugging, and development speed.',
          sampleAnswer:
            "Python's dynamic typing means variables don't have fixed types - the same variable can hold different types at different times. This offers flexibility and faster prototyping since you don't declare types. However, it can lead to runtime type errors that statically-typed languages catch at compile time. Tools like type hints (PEP 484) and mypy can add static type checking while maintaining dynamic runtime behavior, giving you the best of both worlds.",
          keyPoints: [
            'Variables can change types freely',
            'No compile-time type checking',
            'More flexible but can hide bugs until runtime',
            'Type hints (annotations) can add static checking',
          ],
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
        {
          id: 'pf-variables-mc-3',
          question: 'What is the result of: type(3.0) == type(3)?',
          options: ['True', 'False', 'TypeError', '3.0'],
          correctAnswer: 1,
          explanation:
            '3.0 is a float and 3 is an int, so they are different types.',
        },
        {
          id: 'pf-variables-mc-4',
          question: 'Which of the following is NOT a valid variable name?',
          options: ['my_var', '_private', '2fast', 'myVar2'],
          correctAnswer: 2,
          explanation:
            'Variable names cannot start with a number. "2fast" is invalid.',
        },
        {
          id: 'pf-variables-mc-5',
          question: 'What does bool("") evaluate to?',
          options: ['True', 'False', 'None', 'Error'],
          correctAnswer: 1,
          explanation:
            'Empty strings are falsy in Python, so bool("") returns False.',
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
          question:
            'Explain the difference between break, continue, and pass in Python loops. When would you use each?',
          hint: 'Think about what happens to loop execution and when you need each control statement.',
          sampleAnswer:
            "`break` exits the entire loop immediately. Use it when you've found what you're looking for or a condition makes continuing unnecessary. `continue` skips the rest of the current iteration and moves to the next one. Use it to skip processing for certain values. `pass` does nothing - it's a placeholder for code you'll write later, or when syntax requires a statement but you don't want to do anything. Example: When searching a list, use break once found. When processing numbers, use continue to skip negatives. Use pass when defining an empty function stub.",
          keyPoints: [
            'break: exits the loop entirely',
            'continue: skips to next iteration',
            'pass: does nothing, placeholder statement',
            'All serve different purposes in flow control',
          ],
        },
        {
          id: 'pf-control-q-2',
          question:
            'When should you choose a for loop versus a while loop? Can you convert any while loop to a for loop?',
          hint: "Consider when iteration count is known vs unknown, and whether you're iterating over a sequence.",
          sampleAnswer:
            'Use `for` loops when iterating over a known sequence (list, range, string) or when the number of iterations is predetermined. Use `while` loops when iterations depend on a condition that might change unpredictably, like waiting for user input, reading until end of file, or implementing game loops. While technically any while loop can be rewritten as a for loop (using itertools or custom iterators), it often makes code less readable. For example, "while True" with conditional breaks is clearer than forcing it into a for loop structure.',
          keyPoints: [
            'for: iterate over sequences or known ranges',
            'while: condition-based iteration',
            'for loops are more Pythonic for sequences',
            'while loops better for event-driven logic',
          ],
        },
        {
          id: 'pf-control-q-3',
          question:
            'What is the purpose of the else clause in Python loops? How does it differ from putting code after the loop?',
          hint: "Think about when the else block executes and when it doesn't, especially with break statements.",
          sampleAnswer:
            "The `else` clause in loops executes only if the loop completes normally (without hitting a break statement). This is different from code after the loop, which always runs. It's useful for search operations: if you break when finding something, else won't run; if you don't find it, else runs to handle the \"not found\" case. For example: searching for a prime number - if you break after finding a divisor, else doesn't run; if no divisors found, else confirms it's prime. Code after the loop would run regardless of whether you broke out or not.",
          keyPoints: [
            'else runs if loop completes without break',
            'Different from code placed after loop',
            'Useful for search/validation patterns',
            'Eliminates need for flag variables',
          ],
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
        {
          id: 'pf-control-mc-2',
          question: 'What does range(5) produce?',
          options: [
            '[1, 2, 3, 4, 5]',
            '[0, 1, 2, 3, 4]',
            '[0, 1, 2, 3, 4, 5]',
            '(0, 1, 2, 3, 4)',
          ],
          correctAnswer: 1,
          explanation:
            'range(5) generates numbers from 0 up to (but not including) 5.',
        },
        {
          id: 'pf-control-mc-3',
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
        {
          id: 'pf-control-mc-4',
          question:
            'What will this code print?\n\nx = 15\nif x > 10:\n    print("A")\nelif x > 5:\n    print("B")\nelse:\n    print("C")',
          options: ['"A"', '"B"', '"C"', '"A" and "B"'],
          correctAnswer: 0,
          explanation:
            'The first condition (x > 10) is true, so "A" is printed and the rest is skipped.',
        },
        {
          id: 'pf-control-mc-5',
          question: 'What is the difference between == and = in Python?',
          options: [
            'No difference',
            '== is comparison, = is assignment',
            '= is comparison, == is assignment',
            'Both are assignment',
          ],
          correctAnswer: 1,
          explanation:
            '== compares values for equality, while = assigns a value to a variable.',
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
          question:
            'Explain when you should use a list, tuple, set, or dictionary. What are the key trade-offs between them?',
          hint: 'Consider mutability, ordering, uniqueness, and access patterns.',
          sampleAnswer:
            "Use **lists** for ordered, mutable sequences when you need to modify elements, add/remove items, or maintain order. Use **tuples** for immutable sequences, like function returns or dictionary keys, when data shouldn't change. Use **sets** when you need unique elements and don't care about order - great for membership testing and removing duplicates. Use **dictionaries** for key-value mappings when you need fast lookup by key. Trade-offs: lists are flexible but slower for membership tests; tuples are faster and memory-efficient but immutable; sets are fast for membership but unordered; dicts provide fast access but use more memory.",
          keyPoints: [
            'Lists: ordered, mutable, allows duplicates',
            'Tuples: ordered, immutable, hashable',
            'Sets: unordered, mutable, unique elements only',
            'Dicts: key-value pairs, fast lookups',
          ],
        },
        {
          id: 'pf-datastructures-q-2',
          question:
            "Why are list comprehensions considered more Pythonic than traditional for loops? Are there cases where you shouldn't use them?",
          hint: 'Think about readability, performance, and complexity.',
          sampleAnswer:
            'List comprehensions are more Pythonic because they\'re concise, readable (for simple operations), and often faster than equivalent for loops. They express the intent "create a list from a transformation" clearly. However, avoid them when: 1) Logic is complex (multiple conditions, nested loops) - they become hard to read, 2) You need side effects during iteration, 3) The expression is very long, 4) You\'re not actually creating a list (use generator expressions instead). Remember: "Explicit is better than implicit" - if a comprehension is confusing, use a traditional loop.',
          keyPoints: [
            'More concise and often faster',
            'Better for simple transformations and filters',
            'Avoid when logic is complex or needs debugging',
            'Generator expressions for memory efficiency',
          ],
        },
        {
          id: 'pf-datastructures-q-3',
          question:
            'What is the difference between dict.get() and dict[key]? When would you use each method?',
          hint: "Consider what happens when a key doesn't exist and when you want different behaviors.",
          sampleAnswer:
            'dict[key] raises a KeyError if the key doesn\'t exist, while dict.get(key, default) returns None (or a specified default) if the key is missing. Use dict[key] when you expect the key to exist and want to catch programming errors - the KeyError signals a bug. Use dict.get() when missing keys are valid, like when checking optional configuration settings, or when you want to provide a default value. For example: config.get("debug", False) is cleaner than checking "if \'debug\' in config" first.',
          keyPoints: [
            'dict[key]: raises KeyError if missing',
            'dict.get(key, default): returns default if missing',
            'Use [key] when absence is an error',
            'Use .get() for optional values',
          ],
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
        {
          id: 'pf-datastructures-mc-2',
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
          id: 'pf-datastructures-mc-3',
          question: 'How do you create an empty set?',
          options: ['{}', 'set()', '[]', 'Set()'],
          correctAnswer: 1,
          explanation:
            '{} creates an empty dictionary. Use set() for an empty set.',
        },
        {
          id: 'pf-datastructures-mc-4',
          question:
            'What does this code output?\n\na = {1, 2, 3}\nb = {3, 4, 5}\nprint(a & b)',
          options: ['{1, 2, 3, 4, 5}', '{3}', '{1, 2, 4, 5}', '{}'],
          correctAnswer: 1,
          explanation:
            'The & operator performs set intersection, returning elements common to both sets.',
        },
        {
          id: 'pf-datastructures-mc-5',
          question:
            'Which data structure would be most efficient for checking if an item exists?',
          options: ['List', 'Tuple', 'Set', 'String'],
          correctAnswer: 2,
          explanation:
            'Sets use hash tables, making membership testing O(1) average case, much faster than lists or tuples which are O(n).',
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
          question:
            'Explain the difference between *args and **kwargs. When would you use each, and can you use them together?',
          hint: 'Think about positional vs keyword arguments, and how they are unpacked.',
          sampleAnswer:
            '*args collects a variable number of positional arguments into a tuple, while **kwargs collects keyword arguments into a dictionary. Use *args when you want flexibility in the number of positional arguments (like print() or max()). Use **kwargs when you want to accept arbitrary named parameters (like for configuration options or flexible APIs). You can use them together - the order must be: regular args, *args, keyword-only args, **kwargs. For example: def func(a, b, *args, key=None, **kwargs). This allows maximum flexibility while maintaining clear function signatures.',
          keyPoints: [
            '*args: tuple of positional arguments',
            '**kwargs: dictionary of keyword arguments',
            'Can be combined in specific order',
            'Useful for flexible APIs and wrappers',
          ],
        },
        {
          id: 'pf-functions-q-2',
          question:
            'What is the difference between parameters and arguments? How do default parameters work, and what pitfall should you avoid?',
          hint: 'Think about function definition vs function call, and mutable default values.',
          sampleAnswer:
            'Parameters are the variables in the function definition; arguments are the actual values passed when calling the function. Default parameters provide fallback values if no argument is supplied. Critical pitfall: NEVER use mutable objects (like lists or dicts) as default parameters! Python evaluates defaults once at function definition, not at each call. So def func(items=[]): creates ONE shared list across all calls. If you append to it, all future calls see those changes. Use def func(items=None): followed by items = items or [] inside the function instead.',
          keyPoints: [
            'Parameters: in definition, Arguments: when calling',
            'Defaults evaluated once at definition time',
            'Never use mutable defaults (lists, dicts)',
            'Use None as default, then create new object inside',
          ],
        },
        {
          id: 'pf-functions-q-3',
          question:
            'When should you use a lambda function versus a regular function? What are the limitations of lambdas?',
          hint: 'Consider readability, debuggability, and complexity.',
          sampleAnswer:
            "Use lambdas for short, simple operations that are used once, typically as arguments to functions like map(), filter(), sorted(), or in list comprehensions. They're concise for simple transformations like `sorted(items, key=lambda x: x[1])`. However, lambdas are limited to single expressions (no statements), can't contain assignments, and are harder to debug (they don't have names in tracebacks). For anything more complex than a simple transformation, use a regular function with a descriptive name - readability trumps brevity. If you find yourself writing complex lambdas, define a regular function instead.",
          keyPoints: [
            'Lambdas: single expression, anonymous functions',
            'Best for simple, one-time transformations',
            'Cannot contain statements or assignments',
            'Regular functions are more readable and debuggable',
          ],
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
        {
          id: 'pf-functions-mc-2',
          question:
            'What does a function return if no return statement is used?',
          options: ['0', 'Empty string', 'None', 'Error'],
          correctAnswer: 2,
          explanation:
            'Functions without a return statement implicitly return None.',
        },
        {
          id: 'pf-functions-mc-3',
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
          id: 'pf-functions-mc-4',
          question:
            'Which is the correct syntax for a lambda function that squares a number?',
          options: [
            'lambda x: x ** 2',
            'lambda(x): x ** 2',
            'def lambda x: x ** 2',
            'lambda x => x ** 2',
          ],
          correctAnswer: 0,
          explanation: 'Lambda syntax is: lambda arguments: expression',
        },
        {
          id: 'pf-functions-mc-5',
          question:
            'What is the scope of a variable defined inside a function?',
          options: [
            'Global scope',
            'Local scope (function only)',
            'Module scope',
            'Class scope',
          ],
          correctAnswer: 1,
          explanation:
            'Variables defined inside a function are local to that function unless declared global.',
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
          question:
            'Why are strings immutable in Python? What are the implications of this design decision?',
          hint: 'Think about dictionary keys, memory management, and thread safety.',
          sampleAnswer:
            'Strings are immutable to allow them to be hashable and used as dictionary keys and set members. Immutability enables string interning (reusing identical strings in memory) for efficiency, makes strings thread-safe without locks, and simplifies Python\'s implementation. The trade-off is that any "modification" creates a new string object. For frequent string modifications, use lists or StringIO/join() instead of concatenation to avoid creating many intermediate string objects.',
          keyPoints: [
            'Enables use as dictionary keys (hashable)',
            'Allows string interning for memory efficiency',
            'Thread-safe by default',
            'All "modifications" create new strings',
          ],
        },
        {
          id: 'pf-strings-q-2',
          question:
            'Explain the difference between str.format(), f-strings, and % formatting. Which should you use and why?',
          hint: 'Consider readability, performance, Python version requirements, and flexibility.',
          sampleAnswer:
            'Old-style % formatting (like "Hello %s" % name) is C-style but limited and less readable. str.format() (like "Hello {}".format(name)) is more powerful and readable but verbose. F-strings (like f"Hello {name}") are the modern preferred way (Python 3.6+) - they\'re most readable, fastest, and allow expressions inside braces. Use f-strings for new code unless you need Python 3.5 compatibility. str.format() is still useful when the format string comes from user input or configuration (security concern with f-strings).',
          keyPoints: [
            '% formatting: old style, less readable',
            'str.format(): more flexible, verbose',
            'f-strings: fastest, most readable (Python 3.6+)',
            'Prefer f-strings for new code',
          ],
        },
        {
          id: 'pf-strings-q-3',
          question:
            'When would you use str.join() versus string concatenation with +? What about performance considerations?',
          hint: 'Think about building strings in loops and memory allocation.',
          sampleAnswer:
            'Use str.join() when building strings from multiple parts, especially in loops. Since strings are immutable, using + creates a new string object each time, which is O(n²) for n concatenations. str.join() is O(n) as it allocates the final size once. Example: "".join(parts) is much faster than result = ""; for p in parts: result += p. However, for a small fixed number of concatenations (2-3), + is fine and more readable. F-strings are also efficient for combining a few known values.',
          keyPoints: [
            'join(): O(n), efficient for multiple strings',
            '+: O(n²) in loops due to immutability',
            'Use join() for building strings in loops',
            '+ is fine for 2-3 static concatenations',
          ],
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
        {
          id: 'pf-strings-mc-2',
          question: 'What does "Python"[::-1] return?',
          options: ['"Python"', '"nohtyP"', '"Pytho"', 'Error'],
          correctAnswer: 1,
          explanation: '[::-1] reverses the string using negative step.',
        },
        {
          id: 'pf-strings-mc-3',
          question: 'Which method checks if a string contains only digits?',
          options: ['isnum()', 'isdigit()', 'isnumber()', 'isint()'],
          correctAnswer: 1,
          explanation: 'isdigit() returns True if all characters are digits.',
        },
        {
          id: 'pf-strings-mc-4',
          question: 'What is the result of: "hello" + " " + "world"?',
          options: ['"hello world"', '"helloworld"', '"hello  world"', 'Error'],
          correctAnswer: 0,
          explanation: 'String concatenation with + joins strings together.',
        },
        {
          id: 'pf-strings-mc-5',
          question: 'What does "abc" * 3 produce?',
          options: ['"abcabcabc"', '"abc3"', '["abc", "abc", "abc"]', 'Error'],
          correctAnswer: 0,
          explanation: 'String multiplication repeats the string n times.',
        },
      ],
    },
    {
      id: 'none-handling',
      title: 'None and Null Values',
      content: `# None and Null Values

## What is None?

\`None\` is Python's null value - it represents the absence of a value or a null reference.

\`\`\`python
# None is a singleton object
result = None
print(type(None))  # <class 'NoneType'>

# Only one None exists in Python
a = None
b = None
print(a is b)  # True (same object)
\`\`\`

## Common Uses of None

### 1. Function Default Return
\`\`\`python
def greet():
    print("Hello!")
    # No return statement

result = greet()  # Prints "Hello!"
print(result)     # None

# Explicit return
def calculate(x):
    if x < 0:
        return None  # Indicate failure/invalid
    return x * 2
\`\`\`

### 2. Default Function Parameters
\`\`\`python
def log_message(message, timestamp=None):
    """Log message with optional timestamp"""
    if timestamp is None:
        timestamp = datetime.now()
    print(f"[{timestamp}] {message}")

# Use default (None becomes current time)
log_message("Server started")

# Provide custom timestamp
log_message("Error occurred", custom_time)
\`\`\`

### 3. Placeholder for Optional Values
\`\`\`python
class User:
    def __init__(self, name, email=None, phone=None):
        self.name = name
        self.email = email  # Optional
        self.phone = phone  # Optional

user1 = User("Alice", email="alice@example.com")
user2 = User("Bob")  # email and phone are None
\`\`\`

## Checking for None

### Use 'is' Not '=='
\`\`\`python
value = None

# Correct way
if value is None:
    print("Value is None")

if value is not None:
    print("Value exists")

# Wrong way (works but not idiomatic)
if value == None:  # Don't do this!
    print("Value is None")
\`\`\`

**Why 'is' instead of '=='?**
- \`is\` checks object identity (same object in memory)
- \`==\` checks value equality (can be overridden by classes)
- Since None is a singleton, \`is\` is more efficient and correct

### Truthy vs Falsy
\`\`\`python
# None is falsy
if not None:
    print("None is falsy")  # This prints

# But don't use truthiness to check for None!
value = None
if not value:  # Bad - could be 0, "", [], etc.
    print("Might not be None!")

# Better - explicit check
if value is None:
    print("Definitely None")
\`\`\`

## Common Pitfalls

### 1. Mutable Default Arguments
\`\`\`python
# WRONG - dangerous!
def add_item(item, items=[]):
    items.append(item)
    return items

list1 = add_item("a")  # ["a"]
list2 = add_item("b")  # ["a", "b"] - UNEXPECTED!

# RIGHT - use None as default
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

list1 = add_item("a")  # ["a"]
list2 = add_item("b")  # ["b"] - correct!
\`\`\`

**Why?** Default arguments are created once when the function is defined, not each time it's called. Mutable objects (lists, dicts) are shared between calls!

### 2. Forgetting to Return
\`\`\`python
def calculate_discount(price):
    if price > 100:
        return price * 0.9
    # Forgot to return for price <= 100

result = calculate_discount(50)
print(result)  # None - bug!
\`\`\`

### 3. Confusing None with False, 0, or ""
\`\`\`python
def get_user_age(user_id):
    # Returns 0 for baby, None for invalid ID
    if user_id < 0:
        return None
    return 0  # Baby's age

age = get_user_age(1)
if age:  # Wrong! 0 is falsy
    print(f"Age: {age}")
else:
    print("No age")  # Prints for babies!

# Correct - explicit None check
if age is not None:
    print(f"Age: {age}")  # Works for babies
else:
    print("Invalid user")
\`\`\`

## None in Data Structures

### Lists
\`\`\`python
# None as placeholder
data = [1, 2, None, 4, 5]

# Filter out None values
filtered = [x for x in data if x is not None]
print(filtered)  # [1, 2, 4, 5]

# Count None values
none_count = data.count(None)
print(none_count)  # 1
\`\`\`

### Dictionaries
\`\`\`python
config = {
    "host": "localhost",
    "port": 8080,
    "password": None  # Explicit "no password"
}

# Check if key exists vs is None
if "password" in config:
    print("Password key exists")
    if config["password"] is None:
        print("But password is None")

# Distinguish missing vs None
print(config.get("username"))      # None (missing key)
print(config.get("password"))      # None (explicit value)

# Use default to avoid None
username = config.get("username", "guest")  # "guest"
\`\`\`

## None vs Empty Collections

\`\`\`python
# These are different!
value1 = None      # No value
value2 = []        # Empty list (still a value)
value3 = ""        # Empty string (still a value)
value4 = 0         # Zero (still a value)

# All are falsy, but only one is None
print(bool(value1))  # False
print(bool(value2))  # False
print(bool(value3))  # False
print(bool(value4))  # False

print(value1 is None)  # True
print(value2 is None)  # False
print(value3 is None)  # False
print(value4 is None)  # False
\`\`\`

## Best Practices

### 1. Use None for "Not Set" or "Missing"
\`\`\`python
class Config:
    def __init__(self):
        self.database_url = None  # Not configured yet
        self.api_key = None        # Not provided

    def is_configured(self):
        return self.database_url is not None
\`\`\`

### 2. Document When Functions Return None
\`\`\`python
def find_user(user_id: int) -> User | None:
    """
    Find user by ID.
    
    Returns:
        User object if found, None if not found
    """
    # ... search logic
    if not found:
        return None
    return user
\`\`\`

### 3. Avoid Returning None When Possible
\`\`\`python
# Instead of returning None for "not found":
def get_users():
    if no_users:
        return None  # Caller must check

    return users

# Better - return empty list:
def get_users():
    if no_users:
        return []  # Caller can iterate immediately
    
    return users

# Now this always works:
for user in get_users():
    print(user)
\`\`\`

### 4. Use None for Optional Type Hints
\`\`\`python
from typing import Optional

def greet(name: str, title: Optional[str] = None) -> str:
    """
    Optional[str] is equivalent to str | None
    """
    if title is None:
        return f"Hello, {name}!"
    return f"Hello, {title} {name}!"
\`\`\`

## Real-World Patterns

### Null Object Pattern Alternative
\`\`\`python
# Instead of returning None and checking everywhere
def get_user_permissions(user_id):
    user = find_user(user_id)
    if user is None:
        return []  # Empty permissions instead of None
    return user.permissions

# No None check needed
permissions = get_user_permissions(123)
if "admin" in permissions:  # Works even if empty
    print("User is admin")
\`\`\`

### None Guard Pattern
\`\`\`python
def process_data(data=None):
    """Early return for None"""
    if data is None:
        return []  # Or raise ValueError
    
    # Process data knowing it's not None
    return [item * 2 for item in data]
\`\`\`

### Chaining with None
\`\`\`python
# Without None handling
def get_user_city(user_id):
    user = get_user(user_id)
    if user is None:
        return None
    
    address = user.get_address()
    if address is None:
        return None
    
    return address.city

# Better - use default values
def get_user_city(user_id):
    user = get_user(user_id)
    if user is None:
        return "Unknown"
    
    address = user.get_address()
    return address.city if address else "Unknown"
\`\`\`

## Summary

✅ **Do:**
- Use \`is None\` to check for None
- Use None for missing/unset values
- Document when functions can return None
- Use None as default for mutable parameters

❌ **Don't:**
- Use \`== None\` (use \`is None\`)
- Use truthiness to check for None (use explicit \`is None\`)
- Use mutable defaults (use None instead)
- Return None when empty collection is better`,
      quiz: [
        {
          id: 'pf-none-q-1',
          question:
            'Why should you use "is None" instead of "== None" to check for None? What is the fundamental difference?',
          hint: 'Think about identity vs equality and how None is implemented.',
          sampleAnswer:
            '"is None" checks object identity - whether the variable points to the exact same None object in memory. "== None" checks value equality, which can be overridden by implementing __eq__. Since None is a singleton (only one None object exists in Python), "is" is both more efficient (no method call) and more correct (can\'t be fooled by classes that define __eq__ to return True when compared to None). Additionally, "is None" is the idiomatic Pythonic way and is recommended by PEP 8. While "== None" usually works, edge cases exist where custom __eq__ methods could break it.',
          keyPoints: [
            '"is" checks identity (same object in memory)',
            '"==" checks equality (can be overridden)',
            'None is a singleton',
            '"is" is more efficient and correct',
            'PEP 8 recommends "is None"',
          ],
        },
        {
          id: 'pf-none-q-2',
          question:
            'Explain the mutable default argument problem. Why should you use None as default instead of [] or {}?',
          hint: 'Consider when default arguments are created and how they are shared between function calls.',
          sampleAnswer:
            'Default arguments are created once when the function is defined, not each time the function is called. If you use a mutable default like [] or {}, all calls to the function share the same list/dict object. Example: "def add(item, items=[])" - the first call creates the list, and subsequent calls reuse it, causing items to accumulate across calls. This is almost never the intended behavior. The solution is to use None as default and create a new mutable object inside the function: "def add(item, items=None): if items is None: items = []". This ensures each call gets its own fresh list.',
          keyPoints: [
            'Default arguments created once at function definition',
            'Mutable defaults shared between all calls',
            'Causes unexpected accumulation of data',
            'Use None as default, create mutable inside function',
            'Pattern: if items is None: items = []',
          ],
        },
        {
          id: 'pf-none-q-3',
          question:
            'When should you return None vs an empty collection ([], {}, "")? What are the trade-offs?',
          hint: 'Consider how callers will use the return value and what makes their code simpler.',
          sampleAnswer:
            'Return empty collections when the function is querying/searching and the "not found" case is normal, not exceptional. This lets callers iterate or check membership without None checks: "for item in get_items()" works even if no items. Return None when: 1) distinguishing "no result" from "empty result" matters (None = error/not-found, [] = found but empty), 2) the operation failed or is invalid, 3) the value is truly optional/unset. Example: search_users() returning [] means "no matches" (normal), returning None means "search failed" (error). Generally, prefer empty collections for better API usability unless None conveys important semantic information.',
          keyPoints: [
            'Empty collection: "not found" is normal, allows iteration',
            'None: operation failed or value truly missing',
            'Empty collection makes caller code simpler',
            'None when distinguishing error from empty result',
            'Example: [] for "no matches", None for "search failed"',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'pf-none-mc-1',
          question: 'What is the correct way to check if a variable is None?',
          options: [
            'if value == None:',
            'if value is None:',
            'if not value:',
            'if value == null:',
          ],
          correctAnswer: 1,
          explanation:
            '"is None" checks object identity and is the correct, idiomatic way to check for None in Python.',
        },
        {
          id: 'pf-none-mc-2',
          question:
            'What does a function return if it has no return statement?',
          options: ['Nothing', '0', 'None', 'Empty string'],
          correctAnswer: 2,
          explanation:
            'Functions without a return statement (or with just "return") implicitly return None.',
        },
        {
          id: 'pf-none-mc-3',
          question:
            'What is wrong with this code?\n\ndef add_item(item, items=[]):\n    items.append(item)\n    return items',
          options: [
            'Nothing, it works fine',
            'The default list is shared between calls',
            'Syntax error',
            'items must be a tuple',
          ],
          correctAnswer: 1,
          explanation:
            'Mutable default arguments are created once and shared between all function calls. Use items=None instead.',
        },
        {
          id: 'pf-none-mc-4',
          question: 'Which values are falsy in Python?',
          options: [
            'Only None',
            'None, False, 0, "", [], {}',
            'None and False only',
            'All values',
          ],
          correctAnswer: 1,
          explanation:
            'None, False, 0, empty strings, empty lists, and empty dicts are all falsy, but should be checked differently depending on context.',
        },
        {
          id: 'pf-none-mc-5',
          question: 'What is the type of None?',
          options: ['NoneType', 'null', 'object', 'None'],
          correctAnswer: 0,
          explanation:
            'The type of None is NoneType. None is the only value of this type and is a singleton.',
        },
      ],
    },
    {
      id: 'modules-imports',
      title: 'Modules and Imports',
      content: `# Modules and Imports

Modules are Python files containing reusable code. The import system lets you use code from other files and libraries.

## What is a Module?

A module is simply a Python file (.py) containing variables, functions, and classes.

\`\`\`python
# File: my_module.py
def greet(name):
    return f"Hello, {name}!"

PI = 3.14159

class Circle:
    def __init__(self, radius):
        self.radius = radius
\`\`\`

## Basic Imports

### Import Entire Module

\`\`\`python
import math

print(math.sqrt(16))  # 4.0
print(math.pi)        # 3.141592653589793
\`\`\`

### Import Specific Items

\`\`\`python
from math import sqrt, pi

print(sqrt(16))  # 4.0
print(pi)        # 3.141592653589793
\`\`\`

### Import All (Not Recommended)

\`\`\`python
from math import *  # ❌ Avoid - pollutes namespace

print(sqrt(16))  # Works but unclear where sqrt comes from
\`\`\`

### Import with Alias

\`\`\`python
import numpy as np  # Common convention
import pandas as pd

arr = np.array([1, 2, 3])
\`\`\`

## Importing Your Own Modules

\`\`\`python
# File structure:
# my_project/
#   main.py
#   utils.py
#   helpers.py

# In main.py:
import utils
from helpers import calculate

result = utils.process_data()
value = calculate(10)
\`\`\`

## Package Structure

A package is a directory containing multiple modules and a special \`__init__.py\` file.

\`\`\`python
# File structure:
# my_package/
#   __init__.py
#   module1.py
#   module2.py

# Importing from package:
from my_package import module1
from my_package.module2 import some_function
\`\`\`

## The \`__name__\` Variable

Every Python file has a built-in \`__name__\` variable.

\`\`\`python
# File: script.py
def main():
    print("Running main function")

# This code only runs when file is executed directly
if __name__ == "__main__":
    main()
    
# When imported: __name__ is "script"
# When run directly: __name__ is "__main__"
\`\`\`

**Why this matters:**
- Allows file to work as both module and script
- Common pattern in Python projects
- Prevents code from running when imported

## Common Standard Library Modules

### Math and Random

\`\`\`python
import math
import random

print(math.ceil(3.2))        # 4
print(math.floor(3.8))       # 3
print(random.randint(1, 10)) # Random int 1-10
print(random.choice(['a', 'b', 'c']))  # Random choice
\`\`\`

### Datetime

\`\`\`python
from datetime import datetime, timedelta

now = datetime.now()
print(now)  # Current date and time

tomorrow = now + timedelta(days=1)
print(tomorrow)
\`\`\`

### Collections

\`\`\`python
from collections import Counter, defaultdict

counts = Counter(['a', 'b', 'a', 'c', 'b', 'a'])
print(counts)  # Counter({'a': 3, 'b': 2, 'c': 1})

# defaultdict - no KeyError on missing keys
d = defaultdict(list)
d['key'].append('value')  # Works without initializing
\`\`\`

## Relative vs Absolute Imports

\`\`\`python
# File structure:
# project/
#   package/
#     __init__.py
#     module_a.py
#     module_b.py

# Absolute import (from project root):
from package.module_a import function_a

# Relative import (from within package):
from .module_a import function_a  # Same directory
from ..other_package import something  # Parent directory
\`\`\`

## Import Best Practices

✅ **Do:**
- Import at top of file
- Use absolute imports for clarity
- Group imports: standard library → third-party → local
- One import per line for readability

\`\`\`python
# Good
import os
import sys

import numpy as np
import pandas as pd

from my_package import my_module
\`\`\`

❌ **Don't:**
- Use \`from module import *\`
- Import in the middle of code
- Create circular imports

## Installing Third-Party Packages

\`\`\`bash
# Install package
pip install requests

# Install specific version
pip install requests==2.28.0

# Install from requirements.txt
pip install -r requirements.txt

# List installed packages
pip list

# Show package info
pip show requests
\`\`\`

## Creating requirements.txt

\`\`\`bash
# Save current environment packages
pip freeze > requirements.txt
\`\`\`

Example requirements.txt:
\`\`\`
requests==2.28.1
numpy==1.23.0
pandas==1.4.3
\`\`\`

## Common Import Errors

### ModuleNotFoundError

\`\`\`python
import non_existent_module  # ModuleNotFoundError

# Solutions:
# 1. Check spelling
# 2. Install package: pip install package_name
# 3. Check Python path
\`\`\`

### ImportError

\`\`\`python
from math import non_existent_function  # ImportError

# Solution: Check what's available
import math
print(dir(math))  # List all available items
\`\`\`

### Circular Import

\`\`\`python
# File: a.py
import b

# File: b.py  
import a  # Circular import!

# Solution: Restructure code or use import inside function
\`\`\`

## Quick Reference

| Import Type | Syntax | When to Use |
|-------------|--------|-------------|
| **Full module** | \`import math\` | Use module namespace |
| **Specific items** | \`from math import sqrt\` | Use items directly |
| **With alias** | \`import numpy as np\` | Shorter name |
| **Multiple items** | \`from os import path, getcwd\` | Several items |
| **Package** | \`from package import module\` | Organized code |`,
      quiz: [
        {
          id: 'q1',
          question:
            'What is the difference between "import math" and "from math import sqrt"?',
          sampleAnswer:
            '"import math" imports the entire math module, and you access functions using math.sqrt(). "from math import sqrt" imports only the sqrt function, allowing you to use it directly as sqrt() without the module prefix. The first is more explicit about where functions come from, while the second is more concise but can cause naming conflicts if multiple modules have functions with the same name.',
          keyPoints: [
            'import math: use math.sqrt()',
            'from math import sqrt: use sqrt() directly',
            'First is more explicit',
            'Second is more concise',
            'Consider namespace pollution',
          ],
        },
        {
          id: 'q2',
          question: 'Why is the "if __name__ == \'__main__\':" pattern useful?',
          sampleAnswer:
            'This pattern allows a Python file to work as both an importable module and a standalone script. When the file is run directly, __name__ equals "__main__" and the code inside the if block executes. When the file is imported as a module, __name__ is the module name and the code doesn\'t run. This prevents unwanted code execution during imports and is the standard way to structure Python scripts.',
          keyPoints: [
            'File works as both module and script',
            'Code only runs when executed directly',
            'Prevents execution on import',
            'Standard Python pattern',
            '__name__ == "__main__" when run directly',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the import best practices and why should you avoid "from module import *"?',
          sampleAnswer:
            'Best practices: (1) Import at top of file for clear dependencies, (2) Use absolute imports for clarity, (3) Group imports: standard library, then third-party, then local, (4) One import per line for readability. Avoid "from module import *" because it pollutes namespace with potentially hundreds of names, makes it unclear where functions come from (reduces code readability), can cause naming conflicts if multiple modules have same function names, and makes debugging harder. For example, if you do "from math import *" and "from numpy import *", both have "sqrt" function, causing confusion. Better: "import math" and "math.sqrt()" or "from math import sqrt" for specific imports.',
          keyPoints: [
            'Import at top, use absolute imports, group by type',
            '"import *" pollutes namespace',
            'Makes code unclear (where does function come from?)',
            'Can cause naming conflicts between modules',
            'Better: explicit imports or use module prefix',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What does "import math" do?',
          options: [
            'Imports all Python modules',
            'Imports the math module into the current namespace',
            'Creates a new math module',
            'Deletes the math module',
          ],
          correctAnswer: 1,
          explanation:
            'import math loads the math module, making its functions available via math.function_name().',
        },
        {
          id: 'mc2',
          question:
            'What is the difference between "import math" and "from math import *"?',
          options: [
            'No difference',
            '"from math import *" imports all functions directly into namespace',
            '"import math" is faster',
            '"from math import *" is recommended',
          ],
          correctAnswer: 1,
          explanation:
            '"from math import *" imports all functions directly (e.g., sqrt() instead of math.sqrt()), but pollutes namespace and is not recommended.',
        },
        {
          id: 'mc3',
          question:
            'What is the value of __name__ when a Python file is run directly?',
          options: ['The filename', '"__main__"', '"main"', 'None'],
          correctAnswer: 1,
          explanation:
            'When a Python file is executed directly, __name__ is set to "__main__", allowing the if __name__ == "__main__": pattern.',
        },
        {
          id: 'mc4',
          question: 'What is a Python package?',
          options: [
            'A single .py file',
            'A directory containing modules and __init__.py',
            'A compressed file',
            'A function collection',
          ],
          correctAnswer: 1,
          explanation:
            'A package is a directory containing Python modules and an __init__.py file, allowing hierarchical module organization.',
        },
        {
          id: 'mc5',
          question: 'Which import style is generally recommended?',
          options: [
            'from module import *',
            'import module or from module import specific_function',
            'Always use relative imports',
            'Import in the middle of code',
          ],
          correctAnswer: 1,
          explanation:
            'Explicit imports (import module or from module import func) are recommended for clarity. Avoid "import *" and import at the top of files.',
        },
      ],
    },
    {
      id: 'list-comprehensions',
      title: 'List Comprehensions',
      content: `# List Comprehensions

List comprehensions provide a concise way to create lists based on existing lists or other iterables.

## Basic Syntax

\`\`\`python
# Traditional way
squares = []
for i in range(10):
    squares.append(i ** 2)

# List comprehension way
squares = [i ** 2 for i in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
\`\`\`

**Syntax:** \`[expression for item in iterable]\`

## With Conditions (Filtering)

\`\`\`python
# Get even numbers
evens = [i for i in range(10) if i % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]

# Get squares of even numbers only
even_squares = [i ** 2 for i in range(10) if i % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]
\`\`\`

**Syntax with condition:** \`[expression for item in iterable if condition]\`

## With if-else (Transformation)

\`\`\`python
# Replace odd with "odd" and even with the number
result = [i if i % 2 == 0 else "odd" for i in range(10)]
print(result)  # [0, 'odd', 2, 'odd', 4, 'odd', 6, 'odd', 8, 'odd']

# Absolute values
numbers = [-4, -2, 0, 2, 4]
abs_values = [abs(x) for x in numbers]
print(abs_values)  # [4, 2, 0, 2, 4]
\`\`\`

**Syntax with if-else:** \`[expression_if_true if condition else expression_if_false for item in iterable]\`

## String Operations

\`\`\`python
# Convert to uppercase
words = ['hello', 'world', 'python']
upper_words = [word.upper() for word in words]
print(upper_words)  # ['HELLO', 'WORLD', 'PYTHON']

# Get first letter
first_letters = [word[0] for word in words]
print(first_letters)  # ['h', 'w', 'p']

# Filter strings by length
long_words = [word for word in words if len(word) > 5]
print(long_words)  # ['python']
\`\`\`

## Nested List Comprehensions

\`\`\`python
# Create 2D matrix
matrix = [[i * j for j in range(3)] for i in range(3)]
print(matrix)
# [[0, 0, 0],
#  [0, 1, 2],
#  [0, 2, 4]]

# Flatten nested list
nested = [[1, 2, 3], [4, 5], [6, 7, 8]]
flat = [item for sublist in nested for item in sublist]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8]
\`\`\`

## Multiple Iterables

\`\`\`python
# Cartesian product
colors = ['red', 'blue']
sizes = ['S', 'M', 'L']
combinations = [(color, size) for color in colors for size in sizes]
print(combinations)
# [('red', 'S'), ('red', 'M'), ('red', 'L'),
#  ('blue', 'S'), ('blue', 'M'), ('blue', 'L')]

# Using zip
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
people = [f"{name} is {age}" for name, age in zip(names, ages)]
print(people)
# ['Alice is 25', 'Bob is 30', 'Charlie is 35']
\`\`\`

## Dictionary Comprehensions

\`\`\`python
# Create dictionary
squares_dict = {i: i ** 2 for i in range(5)}
print(squares_dict)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Filter dictionary
prices = {'apple': 0.50, 'banana': 0.25, 'orange': 0.75}
expensive = {k: v for k, v in prices.items() if v > 0.40}
print(expensive)  # {'apple': 0.5, 'orange': 0.75}

# Transform values
doubled = {k: v * 2 for k, v in prices.items()}
print(doubled)  # {'apple': 1.0, 'banana': 0.5, 'orange': 1.5}
\`\`\`

## Set Comprehensions

\`\`\`python
# Create set (duplicates removed)
squares_set = {i ** 2 for i in range(-5, 6)}
print(squares_set)  # {0, 1, 4, 9, 16, 25}

# Filter set
numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9}
evens = {n for n in numbers if n % 2 == 0}
print(evens)  # {2, 4, 6, 8}
\`\`\`

## Generator Expressions

Similar to list comprehensions but create generators (lazy evaluation).

\`\`\`python
# List comprehension - creates list immediately
squares_list = [i ** 2 for i in range(1000000)]  # Uses lots of memory

# Generator expression - computes on demand
squares_gen = (i ** 2 for i in range(1000000))  # Uses little memory

# Use in for loop
for sq in squares_gen:
    if sq > 100:
        break
    print(sq)
\`\`\`

## When to Use List Comprehensions

✅ **Use when:**
- Creating new list from existing iterable
- Simple transformation or filtering
- Code is still readable
- Need all results at once

\`\`\`python
# Good - clear and concise
evens = [x for x in range(100) if x % 2 == 0]
\`\`\`

❌ **Don't use when:**
- Logic is complex (use regular loop)
- Multiple statements needed
- Readability suffers

\`\`\`python
# Bad - too complex
result = [func1(func2(x, y)) for x in range(10) 
          for y in range(20) if x != y if x * y < 100]

# Better - use regular loop
result = []
for x in range(10):
    for y in range(20):
        if x != y and x * y < 100:
            result.append(func1(func2(x, y)))
\`\`\`

## Performance Comparison

\`\`\`python
import timeit

# List comprehension is faster
def with_loop():
    result = []
    for i in range(1000):
        result.append(i ** 2)
    return result

def with_comprehension():
    return [i ** 2 for i in range(1000)]

# Comprehension is typically 20-30% faster
print(timeit.timeit(with_loop, number=10000))
print(timeit.timeit(with_comprehension, number=10000))
\`\`\`

## Common Patterns

### Filter then Transform
\`\`\`python
# Get squares of even numbers
result = [x ** 2 for x in range(20) if x % 2 == 0]
\`\`\`

### Transform then Filter  
\`\`\`python
# Not directly possible - use two steps or regular loop
\`\`\`

### Flatten Nested Structure
\`\`\`python
matrix = [[1, 2], [3, 4], [5, 6]]
flat = [item for row in matrix for item in row]
\`\`\`

### Conditional Transformation
\`\`\`python
# Different transformation based on condition
result = ['even' if x % 2 == 0 else 'odd' for x in range(10)]
\`\`\`

## Quick Reference

| Type | Syntax | Example |
|------|--------|---------|
| **List** | \`[expr for x in iter]\` | \`[x*2 for x in range(5)]\` |
| **With if** | \`[expr for x in iter if cond]\` | \`[x for x in range(10) if x%2==0]\` |
| **if-else** | \`[expr1 if cond else expr2 for x in iter]\` | \`[x if x>0 else 0 for x in nums]\` |
| **Dict** | \`{k:v for x in iter}\` | \`{x:x**2 for x in range(5)}\` |
| **Set** | \`{expr for x in iter}\` | \`{x**2 for x in range(5)}\` |
| **Generator** | \`(expr for x in iter)\` | \`(x**2 for x in range(5))\` |`,
      quiz: [
        {
          id: 'q1',
          question:
            'Rewrite this loop as a list comprehension: result = []; for x in range(10): if x % 2 == 0: result.append(x ** 2)',
          sampleAnswer:
            'result = [x ** 2 for x in range(10) if x % 2 == 0]. The list comprehension first defines the expression (x ** 2), then the iteration (for x in range(10)), and finally the condition (if x % 2 == 0). This creates a list of squares of even numbers from 0-9: [0, 4, 16, 36, 64].',
          keyPoints: [
            'Syntax: [expression for item in iterable if condition]',
            'Expression comes first: x ** 2',
            'Then iteration: for x in range(10)',
            'Then filter: if x % 2 == 0',
            'More concise than loop',
          ],
        },
        {
          id: 'q2',
          question:
            'What is the difference between [] and () in list comprehension vs generator expression?',
          sampleAnswer:
            'Square brackets [] create a list comprehension that immediately creates the entire list in memory. Parentheses () create a generator expression that computes values lazily on-demand. For example, [x**2 for x in range(1000000)] creates a list of 1 million integers in memory, while (x**2 for x in range(1000000)) creates a generator object that computes values one at a time as needed. Generators are more memory-efficient for large datasets or when you only need values once.',
          keyPoints: [
            '[] creates list immediately (eager)',
            '() creates generator (lazy)',
            'List uses more memory',
            'Generator computes on-demand',
            'Use generators for large data or single iteration',
          ],
        },
        {
          id: 'q3',
          question:
            'When should you use a regular for loop instead of a list comprehension?',
          sampleAnswer:
            'Use a regular for loop when: (1) Logic is complex - list comprehensions should be simple and readable, if you need nested if/else or multiple conditions, use a loop, (2) You need to perform side effects - like printing, writing to files, or modifying external state, (3) The comprehension becomes too long (> 79 characters or wraps multiple lines) - readability matters more than conciseness, (4) You need to break early or use continue with complex logic. For example, "for x in nums: if complex_condition(x): process(x); do_other_stuff()" is clearer as a loop. List comprehensions are for transforming data into new lists, not for executing complex logic.',
          keyPoints: [
            'Use loop for complex logic (multiple conditions, nested if/else)',
            'Use loop for side effects (print, file I/O, mutations)',
            'Use loop if comprehension becomes too long (> 79 chars)',
            'Readability beats conciseness',
            'Comprehensions are for simple transformations',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the syntax for a basic list comprehension?',
          options: [
            '[x for x in iterable]',
            '{x for x in iterable}',
            '(x for x in iterable)',
            'list(x in iterable)',
          ],
          correctAnswer: 0,
          explanation:
            'List comprehensions use square brackets []: [expression for item in iterable].',
        },
        {
          id: 'mc2',
          question:
            'How do you add a condition to filter items in a list comprehension?',
          options: [
            '[x where x > 0]',
            '[x for x in nums if x > 0]',
            '[x | x > 0 for x in nums]',
            '[x for x if x > 0 in nums]',
          ],
          correctAnswer: 1,
          explanation:
            'Add an if clause at the end: [x for x in nums if condition].',
        },
        {
          id: 'mc3',
          question: 'What does (x**2 for x in range(10)) create?',
          options: ['A list', 'A tuple', 'A generator', 'A set'],
          correctAnswer: 2,
          explanation:
            'Parentheses () create a generator expression, which evaluates lazily. Use [] for lists.',
        },
        {
          id: 'mc4',
          question: 'When should you avoid list comprehensions?',
          options: [
            'Never, always use them',
            'When the logic becomes too complex',
            'When working with numbers',
            'When the list is small',
          ],
          correctAnswer: 1,
          explanation:
            'Avoid list comprehensions when they become too complex or nested. Use regular loops for better readability.',
        },
        {
          id: 'mc5',
          question: 'What is a dictionary comprehension?',
          options: [
            '[key: value for item in iterable]',
            '{key: value for item in iterable}',
            '(key, value for item in iterable)',
            'dict(key=value for item in iterable)',
          ],
          correctAnswer: 1,
          explanation:
            'Dictionary comprehensions use curly braces {}: {key: value for item in iterable}.',
        },
      ],
    },
    {
      id: 'lambda-functions',
      title: 'Lambda Functions',
      content: `# Lambda Functions

Lambda functions are small anonymous functions defined with the \`lambda\` keyword. They're useful for short, simple operations.

## Basic Syntax

\`\`\`python
# Regular function
def square(x):
    return x ** 2

# Lambda equivalent
square = lambda x: x ** 2

print(square(5))  # 25
\`\`\`

**Syntax:** \`lambda arguments: expression\`

## Multiple Parameters

\`\`\`python
# Two parameters
add = lambda x, y: x + y
print(add(3, 5))  # 8

# Three parameters
multiply_three = lambda x, y, z: x * y * z
print(multiply_three(2, 3, 4))  # 24
\`\`\`

## With sorted()

Lambda functions are commonly used with built-in functions like sorted().

\`\`\`python
# Sort list of tuples by second element
pairs = [(1, 'one'), (3, 'three'), (2, 'two')]
sorted_pairs = sorted(pairs, key=lambda x: x[1])
print(sorted_pairs)  # [(1, 'one'), (3, 'three'), (2, 'two')]

# Sort by absolute value
numbers = [-4, -1, 3, -2, 5]
sorted_nums = sorted(numbers, key=lambda x: abs(x))
print(sorted_nums)  # [-1, -2, 3, -4, 5]

# Sort strings by length
words = ['python', 'is', 'awesome']
sorted_words = sorted(words, key=lambda x: len(x))
print(sorted_words)  # ['is', 'python', 'awesome']
\`\`\`

## With map()

Apply function to every item in an iterable.

\`\`\`python
# Square all numbers
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# Convert to uppercase
words = ['hello', 'world']
upper = list(map(lambda x: x.upper(), words))
print(upper)  # ['HELLO', 'WORLD']
\`\`\`

## With filter()

Filter items based on condition.

\`\`\`python
# Get even numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8]

# Get positive numbers
numbers = [-2, -1, 0, 1, 2]
positive = list(filter(lambda x: x > 0, numbers))
print(positive)  # [1, 2]

# Get strings longer than 3 characters
words = ['a', 'ab', 'abc', 'abcd']
long_words = list(filter(lambda x: len(x) > 3, words))
print(long_words)  # ['abcd']
\`\`\`

## With reduce()

Apply function cumulatively to reduce iterable to single value.

\`\`\`python
from functools import reduce

# Sum all numbers
numbers = [1, 2, 3, 4, 5]
total = reduce(lambda x, y: x + y, numbers)
print(total)  # 15

# Find maximum
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
maximum = reduce(lambda x, y: x if x > y else y, numbers)
print(maximum)  # 9

# Concatenate strings
words = ['Hello', ' ', 'World', '!']
sentence = reduce(lambda x, y: x + y, words)
print(sentence)  # 'Hello World!'
\`\`\`

## With max() and min()

\`\`\`python
# Find longest word
words = ['python', 'java', 'javascript', 'c']
longest = max(words, key=lambda x: len(x))
print(longest)  # 'javascript'

# Find person with highest score
people = [
    {'name': 'Alice', 'score': 85},
    {'name': 'Bob', 'score': 92},
    {'name': 'Charlie', 'score': 78}
]
top_scorer = max(people, key=lambda x: x['score'])
print(top_scorer)  # {'name': 'Bob', 'score': 92}
\`\`\`

## Conditional Lambda

\`\`\`python
# If-else in lambda
absolute = lambda x: x if x >= 0 else -x
print(absolute(-5))  # 5
print(absolute(3))   # 3

# Ternary with multiple conditions
grade = lambda score: 'A' if score >= 90 else 'B' if score >= 80 else 'C'
print(grade(95))  # 'A'
print(grade(85))  # 'B'
print(grade(75))  # 'C'
\`\`\`

## Lambda Limitations

❌ **Can't do:**
- Multiple statements
- Annotations
- Complex logic
- Assignments within lambda

\`\`\`python
# ❌ Invalid - multiple statements
# bad = lambda x: x += 1; return x  # Syntax Error

# ❌ Invalid - assignment
# bad = lambda x: y = x * 2; y  # Syntax Error
\`\`\`

✅ **Can do:**
- Single expressions
- Conditional expressions
- Function calls
- Simple operations

## When to Use Lambda vs Regular Functions

### Use Lambda When:
- Function is simple (one line)
- Used once or in specific context
- Passed to higher-order functions (map, filter, sorted)
- Callback functions

\`\`\`python
# Good use of lambda
numbers.sort(key=lambda x: abs(x))
squared = map(lambda x: x ** 2, numbers)
\`\`\`

### Use Regular Function When:
- Function is complex
- Used multiple times
- Needs docstring
- Requires multiple statements

\`\`\`python
# Better as regular function
def process_data(data):
    """Process and validate data."""
    if not data:
        return []
    cleaned = [item.strip() for item in data]
    validated = [item for item in cleaned if len(item) > 0]
    return sorted(validated)
\`\`\`

## Lambda vs List Comprehension

\`\`\`python
# Both achieve same result
numbers = [1, 2, 3, 4, 5]

# With lambda and map
squared1 = list(map(lambda x: x ** 2, numbers))

# With list comprehension (more Pythonic)
squared2 = [x ** 2 for x in numbers]

# List comprehension is preferred in Python
\`\`\`

## Common Patterns

### Sort by Multiple Keys
\`\`\`python
people = [('Alice', 25), ('Bob', 30), ('Charlie', 25)]
# Sort by age, then name
sorted_people = sorted(people, key=lambda x: (x[1], x[0]))
\`\`\`

### Custom Comparison
\`\`\`python
# Sort in descending order
numbers = [3, 1, 4, 1, 5]
desc = sorted(numbers, key=lambda x: -x)
\`\`\`

### Data Transformation
\`\`\`python
# Extract specific fields
users = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}]
names = list(map(lambda u: u['name'], users))
\`\`\`

## Quick Reference

| Use Case | Example |
|----------|---------|
| **Basic lambda** | \`lambda x: x * 2\` |
| **Multiple args** | \`lambda x, y: x + y\` |
| **With sorted** | \`sorted(list, key=lambda x: x[1])\` |
| **With map** | \`map(lambda x: x**2, list)\` |
| **With filter** | \`filter(lambda x: x>0, list)\` |
| **With reduce** | \`reduce(lambda x,y: x+y, list)\` |
| **Conditional** | \`lambda x: 'yes' if x>0 else 'no'\` |

**Remember:** Lambda functions are tools for simple cases. For anything complex, use a regular function with a descriptive name!`,
      quiz: [
        {
          id: 'q1',
          question:
            'When should you use a lambda function vs a regular function?',
          sampleAnswer:
            'Use lambda for simple, one-line operations that are used once or passed to functions like map(), filter(), sorted(). Use regular functions for complex logic, reusable code, or anything needing documentation. For example, use lambda for sorting by a key (sorted(items, key=lambda x: x[1])), but use a regular function for data processing with multiple steps, validation, and error handling. Lambda is for convenience, not complexity.',
          keyPoints: [
            'Lambda: simple, one-line, single use',
            'Lambda: with map, filter, sorted',
            'Regular: complex logic, multiple steps',
            'Regular: reusable, needs documentation',
            'Lambda is convenience, not replacement',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the difference between map() with lambda and list comprehension. Which is more Pythonic?',
          sampleAnswer:
            'Both transform lists but with different syntax. map() with lambda: result = list(map(lambda x: x**2, numbers)). List comprehension: result = [x**2 for x in numbers]. List comprehension is more Pythonic because it is more readable, self-contained, and supports filtering inline (e.g., [x**2 for x in numbers if x > 0]). Map with lambda requires wrapping in list() and filter() separately. However, map can be slightly faster for large datasets and more composable with other functional tools. In practice, Python community prefers list/dict comprehensions for readability.',
          keyPoints: [
            'map(lambda x: x**2, nums) vs [x**2 for x in nums]',
            'List comprehension more Pythonic and readable',
            'Comprehension supports inline filtering',
            'map() can be faster for very large data',
            'Python community prefers comprehensions',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the limitations of lambda functions and why do they exist?',
          sampleAnswer:
            'Lambda limitations: (1) Only single expression, no statements, (2) No assignments inside, (3) No annotations, (4) No docstrings, (5) Harder to debug (shows as <lambda> in tracebacks). These limitations exist by design to keep lambdas simple. They force you to use named functions for complex logic, improving code readability and maintainability. If you need multiple steps, assignments, or complex logic, Python wants you to use def with a descriptive name. This prevents cryptic, unreadable code. Lambdas are for throwaway convenience, not as a replacement for proper functions.',
          keyPoints: [
            'Only single expression allowed',
            'No statements, assignments, or annotations',
            'No docstrings, harder to debug',
            'Limitations by design for simplicity',
            'Forces named functions for complex logic',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is a lambda function?',
          options: [
            'A regular function',
            'An anonymous single-expression function',
            'A class method',
            'A module',
          ],
          correctAnswer: 1,
          explanation:
            'Lambda functions are anonymous (unnamed) functions defined in a single expression using the lambda keyword.',
        },
        {
          id: 'mc2',
          question: 'What is the syntax for a lambda function?',
          options: [
            'lambda x: x*2',
            'def lambda(x): x*2',
            'function(x) => x*2',
            'x => x*2',
          ],
          correctAnswer: 0,
          explanation:
            'Lambda syntax: lambda parameters: expression. For example: lambda x: x*2',
        },
        {
          id: 'mc3',
          question: 'Can a lambda function contain multiple statements?',
          options: [
            'Yes, any number',
            'Yes, up to 10',
            'No, only single expression',
            'Only if using semicolons',
          ],
          correctAnswer: 2,
          explanation:
            'Lambda functions can only contain a single expression, not statements. For complex logic, use regular functions.',
        },
        {
          id: 'mc4',
          question: 'Which function commonly uses lambda as a parameter?',
          options: ['print()', 'sorted()', 'len()', 'input()'],
          correctAnswer: 1,
          explanation:
            'sorted(), map(), filter(), reduce() commonly use lambda functions for key/operation parameters.',
        },
        {
          id: 'mc5',
          question: 'When should you avoid lambda functions?',
          options: [
            'With sorted()',
            'For complex multi-step logic',
            'With map()',
            'For simple operations',
          ],
          correctAnswer: 1,
          explanation:
            'Avoid lambdas for complex logic - use regular named functions for readability and debuggability.',
        },
      ],
    },
    {
      id: 'built-in-functions',
      title: 'Essential Built-in Functions',
      content: `# Essential Built-in Functions

Python provides many powerful built-in functions. Here are the most important ones you should know.

## Iteration Functions

### enumerate()

Add counter to an iterable.

\`\`\`python
# Without enumerate
fruits = ['apple', 'banana', 'cherry']
for i in range(len(fruits)):
    print(f"{i}: {fruits[i]}")

# With enumerate (better!)
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
# Output:
# 0: apple
# 1: banana
# 2: cherry

# Start counting from 1
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}. {fruit}")
# Output:
# 1. apple
# 2. banana
# 3. cherry
\`\`\`

### zip()

Combine multiple iterables element-wise.

\`\`\`python
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
cities = ['NYC', 'LA', 'Chicago']

# Zip together
for name, age, city in zip(names, ages, cities):
    print(f"{name} is {age} and lives in {city}")

# Create dictionary
person_dict = dict(zip(names, ages))
print(person_dict)  # {'Alice': 25, 'Bob': 30, 'Charlie': 35}

# Unzip (transpose)
pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
numbers, letters = zip(*pairs)
print(numbers)  # (1, 2, 3)
print(letters)  # ('a', 'b', 'c')
\`\`\`

**Note:** \`zip\` stops at shortest iterable

\`\`\`python
names = ['Alice', 'Bob']
ages = [25, 30, 35]
result = list(zip(names, ages))
print(result)  # [('Alice', 25), ('Bob', 30)] - only 2 pairs
\`\`\`

### range()

Generate sequence of numbers.

\`\`\`python
# range(stop)
list(range(5))  # [0, 1, 2, 3, 4]

# range(start, stop)
list(range(2, 7))  # [2, 3, 4, 5, 6]

# range(start, stop, step)
list(range(0, 10, 2))  # [0, 2, 4, 6, 8]

# Reverse
list(range(10, 0, -1))  # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
\`\`\`

## Aggregation Functions

### sum()

\`\`\`python
numbers = [1, 2, 3, 4, 5]
print(sum(numbers))  # 15

# With start value
print(sum(numbers, 10))  # 25 (sum + 10)

# Sum of squares
print(sum(x ** 2 for x in numbers))  # 55
\`\`\`

### min() and max()

\`\`\`python
numbers = [3, 1, 4, 1, 5, 9, 2]

print(min(numbers))  # 1
print(max(numbers))  # 9

# With strings (lexicographic)
words = ['apple', 'banana', 'cherry']
print(min(words))  # 'apple'
print(max(words))  # 'cherry'

# With key function
people = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30},
    {'name': 'Charlie', 'age': 20}
]
youngest = min(people, key=lambda x: x['age'])
print(youngest)  # {'name': 'Charlie', 'age': 20}

oldest = max(people, key=lambda x: x['age'])
print(oldest)  # {'name': 'Bob', 'age': 30}
\`\`\`

### any() and all()

Test if any/all elements are truthy.

\`\`\`python
# any() - True if at least one element is True
print(any([False, False, True]))  # True
print(any([False, False, False]))  # False
print(any([]))  # False (empty)

# Check if any number is even
numbers = [1, 3, 5, 8, 9]
has_even = any(n % 2 == 0 for n in numbers)
print(has_even)  # True

# all() - True if all elements are True
print(all([True, True, True]))  # True
print(all([True, False, True]))  # False
print(all([]))  # True (empty!)

# Check if all numbers are positive
numbers = [1, 2, 3, 4, 5]
all_positive = all(n > 0 for n in numbers)
print(all_positive)  # True
\`\`\`

## Transformation Functions

### map()

Apply function to every item.

\`\`\`python
# Square all numbers
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# Convert to integers
strings = ['1', '2', '3']
integers = list(map(int, strings))
print(integers)  # [1, 2, 3]

# Multiple iterables
a = [1, 2, 3]
b = [10, 20, 30]
sums = list(map(lambda x, y: x + y, a, b))
print(sums)  # [11, 22, 33]
\`\`\`

### filter()

Filter items by condition.

\`\`\`python
# Get even numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8]

# Filter None values
values = [1, None, 3, None, 5]
filtered = list(filter(None, values))
print(filtered)  # [1, 3, 5]

# Filter empty strings
strings = ['hello', '', 'world', '', 'python']
non_empty = list(filter(len, strings))
print(non_empty)  # ['hello', 'world', 'python']
\`\`\`

### sorted()

Return sorted list (original unchanged).

\`\`\`python
# Basic sorting
numbers = [3, 1, 4, 1, 5, 9]
sorted_nums = sorted(numbers)
print(sorted_nums)  # [1, 1, 3, 4, 5, 9]

# Reverse
sorted_desc = sorted(numbers, reverse=True)
print(sorted_desc)  # [9, 5, 4, 3, 1, 1]

# Custom key
words = ['apple', 'pie', 'a', 'glance']
by_length = sorted(words, key=len)
print(by_length)  # ['a', 'pie', 'apple', 'glance']

# Complex sorting
people = [('Alice', 25), ('Bob', 30), ('Charlie', 20)]
by_age = sorted(people, key=lambda x: x[1])
print(by_age)  # [('Charlie', 20), ('Alice', 25), ('Bob', 30)]
\`\`\`

## Type Checking

### isinstance()

Check if object is instance of class.

\`\`\`python
x = 5
print(isinstance(x, int))  # True
print(isinstance(x, str))  # False

# Multiple types
print(isinstance(x, (int, float)))  # True

# Custom classes
class Dog:
    pass

my_dog = Dog()
print(isinstance(my_dog, Dog))  # True
\`\`\`

### type()

Get type of object.

\`\`\`python
print(type(5))  # <class 'int'>
print(type(5.0))  # <class 'float'>
print(type('hello'))  # <class 'str'>
print(type([1, 2, 3]))  # <class 'list'>

# Comparison (use isinstance instead)
x = 5
if type(x) == int:  # Works but not recommended
    print("x is an integer")

# Better:
if isinstance(x, int):
    print("x is an integer")
\`\`\`

## String Conversion

### str(), int(), float()

\`\`\`python
# To string
print(str(123))  # '123'
print(str(3.14))  # '3.14'
print(str([1, 2, 3]))  # '[1, 2, 3]'

# To integer
print(int('123'))  # 123
print(int(3.14))  # 3 (truncates)
print(int('FF', 16))  # 255 (hex to int)

# To float
print(float('3.14'))  # 3.14
print(float('123'))  # 123.0
\`\`\`

## Utility Functions

### len()

Get length of sequence.

\`\`\`python
print(len([1, 2, 3]))  # 3
print(len('hello'))  # 5
print(len({'a': 1, 'b': 2}))  # 2
\`\`\`

### reversed()

Reverse an iterable (returns iterator).

\`\`\`python
numbers = [1, 2, 3, 4, 5]
reversed_nums = list(reversed(numbers))
print(reversed_nums)  # [5, 4, 3, 2, 1]

# With strings
word = 'hello'
backwards = ''.join(reversed(word))
print(backwards)  # 'olleh'
\`\`\`

### abs()

Absolute value.

\`\`\`python
print(abs(-5))  # 5
print(abs(3.14))  # 3.14
print(abs(-2.5))  # 2.5
\`\`\`

### round()

Round number.

\`\`\`python
print(round(3.14159))  # 3
print(round(3.14159, 2))  # 3.14
print(round(3.14159, 3))  # 3.142

# Round to nearest 10
print(round(123, -1))  # 120
print(round(127, -1))  # 130
\`\`\`

### pow()

Power function.

\`\`\`python
print(pow(2, 3))  # 8 (2^3)
print(pow(5, 2))  # 25 (5^2)

# With modulo (for large numbers)
print(pow(2, 10, 100))  # 24 (2^10 % 100)
\`\`\`

## Quick Reference

| Function | Purpose | Example |
|----------|---------|---------|
| \`enumerate()\` | Add counter | \`enumerate(['a','b'])\` → \`(0,'a'), (1,'b')\` |
| \`zip()\` | Combine iterables | \`zip([1,2], ['a','b'])\` → \`(1,'a'), (2,'b')\` |
| \`sum()\` | Sum numbers | \`sum([1,2,3])\` → \`6\` |
| \`min()/max()\` | Find min/max | \`min([3,1,2])\` → \`1\` |
| \`any()/all()\` | Boolean checks | \`any([False, True])\` → \`True\` |
| \`map()\` | Apply function | \`map(str, [1,2,3])\` → \`'1','2','3'\` |
| \`filter()\` | Filter items | \`filter(None, [0,1,2])\` → \`1,2\` |
| \`sorted()\` | Sort items | \`sorted([3,1,2])\` → \`[1,2,3]\` |
| \`len()\` | Get length | \`len([1,2,3])\` → \`3\` |
| \`isinstance()\` | Check type | \`isinstance(5, int)\` → \`True\` |

**Pro Tip:** These functions make code more Pythonic. Use them instead of manual loops when possible!`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the difference between map()/filter() and list comprehensions. Which is more Pythonic?',
          sampleAnswer:
            'map() and filter() are functional programming tools that return iterators, while list comprehensions directly create lists and are considered more Pythonic. For example, map(lambda x: x**2, nums) vs [x**2 for x in nums], and filter(lambda x: x>0, nums) vs [x for x in nums if x>0]. List comprehensions are generally preferred in Python because they are more readable and slightly faster. However, map() and filter() are useful when you already have a function defined (like map(int, strings)) rather than using a lambda.',
          keyPoints: [
            'map/filter return iterators',
            'List comprehensions create lists directly',
            'Comprehensions more Pythonic and readable',
            'map/filter useful with existing functions',
            'Comprehensions often faster',
          ],
        },
        {
          id: 'q2',
          question: 'When would you use enumerate() vs range(len())?',
          sampleAnswer:
            "enumerate() is preferred over range(len()) because it's more Pythonic and less error-prone. Compare: for i in range(len(items)): print(i, items[i]) vs for i, item in enumerate(items): print(i, item). enumerate() directly gives you both index and value, avoiding indexing errors and making code clearer. It's also more efficient and works with any iterable, not just sequences with indexing. Use enumerate() when you need both index and value; use plain for item in items when you only need values.",
          keyPoints: [
            'enumerate() more Pythonic',
            'Gives index and value directly',
            'Avoids indexing errors',
            'Works with any iterable',
            'More readable than range(len())',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain why all([]) returns True but any([]) returns False.',
          sampleAnswer:
            "all([]) returns True because of vacuous truth in logic: a statement about all elements of an empty set is considered true since there are no counterexamples. Think: 'all numbers in [] are positive' - technically true because there are no numbers to disprove it. any([]) returns False because there's no element to make it True. This matters in code: if all(validations): can pass with empty validations[], but if any(errors): won't trigger with no errors. Be careful with empty sequences - sometimes you want to explicitly check if not items: first.",
          keyPoints: [
            'all([]): True - vacuous truth, no counterexamples',
            'any([]): False - no element to be True',
            'Based on logical quantifiers',
            'Can cause bugs if not expected',
            'Check for empty sequences explicitly when it matters',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What does enumerate() return?',
          options: [
            'Just indices',
            'Just values',
            'Tuples of (index, value)',
            'A dictionary',
          ],
          correctAnswer: 2,
          explanation:
            'enumerate() returns tuples of (index, value) for each item in an iterable.',
        },
        {
          id: 'mc2',
          question: 'What does zip() do with two lists?',
          options: [
            'Compresses them',
            'Pairs corresponding elements',
            'Combines into one list',
            'Sorts both',
          ],
          correctAnswer: 1,
          explanation:
            'zip() pairs corresponding elements: zip([1,2], ["a","b"]) → [(1,"a"), (2,"b")]',
        },
        {
          id: 'mc3',
          question: 'What is the difference between any() and all()?',
          options: [
            'No difference',
            'any() if ANY True, all() if ALL True',
            'any() is faster',
            'all() works with numbers only',
          ],
          correctAnswer: 1,
          explanation:
            'any() returns True if at least one element is truthy; all() only if all are truthy.',
        },
        {
          id: 'mc4',
          question: 'What does map(func, iterable) return?',
          options: ['A list', 'A map object (iterator)', 'A tuple', 'A set'],
          correctAnswer: 1,
          explanation:
            'map() returns a map object (iterator). Use list(map(...)) to get a list.',
        },
        {
          id: 'mc5',
          question: 'How does isinstance() differ from type()?',
          options: [
            'No difference',
            'isinstance() checks class hierarchy, type() exact type',
            'isinstance() deprecated',
            'type() is faster',
          ],
          correctAnswer: 1,
          explanation:
            'isinstance(obj, Class) checks class hierarchy. type(obj) == Class checks exact type only.',
        },
      ],
    },
  ],
};

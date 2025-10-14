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
  ],
};

/**
 * List Comprehensions Section
 */

export const listcomprehensionsSection = {
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
};

/**
 * Lambda Functions Section
 */

export const lambdafunctionsSection = {
  id: 'lambda-functions',
  title: 'Lambda Functions',
  content: `# Lambda Functions

Lambda functions are small anonymous functions defined with the \`lambda\` keyword. They're useful for short, simple operations.

## Basic Syntax

\`\`\`python
# Regular function
def square (x):
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
sorted_pairs = sorted (pairs, key=lambda x: x[1])
print(sorted_pairs)  # [(1, 'one'), (3, 'three'), (2, 'two')]

# Sort by absolute value
numbers = [-4, -1, 3, -2, 5]
sorted_nums = sorted (numbers, key=lambda x: abs (x))
print(sorted_nums)  # [-1, -2, 3, -4, 5]

# Sort strings by length
words = ['python', 'is', 'awesome']
sorted_words = sorted (words, key=lambda x: len (x))
print(sorted_words)  # ['is', 'python', 'awesome']
\`\`\`

## With map()

Apply function to every item in an iterable.

\`\`\`python
# Square all numbers
numbers = [1, 2, 3, 4, 5]
squared = list (map (lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# Convert to uppercase
words = ['hello', 'world']
upper = list (map (lambda x: x.upper(), words))
print(upper)  # ['HELLO', 'WORLD']
\`\`\`

## With filter()

Filter items based on condition.

\`\`\`python
# Get even numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8]
evens = list (filter (lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8]

# Get positive numbers
numbers = [-2, -1, 0, 1, 2]
positive = list (filter (lambda x: x > 0, numbers))
print(positive)  # [1, 2]

# Get strings longer than 3 characters
words = ['a', 'ab', 'abc', 'abcd']
long_words = list (filter (lambda x: len (x) > 3, words))
print(long_words)  # ['abcd']
\`\`\`

## With reduce()

Apply function cumulatively to reduce iterable to single value.

\`\`\`python
from functools import reduce

# Sum all numbers
numbers = [1, 2, 3, 4, 5]
total = reduce (lambda x, y: x + y, numbers)
print(total)  # 15

# Find maximum
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
maximum = reduce (lambda x, y: x if x > y else y, numbers)
print(maximum)  # 9

# Concatenate strings
words = ['Hello', ' ', 'World', '!']
sentence = reduce (lambda x, y: x + y, words)
print(sentence)  # 'Hello World!'
\`\`\`

## With max() and min()

\`\`\`python
# Find longest word
words = ['python', 'java', 'javascript', 'c']
longest = max (words, key=lambda x: len (x))
print(longest)  # 'javascript'

# Find person with highest score
people = [
    {'name': 'Alice', 'score': 85},
    {'name': 'Bob', 'score': 92},
    {'name': 'Charlie', 'score': 78}
]
top_scorer = max (people, key=lambda x: x['score'])
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
numbers.sort (key=lambda x: abs (x))
squared = map (lambda x: x ** 2, numbers)
\`\`\`

### Use Regular Function When:
- Function is complex
- Used multiple times
- Needs docstring
- Requires multiple statements

\`\`\`python
# Better as regular function
def process_data (data):
    """Process and validate data."""
    if not data:
        return []
    cleaned = [item.strip() for item in data]
    validated = [item for item in cleaned if len (item) > 0]
    return sorted (validated)
\`\`\`

## Lambda vs List Comprehension

\`\`\`python
# Both achieve same result
numbers = [1, 2, 3, 4, 5]

# With lambda and map
squared1 = list (map (lambda x: x ** 2, numbers))

# With list comprehension (more Pythonic)
squared2 = [x ** 2 for x in numbers]

# List comprehension is preferred in Python
\`\`\`

## Common Patterns

### Sort by Multiple Keys
\`\`\`python
people = [('Alice', 25), ('Bob', 30), ('Charlie', 25)]
# Sort by age, then name
sorted_people = sorted (people, key=lambda x: (x[1], x[0]))
\`\`\`

### Custom Comparison
\`\`\`python
# Sort in descending order
numbers = [3, 1, 4, 1, 5]
desc = sorted (numbers, key=lambda x: -x)
\`\`\`

### Data Transformation
\`\`\`python
# Extract specific fields
users = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}]
names = list (map (lambda u: u['name'], users))
\`\`\`

## Quick Reference

| Use Case | Example |
|----------|---------|
| **Basic lambda** | \`lambda x: x * 2\` |
| **Multiple args** | \`lambda x, y: x + y\` |
| **With sorted** | \`sorted (list, key=lambda x: x[1])\` |
| **With map** | \`map (lambda x: x**2, list)\` |
| **With filter** | \`filter (lambda x: x>0, list)\` |
| **With reduce** | \`reduce (lambda x,y: x+y, list)\` |
| **Conditional** | \`lambda x: 'yes' if x>0 else 'no'\` |

**Remember:** Lambda functions are tools for simple cases. For anything complex, use a regular function with a descriptive name!`,
};

/**
 * Variables and Data Types Section
 */

export const variablestypesSection = {
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
};

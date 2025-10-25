/**
 * Data Structures Section
 */

export const datastructuresSection = {
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
len (numbers)           # Length
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
def get_min_max (numbers):
    return min (numbers), max (numbers)

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
word_lengths = {word: len (word) for word in ["hi", "hello", "hey"]}

# Set comprehension
unique_lengths = {len (word) for word in ["hi", "hello", "hey"]}
\`\`\``,
  videoUrl: 'https://www.youtube.com/watch?v=W8KRzm-HUcc',
};

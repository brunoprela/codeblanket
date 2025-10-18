/**
 * Essential Built-in Functions Section
 */

export const builtinfunctionsSection = {
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
};

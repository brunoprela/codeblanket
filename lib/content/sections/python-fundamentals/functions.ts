/**
 * Functions Section
 */

export const functionsSection = {
  id: 'functions',
  title: 'Functions',
  content: `# Functions

## Defining Functions

\`\`\`python
def greet (name):
    """Function with one parameter."""
    return f"Hello, {name}!"

# Calling the function
message = greet("Alice")
print(message)  # Hello, Alice!
\`\`\`

## Parameters and Arguments

\`\`\`python
# Default parameters
def power (base, exponent=2):
    return base ** exponent

power(3)      # 9 (uses default exponent=2)
power(3, 3)   # 27

# Keyword arguments
def describe_pet (animal, name):
    print(f"I have a {animal} named {name}")

describe_pet (animal="dog", name="Max")
describe_pet (name="Max", animal="dog")  # Order doesn't matter

# *args - variable number of positional arguments
def sum_all(*numbers):
    return sum (numbers)

sum_all(1, 2, 3, 4)  # 10

# **kwargs - variable number of keyword arguments
def print_info(**info):
    for key, value in info.items():
        print(f"{key}: {value}")

print_info (name="Alice", age=30, city="NYC")
\`\`\`

## Return Values

\`\`\`python
# Single return value
def square (x):
    return x ** 2

# Multiple return values (tuple)
def get_min_max (numbers):
    return min (numbers), max (numbers)

minimum, maximum = get_min_max([1, 2, 3, 4, 5])

# No return (returns None)
def print_greeting (name):
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

squared = list (map (lambda x: x**2, numbers))
evens = list (filter (lambda x: x % 2 == 0, numbers))

# Sorting with key
people = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
sorted_by_age = sorted (people, key=lambda person: person[1])
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
def calculate_area (length, width):
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
};

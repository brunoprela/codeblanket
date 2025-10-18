/**
 * String Operations Section
 */

export const stringsSection = {
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
};

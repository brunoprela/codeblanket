/**
 * Python Intermediate - Building practical skills beyond the basics
 */

import { Module } from '../types';

export const pythonIntermediateModule: Module = {
  id: 'python-intermediate',
  title: 'Python Intermediate',
  description:
    'Build practical Python skills with file handling, error management, regular expressions, and more.',
  category: 'Python',
  difficulty: 'Intermediate',
  estimatedTime: '10 hours',
  prerequisites: ['python-fundamentals'],
  icon: '🔧',
  keyTakeaways: [
    'Read and write files using context managers',
    'Handle errors gracefully with try-except blocks',
    'Parse and validate data with regular expressions',
    'Work with JSON and CSV data formats',
    'Manipulate dates and times effectively',
    'Create custom exception classes',
    'Build maintainable Python applications',
  ],
  learningObjectives: [
    'Master file I/O operations and context managers',
    'Handle exceptions and create custom error types',
    'Use regular expressions for text processing',
    'Work with JSON and CSV data formats',
    'Understand and use Python modules effectively',
    'Apply functional programming concepts',
    'Handle dates and times in Python',
    'Build simple classes and objects',
    'Process command-line arguments',
    'Validate and transform data',
  ],
  sections: [
    {
      id: 'file-handling',
      title: 'File Handling and I/O',
      content: `# File Handling and I/O

## Opening and Reading Files

\`\`\`python
# Basic file reading
with open('file.txt', 'r') as f:
    content = f.read()  # Read entire file
    
# Read line by line
with open('file.txt', 'r') as f:
    for line in f:
        print(line.strip())

# Read all lines into list
with open('file.txt', 'r') as f:
    lines = f.readlines()

# Read specific number of characters
with open('file.txt', 'r') as f:
    chunk = f.read(100)  # Read first 100 characters
\`\`\`

## Writing to Files

\`\`\`python
# Write (overwrites existing content)
with open('output.txt', 'w') as f:
    f.write('Hello, World!\\n')
    f.write('Second line\\n')

# Append to file
with open('output.txt', 'a') as f:
    f.write('Appended line\\n')

# Write multiple lines
lines = ['Line 1\\n', 'Line 2\\n', 'Line 3\\n']
with open('output.txt', 'w') as f:
    f.writelines(lines)
\`\`\`

## File Modes

- **'r'**: Read (default) - file must exist
- **'w'**: Write - creates new or truncates existing
- **'a'**: Append - adds to end of file
- **'r+'**: Read and write
- **'rb'**: Read binary
- **'wb'**: Write binary

## Context Managers (with statement)

\`\`\`python
# Automatically closes file
with open('file.txt', 'r') as f:
    data = f.read()
# File is closed here

# Multiple files
with open('input.txt', 'r') as fin, open('output.txt', 'w') as fout:
    for line in fin:
        fout.write(line.upper())
\`\`\`

## File Operations

\`\`\`python
import os

# Check if file exists
if os.path.exists('file.txt'):
    print('File exists')

# Get file size
size = os.path.getsize('file.txt')

# Rename file
os.rename('old.txt', 'new.txt')

# Delete file
os.remove('file.txt')

# Get file info
import os.path
modified_time = os.path.getmtime('file.txt')
is_file = os.path.isfile('file.txt')
is_dir = os.path.isdir('folder')
\`\`\`

## Working with Paths

\`\`\`python
from pathlib import Path

# Modern way to handle paths
p = Path('folder/subfolder/file.txt')

# Check existence
if p.exists():
    print('Exists')

# Read/write with Path
content = p.read_text()
p.write_text('New content')

# Path operations
print(p.name)        # 'file.txt'
print(p.stem)        # 'file'
print(p.suffix)      # '.txt'
print(p.parent)      # 'folder/subfolder'

# Join paths
new_path = Path('folder') / 'subfolder' / 'file.txt'
\`\`\`

## Best Practices

1. **Always use context managers** (with statement)
2. **Handle encoding**: \`open('file.txt', 'r', encoding='utf-8')\`
3. **Close files** if not using with statement
4. **Check file existence** before operations
5. **Use pathlib** for modern path handling`,
      videoUrl: 'https://www.youtube.com/watch?v=Uh2ebFW8OYM',
      quiz: [
        {
          id: 'pi-filehandling-q-1',
          question:
            'Explain the difference between text mode and binary mode when working with files. When should you use each?',
          hint: 'Think about encoding, data types, and what kinds of files you might read.',
          sampleAnswer:
            'Text mode (default) reads files as strings and handles encoding/decoding automatically (usually UTF-8). Use it for .txt, .py, .csv, .json files. Binary mode (rb/wb) reads files as bytes without encoding - use it for images, videos, executables, or when you need exact byte-level control. Binary mode is crucial when file encoding is unknown or for non-text data. Opening a binary file in text mode can corrupt data or raise encoding errors.',
          keyPoints: [
            'Text mode: strings, automatic encoding/decoding',
            'Binary mode: bytes, no encoding',
            'Use text for human-readable files',
            'Use binary for images, executables, unknown encoding',
          ],
        },
        {
          id: 'pi-filehandling-q-2',
          question:
            'Why is using "with" statement crucial for file operations? What happens if you forget to close a file?',
          hint: 'Consider resource management, exceptions, and OS limitations.',
          sampleAnswer:
            'The "with" statement (context manager) automatically closes files even if exceptions occur, preventing resource leaks. Without it, files might stay open if your code crashes, potentially causing: 1) Resource exhaustion (OS limits on open files), 2) File locking issues preventing other processes from accessing the file, 3) Data not being flushed to disk (buffered writes), 4) Memory leaks. Always use "with" - it\'s the Pythonic way and ensures proper cleanup.',
          keyPoints: [
            'Automatically closes files, even on exceptions',
            'Prevents resource leaks and file locking',
            'Ensures buffered data is flushed to disk',
            'Context manager protocol (__enter__/__exit__)',
          ],
        },
        {
          id: 'pi-filehandling-q-3',
          question:
            'What are the advantages of using pathlib over os.path? Should you still learn os.path?',
          hint: 'Consider API design, readability, and cross-platform compatibility.',
          sampleAnswer:
            'pathlib (Python 3.4+) provides object-oriented path manipulation with cleaner syntax: path / "subdir" / "file.txt" instead of os.path.join(). It has convenient methods like .read_text(), .write_text(), .glob(), and works seamlessly across platforms. However, os.path is still useful for: 1) Legacy code, 2) Some specific operations not in pathlib, 3) When you need string paths for compatibility. Learn both - pathlib for new code, os.path to understand existing code.',
          keyPoints: [
            'pathlib: object-oriented, cleaner syntax',
            'Supports / operator for path joining',
            'Built-in methods for common operations',
            'os.path still needed for legacy compatibility',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'pi-filehandling-mc-1',
          question:
            'What happens if you open a file in "w" mode that already exists?',
          options: [
            'Error is raised',
            'Content is appended',
            'File is truncated (emptied)',
            'File is renamed',
          ],
          correctAnswer: 2,
          explanation:
            "'w' mode truncates (empties) existing files before writing.",
        },
        {
          id: 'pi-filehandling-mc-2',
          question: 'What does the "with" statement do when opening files?',
          options: [
            'Opens file faster',
            'Automatically closes the file',
            'Makes file read-only',
            'Encrypts the file',
          ],
          correctAnswer: 1,
          explanation:
            'The with statement (context manager) automatically closes the file when the block exits, even if an exception occurs.',
        },
        {
          id: 'pi-filehandling-mc-3',
          question:
            'Which mode should you use to add content to the end of a file?',
          options: ["'r'", "'w'", "'a'", "'x'"],
          correctAnswer: 2,
          explanation:
            "'a' mode opens the file for appending, adding new content to the end without removing existing content.",
        },
        {
          id: 'pi-filehandling-mc-4',
          question:
            'What is the difference between open() and Path.read_text()?',
          options: [
            'No difference',
            'Path.read_text() automatically handles opening and closing',
            'open() is faster',
            'Path.read_text() only works with binary files',
          ],
          correctAnswer: 1,
          explanation:
            'Path.read_text() from pathlib automatically opens, reads, and closes the file in one operation.',
        },
        {
          id: 'pi-filehandling-mc-5',
          question:
            "What happens if you try to read a file that doesn't exist?",
          options: [
            'Returns empty string',
            'Returns None',
            'Raises FileNotFoundError',
            'Creates the file',
          ],
          correctAnswer: 2,
          explanation:
            'Attempting to open a non-existent file in read mode raises a FileNotFoundError.',
        },
      ],
    },
    {
      id: 'exceptions',
      title: 'Exception Handling',
      content: `# Exception Handling

## Try-Except Basics

\`\`\`python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
try:
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Invalid number!")
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Catch multiple exception types
try:
    # risky code
    pass
except (ValueError, TypeError) as e:
    print(f"Error: {e}")
\`\`\`

## Else and Finally

\`\`\`python
try:
    file = open('data.txt', 'r')
    data = file.read()
except FileNotFoundError:
    print("File not found")
else:
    # Runs if no exception occurred
    print("File read successfully")
finally:
    # Always runs, even if exception occurred
    if 'file' in locals():
        file.close()
\`\`\`

## Raising Exceptions

\`\`\`python
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age seems unrealistic")
    return age

# Re-raising exceptions
try:
    # some code
    pass
except Exception as e:
    print(f"Logging error: {e}")
    raise  # Re-raise the same exception
\`\`\`

## Custom Exceptions

\`\`\`python
class InsufficientFundsError(Exception):
    """Raised when account has insufficient funds"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        message = f"Insufficient funds: need {amount}, have {balance}"
        super().__init__(message)

class Account:
    def __init__(self, balance):
        self.balance = balance
    
    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(self.balance, amount)
        self.balance -= amount

# Usage
account = Account(100)
try:
    account.withdraw(150)
except InsufficientFundsError as e:
    print(e)
\`\`\`

## Common Built-in Exceptions

- **ValueError**: Invalid value
- **TypeError**: Wrong type
- **KeyError**: Key not found in dictionary
- **IndexError**: List index out of range
- **FileNotFoundError**: File doesn't exist
- **ZeroDivisionError**: Division by zero
- **AttributeError**: Attribute doesn't exist
- **ImportError**: Module import fails

## Exception Hierarchy

\`\`\`python
# Catch more specific exceptions first
try:
    # code
    pass
except FileNotFoundError:
    # Specific exception
    print("File not found")
except IOError:
    # More general exception
    print("I/O error")
except Exception:
    # Catch-all (use sparingly)
    print("Something went wrong")
\`\`\`

## Best Practices

1. **Be specific**: Catch specific exceptions, not generic \`Exception\`
2. **Don't silence errors**: Always handle or log exceptions
3. **Use custom exceptions**: For domain-specific errors
4. **Clean up resources**: Use finally or context managers
5. **Don't catch what you can't handle**: Let exceptions propagate
6. **Provide context**: Include helpful error messages`,
      videoUrl: 'https://www.youtube.com/watch?v=NIWwJbo-9_8',
      quiz: [
        {
          id: 'pi-exceptions-q-1',
          question:
            'Explain the difference between catching specific exceptions versus using a bare except or except Exception. When is each appropriate?',
          hint: 'Think about debugging, KeyboardInterrupt, SystemExit, and error handling granularity.',
          sampleAnswer:
            'Always catch specific exceptions (like FileNotFoundError, ValueError) when you know what can go wrong and how to handle it. This makes code more maintainable and prevents hiding bugs. "except Exception" catches most errors but not system-critical ones like KeyboardInterrupt or SystemExit. Bare "except:" catches EVERYTHING including Ctrl+C, which is dangerous. Use specific exceptions for normal error handling, "except Exception" only for top-level logging, and bare except almost never (maybe for ensuring cleanup in critical systems).',
          keyPoints: [
            'Specific exceptions: best for known error handling',
            'except Exception: catches most, but not system exits',
            'Bare except: dangerous, catches KeyboardInterrupt too',
            'More specific = better debugging and maintenance',
          ],
        },
        {
          id: 'pi-exceptions-q-2',
          question:
            'What is the purpose of the finally block? How does it differ from putting code after the try-except block?',
          hint: 'Consider what happens when exceptions are raised or when return statements execute.',
          sampleAnswer:
            "The finally block ALWAYS executes, even if: 1) an exception is raised and not caught, 2) a return statement is executed in try/except, 3) break/continue is used in a loop. Code after try-except only runs if the exception was caught or didn't occur. Use finally for cleanup (closing files, releasing locks, database rollback) that must happen regardless of success or failure. However, context managers (with statement) are often cleaner than try-finally for resource management.",
          keyPoints: [
            'finally: always runs, even on return/break',
            'Code after try-except: only if exception caught',
            'Use for mandatory cleanup (files, locks, connections)',
            'Context managers often better than try-finally',
          ],
        },
        {
          id: 'pi-exceptions-q-3',
          question:
            'When should you create custom exceptions? What makes a good custom exception?',
          hint: 'Consider API design, error handling hierarchy, and what information exceptions should carry.',
          sampleAnswer:
            'Create custom exceptions for domain-specific errors that deserve special handling (like ValidationError, InsufficientFundsError, DatabaseConnectionError). Good custom exceptions: 1) Inherit from appropriate base (Exception or a more specific built-in), 2) Have descriptive names ending in "Error", 3) Can carry context (user_id, amount, etc.), 4) Form a logical hierarchy (APIError → HTTPError → NotFoundError). Don\'t create custom exceptions for every error - use built-ins when they fit (ValueError, TypeError). Custom exceptions make error handling more semantic and maintainable.',
          keyPoints: [
            'Use for domain-specific errors needing special handling',
            'Inherit from Exception or specific built-in',
            'Can carry context data for debugging',
            'Create hierarchy for related errors',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'pi-exceptions-mc-1',
          question:
            'What happens if you raise an exception without catching it?',
          options: [
            'Program continues normally',
            'Program terminates with error',
            'Exception is ignored',
            'Exception becomes a warning',
          ],
          correctAnswer: 1,
          explanation:
            'Uncaught exceptions cause the program to terminate and print a traceback.',
        },
        {
          id: 'pi-exceptions-mc-2',
          question: 'When does the "finally" block execute?',
          options: [
            'Only if no exception occurs',
            'Only if an exception occurs',
            'Always, regardless of exceptions',
            'Never',
          ],
          correctAnswer: 2,
          explanation:
            'The finally block always executes, whether an exception occurred or not, making it ideal for cleanup code.',
        },
        {
          id: 'pi-exceptions-mc-3',
          question: 'What is the purpose of custom exceptions?',
          options: [
            'Make code run faster',
            'Provide domain-specific error handling',
            'Replace built-in exceptions',
            'Prevent all errors',
          ],
          correctAnswer: 1,
          explanation:
            'Custom exceptions help represent domain-specific errors in your application, making error handling more meaningful.',
        },
        {
          id: 'pi-exceptions-mc-4',
          question:
            'Which is the correct way to catch multiple exception types?',
          options: [
            'except ValueError, TypeError:',
            'except (ValueError, TypeError):',
            'except ValueError and TypeError:',
            'except ValueError | TypeError:',
          ],
          correctAnswer: 1,
          explanation:
            'Multiple exception types are caught using a tuple: except (Type1, Type2):',
        },
        {
          id: 'pi-exceptions-mc-5',
          question: 'What does the "else" clause in try-except do?',
          options: [
            'Executes if an exception occurs',
            'Executes if no exception occurs',
            'Always executes',
            'Same as finally',
          ],
          correctAnswer: 1,
          explanation:
            'The else clause executes only if no exception was raised in the try block.',
        },
      ],
    },
    {
      id: 'json-csv',
      title: 'Working with JSON and CSV',
      content: `# Working with JSON and CSV

## JSON Operations

\`\`\`python
import json

# Python to JSON
data = {
    'name': 'Alice',
    'age': 30,
    'skills': ['Python', 'JavaScript'],
    'active': True
}

# Convert to JSON string
json_string = json.dumps(data)
print(json_string)

# Convert to JSON string (pretty-printed)
json_string = json.dumps(data, indent=2)

# Write to JSON file
with open('data.json', 'w') as f:
    json.dump(data, f, indent=2)

# Read from JSON file
with open('data.json', 'r') as f:
    loaded_data = json.load(f)

# Parse JSON string
parsed_data = json.loads(json_string)
\`\`\`

## JSON Type Mapping

Python → JSON:
- dict → object
- list/tuple → array
- str → string
- int/float → number
- True/False → true/false
- None → null

## Working with CSV

\`\`\`python
import csv

# Reading CSV
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Get headers
    for row in reader:
        print(row)  # Each row is a list

# Reading CSV as dictionaries
with open('data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)  # Each row is a dict

# Writing CSV
data = [
    ['Name', 'Age', 'City'],
    ['Alice', 30, 'NYC'],
    ['Bob', 25, 'LA']
]

with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

# Writing CSV from dictionaries
data = [
    {'Name': 'Alice', 'Age': 30, 'City': 'NYC'},
    {'Name': 'Bob', 'Age': 25, 'City': 'LA'}
]

with open('output.csv', 'w', newline='') as f:
    fieldnames = ['Name', 'Age', 'City']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
\`\`\`

## Advanced CSV Options

\`\`\`python
# Custom delimiter
with open('data.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\\t')

# Custom quote character
reader = csv.reader(f, quotechar="'")

# Skip header
reader = csv.reader(f)
next(reader)  # Skip first row
\`\`\`

## Data Validation

\`\`\`python
def validate_json_schema(data, required_keys):
    """Validate JSON data has required keys."""
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    return True

# Usage
user_data = json.loads(json_string)
validate_json_schema(user_data, ['name', 'age', 'email'])
\`\`\`

## Best Practices

1. **Handle encoding**: Specify \`encoding='utf-8'\` for non-ASCII data
2. **Validate data**: Check structure before processing
3. **Use DictReader/DictWriter**: More readable than lists
4. **Handle missing fields**: Provide defaults
5. **Close files**: Use context managers
6. **Pretty-print JSON**: Use \`indent\` for readability`,
      videoUrl: 'https://www.youtube.com/watch?v=pTT7HMqDnJw',
      quiz: [
        {
          id: 'pi-jsoncsv-q-1',
          question:
            'Explain the difference between json.dump()/json.load() and json.dumps()/json.loads(). When would you use each pair?',
          hint: 'Think about what the "s" suffix means - string vs file operations.',
          sampleAnswer:
            'json.dump() and json.load() work with file objects - dump() writes JSON directly to a file, load() reads from a file. json.dumps() and json.loads() work with strings - dumps() converts Python to JSON string, loads() parses JSON string to Python. Use dump/load when working with files (most common), use dumps/loads when: 1) sending JSON over network/API, 2) storing JSON in databases, 3) debugging (printing JSON), 4) when you need the JSON as a string for manipulation.',
          keyPoints: [
            'dump/load: file operations',
            'dumps/loads: string operations',
            'dumps = dump + string, loads = load + string',
            'Use dump/load for files, dumps/loads for APIs/strings',
          ],
        },
        {
          id: 'pi-jsoncsv-q-2',
          question:
            'When should you choose CSV over JSON, and vice versa? What are the trade-offs?',
          hint: 'Consider data structure, file size, human readability, and Excel compatibility.',
          sampleAnswer:
            "Use CSV for: 1) Tabular/flat data with rows and columns, 2) Excel/spreadsheet compatibility, 3) Smaller file sizes for simple data, 4) When everyone has same columns. Use JSON for: 1) Nested/hierarchical data, 2) APIs and web services, 3) Mixed data types (preserves numbers, booleans, nulls), 4) Flexible schemas where objects can have different fields. CSV is simpler but can't handle nesting; JSON is more expressive but larger and requires parsing.",
          keyPoints: [
            'CSV: flat tables, Excel, compact',
            'JSON: nested data, APIs, type preservation',
            'CSV simpler for spreadsheets',
            'JSON better for complex, hierarchical data',
          ],
        },
        {
          id: 'pi-jsoncsv-q-3',
          question:
            'Why is DictReader/DictWriter often preferred over basic reader/writer in the CSV module?',
          hint: 'Think about code readability, maintainability, and column ordering.',
          sampleAnswer:
            'DictReader/DictWriter use column names (headers) instead of positional indices, making code more readable and maintainable. Benefits: 1) row["name"] is clearer than row[0], 2) Code doesn\'t break if column order changes, 3) Automatic header handling, 4) Easier to work with when columns are added/removed. Trade-offs: slightly more memory/processing, but worth it for code clarity. Only use basic reader/writer for headerless CSV or when performance is absolutely critical.',
          keyPoints: [
            'Uses column names instead of indices',
            'More readable: row["name"] vs row[0]',
            'Resilient to column reordering',
            'Slightly more overhead, but worth it',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'pi-jsoncsv-mc-1',
          question: 'Why use newline="" when writing CSV files?',
          options: [
            'Makes file smaller',
            'Prevents extra blank lines on Windows',
            'Required for UTF-8',
            'Speeds up writing',
          ],
          correctAnswer: 1,
          explanation:
            'newline="" prevents the csv module from adding extra blank lines on Windows.',
        },
        {
          id: 'pi-jsoncsv-mc-2',
          question: 'What does json.dump() vs json.dumps() do?',
          options: [
            'dump() is faster',
            'dump() writes to file, dumps() returns string',
            'dumps() is newer',
            'They are identical',
          ],
          correctAnswer: 1,
          explanation:
            'json.dump() writes directly to a file object, while json.dumps() returns a JSON string.',
        },
        {
          id: 'pi-jsoncsv-mc-3',
          question: 'What Python value becomes null in JSON?',
          options: ['0', 'False', 'None', '""'],
          correctAnswer: 2,
          explanation: 'Python None is converted to JSON null.',
        },
        {
          id: 'pi-jsoncsv-mc-4',
          question: 'How do you pretty-print JSON with indentation?',
          options: [
            'json.dumps(data, pretty=True)',
            'json.dumps(data, indent=2)',
            'json.dumps(data, format="pretty")',
            'json.pretty(data)',
          ],
          correctAnswer: 1,
          explanation:
            'Use the indent parameter: json.dumps(data, indent=2) for readable JSON.',
        },
        {
          id: 'pi-jsoncsv-mc-5',
          question: 'What does csv.DictReader do?',
          options: [
            'Reads CSV as list of lists',
            'Reads CSV as list of dictionaries',
            'Converts CSV to JSON',
            'Validates CSV data',
          ],
          correctAnswer: 1,
          explanation:
            'DictReader reads each CSV row as a dictionary with column names as keys.',
        },
      ],
    },
    {
      id: 'regex',
      title: 'Regular Expressions',
      content: `# Regular Expressions

## Basic Patterns

\`\`\`python
import re

# Search for pattern
text = "The quick brown fox jumps over the lazy dog"
match = re.search(r'fox', text)
if match:
    print(f"Found at position {match.start()}")

# Match at beginning
if re.match(r'The', text):
    print("Starts with 'The'")

# Find all occurrences
words = re.findall(r'\\w+', text)
print(words)  # List of all words

# Replace
new_text = re.sub(r'fox', 'cat', text)
print(new_text)
\`\`\`

## Pattern Syntax

### Character Classes
- **.**: Any character except newline
- **\\d**: Digit [0-9]
- **\\D**: Non-digit
- **\\w**: Word character [a-zA-Z0-9_]
- **\\W**: Non-word character
- **\\s**: Whitespace
- **\\S**: Non-whitespace

### Quantifiers
- **\***: 0 or more
- **+**: 1 or more
- **?**: 0 or 1 (optional)
- **{n}**: Exactly n times
- **{n,}**: n or more times
- **{n,m}**: Between n and m times

### Anchors
- **^**: Start of string
- **$**: End of string
- **\\b**: Word boundary

\`\`\`python
# Examples
email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
phone_pattern = r'\\d{3}-\\d{3}-\\d{4}'
url_pattern = r'https?://\\S+'

# Validate email
email = "user@example.com"
if re.match(email_pattern, email):
    print("Valid email")

# Extract phone numbers
text = "Call 555-123-4567 or 555-987-6543"
phones = re.findall(phone_pattern, text)
print(phones)
\`\`\`

## Groups and Capturing

\`\`\`python
# Capture groups
pattern = r'(\\d{3})-(\\d{3})-(\\d{4})'
match = re.search(pattern, "555-123-4567")
if match:
    area = match.group(1)      # "555"
    exchange = match.group(2)  # "123"
    number = match.group(3)    # "4567"
    full = match.group(0)      # "555-123-4567"

# Named groups
pattern = r'(?P<area>\\d{3})-(?P<exchange>\\d{3})-(?P<number>\\d{4})'
match = re.search(pattern, "555-123-4567")
if match:
    print(match.group('area'))  # "555"
    print(match.groupdict())    # {'area': '555', ...}
\`\`\`

## Compilation and Flags

\`\`\`python
# Compile pattern for reuse
pattern = re.compile(r'\\d+')
matches = pattern.findall("I have 3 apples and 5 oranges")

# Flags
pattern = re.compile(r'hello', re.IGNORECASE)  # Case-insensitive
pattern = re.compile(r'line 1.*line 2', re.DOTALL)  # . matches newline
pattern = re.compile(r'''
    \\d{3}   # Area code
    -        # Separator
    \\d{4}   # Number
''', re.VERBOSE)  # Allow comments and whitespace
\`\`\`

## Common Patterns

\`\`\`python
# Email
email = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'

# URL
url = r'https?://(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b'

# IP Address
ip = r'^(?:[0-9]{1,3}\\.){3}[0-9]{1,3}$'

# Date (MM/DD/YYYY)
date = r'\\b(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/(\\d{4})\\b'

# Credit Card
cc = r'\\b\\d{4}[ -]?\\d{4}[ -]?\\d{4}[ -]?\\d{4}\\b'
\`\`\`

## Split and Replace

\`\`\`python
# Split by pattern
text = "one,two;three four"
parts = re.split(r'[,;\\s]+', text)  # ['one', 'two', 'three', 'four']

# Replace with function
def capitalize_match(match):
    return match.group(0).upper()

text = "hello world"
result = re.sub(r'\\b\\w+\\b', capitalize_match, text)  # "HELLO WORLD"
\`\`\`

## Best Practices

1. **Use raw strings**: \`r'pattern'\` to avoid escaping backslashes
2. **Compile patterns**: If using multiple times
3. **Test thoroughly**: Regex can be tricky
4. **Keep it simple**: Complex regex is hard to maintain
5. **Use online tools**: regex101.com for testing
6. **Consider alternatives**: Sometimes string methods are clearer`,
      videoUrl: 'https://www.youtube.com/watch?v=K8L6KVGG-7o',
      quiz: [
        {
          id: 'pi-regex-q-1',
          question:
            'Explain the difference between re.search(), re.match(), and re.findall(). When would you use each?',
          hint: 'Think about where in the string they look and what they return.',
          sampleAnswer:
            're.match() only checks if the pattern matches at the START of the string - use for validating entire strings (like "does this string look like an email?"). re.search() finds the pattern ANYWHERE in the string, returning the first match - use when you want to find one occurrence. re.findall() returns ALL matches as a list - use when you need multiple occurrences. For validation, use match(). For finding, use search() or findall(). Most common mistake: using match() when you mean search().',
          keyPoints: [
            're.match(): checks start of string only',
            're.search(): finds first match anywhere',
            're.findall(): returns list of all matches',
            'match() for validation, search()/findall() for finding',
          ],
        },
        {
          id: 'pi-regex-q-2',
          question:
            'Why should you use raw strings (r"...") for regex patterns? What problems does it prevent?',
          hint: 'Think about backslashes and Python string escaping.',
          sampleAnswer:
            'Raw strings treat backslashes literally, preventing double-escaping issues. Without raw strings, to match a literal backslash you\'d need "\\\\\\\\", but with raw strings just r"\\\\". Common patterns like \\d (digit) or \\w (word) work as r"\\d" instead of "\\\\d". Always use raw strings for regex - it makes patterns readable and prevents bugs. Without raw strings, you\'d need to escape every backslash twice: once for Python, once for regex.',
          keyPoints: [
            'Prevents double-escaping of backslashes',
            'Makes patterns more readable',
            'r"\\d" instead of "\\\\d"',
            'Always use raw strings for regex',
          ],
        },
        {
          id: 'pi-regex-q-3',
          question:
            'When should you avoid using regex? What are better alternatives for common tasks?',
          hint: 'Consider simplicity, maintainability, and specialized tools.',
          sampleAnswer:
            'Avoid regex for: 1) Simple string operations - use str.startswith(), str.endswith(), str.split(), "substring" in string instead, 2) Parsing HTML/XML - use BeautifulSoup or lxml (regex can\'t handle nesting), 3) Complex patterns that become unreadable - break into multiple steps or use parsing libraries, 4) When string methods are clearer and faster. Regex is powerful but has a learning curve and can be hard to debug. If a simple string method works, use that. "Some people, when confronted with a problem, think \'I know, I\'ll use regular expressions.\' Now they have two problems."',
          keyPoints: [
            'Use string methods for simple operations',
            'Use parsers (BeautifulSoup) for HTML/XML',
            'Break complex patterns into simpler steps',
            'Regex is powerful but can create maintenance issues',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'pi-regex-mc-1',
          question: 'What does the pattern ^abc$ match?',
          options: [
            'Any string containing "abc"',
            'Strings starting with "abc"',
            'Strings ending with "abc"',
            'Exactly the string "abc"',
          ],
          correctAnswer: 3,
          explanation:
            '^ anchors to start, $ anchors to end, so it matches only "abc" exactly.',
        },
        {
          id: 'pi-regex-mc-2',
          question: 'What does \\d+ match?',
          options: [
            'Exactly one digit',
            'One or more digits',
            'Zero or more digits',
            'Any character',
          ],
          correctAnswer: 1,
          explanation:
            '\\d matches a digit, and + means one or more occurrences.',
        },
        {
          id: 'pi-regex-mc-3',
          question: 'What is the purpose of raw strings (r"...") in regex?',
          options: [
            'Makes regex case-insensitive',
            'Prevents backslash escaping issues',
            'Makes pattern faster',
            'Required for all regex',
          ],
          correctAnswer: 1,
          explanation:
            'Raw strings prevent Python from interpreting backslashes, which are common in regex patterns.',
        },
        {
          id: 'pi-regex-mc-4',
          question: 'What is the difference between * and + quantifiers?',
          options: [
            'No difference',
            '* means 0 or more, + means 1 or more',
            '+ means 0 or more, * means 1 or more',
            '* is faster',
          ],
          correctAnswer: 1,
          explanation:
            '* matches zero or more occurrences, while + requires at least one occurrence.',
        },
        {
          id: 'pi-regex-mc-5',
          question: 'What does re.compile() do?',
          options: [
            'Checks if pattern is valid',
            'Pre-compiles pattern for reuse',
            'Makes pattern case-insensitive',
            'Converts string to regex',
          ],
          correctAnswer: 1,
          explanation:
            're.compile() pre-compiles the pattern, making it more efficient when used multiple times.',
        },
      ],
    },
    {
      id: 'datetime',
      title: 'Date and Time',
      content: `# Date and Time

## datetime Module

\`\`\`python
from datetime import datetime, date, time, timedelta

# Current date and time
now = datetime.now()
print(now)  # 2024-01-15 10:30:45.123456

# Current date
today = date.today()
print(today)  # 2024-01-15

# Create specific datetime
dt = datetime(2024, 1, 15, 10, 30, 45)

# Create specific date
d = date(2024, 1, 15)

# Create specific time
t = time(10, 30, 45)
\`\`\`

## Formatting Dates

\`\`\`python
# Convert to string
now = datetime.now()
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print(formatted)  # "2024-01-15 10:30:45"

# Common format codes:
# %Y - Year (4 digits)
# %m - Month (01-12)
# %d - Day (01-31)
# %H - Hour (00-23)
# %M - Minute (00-59)
# %S - Second (00-59)
# %A - Weekday name (Monday)
# %B - Month name (January)

examples = [
    now.strftime("%B %d, %Y"),          # "January 15, 2024"
    now.strftime("%m/%d/%y"),           # "01/15/24"
    now.strftime("%A, %B %d, %Y"),      # "Monday, January 15, 2024"
    now.strftime("%I:%M %p"),           # "10:30 AM"
]
\`\`\`

## Parsing Dates

\`\`\`python
# Parse string to datetime
date_string = "2024-01-15 10:30:45"
dt = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

# Different formats
formats = [
    ("01/15/2024", "%m/%d/%Y"),
    ("15-Jan-2024", "%d-%b-%Y"),
    ("2024-01-15T10:30:45", "%Y-%m-%dT%H:%M:%S"),
]

for date_str, fmt in formats:
    dt = datetime.strptime(date_str, fmt)
\`\`\`

## Date Arithmetic

\`\`\`python
# Add/subtract time
now = datetime.now()
tomorrow = now + timedelta(days=1)
last_week = now - timedelta(weeks=1)
in_2_hours = now + timedelta(hours=2)

# Difference between dates
date1 = datetime(2024, 1, 15)
date2 = datetime(2024, 1, 10)
diff = date1 - date2
print(diff.days)  # 5

# Time components
delta = timedelta(days=5, hours=3, minutes=30)
print(delta.total_seconds())  # Convert to seconds
\`\`\`

## Working with Timestamps

\`\`\`python
import time

# Get current timestamp
timestamp = time.time()  # Seconds since epoch

# Convert timestamp to datetime
dt = datetime.fromtimestamp(timestamp)

# Convert datetime to timestamp
timestamp = dt.timestamp()

# UTC time
utc_now = datetime.utcnow()
\`\`\`

## Timezone Handling

\`\`\`python
from datetime import timezone
import pytz  # Third-party library for better timezone support

# Create timezone-aware datetime
utc_time = datetime.now(timezone.utc)

# With pytz (install with: pip install pytz)
eastern = pytz.timezone('US/Eastern')
eastern_time = datetime.now(eastern)

# Convert between timezones
utc_time = datetime.now(pytz.UTC)
eastern_time = utc_time.astimezone(eastern)
\`\`\`

## Common Operations

\`\`\`python
# Get day of week
today = date.today()
day_of_week = today.strftime("%A")  # "Monday"
weekday_num = today.weekday()       # 0 = Monday

# First day of month
first_day = today.replace(day=1)

# Last day of month
from calendar import monthrange
last_day_num = monthrange(today.year, today.month)[1]
last_day = today.replace(day=last_day_num)

# Age calculation
birthdate = date(1990, 5, 15)
today = date.today()
age = today.year - birthdate.year
if (today.month, today.day) < (birthdate.month, birthdate.day):
    age -= 1
\`\`\`

## Best Practices

1. **Use datetime for timestamps**: More features than time module
2. **Store as UTC**: Convert to local timezone only for display
3. **Use ISO format**: \`%Y-%m-%d %H:%M:%S\` is unambiguous
4. **Handle timezones**: Use pytz for production code
5. **Validate dates**: Check for valid date ranges
6. **Use timedelta**: For date arithmetic`,
      videoUrl: 'https://www.youtube.com/watch?v=eirjjyP2qcQ',
      quiz: [
        {
          id: 'pi-datetime-q-1',
          question:
            'Explain the difference between datetime, date, and time objects. When should you use each?',
          hint: 'Think about what information each stores and typical use cases.',
          sampleAnswer:
            "datetime stores both date AND time (year, month, day, hour, minute, second, microsecond) - use for timestamps, events, logging. date stores only the date (year, month, day) - use for birthdays, schedules, appointments where time doesn't matter. time stores only time (hour, minute, second) - use for recurring events like \"daily at 9 AM\" without a specific date. Most common is datetime since it's the most complete. Use date when you explicitly don't care about time, and time for time-of-day patterns.",
          keyPoints: [
            'datetime: full timestamp with date and time',
            'date: just year/month/day',
            'time: just hour/minute/second',
            'datetime is most versatile and commonly used',
          ],
        },
        {
          id: 'pi-datetime-q-2',
          question:
            'Why should you always store timestamps in UTC and convert to local timezone only for display?',
          hint: 'Consider daylight saving time, consistency, and timezone conversions.',
          sampleAnswer:
            "Storing in UTC avoids ambiguity: 1) No daylight saving time issues (clock shifts, missing hours), 2) Consistent reference point globally, 3) Easy to convert to any timezone for display, 4) Prevents bugs when users travel or move. Common mistake: storing local time causes problems when DST changes or when comparing times across timezones. Always: store UTC in database, convert to user's timezone only when displaying. This is a critical best practice for any application with users in multiple locations.",
          keyPoints: [
            'UTC avoids DST ambiguity and clock shifts',
            'Consistent global reference point',
            'Easy to convert to any local timezone',
            'Store UTC, display in local timezone',
          ],
        },
        {
          id: 'pi-datetime-q-3',
          question:
            'What is timedelta and how is it used for date arithmetic? Why not just add/subtract integers?',
          hint: 'Think about handling months, leap years, and DST.',
          sampleAnswer:
            'timedelta represents a duration (like "3 days" or "2 hours") and handles date arithmetic correctly - accounting for months with different days, leap years, and DST. You can\'t just add integers because: tomorrow isn\'t always today+1 (DST shifts), next month isn\'t always +30 days (different month lengths), next year isn\'t always +365 days (leap years). Use timedelta for: adding/subtracting time (today + timedelta(days=7)), calculating differences (end_date - start_date), measuring elapsed time. It handles all calendar complexity automatically.',
          keyPoints: [
            'Represents duration, not a point in time',
            'Handles calendar complexity (DST, leap years)',
            'Use for adding/subtracting time periods',
            'Returns from subtracting two datetimes',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'pi-datetime-mc-1',
          question: 'What does datetime.now() return?',
          options: [
            'Current date only',
            'Current time only',
            'Current date and time',
            'Current timestamp',
          ],
          correctAnswer: 2,
          explanation:
            'datetime.now() returns a datetime object with both date and time.',
        },
        {
          id: 'pi-datetime-mc-2',
          question: 'What does timedelta represent?',
          options: [
            'A specific point in time',
            'A duration or difference between times',
            'A timezone',
            'A calendar date',
          ],
          correctAnswer: 1,
          explanation:
            'timedelta represents a duration - the difference between two dates or times.',
        },
        {
          id: 'pi-datetime-mc-3',
          question: 'What is the advantage of storing times in UTC?',
          options: [
            'Takes less space',
            'Faster processing',
            'Avoids timezone conversion issues',
            'Required by Python',
          ],
          correctAnswer: 2,
          explanation:
            'Storing in UTC avoids daylight saving time issues and makes it easy to convert to any local timezone.',
        },
        {
          id: 'pi-datetime-mc-4',
          question: 'How do you add 5 days to a datetime object "dt"?',
          options: [
            'dt + 5',
            'dt.add(5)',
            'dt + timedelta(days=5)',
            'dt.addDays(5)',
          ],
          correctAnswer: 2,
          explanation:
            'Use timedelta for date arithmetic: dt + timedelta(days=5)',
        },
        {
          id: 'pi-datetime-mc-5',
          question: 'What does strftime() do?',
          options: [
            'Parses string to datetime',
            'Formats datetime as string',
            'Converts timezone',
            'Returns current time',
          ],
          correctAnswer: 1,
          explanation:
            'strftime() formats a datetime object as a string with a specified format.',
        },
      ],
    },
  ],
};

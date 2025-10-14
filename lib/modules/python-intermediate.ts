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
    {
      id: 'logging',
      title: 'Logging and Debugging',
      content: `# Logging and Debugging

## Why Use Logging?

Logging is better than \`print()\` for production code because:
- **Levels**: Control verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Formatting**: Timestamps, filenames, line numbers automatically
- **Output**: Can log to files, console, network, etc.
- **Performance**: Can be turned off without code changes
- **Thread-safe**: Safe for multi-threaded applications

## Basic Logging

\`\`\`python
import logging

# Basic configuration (do this once at start)
logging.basicConfig(level=logging.DEBUG)

# Log at different levels
logging.debug("Detailed information for debugging")
logging.info("General informational messages")
logging.warning("Warning messages")
logging.error("Error messages")
logging.critical("Critical errors that may cause termination")

# By default, only WARNING and above are shown
\`\`\`

## Configuring Logging

\`\`\`python
import logging

# Configure format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
logger.info("This is an info message")
# Output: 2024-01-15 10:30:45 - __main__ - INFO - This is an info message
\`\`\`

## Logging to Files

\`\`\`python
import logging

# Log to file
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("This goes to the file")
logging.error("So does this error")

# Append mode (default) vs write mode
logging.basicConfig(filename='app.log', filemode='w')  # Overwrites
\`\`\`

## Multiple Handlers

\`\`\`python
import logging

# Create logger
logger = logging.getLogger('my_app')
logger.setLevel(logging.DEBUG)

# Console handler (INFO and above)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# File handler (DEBUG and above)
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Log messages
logger.debug("Detailed debug info")  # Only in file
logger.info("Important info")       # Console and file
logger.error("An error occurred")   # Console and file
\`\`\`

## Logging with Variables

\`\`\`python
import logging

logger = logging.getLogger(__name__)

# Old way (string formatting happens even if not logged)
logging.debug("User: " + username + ", Age: " + str(age))

# Better way (lazy evaluation)
logging.debug("User: %s, Age: %d", username, age)

# f-strings work but aren't lazy
logging.debug(f"User: {username}, Age: {age}")

# Extra context with exc_info
try:
    result = 10 / 0
except Exception as e:
    logging.error("Division failed", exc_info=True)  # Includes traceback
    # Or use logging.exception() which does this automatically
    logging.exception("Division failed")
\`\`\`

## Logging in Modules

\`\`\`python
# my_module.py
import logging

# Use __name__ so log shows which module logged it
logger = logging.getLogger(__name__)

def process_data(data):
    logger.info("Processing %d items", len(data))
    try:
        # Process data
        result = do_something(data)
        logger.debug("Result: %s", result)
        return result
    except Exception:
        logger.exception("Failed to process data")
        raise

# main.py
import logging
import my_module

logging.basicConfig(level=logging.INFO)
my_module.process_data([1, 2, 3])
# Output: my_module - INFO - Processing 3 items
\`\`\`

## Rotating Log Files

\`\`\`python
import logging
from logging.handlers import RotatingFileHandler

# Create logger
logger = logging.getLogger('my_app')
logger.setLevel(logging.DEBUG)

# Rotating file handler (5 MB per file, keep 3 backups)
handler = RotatingFileHandler(
    'app.log',
    maxBytes=5*1024*1024,  # 5 MB
    backupCount=3
)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Files: app.log, app.log.1, app.log.2, app.log.3
\`\`\`

## Time-Based Rotation

\`\`\`python
import logging
from logging.handlers import TimedRotatingFileHandler

# Rotate daily at midnight, keep 7 days
handler = TimedRotatingFileHandler(
    'app.log',
    when='midnight',
    interval=1,
    backupCount=7
)

# Other options for 'when':
# 'S' - Seconds
# 'M' - Minutes
# 'H' - Hours
# 'D' - Days
# 'midnight' - Roll over at midnight
# 'W0'-'W6' - Weekday (0=Monday)
\`\`\`

## Structured Logging

\`\`\`python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

# Use JSON formatter
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger('app')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info("User login", extra={'user_id': 123, 'ip': '192.168.1.1'})
# Output: {"timestamp": "2024-01-15 10:30:00", "level": "INFO", ...}
\`\`\`

## Logging Best Practices

**1. Choose the Right Level:**
- **DEBUG**: Detailed diagnostic info (algorithm steps, variable values)
- **INFO**: Confirmation things are working (server started, job completed)
- **WARNING**: Something unexpected but not breaking (deprecated feature, fallback used)
- **ERROR**: Serious problem, function failed
- **CRITICAL**: Program may crash, data corruption

**2. Use Lazy Evaluation:**
\`\`\`python
# Good - string only built if logged
logging.debug("Processing %s with %d items", name, len(items))

# Bad - string always built
logging.debug(f"Processing {name} with {len(items)} items")
\`\`\`

**3. Log Exceptions Properly:**
\`\`\`python
try:
    risky_operation()
except Exception:
    logging.exception("Operation failed")  # Includes traceback
\`\`\`

**4. Don't Log Sensitive Data:**
\`\`\`python
# Bad
logging.info(f"User logged in: {username}, password: {password}")

# Good
logging.info(f"User logged in: {username}")
\`\`\`

**5. Use Module-Level Loggers:**
\`\`\`python
# Good - shows which module logged
logger = logging.getLogger(__name__)

# Bad - all logs show same name
logger = logging.getLogger('app')
\`\`\`

## Common Patterns

**Configuration File:**
\`\`\`python
import logging
import logging.config

# logging.ini
'''
[loggers]
keys=root

[handlers]
keys=console,file

[formatters]
keys=simple,detailed

[logger_root]
level=DEBUG
handlers=console,file

[handler_console]
class=StreamHandler
level=INFO
formatter=simple
args=(sys.stdout,)

[handler_file]
class=FileHandler
level=DEBUG
formatter=detailed
args=('app.log', 'a')

[formatter_simple]
format=%(levelname)s - %(message)s

[formatter_detailed]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
'''

# Load configuration
logging.config.fileConfig('logging.ini')
logger = logging.getLogger()
\`\`\`

**Testing/Development vs Production:**
\`\`\`python
import logging
import os

# Set level based on environment
level = logging.DEBUG if os.getenv('ENV') == 'development' else logging.INFO

logging.basicConfig(
    level=level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
\`\`\``,
      quiz: [
        {
          id: 'q1',
          question:
            'Why is logging better than print() for production applications? List at least 4 advantages.',
          hint: 'Think about control, formatting, output destinations, and performance.',
          sampleAnswer:
            'Logging is superior to print() for production code because: 1) **Levels**: You can control verbosity with DEBUG/INFO/WARNING/ERROR/CRITICAL and filter at runtime without code changes, 2) **Formatting**: Automatically includes timestamps, module names, line numbers, and exception tracebacks, 3) **Multiple outputs**: Can simultaneously log to console, files, network services, or cloud logging, 4) **Performance**: Can be disabled in production without removing code, and lazy evaluation means expensive string formatting only happens if the message will be logged, 5) **Thread-safe**: Safe for concurrent applications. For example, you can set level to ERROR in production to only log serious issues, then switch to DEBUG in development without changing any code.',
          keyPoints: [
            'Levels allow runtime verbosity control',
            'Automatic formatting with timestamps and context',
            'Can output to multiple destinations',
            'Lazy evaluation and can be disabled',
            'Thread-safe for concurrent applications',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the difference between the five logging levels. Give an example use case for each.',
          hint: 'Consider severity and what action each level implies.',
          sampleAnswer:
            '**DEBUG**: Detailed diagnostic information, used during development to trace program flow. Example: "Entering function calculate_score with args: user_id=123". **INFO**: Confirmation that things are working as expected. Example: "Server started on port 8080" or "Processing batch 5 of 20". **WARNING**: Something unexpected happened but the program continues. Example: "Deprecated API endpoint used" or "Cache miss, loading from database". **ERROR**: A serious problem caused a function to fail, but the application continues. Example: "Failed to send email to user@example.com: SMTP timeout". **CRITICAL**: A serious error that may cause the program to crash or corrupt data. Example: "Database connection lost, shutting down" or "Out of memory". In production, typically log INFO and above; in development, use DEBUG.',
          keyPoints: [
            'DEBUG: detailed diagnostics for development',
            'INFO: confirmation of normal operation',
            'WARNING: unexpected but not breaking',
            'ERROR: function failed but app continues',
            'CRITICAL: may cause crash or data corruption',
          ],
        },
        {
          id: 'q3',
          question:
            'What is lazy evaluation in logging and why is it important? Show examples of correct and incorrect usage.',
          hint: 'Think about when string formatting happens and performance implications.',
          sampleAnswer:
            'Lazy evaluation means the log message string is only built if it will actually be logged (based on level). This matters for performance when logging is disabled or filtered. **Correct**: logging.debug("User %s has %d items", username, len(items)) - the string is only formatted if DEBUG is enabled. **Incorrect**: logging.debug(f"User {username} has {len(items)} items") - f-string always evaluates, even if DEBUG is disabled. For expensive operations like database queries or large JSON serialization, this difference is huge. Example: logging.debug(f"Data: {json.dumps(huge_dict)}") always serializes even if not logged, but logging.debug("Data: %s", json.dumps(huge_dict)) doesn\'t. Use % or comma-separated args for lazy evaluation.',
          keyPoints: [
            'Message only built if it will be logged',
            'Critical for performance with expensive formatting',
            'Use % formatting or comma args, not f-strings',
            'Example: logging.debug("Value: %s", expensive_func())',
            'Especially important for disabled DEBUG messages',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the default logging level?',
          options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
          correctAnswer: 2,
          explanation:
            'By default, logging is set to WARNING level, meaning only WARNING, ERROR, and CRITICAL messages are shown. INFO and DEBUG are not displayed unless you configure it.',
        },
        {
          id: 'mc2',
          question:
            'Which method should you use to log an exception with traceback?',
          options: [
            'logging.error() with exc_info=True',
            'logging.exception()',
            'Both A and B',
            'logging.traceback()',
          ],
          correctAnswer: 2,
          explanation:
            'Both logging.error() with exc_info=True and logging.exception() will log the exception with full traceback. logging.exception() is a shorthand that automatically includes the traceback.',
        },
        {
          id: 'mc3',
          question: 'Why should you use logging.getLogger(__name__)?',
          options: [
            "It's faster",
            'It shows which module the log came from',
            "It's required by Python",
            'It enables colored output',
          ],
          correctAnswer: 1,
          explanation:
            'Using __name__ creates a logger specific to your module, so logs show which module generated them. This makes debugging much easier in large applications.',
        },
        {
          id: 'mc4',
          question: 'What does RotatingFileHandler do?',
          options: [
            'Rotates log messages',
            'Automatically creates new log files when size limit is reached',
            'Encrypts log files',
            'Compresses old logs',
          ],
          correctAnswer: 1,
          explanation:
            'RotatingFileHandler automatically creates new log files when the current file reaches a specified size, keeping a configured number of backup files.',
        },
        {
          id: 'mc5',
          question: 'Which is the correct way for lazy evaluation in logging?',
          options: [
            'logging.info(f"Value: {value}")',
            'logging.info("Value: " + str(value))',
            'logging.info("Value: %s", value)',
            'logging.info("Value: {}".format(value))',
          ],
          correctAnswer: 2,
          explanation:
            'Using % formatting (logging.info("Value: %s", value)) or comma-separated arguments enables lazy evaluation—the string is only built if the message will be logged.',
        },
      ],
    },
    {
      id: 'virtual-environments',
      title: 'Virtual Environments & Package Management',
      content: `# Virtual Environments & Package Management

## Why Virtual Environments?

Virtual environments solve the "dependency hell" problem:
- **Isolation**: Each project has its own dependencies
- **No conflicts**: Different projects can use different package versions
- **Reproducibility**: Easy to replicate the exact environment
- **Clean system**: Don't pollute global Python installation

## Creating Virtual Environments

**Using venv (built-in, Python 3.3+):**
\`\`\`bash
# Create virtual environment
python -m venv myenv

# Or specify Python version
python3.11 -m venv myenv

# Creates directory structure:
# myenv/
#   bin/          # Scripts (Linux/Mac)
#   Scripts/      # Scripts (Windows)
#   lib/          # Installed packages
#   include/      # C headers
#   pyvenv.cfg    # Configuration
\`\`\`

## Activating Virtual Environments

\`\`\`bash
# Linux/Mac
source myenv/bin/activate

# Windows Command Prompt
myenv\\Scripts\\activate.bat

# Windows PowerShell
myenv\\Scripts\\Activate.ps1

# After activation, your prompt changes:
(myenv) $

# Check you're in the virtual environment
which python  # Should point to myenv/bin/python
python --version
\`\`\`

## Deactivating

\`\`\`bash
# From any directory
deactivate

# Your prompt returns to normal
$
\`\`\`

## Installing Packages

\`\`\`bash
# Activate environment first
source myenv/bin/activate

# Install packages
pip install requests
pip install numpy pandas matplotlib

# Install specific version
pip install Django==4.2.0

# Install from requirements.txt
pip install -r requirements.txt

# Upgrade package
pip install --upgrade requests

# Uninstall
pip uninstall requests
\`\`\`

## Managing Dependencies

**requirements.txt:**
\`\`\`bash
# Generate requirements.txt (all installed packages)
pip freeze > requirements.txt

# Example requirements.txt:
'''
requests==2.31.0
numpy==1.24.3
pandas==2.0.2
Django==4.2.0
'''

# Install from requirements.txt
pip install -r requirements.txt
\`\`\`

**Better: Separate dev and prod dependencies:**
\`\`\`bash
# requirements.txt (production)
requests==2.31.0
Django==4.2.0

# requirements-dev.txt (development)
-r requirements.txt  # Include production requirements
pytest==7.4.0
black==23.3.0
mypy==1.3.0
\`\`\`

## Viewing Installed Packages

\`\`\`bash
# List all installed packages
pip list

# Show package details
pip show requests

# Output:
# Name: requests
# Version: 2.31.0
# Summary: HTTP library
# Home-page: https://requests.readthedocs.io
# Location: /path/to/myenv/lib/python3.11/site-packages
# Requires: charset-normalizer, idna, urllib3, certifi
\`\`\`

## Alternative: virtualenv

virtualenv is a more powerful third-party tool:
\`\`\`bash
# Install virtualenv
pip install virtualenv

# Create virtual environment
virtualenv myenv

# Can use different Python versions
virtualenv -p python3.10 myenv

# Or specify full path
virtualenv -p /usr/bin/python3.11 myenv
\`\`\`

## Alternative: Poetry

Poetry is a modern dependency management tool:
\`\`\`bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Create new project
poetry new my-project
cd my-project

# Initialize in existing project
poetry init

# Add dependencies
poetry add requests
poetry add pytest --dev  # Development dependency

# Install all dependencies
poetry install

# pyproject.toml (Poetry's config file):
'''
[tool.poetry]
name = "my-project"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
requests = "^2.31.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
'''

# Poetry automatically creates and manages virtual environment
poetry run python script.py
poetry shell  # Activate virtual environment
\`\`\`

## Alternative: Conda

Conda is popular in data science (includes non-Python dependencies):
\`\`\`bash
# Create environment
conda create -n myenv python=3.11

# Activate
conda activate myenv

# Install packages
conda install numpy pandas matplotlib

# Install from conda-forge
conda install -c conda-forge opencv

# Mix conda and pip
conda install numpy
pip install some-package-not-in-conda

# Export environment
conda env export > environment.yml

# Create from environment.yml
conda env create -f environment.yml

# List environments
conda env list

# Remove environment
conda env remove -n myenv
\`\`\`

## Best Practices

**1. One virtual environment per project:**
\`\`\`bash
my-project/
  venv/          # Virtual environment
  src/           # Source code
  tests/         # Tests
  requirements.txt
  README.md
\`\`\`

**2. Don't commit virtual environment to git:**
\`\`\`bash
# .gitignore
venv/
env/
*.pyc
__pycache__/
\`\`\`

**3. Pin versions in production:**
\`\`\`bash
# Development: allow minor updates
requests>=2.31.0,<3.0.0

# Production: pin exact versions
requests==2.31.0
\`\`\`

**4. Document Python version:**
\`\`\`bash
# .python-version (for pyenv)
3.11.4

# Or in README:
'''
Requires Python 3.11+
'''
\`\`\`

**5. Use requirements.txt for simple projects, Poetry for complex ones:**
- Small scripts, tutorials: requirements.txt
- Libraries, applications: Poetry or pipenv

## Common Workflows

**Starting a new project:**
\`\`\`bash
# Create project directory
mkdir my-project
cd my-project

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate    # Windows

# Install packages
pip install requests pytest

# Save dependencies
pip freeze > requirements.txt

# Create .gitignore
echo "venv/" > .gitignore
\`\`\`

**Cloning an existing project:**
\`\`\`bash
# Clone repository
git clone https://github.com/user/project.git
cd project

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
\`\`\`

**Upgrading dependencies:**
\`\`\`bash
# Check outdated packages
pip list --outdated

# Upgrade specific package
pip install --upgrade requests

# Upgrade all (careful!)
pip list --outdated --format=freeze | grep -v '^\\-e' | cut -d = -f 1 | xargs -n1 pip install -U

# Update requirements.txt
pip freeze > requirements.txt
\`\`\`

## Troubleshooting

**Virtual environment not activating:**
\`\`\`bash
# Linux/Mac: check permissions
ls -la venv/bin/activate
chmod +x venv/bin/activate

# Windows PowerShell: enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
\`\`\`

**Wrong Python version:**
\`\`\`bash
# Specify exact Python version
python3.11 -m venv venv

# Or with virtualenv
virtualenv -p /usr/bin/python3.11 venv
\`\`\`

**Package conflicts:**
\`\`\`bash
# Start fresh
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\``,
      quiz: [
        {
          id: 'q1',
          question:
            'What problems do virtual environments solve? Why should every Python project use one?',
          hint: 'Think about package versions, system cleanliness, and reproducibility.',
          sampleAnswer:
            'Virtual environments solve several critical problems: 1) **Version conflicts**: Project A needs Django 3.2, Project B needs Django 4.2—without virtual environments, only one can be installed. 2) **Reproducibility**: requirements.txt ensures everyone working on the project has identical dependencies. 3) **Clean system**: Installing packages globally pollutes your system Python, potentially breaking system tools. 4) **Experimentation**: Safely try new packages without affecting other projects. 5) **Multiple Python versions**: Different projects can use Python 3.9, 3.10, 3.11 side-by-side. For example, data science projects often need specific NumPy/Pandas versions that conflict with web projects. Virtual environments make this trivial.',
          keyPoints: [
            'Isolates dependencies per project',
            'Prevents version conflicts',
            'Enables reproducibility with requirements.txt',
            'Keeps global Python clean',
            'Allows multiple Python versions',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the difference between pip freeze and manually maintaining requirements.txt. What are the pros and cons?',
          hint: 'Consider transitive dependencies, version pinning, and maintainability.',
          sampleAnswer:
            "**pip freeze**: Outputs ALL installed packages including transitive dependencies (dependencies of dependencies) with exact versions. **Pros**: Completely reproducible—guarantees exact environment. **Cons**: Includes packages you didn't explicitly install, making it hard to see actual dependencies; updates are all-or-nothing. **Manual requirements.txt**: List only direct dependencies, optionally with version ranges (>=2.0,<3.0). **Pros**: Clear what your project actually needs; allows minor updates for security patches. **Cons**: Different Python versions or platforms might resolve dependencies differently. **Best practice**: Use pip freeze for production (exact reproducibility), manual for development (flexibility). Or use Poetry which tracks both: pyproject.toml for direct deps, poetry.lock for exact versions.",
          keyPoints: [
            'pip freeze: all packages with exact versions',
            'Manual: only direct dependencies',
            'freeze pros: exact reproducibility',
            'Manual pros: clarity and flexibility',
            'Use Poetry for best of both worlds',
          ],
        },
        {
          id: 'q3',
          question:
            'When should you use venv vs virtualenv vs Poetry vs Conda? What are the use cases for each?',
          hint: 'Consider built-in vs third-party, Python-only vs non-Python deps, simplicity vs features.',
          sampleAnswer:
            '**venv**: Built into Python 3.3+, perfect for most projects. Use for: simple projects, when you want standard tooling, teaching. **virtualenv**: Third-party, more features than venv (faster, supports older Python). Use for: need Python 2.7 support (legacy), need certain advanced features. **Poetry**: Modern dependency management with lockfiles. Use for: libraries (publishing to PyPI), complex applications, teams that need reproducible builds. Handles dependency resolution better than pip. **Conda**: Includes non-Python dependencies (C libraries, system tools). Use for: data science (needs NumPy, SciPy with optimized binaries), projects mixing Python with R/Julia, cross-platform scientific computing. **Recommendation**: Start with venv for learning, move to Poetry for serious projects, use Conda only for data science.',
          keyPoints: [
            'venv: built-in, simple, most common',
            'virtualenv: more features, older Python support',
            'Poetry: modern, dependency resolution, lockfiles',
            'Conda: data science, non-Python dependencies',
            'Choose based on project complexity and needs',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What command creates a virtual environment with venv?',
          options: [
            'python -m venv myenv',
            'python create-venv myenv',
            'pip install venv myenv',
            'venv create myenv',
          ],
          correctAnswer: 0,
          explanation:
            'python -m venv myenv is the correct command to create a virtual environment named "myenv" using the built-in venv module.',
        },
        {
          id: 'mc2',
          question: 'How do you activate a virtual environment on Linux/Mac?',
          options: [
            'activate myenv',
            'source myenv/bin/activate',
            'python myenv',
            'myenv/activate',
          ],
          correctAnswer: 1,
          explanation:
            'source myenv/bin/activate is the correct command to activate a virtual environment on Linux and Mac systems.',
        },
        {
          id: 'mc3',
          question: 'What does pip freeze > requirements.txt do?',
          options: [
            "Freezes pip so it can't be updated",
            'Saves all installed packages and versions to a file',
            'Locks the Python version',
            'Creates a backup of pip',
          ],
          correctAnswer: 1,
          explanation:
            'pip freeze > requirements.txt outputs all installed packages with their exact versions and saves them to requirements.txt, enabling environment recreation.',
        },
        {
          id: 'mc4',
          question:
            'Should you commit your virtual environment folder (venv/) to git?',
          options: [
            'Yes, always',
            'No, add it to .gitignore',
            'Only for small projects',
            'Only the bin/ directory',
          ],
          correctAnswer: 1,
          explanation:
            'No, never commit virtual environments to git. They are large, platform-specific, and can be recreated from requirements.txt. Add venv/ to .gitignore.',
        },
        {
          id: 'mc5',
          question: 'What is the advantage of Poetry over pip?',
          options: [
            'Faster installation',
            'Uses less disk space',
            'Better dependency resolution and lockfiles',
            'No advantages',
          ],
          correctAnswer: 2,
          explanation:
            'Poetry provides better dependency resolution (solves conflicts automatically), lockfiles for reproducibility, and cleaner dependency management with pyproject.toml.',
        },
      ],
    },
    {
      id: 'collections-module',
      title: 'Collections Module - Advanced Data Structures',
      content: `# Collections Module - Advanced Data Structures

The \`collections\` module provides specialized container datatypes beyond the built-in list, dict, tuple, and set. These are essential for interviews and production code.

## Counter - Frequency Counting

\`Counter\` is a dict subclass for counting hashable objects. **Extremely common in coding interviews!**

### Basic Usage

\`\`\`python
from collections import Counter

# Count elements in list
fruits = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
counts = Counter(fruits)
print(counts)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})

# Access counts
print(counts['apple'])  # 3
print(counts['grape'])  # 0 (no KeyError!)

# Count characters in string
text = 'hello world'
char_counts = Counter(text)
print(char_counts)  # Counter({'l': 3, 'o': 2, 'h': 1, ...})
\`\`\`

### Most Common Elements

\`\`\`python
# Get most common
numbers = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4]
counter = Counter(numbers)

# Top 2 most common
print(counter.most_common(2))  # [(4, 4), (1, 3)]

# All elements ordered by frequency
print(counter.most_common())  # [(4, 4), (1, 3), (2, 2), (3, 1)]
\`\`\`

### Counter Operations

\`\`\`python
c1 = Counter(['a', 'b', 'c', 'a', 'b', 'b'])
c2 = Counter(['a', 'b', 'b', 'd'])

# Addition
print(c1 + c2)  # Counter({'b': 5, 'a': 3, 'c': 1, 'd': 1})

# Subtraction (keeps only positive)
print(c1 - c2)  # Counter({'b': 1, 'c': 1, 'a': 0})

# Intersection (min of counts)
print(c1 & c2)  # Counter({'b': 2, 'a': 1})

# Union (max of counts)
print(c1 | c2)  # Counter({'b': 3, 'a': 2, 'c': 1, 'd': 1})
\`\`\`

### Interview Applications

\`\`\`python
# Check if two strings are anagrams
def is_anagram(s1, s2):
    return Counter(s1) == Counter(s2)

print(is_anagram('listen', 'silent'))  # True

# Find first non-repeating character
def first_unique_char(s):
    counts = Counter(s)
    for char in s:
        if counts[char] == 1:
            return char
    return None

print(first_unique_char('leetcode'))  # 'l'

# Top K frequent elements
def top_k_frequent(nums, k):
    counter = Counter(nums)
    return [num for num, count in counter.most_common(k)]

print(top_k_frequent([1,1,1,2,2,3], 2))  # [1, 2]
\`\`\`

---

## defaultdict - No More KeyError

\`defaultdict\` is a dict subclass that provides default values for missing keys.

### Basic Usage

\`\`\`python
from collections import defaultdict

# Regular dict - KeyError
regular = {}
# regular['key'].append('value')  # KeyError!

# defaultdict with list
d = defaultdict(list)
d['key'].append('value')  # Works! Auto-creates empty list
print(d)  # defaultdict(<class 'list'>, {'key': ['value']})

# defaultdict with int (default 0)
counts = defaultdict(int)
counts['a'] += 1  # Works! Starts from 0
counts['b'] += 1
print(counts)  # defaultdict(<class 'int'>, {'a': 1, 'b': 1})

# defaultdict with set
groups = defaultdict(set)
groups['team1'].add('Alice')
groups['team1'].add('Bob')
print(groups)  # defaultdict(<class 'set'>, {'team1': {'Alice', 'Bob'}})
\`\`\`

### Common Use Cases

**1. Grouping items:**
\`\`\`python
# Group words by first letter
words = ['apple', 'apricot', 'banana', 'blueberry', 'cherry']
groups = defaultdict(list)

for word in words:
    groups[word[0]].append(word)

print(dict(groups))
# {'a': ['apple', 'apricot'], 
#  'b': ['banana', 'blueberry'], 
#  'c': ['cherry']}
\`\`\`

**2. Graph adjacency list:**
\`\`\`python
# Build graph
graph = defaultdict(list)
edges = [(1, 2), (1, 3), (2, 4), (3, 4)]

for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)  # Undirected

print(dict(graph))
# {1: [2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3]}
\`\`\`

**3. Counting with categories:**
\`\`\`python
# Track scores by player
scores = defaultdict(list)
scores['Alice'].append(10)
scores['Bob'].append(15)
scores['Alice'].append(20)

# Calculate averages
for player, score_list in scores.items():
    avg = sum(score_list) / len(score_list)
    print(f"{player}: {avg}")
\`\`\`

### Factory Functions

\`\`\`python
# Default to specific value
d = defaultdict(lambda: 'N/A')
print(d['missing'])  # 'N/A'

# Default to 0
counts = defaultdict(int)

# Default to empty dict
nested = defaultdict(dict)
nested['level1']['level2'] = 'value'
\`\`\`

---

## deque - Double-Ended Queue

\`deque\` (pronounced "deck") is optimized for fast appends/pops from both ends. **Essential for queues and sliding windows!**

### Why deque?

\`\`\`python
# List: O(n) to pop from front
my_list = [1, 2, 3, 4, 5]
my_list.pop(0)  # O(n) - shifts all elements!

# deque: O(1) for both ends
from collections import deque
my_deque = deque([1, 2, 3, 4, 5])
my_deque.popleft()  # O(1) - efficient!
\`\`\`

### Basic Operations

\`\`\`python
from collections import deque

# Create deque
dq = deque([1, 2, 3])

# Add to right
dq.append(4)  # [1, 2, 3, 4]

# Add to left
dq.appendleft(0)  # [0, 1, 2, 3, 4]

# Remove from right
dq.pop()  # Returns 4, deque: [0, 1, 2, 3]

# Remove from left
dq.popleft()  # Returns 0, deque: [1, 2, 3]

# Extend both ends
dq.extend([4, 5])  # [1, 2, 3, 4, 5]
dq.extendleft([0, -1])  # [-1, 0, 1, 2, 3, 4, 5]
\`\`\`

### Queue Implementation

\`\`\`python
# Perfect for BFS queues
queue = deque()
queue.append(1)  # Enqueue
queue.append(2)
first = queue.popleft()  # Dequeue - O(1)!
\`\`\`

### Sliding Window Maximum

\`\`\`python
def max_sliding_window(nums, k):
    """Find max in each sliding window"""
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove out-of-window indices
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))
# [3, 3, 5, 5, 6, 7]
\`\`\`

### Rotation

\`\`\`python
dq = deque([1, 2, 3, 4, 5])

# Rotate right
dq.rotate(2)  # [4, 5, 1, 2, 3]

# Rotate left
dq.rotate(-2)  # [1, 2, 3, 4, 5]
\`\`\`

### Max Length

\`\`\`python
# Limited-size deque (circular buffer)
dq = deque(maxlen=3)
dq.append(1)
dq.append(2)
dq.append(3)
dq.append(4)  # Removes 1 automatically
print(dq)  # deque([2, 3, 4], maxlen=3)
\`\`\`

---

## OrderedDict - Remembers Insertion Order

**Note:** Python 3.7+ dicts maintain insertion order, but OrderedDict has extra features.

\`\`\`python
from collections import OrderedDict

# Maintains order
od = OrderedDict()
od['b'] = 2
od['a'] = 1
od['c'] = 3
print(list(od.keys()))  # ['b', 'a', 'c']

# Move to end
od.move_to_end('a')  # a moved to end
print(list(od.keys()))  # ['b', 'c', 'a']

# Move to beginning
od.move_to_end('a', last=False)
print(list(od.keys()))  # ['a', 'b', 'c']

# Pop last item
od.popitem(last=True)  # Remove from end
od.popitem(last=False)  # Remove from beginning
\`\`\`

### LRU Cache Implementation

\`\`\`python
class LRUCache:
    """Least Recently Used Cache"""
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return -1
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove LRU
\`\`\`

---

## namedtuple - Lightweight Objects

Create simple classes without defining full class.

\`\`\`python
from collections import namedtuple

# Define structure
Point = namedtuple('Point', ['x', 'y'])
Person = namedtuple('Person', 'name age city')  # Space-separated

# Create instances
p1 = Point(10, 20)
person = Person('Alice', 30, 'NYC')

# Access by name or index
print(p1.x, p1.y)  # 10 20
print(p1[0], p1[1])  # 10 20

# Unpack
x, y = p1

# Immutable (like tuples)
# p1.x = 15  # AttributeError

# Convert to dict
print(person._asdict())
# {'name': 'Alice', 'age': 30, 'city': 'NYC'}

# Replace values (creates new instance)
person2 = person._replace(age=31)
\`\`\`

### Use Cases

\`\`\`python
# Function returns
def get_stats():
    Stats = namedtuple('Stats', 'mean median mode')
    return Stats(mean=10, median=9, mode=8)

stats = get_stats()
print(f"Mean: {stats.mean}")

# CSV/Database rows
Employee = namedtuple('Employee', 'id name department salary')
employees = [
    Employee(1, 'Alice', 'Engineering', 100000),
    Employee(2, 'Bob', 'Sales', 80000),
]

for emp in employees:
    print(f"{emp.name} in {emp.department}")
\`\`\`

---

## ChainMap - Combined Views

Combine multiple dicts into single view.

\`\`\`python
from collections import ChainMap

# Multiple dicts
defaults = {'color': 'red', 'user': 'guest'}
config = {'user': 'admin'}
cli_args = {'debug': True}

# Chain them (first dict takes priority)
combined = ChainMap(cli_args, config, defaults)

print(combined['color'])  # 'red' (from defaults)
print(combined['user'])   # 'admin' (from config, overrides defaults)
print(combined['debug'])  # True (from cli_args)

# Update
combined['user'] = 'root'  # Updates first dict (cli_args)

# Add new dict to front
combined = combined.new_child({'temp': 'value'})
\`\`\`

---

## Quick Reference

| Collection | Use Case | Key Feature |
|------------|----------|-------------|
| **Counter** | Frequency counting | \`most_common()\`, math operations |
| **defaultdict** | Auto-initialize missing keys | No KeyError |
| **deque** | Queue, stack, both ends | O(1) append/pop from both ends |
| **OrderedDict** | Order matters + operations | \`move_to_end()\`, \`popitem()\` |
| **namedtuple** | Lightweight objects | Named access, immutable |
| **ChainMap** | Multiple dicts | Layered lookups |

## Performance Comparison

\`\`\`python
import timeit

# List vs deque for queue operations
def list_queue():
    q = []
    for i in range(1000):
        q.append(i)
    for i in range(1000):
        q.pop(0)  # O(n) each time!

def deque_queue():
    from collections import deque
    q = deque()
    for i in range(1000):
        q.append(i)
    for i in range(1000):
        q.popleft()  # O(1) each time!

# deque is ~100x faster for this!
print(timeit.timeit(list_queue, number=100))
print(timeit.timeit(deque_queue, number=100))
\`\`\`

---

## Interview Patterns

**1. Use Counter for:**
- Anagrams
- Top K frequent elements
- Character frequency

**2. Use defaultdict for:**
- Grouping
- Graph adjacency lists
- Nested structures

**3. Use deque for:**
- BFS queues
- Sliding window
- Both-ends operations

**4. Use namedtuple for:**
- Coordinate pairs
- Return multiple values
- Lightweight objects

**Remember:** These are in Python standard library - no pip install needed!`,
      quiz: [
        {
          id: 'q1',
          question:
            'When would you use Counter vs defaultdict(int) for counting?',
          sampleAnswer:
            "Use Counter when you need counting-specific features like most_common(), arithmetic operations (+, -, &, |), or when you want to emphasize that you're counting. Counter also returns 0 for missing keys instead of raising KeyError. Use defaultdict(int) when counting is part of a larger operation and you need the auto-initialization behavior for general integer operations. Counter is more expressive for pure counting tasks: Counter([1,2,1]).most_common() is clearer than manually sorting defaultdict items.",
          keyPoints: [
            'Counter: pure counting, has most_common()',
            'Counter: math operations between counters',
            'Counter: more explicit intent',
            'defaultdict(int): part of larger logic',
            'Both avoid KeyError for missing keys',
          ],
        },
        {
          id: 'q2',
          question: 'Why is deque better than list for implementing a queue?',
          sampleAnswer:
            'deque.popleft() is O(1) while list.pop(0) is O(n). When removing from the front of a list, Python must shift all remaining elements left, taking O(n) time. With deque, both ends are optimized for O(1) operations using a doubly-linked list of blocks internally. For a queue with 10,000 operations, list would take ~50 million operations total (O(n²)) while deque takes just 10,000 (O(n)). This makes deque essential for BFS, sliding windows, and any FIFO data structure.',
          keyPoints: [
            'deque.popleft(): O(1)',
            'list.pop(0): O(n) - shifts elements',
            'Huge performance difference for queues',
            'Essential for BFS and sliding windows',
            'deque optimized for both ends',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain the practical differences between OrderedDict and regular dict in Python 3.7+.',
          sampleAnswer:
            'In Python 3.7+, regular dicts maintain insertion order, making OrderedDict less critical. However, OrderedDict still has unique features: (1) move_to_end(key) to reorder items, (2) popitem(last=False/True) to remove from specific end, (3) Explicit ordering semantics - code intent is clearer, (4) Equality checks consider order: OrderedDict(a=1, b=2) != OrderedDict(b=2, a=1), but regular dicts with same items are equal regardless of order. Use OrderedDict when you need these operations or want to explicitly signal that order matters for correctness (e.g., LRU cache implementation).',
          keyPoints: [
            'Both maintain insertion order in Python 3.7+',
            'OrderedDict has move_to_end() and directional popitem()',
            'OrderedDict equality considers order',
            'Regular dict equality ignores order',
            'Use OrderedDict when order operations are needed',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What does Counter do?',
          options: [
            'Counts function calls',
            'Counts occurrences of elements',
            'Creates numbered lists',
            'Measures performance',
          ],
          correctAnswer: 1,
          explanation:
            'Counter counts occurrences of hashable elements: Counter(["a","b","a"]) → Counter({"a":2, "b":1})',
        },
        {
          id: 'mc2',
          question: 'What is the advantage of defaultdict?',
          options: [
            'It is faster',
            'It never raises KeyError for missing keys',
            'It sorts keys automatically',
            'It uses less memory',
          ],
          correctAnswer: 1,
          explanation:
            'defaultdict provides a default value for missing keys, avoiding KeyError: d = defaultdict(list); d["key"].append(1)',
        },
        {
          id: 'mc3',
          question: 'What does deque stand for?',
          options: [
            'Decimal Queue',
            'Double-ended queue',
            'Data Equipment',
            'Delete Queue',
          ],
          correctAnswer: 1,
          explanation:
            'deque stands for double-ended queue - allows O(1) append/pop from both ends.',
        },
        {
          id: 'mc4',
          question: 'Which is faster for queue operations: list or deque?',
          options: ['list', 'deque', 'Same speed', 'Depends on size'],
          correctAnswer: 1,
          explanation:
            'deque is faster for queue operations - O(1) for both ends vs O(n) for list.pop(0).',
        },
        {
          id: 'mc5',
          question: 'What is namedtuple?',
          options: [
            'A tuple with named fields',
            'A dictionary',
            'A list with names',
            'A class',
          ],
          correctAnswer: 0,
          explanation:
            'namedtuple creates tuple subclasses with named fields: Point = namedtuple("Point", ["x", "y"])',
        },
      ],
    },
    {
      id: 'testing-debugging',
      title: 'Testing & Debugging',
      content: `# Testing & Debugging

Writing tests and debugging effectively are critical skills for production code and interview success.

## Why Testing Matters

1. **Catches bugs early** - Find issues before they reach production
2. **Enables refactoring** - Change code confidently
3. **Documents behavior** - Tests serve as examples
4. **Interview advantage** - Shows professionalism and thoroughness

## Unit Testing with \`unittest\`

Python's built-in testing framework.

\`\`\`python
import unittest

def add(a, b):
    return a + b

class TestMathOperations(unittest.TestCase):
    def test_add_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)
    
    def test_add_negative_numbers(self):
        self.assertEqual(add(-1, -1), -2)
    
    def test_add_zero(self):
        self.assertEqual(add(5, 0), 5)
    
    def test_add_floats(self):
        self.assertAlmostEqual(add(0.1, 0.2), 0.3, places=7)

if __name__ == '__main__':
    unittest.main()
\`\`\`

### Common Assertions

\`\`\`python
# Equality
self.assertEqual(a, b)
self.assertNotEqual(a, b)

# Truth
self.assertTrue(x)
self.assertFalse(x)

# Identity
self.assertIs(a, b)       # Same object
self.assertIsNot(a, b)

# Membership
self.assertIn(item, collection)
self.assertNotIn(item, collection)

# Exceptions
with self.assertRaises(ValueError):
    function_that_raises()

# Floating point
self.assertAlmostEqual(a, b, places=7)

# None
self.assertIsNone(x)
self.assertIsNotNone(x)
\`\`\`

### Setup and Teardown

\`\`\`python
class TestDatabase(unittest.TestCase):
    def setUp(self):
        """Run before each test"""
        self.db = Database('test.db')
        self.db.connect()
    
    def tearDown(self):
        """Run after each test"""
        self.db.close()
        os.remove('test.db')
    
    def test_insert(self):
        self.db.insert('key', 'value')
        self.assertEqual(self.db.get('key'), 'value')
\`\`\`

## \`pytest\` - Modern Testing

More concise and powerful than unittest.

\`\`\`python
# test_math.py
def add(a, b):
    return a + b

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -1) == -2

def test_add_zero():
    assert add(5, 0) == 5

# Parametrized tests
import pytest

@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    (-1, -1, -2),
    (0, 0, 0),
    (100, 200, 300),
])
def test_add_parametrized(a, b, expected):
    assert add(a, b) == expected
\`\`\`

### Fixtures (pytest)

\`\`\`python
@pytest.fixture
def sample_list():
    """Reusable test data"""
    return [1, 2, 3, 4, 5]

def test_length(sample_list):
    assert len(sample_list) == 5

def test_sum(sample_list):
    assert sum(sample_list) == 15

# Fixture with cleanup
@pytest.fixture
def temp_file():
    f = open('temp.txt', 'w')
    yield f  # Test runs here
    f.close()
    os.remove('temp.txt')
\`\`\`

## Debugging Techniques

### 1. Print Debugging

\`\`\`python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        print(f"Debug: left={left}, right={right}, mid={mid}, arr[mid]={arr[mid]}")
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
\`\`\`

### 2. Logging Module

Better than print statements for production code.

\`\`\`python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_data(data):
    logging.debug(f"Processing {len(data)} items")
    
    for item in data:
        try:
            result = expensive_operation(item)
            logging.info(f"Processed {item}: {result}")
        except Exception as e:
            logging.error(f"Failed to process {item}: {e}")

# Levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
\`\`\`

### 3. Python Debugger (pdb)

Interactive debugging.

\`\`\`python
import pdb

def problematic_function(data):
    result = []
    for item in data:
        pdb.set_trace()  # Breakpoint here
        processed = item * 2
        result.append(processed)
    return result

# In pdb:
# n - next line
# s - step into function
# c - continue
# p variable - print variable
# l - list code
# q - quit
\`\`\`

### 4. Assert Statements

Check assumptions during development.

\`\`\`python
def divide(a, b):
    assert b != 0, "Division by zero!"
    return a / b

def binary_search(arr, target):
    assert len(arr) > 0, "Array must not be empty"
    assert all(arr[i] <= arr[i+1] for i in range(len(arr)-1)), "Array must be sorted"
    # ... implementation
\`\`\`

## Test-Driven Development (TDD)

Write tests before code!

\`\`\`python
# Step 1: Write test (it fails)
def test_is_palindrome():
    assert is_palindrome("racecar") == True
    assert is_palindrome("hello") == False
    assert is_palindrome("") == True

# Step 2: Write minimal code to pass
def is_palindrome(s):
    return s == s[::-1]

# Step 3: Refactor if needed
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
\`\`\`

## Edge Cases to Test

Always test:

1. **Empty input:** \`[], "", {}, None\`
2. **Single element:** \`[1], "a"\`
3. **Duplicates:** \`[1, 1, 1]\`
4. **Negative numbers:** \`[-1, -5]\`
5. **Large inputs:** Stress test
6. **Boundary values:** \`0, MAX_INT, MIN_INT\`
7. **Invalid input:** Wrong types, out of range

\`\`\`python
def test_find_max():
    # Normal case
    assert find_max([1, 5, 3]) == 5
    
    # Edge cases
    assert find_max([1]) == 1              # Single element
    assert find_max([-5, -1, -10]) == -1   # All negative
    assert find_max([5, 5, 5]) == 5        # All same
    
    # Error cases
    with pytest.raises(ValueError):
        find_max([])  # Empty list
\`\`\`

## Debugging Strategies

1. **Read error messages carefully**
   - Line number
   - Error type
   - Traceback

2. **Binary search debugging**
   - Comment out half the code
   - Find which half has the bug
   - Repeat

3. **Rubber duck debugging**
   - Explain code line-by-line to someone (or a rubber duck)
   - Often reveals the bug

4. **Minimal reproducible example**
   - Reduce to smallest code that shows the bug
   - Easier to understand and fix

5. **Check assumptions**
   - Is input what you expect?
   - Are variables the right type?
   - Is the algorithm correct?

## Interview Testing Tips

During interviews:

1. **Ask about testing expectations**
   - Should I write tests?
   - How thorough?

2. **Test as you code**
   - Test each piece immediately
   - Don't wait until the end

3. **Vocalize edge cases**
   - "What if the array is empty?"
   - "Should I handle negative numbers?"

4. **Manual test cases**
   - Walk through your code with example inputs
   - Trace variable values

5. **Think about failure modes**
   - What could go wrong?
   - What inputs would break this?

## Code Coverage

Measure how much code is tested.

\`\`\`bash
# Install coverage
pip install coverage

# Run tests with coverage
coverage run -m pytest

# View report
coverage report

# HTML report
coverage html
\`\`\`

Aim for >80% coverage, but quality > quantity!`,
      quiz: [
        {
          id: 'q1',
          question:
            'Why is logging better than print statements for debugging production code?',
          sampleAnswer:
            'Logging provides: 1) Configurable levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) to filter messages, 2) Timestamps and context automatically, 3) Can write to files, not just console, 4) Can be disabled in production without changing code, 5) Structured format for parsing/analysis. Print statements clutter code, are hard to remove, provide no context, and go to stdout only. In production, you want logging for monitoring and troubleshooting without affecting performance or adding noise.',
          keyPoints: [
            'Logging has configurable levels (DEBUG to CRITICAL)',
            'Automatic timestamps and formatting',
            'Can write to files and multiple outputs',
            'Easy to disable without code changes',
            'Print statements clutter and are hard to manage',
          ],
        },
        {
          id: 'q2',
          question:
            'What edge cases should you always test when writing a function that processes a list?',
          sampleAnswer:
            'Always test: 1) Empty list [], 2) Single element list [x], 3) Two elements (minimum for comparison), 4) All same elements [5,5,5], 5) Already sorted vs unsorted (if relevant), 6) Negative numbers, 7) Duplicates, 8) Large lists (performance), 9) None as input, 10) Wrong type (non-list). These edge cases catch off-by-one errors, null pointer issues, and ensure robust error handling.',
          keyPoints: [
            'Empty list - most common edge case',
            'Single element - boundary condition',
            'All same elements - degenerate case',
            'Negative numbers and duplicates',
            'Type errors and None handling',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain the difference between assertions and exceptions for error handling.',
          sampleAnswer:
            'Assertions (assert) are for debugging and catching programmer errors during development. They check assumptions that should never fail if code is correct, and can be disabled with python -O. Use for: preconditions, postconditions, invariants. Exceptions (try/except, raise) are for handling runtime errors that might legitimately occur: user input errors, network failures, file not found. They stay enabled in production. Rule: Use assert for "this should never happen if my code is correct", use exceptions for "this might happen due to external factors". Never use assert for input validation in production!',
          keyPoints: [
            'assert: debug-time, for programmer errors',
            'Assertions can be disabled (-O flag)',
            'Exceptions: runtime, for expected failures',
            'Exceptions always active in production',
            'Never use assert for production input validation',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the difference between unittest and pytest?',
          options: [
            'No difference',
            'pytest has simpler syntax and better features',
            'unittest is newer',
            'pytest only works with classes',
          ],
          correctAnswer: 1,
          explanation:
            'pytest offers simpler syntax (plain assert), fixtures, parametrization, and better error messages than unittest.',
        },
        {
          id: 'mc2',
          question: 'What does pytest fixture do?',
          options: [
            'Tests the code',
            'Provides reusable setup/teardown',
            'Fixes bugs',
            'Measures performance',
          ],
          correctAnswer: 1,
          explanation:
            'Fixtures provide reusable setup/teardown logic for tests, like database connections or test data.',
        },
        {
          id: 'mc3',
          question: 'What is TDD?',
          options: [
            'Test-Driven Development',
            'Test-Debug-Deploy',
            'Total Development Design',
            'Technical Design Document',
          ],
          correctAnswer: 0,
          explanation:
            'TDD (Test-Driven Development): write tests before code, ensuring testability and clear requirements.',
        },
        {
          id: 'mc4',
          question: 'What does pdb.set_trace() do?',
          options: [
            'Traces function calls',
            'Sets a breakpoint for debugging',
            'Measures execution time',
            'Logs errors',
          ],
          correctAnswer: 1,
          explanation:
            'pdb.set_trace() sets a breakpoint where code execution pauses, allowing interactive debugging.',
        },
        {
          id: 'mc5',
          question: 'What is pytest parametrize used for?',
          options: [
            'Measuring parameters',
            'Running same test with different inputs',
            'Setting up fixtures',
            'Debugging tests',
          ],
          correctAnswer: 1,
          explanation:
            '@pytest.mark.parametrize runs the same test with multiple input sets, reducing code duplication.',
        },
      ],
    },
    {
      id: 'common-pitfalls',
      title: 'Common Python Pitfalls',
      content: `# Common Python Pitfalls

Understanding common mistakes helps you avoid them and debug faster.

## 1. Mutable Default Arguments

### The Problem

\`\`\`python
# ❌ BAD - Mutable default argument
def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2]  # Unexpected!
print(add_item(3))  # [1, 2, 3]  # Same list!

# Why? Default [] is created once when function is defined,
# not each time function is called
\`\`\`

### The Fix

\`\`\`python
# ✅ GOOD - Use None and create new list inside
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item(1))  # [1]
print(add_item(2))  # [2]  # New list each time
print(add_item(3))  # [3]
\`\`\`

**Rule:** Never use mutable objects (\`list\`, \`dict\`, \`set\`) as default arguments. Use \`None\` instead.

---

## 2. Late Binding Closures

### The Problem

\`\`\`python
# ❌ BAD - All functions use final value of i
functions = []
for i in range(3):
    functions.append(lambda: i)

print([f() for f in functions])  # [2, 2, 2]
# Expected [0, 1, 2] but all return 2!

# Why? Lambda captures i by reference, not value
# By the time lambda is called, loop has finished and i=2
\`\`\`

### The Fix

\`\`\`python
# ✅ GOOD - Capture i by value using default argument
functions = []
for i in range(3):
    functions.append(lambda x=i: x)  # x=i captures current value

print([f() for f in functions])  # [0, 1, 2]

# Alternative: Use functools.partial
from functools import partial
functions = [partial(lambda x: x, i) for i in range(3)]
\`\`\`

---

## 3. Modifying List While Iterating

### The Problem

\`\`\`python
# ❌ BAD - Modifying list during iteration
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)

print(numbers)  # [1, 3, 4, 5]  # 4 still there!
# Iterator gets confused when list changes
\`\`\`

### The Fix

\`\`\`python
# ✅ GOOD - Iterate over copy
numbers = [1, 2, 3, 4, 5]
for num in numbers[:]:  # Slice creates copy
    if num % 2 == 0:
        numbers.remove(num)

print(numbers)  # [1, 3, 5]

# ✅ BETTER - List comprehension
numbers = [1, 2, 3, 4, 5]
numbers = [num for num in numbers if num % 2 != 0]
print(numbers)  # [1, 3, 5]
\`\`\`

---

## 4. Integer Division Gotchas

\`\`\`python
# Python 2 vs Python 3
# ❌ In Python 2: 5 / 2 = 2 (integer division)
# ✅ In Python 3: 5 / 2 = 2.5 (float division)

# Use // for integer division in both versions
print(5 // 2)   # 2
print(5 / 2)    # 2.5

# Be careful with negative numbers
print(-7 // 2)  # -4 (not -3!)
# Python rounds toward negative infinity, not toward zero

# For ceiling division
import math
print(math.ceil(7 / 2))  # 4
\`\`\`

---

## 5. Shallow vs Deep Copy

\`\`\`python
# ❌ Assignment doesn't copy, just creates reference
list1 = [[1, 2], [3, 4]]
list2 = list1
list2[0][0] = 99
print(list1)  # [[99, 2], [3, 4]]  # Original changed!

# ✅ Shallow copy (copies outer list only)
import copy
list1 = [[1, 2], [3, 4]]
list2 = copy.copy(list1)  # or list1.copy() or list1[:]
list2[0][0] = 99
print(list1)  # [[99, 2], [3, 4]]  # Inner lists still shared!

# ✅ Deep copy (copies everything)
list1 = [[1, 2], [3, 4]]
list2 = copy.deepcopy(list1)
list2[0][0] = 99
print(list1)  # [[1, 2], [3, 4]]  # Original unchanged
\`\`\`

---

## 6. Name Shadowing

\`\`\`python
# ❌ BAD - Shadowing built-in names
list = [1, 2, 3]  # Shadows built-in list!
# list([1, 2, 3])  # TypeError! Can't use list() anymore

# ✅ GOOD - Use different names
my_list = [1, 2, 3]

# Common names NOT to shadow:
# list, dict, set, str, int, float, bool, type, id, sum, min, max, all, any
\`\`\`

---

## 7. String Concatenation in Loops

\`\`\`python
# ❌ BAD - O(n²) due to string immutability
result = ""
for i in range(10000):
    result += str(i)  # Creates new string each time!

# ✅ GOOD - O(n) using list and join
parts = []
for i in range(10000):
    parts.append(str(i))
result = ''.join(parts)

# ✅ BEST - List comprehension
result = ''.join([str(i) for i in range(10000)])
\`\`\`

---

## 8. Forgetting to Return

\`\`\`python
# ❌ BAD - Forgot return statement
def add(a, b):
    result = a + b  # Calculated but not returned!

print(add(2, 3))  # None

# ✅ GOOD - Return the result
def add(a, b):
    return a + b

print(add(2, 3))  # 5
\`\`\`

---

## 9. Using \`is\` Instead of \`==\`

\`\`\`python
# is checks identity (same object), == checks equality (same value)

a = [1, 2, 3]
b = [1, 2, 3]

print(a == b)  # True (same values)
print(a is b)  # False (different objects)

# ❌ Common mistake with integers
a = 1000
b = 1000
print(a is b)  # False! (CPython caches -5 to 256 only)

# ✅ Use == for value comparison
print(a == b)  # True

# ✅ Use is only for None, True, False
if value is None:
    pass
\`\`\`

---

## 10. Catching All Exceptions

\`\`\`python
# ❌ BAD - Catches everything, even KeyboardInterrupt
try:
    risky_operation()
except:  # Bare except catches ALL exceptions!
    pass

# ✅ GOOD - Catch specific exceptions
try:
    risky_operation()
except ValueError:
    handle_value_error()
except KeyError:
    handle_key_error()

# ✅ If you must catch all, use Exception
try:
    risky_operation()
except Exception as e:
    log_error(e)
    # Still allows KeyboardInterrupt, SystemExit to pass through
\`\`\`

---

## 11. Mixing Tabs and Spaces

\`\`\`python
# ❌ BAD - Invisible but causes IndentationError
def func():
    if True:
→   print("tab")    # Uses tab
        print("spaces")  # Uses spaces
# IndentationError: inconsistent use of tabs and spaces

# ✅ GOOD - Use spaces consistently (4 spaces per PEP 8)
def func():
    if True:
        print("spaces")
        print("spaces")
\`\`\`

---

## 12. Circular Imports

\`\`\`python
# module_a.py
from module_b import func_b

def func_a():
    return func_b()

# module_b.py
from module_a import func_a  # Circular import!

def func_b():
    return func_a()

# ❌ ImportError: cannot import name 'func_a'

# ✅ FIX: Import inside function (lazy import)
# module_b.py
def func_b():
    from module_a import func_a  # Import when called
    return func_a()
\`\`\`

---

## 13. Global Variables

\`\`\`python
# ❌ BAD - Modifying global without declaration
count = 0

def increment():
    count = count + 1  # UnboundLocalError!
    # Python sees assignment, treats count as local

# ✅ GOOD - Declare global
count = 0

def increment():
    global count
    count = count + 1

# ✅ BETTER - Avoid globals, use return values
def increment(count):
    return count + 1

count = increment(count)
\`\`\`

---

## How to Avoid Pitfalls

1. **Use linters:** \`pylint\`, \`flake8\`, \`mypy\`
2. **Follow PEP 8:** Python style guide
3. **Write tests:** Catch bugs early
4. **Code reviews:** Learn from others
5. **Read error messages:** They're usually clear
6. **Use IDE warnings:** They catch many issues
7. **Keep learning:** Python has many gotchas

## Interview Relevance

Interviewers may:
- Introduce these bugs in code review questions
- Ask "what's wrong with this code?"
- Test if you can spot issues quickly

Know these pitfalls to debug faster and write cleaner code!`,
      quiz: [
        {
          id: 'q1',
          question:
            'Why do mutable default arguments cause unexpected behavior?',
          sampleAnswer:
            "Default arguments are evaluated once when the function is defined, not each time it's called. For mutable defaults like [] or {}, the same object is reused across all calls. When you modify it (e.g., append), those changes persist. Example: def f(l=[]): l.append(1); return l → f() returns [1], f() returns [1,1], etc. Fix: Use None and create new object inside: def f(l=None): if l is None: l = []. This creates a fresh list each call.",
          keyPoints: [
            'Default args evaluated at function definition, not call time',
            'Same mutable object reused across calls',
            'Modifications persist between calls',
            'Fix: Use None, create new object inside',
            'Common with [], {}, but also class instances',
          ],
        },
        {
          id: 'q2',
          question: 'What is the difference between "is" and "==" in Python?',
          sampleAnswer:
            '"is" checks identity (same object in memory), "==" checks equality (same value). Example: a = [1,2]; b = [1,2] → a == b is True (same value) but a is b is False (different objects). Use "==" for value comparison (99% of cases). Use "is" only for singletons: None, True, False. Python caches small integers (-5 to 256), so a=100; b=100; a is b might be True, but a=1000; b=1000; a is b is False. Never rely on "is" for numbers/strings.',
          keyPoints: [
            'is: identity check (same object)',
            '==: equality check (same value)',
            'Use == for value comparison',
            'Use is only for None, True, False',
            "Small integers cached, don't rely on identity",
          ],
        },
        {
          id: 'q3',
          question:
            'Why is string concatenation in a loop inefficient and how do you fix it?',
          sampleAnswer:
            'String concatenation in loops is O(n²) because strings are immutable in Python. Each s += "x" creates a new string by copying all existing characters plus the new one. For n iterations: 1st copy=1 char, 2nd=2 chars, ..., nth=n chars. Total: 1+2+3+...+n = O(n²). Fix: Build a list and use join(): parts = []; for x in items: parts.append(str(x)); result = "".join(parts). This is O(n) because join() only copies characters once. Example: 10,000 concatenations take ~50 million operations with +=, but only 10,000 with join().',
          keyPoints: [
            'Strings immutable - each += copies all chars',
            'Loop concatenation: O(n²) time',
            'Solution: Build list, use join()',
            'join() is O(n) - copies once',
            'Huge performance difference for large strings',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is wrong with def func(x, lst=[])?',
          options: [
            'Nothing',
            'Mutable default shared across calls',
            'Lists cannot be default',
            'Syntax error',
          ],
          correctAnswer: 1,
          explanation:
            'Mutable defaults are shared. Use None: def func(x, lst=None): lst = lst or []',
        },
        {
          id: 'mc2',
          question: 'What is the difference between is and ==?',
          options: [
            'No difference',
            'is checks identity, == checks value',
            'is is faster',
            '== checks types',
          ],
          correctAnswer: 1,
          explanation:
            'is checks if same object in memory. == checks if values are equal.',
        },
        {
          id: 'mc3',
          question: 'What is wrong with: for item in list: list.remove(item)?',
          options: [
            'Nothing',
            'Modifying list while iterating skips elements',
            'Syntax error',
            'Too slow',
          ],
          correctAnswer: 1,
          explanation:
            'Never modify list while iterating - indices shift, skipping elements. Use list comprehension.',
        },
        {
          id: 'mc4',
          question: 'What is the difference between copy() and deepcopy()?',
          options: [
            'No difference',
            'copy() shallow, deepcopy() copies nested too',
            'deepcopy() faster',
            'copy() makes backups',
          ],
          correctAnswer: 1,
          explanation:
            'copy() is shallow. deepcopy() recursively copies all nested objects.',
        },
        {
          id: 'mc5',
          question: 'What is the difference between / and //?',
          options: [
            'No difference',
            '/ is float division, // is integer division',
            '// is deprecated',
            '/ rounds up',
          ],
          correctAnswer: 1,
          explanation:
            '/ always returns float. // returns integer (floor division): 5//2 = 2, 5/2 = 2.5',
        },
      ],
    },
  ],
};

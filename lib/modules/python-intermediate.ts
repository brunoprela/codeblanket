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
          question:
            'Which mode should you use to add content to the end of a file?',
          options: ["'r'", "'w'", "'a'", "'x'"],
          correctAnswer: 2,
          explanation:
            "'a' mode opens the file for appending, adding new content to the end without removing existing content.",
        },
      ],
      discussion: [
        {
          question: 'Why use pathlib instead of os.path?',
          answer:
            "pathlib provides an object-oriented interface that's more intuitive and readable. It supports the / operator for joining paths and has convenient methods like read_text() and write_text(). It's the modern, recommended approach.",
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
      ],
      discussion: [
        {
          question: 'When should you use a bare except clause?',
          answer:
            'Rarely! Bare except catches all exceptions including system exits and keyboard interrupts, which you usually want to propagate. Use specific exceptions or "except Exception" at most. Only use bare except for logging at the top level.',
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
          question: 'What Python value becomes null in JSON?',
          options: ['0', 'False', 'None', '""'],
          correctAnswer: 2,
          explanation: 'Python None is converted to JSON null.',
        },
      ],
      discussion: [
        {
          question: 'When should you use CSV vs JSON?',
          answer:
            'Use CSV for tabular data, especially when working with spreadsheets or databases. Use JSON for hierarchical data, APIs, or when you need to preserve data types. JSON is more flexible but CSV is more compact for simple tables.',
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
      ],
      discussion: [
        {
          question: 'When should you avoid using regex?',
          answer:
            'Avoid regex for simple string operations (use str.startswith(), str.split(), etc.), for parsing complex structures like HTML/XML (use dedicated parsers), or when the pattern becomes too complex to understand. Regex is powerful but can be hard to maintain.',
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
      ],
      discussion: [
        {
          question:
            'Why use datetime instead of just storing timestamps as integers?',
          answer:
            'datetime provides timezone awareness, easier date arithmetic, readable formatting, and validation. Raw timestamps are just numbers and require manual calculations for anything beyond basic storage.',
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
      ],
    },
  ],
};

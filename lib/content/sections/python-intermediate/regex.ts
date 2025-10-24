/**
 * Regular Expressions Section
 */

export const regexSection = {
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
- *****: 0 or more
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
pattern = re.compile(r''
    \\d{3}   # Area code
    -        # Separator
    \\d{4}   # Number
'', re.VERBOSE)  # Allow comments and whitespace
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
};

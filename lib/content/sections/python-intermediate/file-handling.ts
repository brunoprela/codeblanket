/**
 * File Handling and I/O Section
 */

export const filehandlingSection = {
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
    f.writelines (lines)
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
        fout.write (line.upper())
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
};

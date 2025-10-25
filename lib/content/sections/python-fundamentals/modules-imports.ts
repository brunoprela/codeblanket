/**
 * Modules and Imports Section
 */

export const modulesimportsSection = {
  id: 'modules-imports',
  title: 'Modules and Imports',
  content: `# Modules and Imports

Modules are Python files containing reusable code. The import system lets you use code from other files and libraries.

## What is a Module?

A module is simply a Python file (.py) containing variables, functions, and classes.

\`\`\`python
# File: my_module.py
def greet (name):
    return f"Hello, {name}!"

PI = 3.14159

class Circle:
    def __init__(self, radius):
        self.radius = radius
\`\`\`

## Basic Imports

### Import Entire Module

\`\`\`python
import math

print(math.sqrt(16))  # 4.0
print(math.pi)        # 3.141592653589793
\`\`\`

### Import Specific Items

\`\`\`python
from math import sqrt, pi

print(sqrt(16))  # 4.0
print(pi)        # 3.141592653589793
\`\`\`

### Import All (Not Recommended)

\`\`\`python
from math import *  # ❌ Avoid - pollutes namespace

print(sqrt(16))  # Works but unclear where sqrt comes from
\`\`\`

### Import with Alias

\`\`\`python
import numpy as np  # Common convention
import pandas as pd

arr = np.array([1, 2, 3])
\`\`\`

## Importing Your Own Modules

\`\`\`python
# File structure:
# my_project/
#   main.py
#   utils.py
#   helpers.py

# In main.py:
import utils
from helpers import calculate

result = utils.process_data()
value = calculate(10)
\`\`\`

## Package Structure

A package is a directory containing multiple modules and a special \`__init__.py\` file.

\`\`\`python
# File structure:
# my_package/
#   __init__.py
#   module1.py
#   module2.py

# Importing from package:
from my_package import module1
from my_package.module2 import some_function
\`\`\`

## The \`__name__\` Variable

Every Python file has a built-in \`__name__\` variable.

\`\`\`python
# File: script.py
def main():
    print("Running main function")

# This code only runs when file is executed directly
if __name__ == "__main__":
    main()
    
# When imported: __name__ is "script"
# When run directly: __name__ is "__main__"
\`\`\`

**Why this matters:**
- Allows file to work as both module and script
- Common pattern in Python projects
- Prevents code from running when imported

## Common Standard Library Modules

### Math and Random

\`\`\`python
import math
import random

print(math.ceil(3.2))        # 4
print(math.floor(3.8))       # 3
print(random.randint(1, 10)) # Random int 1-10
print(random.choice(['a', 'b', 'c']))  # Random choice
\`\`\`

### Datetime

\`\`\`python
from datetime import datetime, timedelta

now = datetime.now()
print(now)  # Current date and time

tomorrow = now + timedelta (days=1)
print(tomorrow)
\`\`\`

### Collections

\`\`\`python
from collections import Counter, defaultdict

counts = Counter(['a', 'b', 'a', 'c', 'b', 'a'])
print(counts)  # Counter({'a': 3, 'b': 2, 'c': 1})

# defaultdict - no KeyError on missing keys
d = defaultdict (list)
d['key'].append('value')  # Works without initializing
\`\`\`

## Relative vs Absolute Imports

\`\`\`python
# File structure:
# project/
#   package/
#     __init__.py
#     module_a.py
#     module_b.py

# Absolute import (from project root):
from package.module_a import function_a

# Relative import (from within package):
from .module_a import function_a  # Same directory
from ..other_package import something  # Parent directory
\`\`\`

## Import Best Practices

✅ **Do:**
- Import at top of file
- Use absolute imports for clarity
- Group imports: standard library → third-party → local
- One import per line for readability

\`\`\`python
# Good
import os
import sys

import numpy as np
import pandas as pd

from my_package import my_module
\`\`\`

❌ **Don't:**
- Use \`from module import *\`
- Import in the middle of code
- Create circular imports

## Installing Third-Party Packages

\`\`\`bash
# Install package
pip install requests

# Install specific version
pip install requests==2.28.0

# Install from requirements.txt
pip install -r requirements.txt

# List installed packages
pip list

# Show package info
pip show requests
\`\`\`

## Creating requirements.txt

\`\`\`bash
# Save current environment packages
pip freeze > requirements.txt
\`\`\`

Example requirements.txt:
\`\`\`
requests==2.28.1
numpy==1.23.0
pandas==1.4.3
\`\`\`

## Common Import Errors

### ModuleNotFoundError

\`\`\`python
import non_existent_module  # ModuleNotFoundError

# Solutions:
# 1. Check spelling
# 2. Install package: pip install package_name
# 3. Check Python path
\`\`\`

### ImportError

\`\`\`python
from math import non_existent_function  # ImportError

# Solution: Check what's available
import math
print(dir (math))  # List all available items
\`\`\`

### Circular Import

\`\`\`python
# File: a.py
import b

# File: b.py  
import a  # Circular import!

# Solution: Restructure code or use import inside function
\`\`\`

## Quick Reference

| Import Type | Syntax | When to Use |
|-------------|--------|-------------|
| **Full module** | \`import math\` | Use module namespace |
| **Specific items** | \`from math import sqrt\` | Use items directly |
| **With alias** | \`import numpy as np\` | Shorter name |
| **Multiple items** | \`from os import path, getcwd\` | Several items |
| **Package** | \`from package import module\` | Organized code |`,
};

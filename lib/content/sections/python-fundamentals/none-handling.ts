/**
 * None and Null Values Section
 */

export const nonehandlingSection = {
  id: 'none-handling',
  title: 'None and Null Values',
  content: `# None and Null Values

## What is None?

\`None\` is Python\'s null value - it represents the absence of a value or a null reference.

\`\`\`python
# None is a singleton object
result = None
print(type(None))  # <class 'NoneType'>

# Only one None exists in Python
a = None
b = None
print(a is b)  # True (same object)
\`\`\`

## Common Uses of None

### 1. Function Default Return
\`\`\`python
def greet():
    print("Hello!")
    # No return statement

result = greet()  # Prints "Hello!"
print(result)     # None

# Explicit return
def calculate (x):
    if x < 0:
        return None  # Indicate failure/invalid
    return x * 2
\`\`\`

### 2. Default Function Parameters
\`\`\`python
def log_message (message, timestamp=None):
    """Log message with optional timestamp"""
    if timestamp is None:
        timestamp = datetime.now()
    print(f"[{timestamp}] {message}")

# Use default (None becomes current time)
log_message("Server started")

# Provide custom timestamp
log_message("Error occurred", custom_time)
\`\`\`

### 3. Placeholder for Optional Values
\`\`\`python
class User:
    def __init__(self, name, email=None, phone=None):
        self.name = name
        self.email = email  # Optional
        self.phone = phone  # Optional

user1 = User("Alice", email="alice@example.com")
user2 = User("Bob")  # email and phone are None
\`\`\`

## Checking for None

### Use 'is' Not '=='
\`\`\`python
value = None

# Correct way
if value is None:
    print("Value is None")

if value is not None:
    print("Value exists")

# Wrong way (works but not idiomatic)
if value == None:  # Don't do this!
    print("Value is None")
\`\`\`

**Why 'is' instead of '=='?**
- \`is\` checks object identity (same object in memory)
- \`==\` checks value equality (can be overridden by classes)
- Since None is a singleton, \`is\` is more efficient and correct

### Truthy vs Falsy
\`\`\`python
# None is falsy
if not None:
    print("None is falsy")  # This prints

# But don't use truthiness to check for None!
value = None
if not value:  # Bad - could be 0, "", [], etc.
    print("Might not be None!")

# Better - explicit check
if value is None:
    print("Definitely None")
\`\`\`

## Common Pitfalls

### 1. Mutable Default Arguments
\`\`\`python
# WRONG - dangerous!
def add_item (item, items=[]):
    items.append (item)
    return items

list1 = add_item("a")  # ["a"]
list2 = add_item("b")  # ["a", "b"] - UNEXPECTED!

# RIGHT - use None as default
def add_item (item, items=None):
    if items is None:
        items = []
    items.append (item)
    return items

list1 = add_item("a")  # ["a"]
list2 = add_item("b")  # ["b"] - correct!
\`\`\`

**Why?** Default arguments are created once when the function is defined, not each time it's called. Mutable objects (lists, dicts) are shared between calls!

### 2. Forgetting to Return
\`\`\`python
def calculate_discount (price):
    if price > 100:
        return price * 0.9
    # Forgot to return for price <= 100

result = calculate_discount(50)
print(result)  # None - bug!
\`\`\`

### 3. Confusing None with False, 0, or ""
\`\`\`python
def get_user_age (user_id):
    # Returns 0 for baby, None for invalid ID
    if user_id < 0:
        return None
    return 0  # Baby\'s age

age = get_user_age(1)
if age:  # Wrong! 0 is falsy
    print(f"Age: {age}")
else:
    print("No age")  # Prints for babies!

# Correct - explicit None check
if age is not None:
    print(f"Age: {age}")  # Works for babies
else:
    print("Invalid user")
\`\`\`

## None in Data Structures

### Lists
\`\`\`python
# None as placeholder
data = [1, 2, None, 4, 5]

# Filter out None values
filtered = [x for x in data if x is not None]
print(filtered)  # [1, 2, 4, 5]

# Count None values
none_count = data.count(None)
print(none_count)  # 1
\`\`\`

### Dictionaries
\`\`\`python
config = {
    "host": "localhost",
    "port": 8080,
    "password": None  # Explicit "no password"
}

# Check if key exists vs is None
if "password" in config:
    print("Password key exists")
    if config["password"] is None:
        print("But password is None")

# Distinguish missing vs None
print(config.get("username"))      # None (missing key)
print(config.get("password"))      # None (explicit value)

# Use default to avoid None
username = config.get("username", "guest")  # "guest"
\`\`\`

## None vs Empty Collections

\`\`\`python
# These are different!
value1 = None      # No value
value2 = []        # Empty list (still a value)
value3 = ""        # Empty string (still a value)
value4 = 0         # Zero (still a value)

# All are falsy, but only one is None
print(bool (value1))  # False
print(bool (value2))  # False
print(bool (value3))  # False
print(bool (value4))  # False

print(value1 is None)  # True
print(value2 is None)  # False
print(value3 is None)  # False
print(value4 is None)  # False
\`\`\`

## Best Practices

### 1. Use None for "Not Set" or "Missing"
\`\`\`python
class Config:
    def __init__(self):
        self.database_url = None  # Not configured yet
        self.api_key = None        # Not provided

    def is_configured (self):
        return self.database_url is not None
\`\`\`

### 2. Document When Functions Return None
\`\`\`python
def find_user (user_id: int) -> User | None:
    """
    Find user by ID.
    
    Returns:
        User object if found, None if not found
    """
    # ... search logic
    if not found:
        return None
    return user
\`\`\`

### 3. Avoid Returning None When Possible
\`\`\`python
# Instead of returning None for "not found":
def get_users():
    if no_users:
        return None  # Caller must check

    return users

# Better - return empty list:
def get_users():
    if no_users:
        return []  # Caller can iterate immediately
    
    return users

# Now this always works:
for user in get_users():
    print(user)
\`\`\`

### 4. Use None for Optional Type Hints
\`\`\`python
from typing import Optional

def greet (name: str, title: Optional[str] = None) -> str:
    """
    Optional[str] is equivalent to str | None
    """
    if title is None:
        return f"Hello, {name}!"
    return f"Hello, {title} {name}!"
\`\`\`

## Real-World Patterns

### Null Object Pattern Alternative
\`\`\`python
# Instead of returning None and checking everywhere
def get_user_permissions (user_id):
    user = find_user (user_id)
    if user is None:
        return []  # Empty permissions instead of None
    return user.permissions

# No None check needed
permissions = get_user_permissions(123)
if "admin" in permissions:  # Works even if empty
    print("User is admin")
\`\`\`

### None Guard Pattern
\`\`\`python
def process_data (data=None):
    """Early return for None"""
    if data is None:
        return []  # Or raise ValueError
    
    # Process data knowing it's not None
    return [item * 2 for item in data]
\`\`\`

### Chaining with None
\`\`\`python
# Without None handling
def get_user_city (user_id):
    user = get_user (user_id)
    if user is None:
        return None
    
    address = user.get_address()
    if address is None:
        return None
    
    return address.city

# Better - use default values
def get_user_city (user_id):
    user = get_user (user_id)
    if user is None:
        return "Unknown"
    
    address = user.get_address()
    return address.city if address else "Unknown"
\`\`\`

## Summary

✅ **Do:**
- Use \`is None\` to check for None
- Use None for missing/unset values
- Document when functions can return None
- Use None as default for mutable parameters

❌ **Don't:**
- Use \`== None\` (use \`is None\`)
- Use truthiness to check for None (use explicit \`is None\`)
- Use mutable defaults (use None instead)
- Return None when empty collection is better`,
};

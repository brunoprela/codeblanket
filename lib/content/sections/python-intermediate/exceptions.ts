/**
 * Exception Handling Section
 */

export const exceptionsSection = {
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
    value = int (input("Enter a number: "))
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
def validate_age (age):
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
    
    def withdraw (self, amount):
        if amount > self.balance:
            raise InsufficientFundsError (self.balance, amount)
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
};

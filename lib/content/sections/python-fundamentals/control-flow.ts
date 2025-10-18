/**
 * Control Flow Section
 */

export const controlflowSection = {
  id: 'control-flow',
  title: 'Control Flow',
  content: `# Control Flow

## If Statements

\`\`\`python
age = 18

if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

# Inline if (ternary operator)
status = "Adult" if age >= 18 else "Minor"
\`\`\`

## Comparison Operators

\`\`\`python
==  # Equal to
!=  # Not equal to
>   # Greater than
<   # Less than
>=  # Greater than or equal to
<=  # Less than or equal to

# Chaining comparisons
if 0 < x < 10:
    print("x is between 0 and 10")
\`\`\`

## Logical Operators

\`\`\`python
and  # Both conditions must be True
or   # At least one condition must be True
not  # Inverts the boolean value

# Examples
if age >= 18 and has_license:
    print("Can drive")

if is_weekend or is_holiday:
    print("No work!")
\`\`\`

## For Loops

\`\`\`python
# Iterate over sequence
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Using range()
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 6):  # 1, 2, 3, 4, 5
    print(i)

for i in range(0, 10, 2):  # 0, 2, 4, 6, 8
    print(i)

# Enumerate for index and value
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
\`\`\`

## While Loops

\`\`\`python
count = 0
while count < 5:
    print(count)
    count += 1

# Infinite loop with break
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input == 'quit':
        break
    print(f"You entered: {user_input}")
\`\`\`

## Loop Control

\`\`\`python
# break - exit loop completely
for i in range(10):
    if i == 5:
        break  # Stop at 5
    print(i)  # Prints 0, 1, 2, 3, 4

# continue - skip to next iteration
for i in range(5):
    if i == 2:
        continue  # Skip 2
    print(i)  # Prints 0, 1, 3, 4

# else clause - executes if loop completes without break
for i in range(5):
    if i == 10:
        break
else:
    print("Loop completed")  # This will print
\`\`\`

## Match-Case (Python 3.10+)

\`\`\`python
def http_status(status):
    match status:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case 500:
            return "Server Error"
        case _:
            return "Unknown"
\`\`\``,
  videoUrl: 'https://www.youtube.com/watch?v=Zp5MuPOtsSY',
};

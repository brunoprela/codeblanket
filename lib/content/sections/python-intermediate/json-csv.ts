/**
 * Working with JSON and CSV Section
 */

export const jsoncsvSection = {
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
};

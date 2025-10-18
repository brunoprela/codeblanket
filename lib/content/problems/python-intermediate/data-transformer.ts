/**
 * Multi-Format Data Transformer
 * Problem ID: intermediate-data-transformer
 * Order: 8
 */

import { Problem } from '../../../types';

export const intermediate_data_transformerProblem: Problem = {
  id: 'intermediate-data-transformer',
  title: 'Multi-Format Data Transformer',
  difficulty: 'Hard',
  description: `Convert data between JSON, CSV, and Python dict formats.

**Supported Conversions:**
- JSON ↔ CSV
- JSON ↔ Dict
- CSV ↔ Dict

**Requirements:**
- Handle nested JSON for CSV conversion (flatten keys)
- Preserve data types where possible
- Handle errors gracefully

**Example:**
\`\`\`python
# JSON to CSV
json_data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
csv_string = json_to_csv(json_data)
\`\`\``,
  examples: [
    {
      input: 'json_to_csv([{"name": "Alice", "age": 30}])',
      output: '"name,age\\nAlice,30"',
    },
  ],
  constraints: [
    'Handle nested JSON objects',
    'Preserve data types',
    'Validate input formats',
  ],
  hints: [
    'Use json.dumps/loads for JSON',
    'Use csv.DictWriter for CSV',
    'Flatten nested dicts with dot notation',
  ],
  starterCode: `import json
import csv
from io import StringIO

def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Key prefix for nested keys
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
        
    Examples:
        >>> flatten_dict({"a": {"b": 1}})
        {"a.b": 1}
    """
    pass


def json_to_csv(json_data):
    """
    Convert JSON array to CSV string.
    
    Args:
        json_data: List of dictionaries
        
    Returns:
        CSV formatted string
    """
    pass


def csv_to_json(csv_string):
    """
    Convert CSV string to JSON array.
    
    Args:
        csv_string: CSV formatted string
        
    Returns:
        List of dictionaries
    """
    pass


def dict_to_json_file(data, filename):
    """Write dictionary to JSON file."""
    pass


def json_file_to_dict(filename):
    """Read JSON file to dictionary."""
    pass


# Test
data = [
    {"name": "Alice", "age": 30, "address": {"city": "NYC"}},
    {"name": "Bob", "age": 25, "address": {"city": "LA"}}
]

csv_string = json_to_csv(data)
print("CSV output:")
print(csv_string)

json_data = csv_to_json(csv_string)
print("\\nJSON output:")
print(json_data)
`,
  testCases: [
    {
      input: [[{ name: 'Alice', age: 30 }]],
      expected: 'name,age\\nAlice,30',
    },
  ],
  solution: `import json
import csv
from io import StringIO

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def json_to_csv(json_data):
    if not json_data:
        return ""
    
    # Flatten all dictionaries
    flattened = [flatten_dict(item) for item in json_data]
    
    # Get all unique keys
    fieldnames = set()
    for item in flattened:
        fieldnames.update(item.keys())
    fieldnames = sorted(fieldnames)
    
    # Write to CSV
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(flattened)
    
    return output.getvalue()


def csv_to_json(csv_string):
    input_stream = StringIO(csv_string)
    reader = csv.DictReader(input_stream)
    return list(reader)


def dict_to_json_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def json_file_to_dict(filename):
    with open(filename, 'r') as f:
        return json.load(f)`,
  timeComplexity: 'O(n*k) where n is records, k is keys per record',
  spaceComplexity: 'O(n*k)',
  order: 8,
  topic: 'Python Intermediate',
};

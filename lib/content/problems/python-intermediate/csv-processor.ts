/**
 * CSV Data Processor
 * Problem ID: intermediate-csv-processor
 * Order: 4
 */

import { Problem } from '../../../types';

export const intermediate_csv_processorProblem: Problem = {
  id: 'intermediate-csv-processor',
  title: 'CSV Data Processor',
  difficulty: 'Medium',
  description: `Process CSV data with filtering and aggregation.

**Tasks:**
- Read CSV file
- Filter rows based on condition
- Calculate aggregates (sum, average, count)
- Write results to new CSV

**Example CSV:**
\`\`\`
name,age,salary,department
Alice,30,70000,Engineering
Bob,25,60000,Sales
Charlie,35,80000,Engineering
\`\`\`

Create functions to:
1. Filter by department
2. Calculate average salary
3. Export filtered data`,
  examples: [
    {
      input: 'filter_by_department("data.csv", "Engineering")',
      output: '[{"name": "Alice", "age": 30, ...}, ...]',
    },
  ],
  constraints: [
    'Use csv.DictReader',
    'Handle missing fields',
    'Write output as CSV',
  ],
  hints: [
    'DictReader treats first row as headers',
    'Convert numeric strings to numbers',
    'Use csv.DictWriter for output',
  ],
  starterCode: `import csv

def filter_by_department(input_file, department):
    """
    Filter CSV rows by department.
    
    Args:
        input_file: Input CSV filename
        department: Department to filter by
        
    Returns:
        List of dictionaries matching department
    """
    pass


def calculate_average_salary(input_file, department=None):
    """
    Calculate average salary, optionally filtered by department.
    
    Args:
        input_file: Input CSV filename
        department: Optional department filter
        
    Returns:
        Average salary as float
    """
    pass


def export_filtered_data(input_file, output_file, department):
    """
    Export filtered data to new CSV file.
    
    Args:
        input_file: Input CSV filename
        output_file: Output CSV filename
        department: Department to filter by
    """
    pass


# Test
employees = filter_by_department("employees.csv", "Engineering")
print(f"Found {len(employees)} engineers")

avg = calculate_average_salary("employees.csv", "Engineering")
print(f"Average salary: {avg:,.2f}")

export_filtered_data("employees.csv", "engineers.csv", "Engineering")
`,
  testCases: [
    {
      input: ['employees.csv', 'Engineering'],
      expected: 2,
    },
  ],
  solution: `import csv

def filter_by_department(input_file, department):
    results = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('department') == department:
                # Convert numeric fields
                row['age'] = int(row['age'])
                row['salary'] = float(row['salary'])
                results.append(row)
    return results


def calculate_average_salary(input_file, department=None):
    salaries = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if department is None or row.get('department') == department:
                salaries.append(float(row['salary']))
    
    return sum(salaries) / len(salaries) if salaries else 0.0


def export_filtered_data(input_file, output_file, department):
    filtered = filter_by_department(input_file, department)
    
    if not filtered:
        return
    
    with open(output_file, 'w', newline=') as f:
        fieldnames = filtered[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)`,
  timeComplexity: 'O(n) where n is number of rows',
  spaceComplexity: 'O(n) for filtered results',
  order: 4,
  topic: 'Python Intermediate',
};

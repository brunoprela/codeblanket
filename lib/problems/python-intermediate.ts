/**
 * Python Intermediate problems - Building practical Python skills
 */

import { Problem } from '../types';
import { pythonIntermediateBatch1 } from './python-intermediate-batch1';
import { pythonIntermediateBatch2 } from './python-intermediate-batch2';
import { pythonIntermediateBatch3 } from './python-intermediate-batch3';

export const pythonIntermediateProblems: Problem[] = [
  {
    id: 'intermediate-file-word-frequency',
    title: 'File Word Frequency Counter',
    difficulty: 'Medium',
    description: `Read a text file and count the frequency of each word.

**Note:** A virtual test file is created for you in the browser environment using Pyodide's filesystem.

**Requirements:**
- Case-insensitive counting
- Ignore punctuation
- Handle file not found errors
- Return dictionary of word frequencies sorted by count

**Example File Content:**
\`\`\`
The quick brown fox jumps over the lazy dog.
The dog was not that lazy.
\`\`\`

**Expected Output:**
\`\`\`python
{'the': 3, 'dog': 2, 'lazy': 2, 'quick': 1, 'brown': 1, ...}
\`\`\``,
    examples: [
      {
        input: 'filename = "text.txt"',
        output: "{'the': 3, 'dog': 2, 'lazy': 2, ...}",
      },
    ],
    constraints: [
      'Handle FileNotFoundError',
      'Case-insensitive',
      'Remove punctuation',
    ],
    hints: [
      'Use string.punctuation for punctuation',
      'Convert to lowercase before counting',
      'Use try-except for file operations',
    ],
    starterCode: `# Setup: Create virtual test file (for browser environment)
with open('test.txt', 'w') as f:
    f.write("""The quick brown fox jumps over the lazy dog.
The dog was not that lazy.""")

def count_word_frequency(filename):
    """
    Count word frequency in a text file.
    
    Args:
        filename: Path to text file
        
    Returns:
        Dictionary of word -> count, sorted by count (descending)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Examples:
        >>> count_word_frequency("text.txt")
        {'the': 3, 'dog': 2, 'lazy': 2, ...}
    """
    pass


# Test
try:
    result = count_word_frequency("test.txt")
    print(result)
except FileNotFoundError as e:
    print(f"Error: {e}")
`,
    testCases: [
      {
        input: ['test.txt'],
        expected: {
          the: 3,
          dog: 2,
          lazy: 2,
          brown: 1,
          fox: 1,
          jumps: 1,
          not: 1,
          over: 1,
          quick: 1,
          that: 1,
          was: 1,
        },
      },
    ],
    solution: `# Setup: Create virtual test file (for browser environment)
with open('test.txt', 'w') as f:
    f.write("""The quick brown fox jumps over the lazy dog.
The dog was not that lazy.""")

import string
from collections import Counter

def count_word_frequency(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found")
    
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator).lower()
    
    # Count words
    words = text.split()
    word_count = Counter(words)
    
    # Sort by frequency (descending), then alphabetically for ties
    sorted_words = dict(sorted(word_count.items(), 
                               key=lambda x: (-x[1], x[0])))
    
    return sorted_words`,
    timeComplexity: 'O(n) where n is file size',
    spaceComplexity: 'O(w) where w is number of unique words',
    order: 1,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-custom-validator',
    title: 'Data Validator with Custom Exceptions',
    difficulty: 'Medium',
    description: `Create a data validator that validates user input and raises custom exceptions.

**Custom Exceptions:**
- \`InvalidEmailError\` - for invalid email format
- \`InvalidAgeError\` - for ages outside 0-150 range
- \`InvalidPhoneError\` - for invalid phone format

**Validation Rules:**
- Email: must contain @ and .
- Age: integer between 0 and 150
- Phone: format XXX-XXX-XXXX (X = digit)

Create a \`validate_user_data\` function that checks all fields.`,
    examples: [
      {
        input: 'validate_user_data("test@email.com", 25, "555-123-4567")',
        output: 'Returns True if all valid',
      },
      {
        input: 'validate_user_data("invalid", 25, "555-123-4567")',
        output: 'Raises InvalidEmailError',
      },
    ],
    constraints: [
      'Create custom exception classes',
      'Validate all three fields',
      'Provide descriptive error messages',
    ],
    hints: [
      'Inherit from Exception class',
      'Use regex for phone validation',
      'Check email contains @ and .',
    ],
    starterCode: `import re

class InvalidEmailError(Exception):
    """Raised when email format is invalid."""
    pass


class InvalidAgeError(Exception):
    """Raised when age is out of valid range."""
    pass


class InvalidPhoneError(Exception):
    """Raised when phone format is invalid."""
    pass


def validate_user_data(email, age, phone):
    """
    Validate user data.
    
    Args:
        email: Email address string
        age: Age integer
        phone: Phone string in XXX-XXX-XXXX format
        
    Returns:
        True if all validations pass
        
    Raises:
        InvalidEmailError: If email is invalid
        InvalidAgeError: If age is out of range
        InvalidPhoneError: If phone format is wrong
        
    Examples:
        >>> validate_user_data("test@example.com", 25, "555-123-4567")
        True
    """
    pass


# Test
try:
    validate_user_data("test@example.com", 25, "555-123-4567")
    print("Valid data")
except (InvalidEmailError, InvalidAgeError, InvalidPhoneError) as e:
    print(f"Validation error: {e}")
`,
    testCases: [
      {
        input: ['test@example.com', 25, '555-123-4567'],
        expected: true,
      },
      {
        input: ['invalid', 25, '555-123-4567'],
        expected: 'InvalidEmailError',
      },
    ],
    solution: `import re

class InvalidEmailError(Exception):
    """Raised when email format is invalid."""
    pass


class InvalidAgeError(Exception):
    """Raised when age is out of valid range."""
    pass


class InvalidPhoneError(Exception):
    """Raised when phone format is invalid."""
    pass


def validate_user_data(email, age, phone):
    # Validate email
    if '@' not in email or '.' not in email:
        raise InvalidEmailError(f"Invalid email format: {email}")
    
    # Validate age
    if not isinstance(age, int) or age < 0 or age > 150:
        raise InvalidAgeError(f"Age must be between 0 and 150, got: {age}")
    
    # Validate phone
    phone_pattern = r'^\\d{3}-\\d{3}-\\d{4}$'
    if not re.match(phone_pattern, phone):
        raise InvalidPhoneError(f"Phone must be XXX-XXX-XXXX format, got: {phone}")
    
    return True`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 2,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-json-config',
    title: 'JSON Configuration Manager',
    difficulty: 'Medium',
    description: `Create a configuration manager that reads, writes, and updates JSON configuration files.

**Features:**
- Load configuration from JSON file
- Get configuration value by key (support nested keys with dot notation)
- Set configuration value
- Save configuration back to file
- Handle missing files by creating defaults

**Example:**
\`\`\`python
config = ConfigManager("config.json")
db_host = config.get("database.host")  # Nested key
config.set("database.port", 5432)
config.save()
\`\`\``,
    examples: [
      {
        input: 'config.get("database.host")',
        output: '"localhost"',
      },
    ],
    constraints: [
      'Support nested keys with dot notation',
      'Create file if not exists',
      'Validate JSON format',
    ],
    hints: [
      'Split dot notation into nested keys',
      'Use dict.get() for safe access',
      'Handle FileNotFoundError for new files',
    ],
    starterCode: `import json

class ConfigManager:
    """Manage JSON configuration files."""
    
    def __init__(self, filename):
        """
        Initialize configuration manager.
        
        Args:
            filename: Path to JSON config file
        """
        self.filename = filename
        self.config = self.load()
    
    def load(self):
        """Load configuration from file."""
        # TODO: Load JSON from file, handle FileNotFoundError
        return {}  # Return empty dict for now to prevent crashes
    
    def get(self, key, default=None):
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config.get("database.host")
            "localhost"
        """
        # TODO: Split key by '.' and navigate nested dicts
        pass
    
    def set(self, key, value):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        # TODO: Split key by '.' and set value in nested dict
        pass
    
    def save(self):
        """Save configuration to file."""
        # TODO: Write config dict to JSON file
        pass


# Create a virtual config file for testing
with open('config.json', 'w') as f:
    f.write('{"database": {"host": "localhost", "port": 3306}}')

# Test
config = ConfigManager("config.json")
print(config.get("database.host", "localhost"))
config.set("database.port", 5432)
config.save()


# Test helper function (for automated testing)
def test_config_manager(filename, key):
    """Test function for ConfigManager - implement the class methods above first!"""
    try:
        config = ConfigManager(filename)
        return config.get(key, 'localhost')
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: ['config.json', 'database.host'],
        expected: 'localhost',
        functionName: 'test_config_manager',
      },
    ],
    solution: `import json

class ConfigManager:
    def __init__(self, filename):
        self.filename = filename
        self.config = self.load()
    
    def load(self):
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return empty config if file doesn't exist
            return {}
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {self.filename}")
    
    def get(self, key, default=None):
        # Split dot notation
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key, value):
        keys = key.split('.')
        config = self.config
        
        # Navigate to nested dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
    
    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.config, f, indent=2)


# Create a virtual config file for testing
with open('config.json', 'w') as f:
    f.write('{"database": {"host": "localhost", "port": 3306}}')


# Test helper function (for automated testing)
def test_config_manager(filename, key):
    """Test function for ConfigManager."""
    config = ConfigManager(filename)
    return config.get(key, 'localhost')`,
    timeComplexity: 'O(d) where d is depth of nested keys',
    spaceComplexity: 'O(n) where n is config size',
    order: 3,
    topic: 'Python Intermediate',
  },
  {
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
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = filtered[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)`,
    timeComplexity: 'O(n) where n is number of rows',
    spaceComplexity: 'O(n) for filtered results',
    order: 4,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-email-extractor',
    title: 'Email Address Extractor',
    difficulty: 'Medium',
    description: `Extract and validate email addresses from text using regular expressions.

**Requirements:**
- Extract all email addresses from text
- Validate email format
- Group emails by domain
- Remove duplicates

**Valid Email Pattern:**
- Username: letters, numbers, dots, underscores, hyphens
- @ symbol
- Domain: letters, numbers, hyphens, dots
- TLD: 2-6 letters

**Example:**
\`\`\`
Contact us at support@example.com or sales@example.com
For urgent matters: admin@urgent-support.co.uk
\`\`\``,
    examples: [
      {
        input: 'text with emails',
        output: "{'example.com': ['support', 'sales'], ...}",
      },
    ],
    constraints: [
      'Use regex for extraction',
      'Validate email format',
      'Group by domain',
    ],
    hints: [
      'Pattern: [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}',
      'Use re.findall()',
      'Split email at @ to get domain',
    ],
    starterCode: `import re
from collections import defaultdict

def extract_emails(text):
    """
    Extract all valid email addresses from text.
    
    Args:
        text: Text containing emails
        
    Returns:
        List of unique email addresses
        
    Examples:
        >>> extract_emails("Contact support@example.com")
        ['support@example.com']
    """
    pass


def validate_email(email):
    """
    Validate email format.
    
    Args:
        email: Email address string
        
    Returns:
        True if valid, False otherwise
    """
    pass


def group_by_domain(emails):
    """
    Group email addresses by domain.
    
    Args:
        emails: List of email addresses
        
    Returns:
        Dict mapping domain to list of usernames
        
    Examples:
        >>> group_by_domain(['user1@example.com', 'user2@example.com'])
        {'example.com': ['user1', 'user2']}
    """
    pass


# Test
text = """
Contact us at support@example.com or sales@example.com.
For urgent matters: admin@urgent-support.co.uk
Invalid emails: not-an-email, missing@domain
"""

emails = extract_emails(text)
print(f"Found {len(emails)} emails")

grouped = group_by_domain(emails)
for domain, users in grouped.items():
    print(f"{domain}: {users}")
`,
    testCases: [
      {
        input: ['support@example.com sales@example.com'],
        expected: ['support@example.com', 'sales@example.com'],
      },
    ],
    solution: `import re
from collections import defaultdict

def extract_emails(text):
    # Email regex pattern
    pattern = r'\\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}\\b'
    emails = re.findall(pattern, text)
    
    # Remove duplicates and validate
    unique_emails = []
    seen = set()
    for email in emails:
        if email.lower() not in seen and validate_email(email):
            unique_emails.append(email)
            seen.add(email.lower())
    
    return unique_emails


def validate_email(email):
    # More strict validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False
    
    # Check for valid domain
    username, domain = email.split('@')
    if not username or not domain:
        return False
    
    # Domain must have at least one dot
    if '.' not in domain:
        return False
    
    return True


def group_by_domain(emails):
    grouped = defaultdict(list)
    for email in emails:
        username, domain = email.split('@')
        grouped[domain].append(username)
    return dict(grouped)`,
    timeComplexity: 'O(n*m) where n is text length, m is number of emails',
    spaceComplexity: 'O(e) where e is number of emails',
    order: 5,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-date-calculator',
    title: 'Date Range Calculator',
    difficulty: 'Medium',
    description: `Create a utility for common date calculations.

**Functions to Implement:**
1. Calculate age from birthdate
2. Find business days between two dates (exclude weekends)
3. Get all dates in a month
4. Check if date is in range

**Date Format:** YYYY-MM-DD (ISO format)

**Business Days:** Monday-Friday only (ignore holidays)`,
    examples: [
      {
        input: 'calculate_age("1990-05-15")',
        output: '33 (or current age)',
      },
      {
        input: 'business_days_between("2024-01-01", "2024-01-05")',
        output: '4 (excluding weekend)',
      },
    ],
    constraints: [
      'Use datetime module',
      'Handle invalid dates',
      'ISO format (YYYY-MM-DD)',
    ],
    hints: [
      'Use datetime.strptime() to parse',
      'timedelta for date arithmetic',
      'weekday() returns 0-6 (Monday-Sunday)',
    ],
    starterCode: `from datetime import datetime, timedelta, date

def calculate_age(birthdate_str):
    """
    Calculate age from birthdate.
    
    Args:
        birthdate_str: Birthdate in YYYY-MM-DD format
        
    Returns:
        Age in years as integer
        
    Examples:
        >>> calculate_age("1990-05-15")
        33
    """
    pass


def business_days_between(start_str, end_str):
    """
    Count business days between two dates (exclude weekends).
    
    Args:
        start_str: Start date in YYYY-MM-DD format
        end_str: End date in YYYY-MM-DD format
        
    Returns:
        Number of business days
        
    Examples:
        >>> business_days_between("2024-01-01", "2024-01-05")
        4
    """
    pass


def get_month_dates(year, month):
    """
    Get all dates in a given month.
    
    Args:
        year: Year as integer
        month: Month as integer (1-12)
        
    Returns:
        List of date objects for each day in month
        
    Examples:
        >>> len(get_month_dates(2024, 1))
        31
    """
    pass


def is_date_in_range(check_date_str, start_str, end_str):
    """
    Check if date is within range (inclusive).
    
    Args:
        check_date_str: Date to check
        start_str: Range start date
        end_str: Range end date
        
    Returns:
        True if date is in range, False otherwise
    """
    pass


# Test
print(f"Age: {calculate_age('1990-05-15')}")
print(f"Business days: {business_days_between('2024-01-01', '2024-01-10')}")
print(f"Days in Jan 2024: {len(get_month_dates(2024, 1))}")
print(f"In range: {is_date_in_range('2024-06-15', '2024-01-01', '2024-12-31')}")
`,
    testCases: [
      {
        input: ['1990-05-15'],
        expected: 33,
      },
      {
        input: ['2024-01-01', '2024-01-05'],
        expected: 4,
      },
    ],
    solution: `from datetime import datetime, timedelta, date
from calendar import monthrange

def calculate_age(birthdate_str):
    birthdate = datetime.strptime(birthdate_str, "%Y-%m-%d").date()
    today = date.today()
    
    age = today.year - birthdate.year
    # Adjust if birthday hasn't occurred this year
    if (today.month, today.day) < (birthdate.month, birthdate.day):
        age -= 1
    
    return age


def business_days_between(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    
    # Count business days
    count = 0
    current = start
    while current <= end:
        # weekday() returns 0-6 (Monday-Sunday), so 0-4 are weekdays
        if current.weekday() < 5:
            count += 1
        current += timedelta(days=1)
    
    return count


def get_month_dates(year, month):
    # Get number of days in month
    _, num_days = monthrange(year, month)
    
    # Create list of all dates
    dates = []
    for day in range(1, num_days + 1):
        dates.append(date(year, month, day))
    
    return dates


def is_date_in_range(check_date_str, start_str, end_str):
    check_date = datetime.strptime(check_date_str, "%Y-%m-%d").date()
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    
    return start <= check_date <= end`,
    timeComplexity: 'O(d) for business_days_between where d is days between',
    spaceComplexity: 'O(d) for get_month_dates',
    order: 6,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-log-parser',
    title: 'Log File Analyzer',
    difficulty: 'Medium',
    description: `Parse and analyze log files to extract statistics.

**Log Format:**
\`\`\`
2024-01-15 10:30:45 [ERROR] Database connection failed
2024-01-15 10:31:12 [INFO] User login successful
2024-01-15 10:32:30 [WARNING] High memory usage detected
2024-01-15 10:33:05 [ERROR] API timeout
\`\`\`

**Tasks:**
- Count logs by level (ERROR, WARNING, INFO)
- Find all ERROR messages
- Get logs within time range
- Calculate error rate

**Pattern:** \`YYYY-MM-DD HH:MM:SS [LEVEL] message\``,
    examples: [
      {
        input: 'analyze_logs("app.log")',
        output: "{'ERROR': 2, 'WARNING': 1, 'INFO': 1}",
      },
    ],
    constraints: [
      'Use regex for parsing',
      'Handle malformed lines',
      'Support time range filtering',
    ],
    hints: [
      'Regex pattern: (\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) \\[(\\w+)\\] (.*)',
      'Group by log level',
      'Use datetime for time comparison',
    ],
    starterCode: `import re
from datetime import datetime
from collections import defaultdict

def parse_log_line(line):
    """
    Parse a single log line.
    
    Args:
        line: Log line string
        
    Returns:
        Dict with 'timestamp', 'level', 'message' or None if invalid
        
    Examples:
        >>> parse_log_line("2024-01-15 10:30:45 [ERROR] Failed")
        {'timestamp': '2024-01-15 10:30:45', 'level': 'ERROR', 'message': 'Failed'}
    """
    pass


def count_by_level(filename):
    """
    Count log entries by level.
    
    Args:
        filename: Path to log file
        
    Returns:
        Dict mapping level to count
    """
    pass


def find_errors(filename):
    """
    Find all ERROR level messages.
    
    Args:
        filename: Path to log file
        
    Returns:
        List of error messages with timestamps
    """
    pass


def filter_by_time_range(filename, start_time, end_time):
    """
    Get logs within time range.
    
    Args:
        filename: Path to log file
        start_time: Start time string (YYYY-MM-DD HH:MM:SS)
        end_time: End time string (YYYY-MM-DD HH:MM:SS)
        
    Returns:
        List of log entries in range
    """
    pass


# Test
counts = count_by_level("app.log")
print(f"Log counts: {counts}")

errors = find_errors("app.log")
print(f"Found {len(errors)} errors")

filtered = filter_by_time_range("app.log", 
                                "2024-01-15 10:30:00",
                                "2024-01-15 10:35:00")
print(f"Logs in range: {len(filtered)}")
`,
    testCases: [
      {
        input: ['app.log'],
        expected: { ERROR: 2, WARNING: 1, INFO: 1 },
      },
    ],
    solution: `import re
from datetime import datetime
from collections import defaultdict

def parse_log_line(line):
    pattern = r'(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) \\[(\\w+)\\] (.*)'
    match = re.match(pattern, line.strip())
    
    if match:
        return {
            'timestamp': match.group(1),
            'level': match.group(2),
            'message': match.group(3)
        }
    return None


def count_by_level(filename):
    counts = defaultdict(int)
    
    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                counts[parsed['level']] += 1
    
    return dict(counts)


def find_errors(filename):
    errors = []
    
    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed and parsed['level'] == 'ERROR':
                errors.append(parsed)
    
    return errors


def filter_by_time_range(filename, start_time, end_time):
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    
    filtered = []
    
    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                log_dt = datetime.strptime(parsed['timestamp'], "%Y-%m-%d %H:%M:%S")
                if start_dt <= log_dt <= end_dt:
                    filtered.append(parsed)
    
    return filtered`,
    timeComplexity: 'O(n) where n is number of log lines',
    spaceComplexity: 'O(m) where m is matching lines',
    order: 7,
    topic: 'Python Intermediate',
  },
  {
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
  },
  {
    id: 'intermediate-bank-account',
    title: 'Bank Account with File Persistence',
    difficulty: 'Medium',
    description: `Create a BankAccount class that persists transactions to a file.

**Features:**
- Deposit and withdraw money
- Check balance
- View transaction history
- Save/load state from JSON file
- Prevent overdrafts

**Transaction Format:**
\`\`\`python
{
    "timestamp": "2024-01-15 10:30:45",
    "type": "deposit",
    "amount": 100.00,
    "balance": 1100.00
}
\`\`\``,
    examples: [
      {
        input: 'account.deposit(100)',
        output: 'New balance: 1100.00',
      },
    ],
    constraints: [
      'Prevent negative balance',
      'Track all transactions',
      'Persist to file',
    ],
    hints: [
      'Use datetime for timestamps',
      'Store transactions as list',
      'Use JSON for persistence',
    ],
    starterCode: `import json
from datetime import datetime

class InsufficientFundsError(Exception):
    """Raised when withdrawal exceeds balance."""
    pass


class BankAccount:
    """Bank account with file persistence."""
    
    def __init__(self, account_number, initial_balance=0, filename=None):
        """
        Initialize bank account.
        
        Args:
            account_number: Account identifier
            initial_balance: Starting balance
            filename: Optional file for persistence
        """
        self.account_number = account_number
        self.balance = initial_balance
        self.transactions = []
        self.filename = filename or f"account_{account_number}.json"
        # Note: load() intentionally not called in starter to avoid file errors
    
    def deposit(self, amount):
        """
        Deposit money into account.
        
        Args:
            amount: Amount to deposit
            
        Raises:
            ValueError: If amount is negative
        """
        # TODO: Implement deposit logic
        # - Validate amount is positive
        # - Add to balance
        # - Add transaction
        # - Save to file
        pass
    
    def withdraw(self, amount):
        """
        Withdraw money from account.
        
        Args:
            amount: Amount to withdraw
            
        Raises:
            ValueError: If amount is negative
            InsufficientFundsError: If balance is insufficient
        """
        # TODO: Implement withdrawal logic
        # - Validate amount is positive
        # - Check sufficient funds
        # - Subtract from balance
        # - Add transaction
        # - Save to file
        pass
    
    def get_balance(self):
        """Get current balance."""
        return self.balance
    
    def get_transactions(self):
        """Get transaction history."""
        return self.transactions
    
    def save(self):
        """Save account state to file."""
        # TODO: Implement save logic
        # - Create dict with account data
        # - Write to JSON file
        pass
    
    def load(self):
        """Load account state from file."""
        # TODO: Implement load logic
        # - Read from JSON file
        # - Update balance and transactions
        # - Handle FileNotFoundError
        pass
    
    def _add_transaction(self, trans_type, amount):
        """Add transaction to history."""
        # TODO: Implement transaction logging
        # - Create transaction dict with timestamp, type, amount, balance
        # - Append to transactions list
        pass


# Test helper function (for automated testing)
def test_bank_account(initial_balance, deposit_amount):
    """Test function for BankAccount - implement the class methods above first!"""
    try:
        account = BankAccount("test123", initial_balance, "test_account.json")
        account.deposit(deposit_amount)
        return account.get_balance()
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: [1000, 500],
        expected: 1500,
        functionName: 'test_bank_account',
      },
    ],
    solution: `import json
from datetime import datetime

class InsufficientFundsError(Exception):
    """Raised when withdrawal exceeds balance."""
    pass


class BankAccount:
    def __init__(self, account_number, initial_balance=0, filename=None):
        self.account_number = account_number
        self.balance = initial_balance
        self.transactions = []
        self.filename = filename or f"account_{account_number}.json"

        self.load()

    def deposit(self, amount):
        if amount < 0:
            raise ValueError("Deposit amount must be positive")

        self.balance += amount
        self._add_transaction("deposit", amount)
        self.save()

    def withdraw(self, amount):
        if amount < 0:
            raise ValueError("Withdrawal amount must be positive")

        if amount > self.balance:
            raise InsufficientFundsError(
                f"Insufficient funds: balance {self.balance:.2f}, "
                f"withdrawal {amount:.2f}"
            )

        self.balance -= amount
        self._add_transaction("withdrawal", amount)
        self.save()

    def get_balance(self):
        return self.balance

    def get_transactions(self):
        return self.transactions

    def save(self):
        data = {
            "account_number": self.account_number,
            "balance": self.balance,
            "transactions": self.transactions
        }
        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                self.balance = data.get("balance", self.balance)
                self.transactions = data.get("transactions", [])
        except FileNotFoundError:
            # New account
            pass

    def _add_transaction(self, trans_type, amount):
        transaction = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": trans_type,
            "amount": amount,
            "balance": self.balance
        }
        self.transactions.append(transaction)


def test_bank_account(initial_balance, deposit_amount):
    """Test function for BankAccount."""
    account = BankAccount("test123", initial_balance, "test_account.json")
    account.deposit(deposit_amount)
    return account.get_balance()`,
    timeComplexity: 'O(1) for deposit/withdraw, O(n) for save',
    spaceComplexity: 'O(t) where t is number of transactions',
    order: 9,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-url-parser',
    title: 'URL Parser and Validator',
    difficulty: 'Medium',
    description: `Parse and validate URLs, extracting components and query parameters.

**URL Components:**
- Protocol (http, https)
- Domain
- Path
- Query parameters
- Fragment

**Example URL:**
\`\`\`
https://example.com/path/to/page?id=123&sort=name#section
\`\`\`

**Tasks:**
- Parse URL into components
- Validate URL format
- Extract query parameters as dict
- Rebuild URL from components`,
    examples: [
      {
        input: 'parse_url("https://example.com/page?id=123")',
        output: "{'protocol': 'https', 'domain': 'example.com', ...}",
      },
    ],
    constraints: [
      'Use regex for parsing',
      'Validate URL format',
      'Handle missing components',
    ],
    hints: [
      'URL pattern: (https?)://([^/]+)(/[^?#]*)?(?:\\?([^#]*))?(?:#(.*))?',
      'Split query string on & and =',
      'Use urllib.parse for robust parsing',
    ],
    starterCode: `import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

def parse_url(url):
    """
    Parse URL into components.
    
    Args:
        url: URL string
        
    Returns:
        Dict with protocol, domain, path, query, fragment
        
    Examples:
        >>> parse_url("https://example.com/page?id=123")
        {'protocol': 'https', 'domain': 'example.com', 'path': '/page', 
         'query': {'id': '123'}, 'fragment': ''}
    """
    pass


def validate_url(url):
    """
    Validate URL format.
    
    Args:
        url: URL string
        
    Returns:
        True if valid, False otherwise
    """
    pass


def extract_query_params(url):
    """
    Extract query parameters as dictionary.
    
    Args:
        url: URL string
        
    Returns:
        Dict of query parameters
        
    Examples:
        >>> extract_query_params("https://example.com?id=123&sort=name")
        {'id': '123', 'sort': 'name'}
    """
    pass


def build_url(protocol, domain, path='', query_params=None, fragment=''):
    """
    Build URL from components.
    
    Args:
        protocol: 'http' or 'https'
        domain: Domain name
        path: URL path
        query_params: Dict of query parameters
        fragment: Fragment identifier
        
    Returns:
        Complete URL string
        
    Examples:
        >>> build_url('https', 'example.com', '/page', {'id': '123'})
        'https://example.com/page?id=123'
    """
    pass


# Test
url = "https://example.com/path/to/page?id=123&sort=name&page=2#section"

parsed = parse_url(url)
print(f"Protocol: {parsed['protocol']}")
print(f"Domain: {parsed['domain']}")
print(f"Path: {parsed['path']}")
print(f"Query: {parsed['query']}")
print(f"Fragment: {parsed['fragment']}")

print(f"\\nValid URL: {validate_url(url)}")

params = extract_query_params(url)
print(f"\\nQuery params: {params}")

rebuilt = build_url('https', 'example.com', '/page', {'id': '456'}, 'top')
print(f"\\nRebuilt URL: {rebuilt}")
`,
    testCases: [
      {
        input: ['https://example.com/page?id=123'],
        expected: { protocol: 'https', domain: 'example.com' },
      },
    ],
    solution: `import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

def parse_url(url):
    parsed = urlparse(url)
    
    # Parse query string into dict (parse_qs returns lists, we want single values)
    query_params = {}
    if parsed.query:
        parsed_qs = parse_qs(parsed.query)
        query_params = {k: v[0] if len(v) == 1 else v for k, v in parsed_qs.items()}
    
    return {
        'protocol': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path,
        'query': query_params,
        'fragment': parsed.fragment
    }


def validate_url(url):
    # Basic URL regex
    pattern = r'^(https?://)([a-zA-Z0-9.-]+)(:[0-9]+)?(/.*)?$'
    if not re.match(pattern, url):
        return False
    
    # Additional validation using urlparse
    parsed = urlparse(url)
    return all([parsed.scheme, parsed.netloc])


def extract_query_params(url):
    parsed = urlparse(url)
    if not parsed.query:
        return {}
    
    params = parse_qs(parsed.query)
    # Convert lists to single values for simplicity
    return {k: v[0] if len(v) == 1 else v for k, v in params.items()}


def build_url(protocol, domain, path='', query_params=None, fragment=''):
    # Ensure path starts with /
    if path and not path.startswith('/'):
        path = '/' + path
    
    # Build query string
    query = ''
    if query_params:
        query = urlencode(query_params)
    
    # Build URL using urlunparse
    url_parts = (protocol, domain, path, '', query, fragment)
    return urlunparse(url_parts)`,
    timeComplexity: 'O(n) where n is URL length',
    spaceComplexity: 'O(p) where p is number of query parameters',
    order: 10,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-decorator-timer',
    title: 'Function Timer Decorator',
    difficulty: 'Medium',
    description: `Create a decorator that measures and logs function execution time.

**Requirements:**
- Measure execution time in milliseconds
- Log function name and arguments
- Support both sync functions
- Optionally repeat execution N times and average

**Example:**
\`\`\`python
@timer(repeat=3)
def slow_function(n):
    time.sleep(n)
    return n * 2
\`\`\``,
    examples: [
      {
        input: '@timer(repeat=1)\\ndef add(a, b): return a + b',
        output: 'Function add(2, 3) took 0.02ms',
      },
    ],
    constraints: [
      'Use functools.wraps',
      'Preserve function signature',
      'Handle exceptions',
    ],
    hints: [
      'Use time.perf_counter() for precision',
      'functools.wraps preserves metadata',
      'Decorator with arguments needs nested functions',
    ],
    starterCode: `import time
import functools

def timer(repeat=1):
    """
    Decorator to measure function execution time.
    
    Args:
        repeat: Number of times to execute and average
        
    Returns:
        Decorated function
        
    Examples:
        >>> @timer(repeat=3)
        ... def add(a, b):
        ...     return a + b
        >>> add(2, 3)
        5
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pass
        return wrapper
    return decorator


# Test
@timer(repeat=1)
def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

@timer(repeat=5)
def quick_math(x, y):
    """Perform quick calculation."""
    return x ** 2 + y ** 2


def test_timer():
    """Test function that validates timer decorator"""
    import sys
    from io import StringIO
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Call decorated function
        result = fibonacci(10)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify function returned correct value
        if result != 55:
            return f"FAIL: Wrong result: {result}"
        
        # Verify decorator printed timing information
        if not output or "took" not in output.lower() and "time" not in output.lower():
            return "FAIL: Decorator should print timing info"
        
        return 55
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"
`,
    testCases: [
      {
        input: [],
        expected: 55,
        functionName: 'test_timer',
      },
    ],
    solution: `import time
import functools

def timer(repeat=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Format arguments for display
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            
            # Run function multiple times
            times = []
            result = None
            
            for _ in range(repeat):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate average
            avg_time = sum(times) / len(times)
            
            # Log execution
            if repeat > 1:
                print(f"Function {func.__name__}({signature}) "
                      f"took {avg_time:.4f}ms (avg of {repeat} runs)")
            else:
                print(f"Function {func.__name__}({signature}) "
                      f"took {avg_time:.4f}ms")
            
            return result
        return wrapper
    return decorator


# Test
@timer(repeat=1)
def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

@timer(repeat=5)
def quick_math(x, y):
    """Perform quick calculation."""
    return x ** 2 + y ** 2


def test_timer():
    """Test function that validates timer decorator"""
    import sys
    from io import StringIO
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Call decorated function
        result = fibonacci(10)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify function returned correct value
        if result != 55:
            return f"FAIL: Wrong result: {result}"
        
        # Verify decorator printed timing information
        if not output or "took" not in output.lower() and "time" not in output.lower():
            return "FAIL: Decorator should print timing info"
        
        return 55
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"`,
    timeComplexity: 'O(r*f) where r is repeats, f is function complexity',
    spaceComplexity: 'O(r) for storing times',
    order: 11,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-context-manager',
    title: 'Custom Context Manager',
    difficulty: 'Medium',
    description: `Create a context manager that temporarily changes directory and ensures cleanup.

**Requirements:**
- Change to specified directory
- Automatically return to original directory
- Handle errors gracefully
- Support both class-based and function-based implementations

**Usage:**
\`\`\`python
with ChangeDirectory('/tmp'):
    # Working in /tmp
    print(os.getcwd())
# Automatically back to original directory
\`\`\``,
    examples: [
      {
        input: 'with ChangeDirectory("/tmp"): pass',
        output: 'Changes dir and returns automatically',
      },
    ],
    constraints: [
      'Implement __enter__ and __exit__',
      'Restore original directory even on error',
      'Support with statement',
    ],
    hints: [
      'Save os.getcwd() before changing',
      'Use try/finally in __exit__',
      'Or use @contextmanager decorator',
    ],
    starterCode: `import os
from contextlib import contextmanager

class ChangeDirectory:
    """
    Context manager to temporarily change directory.
    
    Examples:
        >>> with ChangeDirectory('/tmp'):
        ...     print(os.getcwd())
        '/tmp'
    """
    
    def __init__(self, path):
        """
        Initialize with target directory.
        
        Args:
            path: Directory to change to
        """
        # TODO: Store the path and initialize original_dir to None
        self.path = path
        self.original_dir = None
    
    def __enter__(self):
        """Enter context - change directory."""
        # TODO: Save current directory and change to new path
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context - restore directory.
        
        Args:
            exc_type: Exception type if any
            exc_val: Exception value if any
            exc_tb: Exception traceback if any
            
        Returns:
            False to propagate exceptions
        """
        # TODO: Restore original directory
        pass


@contextmanager
def change_directory(path):
    """
    Function-based context manager using decorator.
    
    Args:
        path: Directory to change to
        
    Yields:
        None
        
    Examples:
        >>> with change_directory('/tmp'):
        ...     pass
    """
    # TODO: Implement using yield
    pass


# Test
print(f"Original directory: {os.getcwd()}")

try:
    with ChangeDirectory('/tmp'):
        print(f"Inside context: {os.getcwd()}")
        # Could raise exception here
finally:
    print(f"After context: {os.getcwd()}")


# Test helper function (for automated testing)
def test_change_directory(target_path):
    """Test function for ChangeDirectory - implement the class methods above first!"""
    try:
        original = os.getcwd()
        with ChangeDirectory(target_path):
            changed = os.getcwd()
        restored = os.getcwd()
        # Return True if we successfully changed and restored
        return changed == target_path and restored == original
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: ['/tmp'],
        expected: true,
        functionName: 'test_change_directory',
      },
    ],
    solution: `import os
from contextlib import contextmanager

class ChangeDirectory:
    def __init__(self, path):
        self.path = path
        self.original_dir = None
    
    def __enter__(self):
        self.original_dir = os.getcwd()
        os.chdir(self.path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always restore original directory
        os.chdir(self.original_dir)
        # Return False to propagate exceptions
        return False


@contextmanager
def change_directory(path):
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)


# Test helper function (for automated testing)
def test_change_directory(target_path):
    """Test function for ChangeDirectory."""
    original = os.getcwd()
    with ChangeDirectory(target_path):
        changed = os.getcwd()
    restored = os.getcwd()
    return changed == target_path and restored == original


# More advanced: File opener with automatic cleanup
class FileOpener:
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 12,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-generator-pipeline',
    title: 'Data Processing Pipeline with Generators',
    difficulty: 'Medium',
    description: `Create a memory-efficient data processing pipeline using generators.

**Pipeline Steps:**
1. Read lines from file (generator)
2. Filter lines matching pattern (generator)
3. Transform lines (generator)
4. Aggregate results

**Benefits:**
- Memory efficient (processes one item at a time)
- Lazy evaluation
- Composable operations

**Example:**
\`\`\`python
lines = read_lines('data.txt')
filtered = filter_lines(lines, pattern='ERROR')
transformed = transform_lines(filtered, str.upper)
result = list(transformed)
\`\`\``,
    examples: [
      {
        input: 'Pipeline processes large file',
        output: 'Memory-efficient streaming',
      },
    ],
    constraints: [
      'Use yield keyword',
      'Chain generators',
      'Process one item at a time',
    ],
    hints: [
      'yield returns values lazily',
      'Generators can be chained',
      'Use generator expressions',
    ],
    starterCode: `import re

def read_lines(filename):
    """
    Generator that yields lines from file.
    
    Args:
        filename: Path to file
        
    Yields:
        Individual lines from file
        
    Examples:
        >>> for line in read_lines('data.txt'):
        ...     print(line)
    """
    pass


def filter_lines(lines, pattern):
    """
    Generator that filters lines matching pattern.
    
    Args:
        lines: Iterator of lines
        pattern: Regex pattern to match
        
    Yields:
        Lines matching pattern
    """
    pass


def transform_lines(lines, transform_func):
    """
    Generator that transforms each line.
    
    Args:
        lines: Iterator of lines
        transform_func: Function to apply to each line
        
    Yields:
        Transformed lines
    """
    pass


def batch_lines(lines, batch_size):
    """
    Generator that groups lines into batches.
    
    Args:
        lines: Iterator of lines
        batch_size: Number of lines per batch
        
    Yields:
        Lists of lines (batches)
        
    Examples:
        >>> for batch in batch_lines(lines, 10):
        ...     process_batch(batch)
    """
    pass


# Test with virtual file
# Create test file
with open('test_data.txt', 'w') as f:
    f.write("""ERROR: Connection failed
INFO: System starting
ERROR: Database timeout
WARNING: Low memory
INFO: User logged in
ERROR: API error
""")

# Build pipeline
lines = read_lines('test_data.txt')
errors = filter_lines(lines, r'ERROR')
uppercase = transform_lines(errors, str.upper)

print("Filtered and transformed lines:")
for line in uppercase:
    print(line)

# Example with batching
lines2 = read_lines('test_data.txt')
batches = batch_lines(lines2, 2)
for i, batch in enumerate(batches, 1):
    print(f"\\nBatch {i}:")
    for line in batch:
        print(f"  {line}", end='')
`,
    testCases: [
      {
        input: ['test_data.txt', 'ERROR'],
        expected: 3,
        functionName: 'test_data_processing',
      },
    ],
    solution: `import re

# Create test file (for browser environment)
with open('test_data.txt', 'w') as f:
    f.write("""ERROR: Connection failed
INFO: System starting
ERROR: Database timeout
WARNING: Low memory
INFO: User logged in
ERROR: API error
""")


def read_lines(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line.rstrip('\\n')


def filter_lines(lines, pattern):
    regex = re.compile(pattern)
    for line in lines:
        if regex.search(line):
            yield line


def transform_lines(lines, transform_func):
    for line in lines:
        yield transform_func(line)


def batch_lines(lines, batch_size):
    batch = []
    for line in lines:
        batch.append(line)
        if len(batch) == batch_size:
            yield batch
            batch = []
    
    # Yield remaining items
    if batch:
        yield batch


# Test helper function (for automated testing)
def test_data_processing(filename, pattern):
    """Test function for data processing pipeline."""
    lines = read_lines(filename)
    filtered = filter_lines(lines, pattern)
    return len(list(filtered))


# Advanced: Generator with send()
def running_average():
    """Generator that calculates running average."""
    total = 0
    count = 0
    average = None
    
    while True:
        value = yield average
        total += value
        count += 1
        average = total / count


# Advanced: Generator expression examples
def process_large_file(filename):
    """Process file using generator expressions."""
    # Generator expression - memory efficient
    lines = (line.strip() for line in open(filename))
    errors = (line for line in lines if 'ERROR' in line)
    uppercase = (line.upper() for line in errors)
    
    return list(uppercase)`,
    timeComplexity: 'O(n) where n is number of lines',
    spaceComplexity: 'O(1) for generators, O(b) for batches',
    order: 13,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-cache-decorator',
    title: 'LRU Cache Decorator',
    difficulty: 'Hard',
    description: `Implement a Least Recently Used (LRU) cache decorator.

**Features:**
- Cache function results
- Limit cache size
- Evict least recently used items when full
- Track cache hits/misses
- Provide cache statistics

**LRU means:** When cache is full, remove the item that was accessed longest ago.

**Example:**
\`\`\`python
@lru_cache(maxsize=3)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
\`\`\``,
    examples: [
      {
        input: 'Cached fibonacci(10)',
        output: 'Much faster than uncached',
      },
    ],
    constraints: [
      'Implement LRU eviction',
      'Track access order',
      'Support cache stats',
    ],
    hints: [
      'Use OrderedDict for LRU tracking',
      'Move accessed items to end',
      'Check size before adding',
    ],
    starterCode: `from functools import wraps
from collections import OrderedDict

def lru_cache(maxsize=128):
    """
    LRU cache decorator.
    
    Args:
        maxsize: Maximum number of cached items
        
    Returns:
        Decorated function with caching
        
    Examples:
        >>> @lru_cache(maxsize=3)
        ... def add(a, b):
        ...     return a + b
    """
    def decorator(func):
        cache = OrderedDict()
        hits = 0
        misses = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal hits, misses
            # Create cache key from arguments
            # Implement LRU logic here
            pass
        
        def cache_info():
            """Return cache statistics."""
            return {
                'hits': hits,
                'misses': misses,
                'maxsize': maxsize,
                'currsize': len(cache)
            }
        
        def cache_clear():
            """Clear the cache."""
            nonlocal hits, misses
            cache.clear()
            hits = misses = 0
        
        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        
        return wrapper
    return decorator


# Test
@lru_cache(maxsize=3)
def fibonacci(n):
    """Calculate Fibonacci number."""
    print(f"Computing fibonacci({n})")
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

@lru_cache(maxsize=5)
def expensive_operation(x, y):
    """Simulate expensive computation."""
    print(f"Computing expensive_operation({x}, {y})")
    import time
    time.sleep(0.1)  # Simulate slow operation
    return x ** y

def test_cache():
    """Test function that validates LRU cache decorator"""
    # First test: verify fibonacci works
    result = fibonacci(10)
    if result != 55:
        return f"FAIL: Wrong fibonacci result: {result}"
    
    # Second test: verify caching works by checking cache hits
    fibonacci.cache_clear()  # Clear any existing cache
    
    # Call fibonacci(5) multiple times
    fibonacci(5)  # Miss
    fibonacci(5)  # Should be a hit
    fibonacci(5)  # Should be a hit
    
    info = fibonacci.cache_info()
    
    # Verify cache stats exist
    if 'hits' not in info or 'misses' not in info:
        return "FAIL: Cache stats not available"
    
    # Verify we have at least 1 cache hit (from repeated calls)
    if info['hits'] < 1:
        return f"FAIL: No cache hits detected. Got {info}"
    
    return 55
`,
    testCases: [
      {
        input: [],
        expected: 55,
        functionName: 'test_cache',
      },
    ],
    solution: `from functools import wraps
from collections import OrderedDict

def lru_cache(maxsize=128):
    def decorator(func):
        cache = OrderedDict()
        hits = 0
        misses = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal hits, misses
            
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check if in cache
            if key in cache:
                hits += 1
                # Move to end (most recently used)
                cache.move_to_end(key)
                return cache[key]
            
            # Not in cache - compute result
            misses += 1
            result = func(*args, **kwargs)
            
            # Add to cache
            cache[key] = result
            cache.move_to_end(key)
            
            # Evict least recently used if over size
            if len(cache) > maxsize:
                cache.popitem(last=False)  # Remove first (oldest)
            
            return result
        
        def cache_info():
            return {
                'hits': hits,
                'misses': misses,
                'maxsize': maxsize,
                'currsize': len(cache),
                'hit_rate': hits / (hits + misses) if (hits + misses) > 0 else 0
            }
        
        def cache_clear():
            nonlocal hits, misses
            cache.clear()
            hits = misses = 0
        
        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        
        return wrapper
    return decorator


@lru_cache(maxsize=3)
def fibonacci(n):
    """Calculate Fibonacci number."""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)`,
    timeComplexity: 'O(1) for cache lookup, O(n) for function execution',
    spaceComplexity: 'O(maxsize)',
    order: 14,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-class-property',
    title: 'Property Decorators and Validation',
    difficulty: 'Medium',
    description: `Create a class using property decorators with validation.

**Requirements:**
- Use @property for getters
- Use @setter for validation
- Implement computed properties
- Add custom validation logic

**Example:**
\`\`\`python
class Temperature:
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Below absolute zero!")
        self._celsius = value
\`\`\``,
    examples: [
      {
        input: 'temp.celsius = 25',
        output: 'Sets with validation',
      },
    ],
    constraints: [
      'Use @property decorator',
      'Validate in setter',
      'Provide computed properties',
    ],
    hints: [
      '@property creates getter',
      '@name.setter creates setter',
      'Computed properties calculate on access',
    ],
    starterCode: `class Rectangle:
    """
    Rectangle with validated dimensions and computed properties.
    
    Examples:
        >>> rect = Rectangle(5, 10)
        >>> rect.width
        5
        >>> rect.area
        50
        >>> rect.width = -5  # Raises ValueError
    """
    
    def __init__(self, width, height):
        """
        Initialize rectangle.
        
        Args:
            width: Width (must be positive)
            height: Height (must be positive)
        """
        # TODO: Set width and height using the setters below
        # This allows validation to happen during initialization
        self._width = width  # Temporary: use setters instead
        self._height = height  # Temporary: use setters instead
    
    @property
    def width(self):
        """Get width."""
        # TODO: Return the width
        return self._width
    
    @width.setter
    def width(self, value):
        """
        Set width with validation.
        
        Args:
            value: New width
            
        Raises:
            ValueError: If width is not positive
        """
        # TODO: Implement validation and set _width
        # - Check if value is positive
        # - Raise ValueError if not
        # - Set self._width if valid
        pass
    
    @property
    def height(self):
        """Get height."""
        # TODO: Return the height
        return self._height
    
    @height.setter
    def height(self, value):
        """
        Set height with validation.
        
        Args:
            value: New height
            
        Raises:
            ValueError: If height is not positive
        """
        # TODO: Implement validation and set _height
        # - Check if value is positive
        # - Raise ValueError if not
        # - Set self._height if valid
        pass
    
    @property
    def area(self):
        """Calculate and return area (computed property)."""
        # TODO: Calculate and return width * height
        pass
    
    @property
    def perimeter(self):
        """Calculate and return perimeter (computed property)."""
        # TODO: Calculate and return 2 * (width + height)
        pass
    
    @property
    def diagonal(self):
        """Calculate and return diagonal length (computed property)."""
        # TODO: Calculate and return diagonal using Pythagorean theorem
        # Hint: import math and use math.sqrt()
        pass
    
    def __str__(self):
        """String representation."""
        return f"Rectangle({self.width}x{self.height})"


# Test helper function (for automated testing)
def test_rectangle(width, height):
    """Test function for Rectangle - implement the class methods above first!"""
    try:
        rect = Rectangle(width, height)
        return rect.area
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: [5, 10],
        expected: 50,
        functionName: 'test_rectangle',
      },
    ],
    solution: `import math

class Rectangle:
    def __init__(self, width, height):
        self.width = width  # Uses setter
        self.height = height  # Uses setter

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value <= 0:
            raise ValueError(f"Width must be positive, got {value}")
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if value <= 0:
            raise ValueError(f"Height must be positive, got {value}")
        self._height = value

    @property
    def area(self):
        return self._width * self._height

    @property
    def perimeter(self):
        return 2 * (self._width + self._height)

    @property
    def diagonal(self):
        return math.sqrt(self._width ** 2 + self._height ** 2)

    def __str__(self):
        return f"Rectangle({self.width}x{self.height})"

    def __repr__(self):
        return f"Rectangle(width={self.width}, height={self.height})"


# More advanced example: Temperature converter
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9
    
    @property
    def kelvin(self):
        return self._celsius + 273.15

    @kelvin.setter
    def kelvin(self, value):
        self.celsius = value - 273.15


def test_rectangle(width, height):
    """Test function for the Rectangle class."""
    rect = Rectangle(width, height)
    return rect.area`,
    timeComplexity: 'O(1) for all operations',
    spaceComplexity: 'O(1)',
    order: 15,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-sqlite-manager',
    title: 'SQLite Database Manager',
    difficulty: 'Hard',
    description: `Create a simple database manager for SQLite operations.

**Features:**
- Create tables
- Insert, update, delete records
- Query with filters
- Use context manager for connections
- Handle transactions

**Example:**
\`\`\`python
with DatabaseManager('users.db') as db:
    db.create_table('users', ['id INTEGER PRIMARY KEY', 'name TEXT', 'age INTEGER'])
    db.insert('users', {'name': 'Alice', 'age': 30})
    users = db.query('users', where={'age': 30})
\`\`\``,
    examples: [
      {
        input: "db.insert('users', {'name': 'Bob', 'age': 25})",
        output: 'Inserts record into database',
      },
    ],
    constraints: [
      'Use sqlite3 module',
      'Implement context manager',
      'Handle SQL injection safely',
    ],
    hints: [
      'Use parameterized queries (? placeholders)',
      'Implement __enter__ and __exit__',
      'Commit transactions in __exit__',
    ],
    starterCode: `import sqlite3

class DatabaseManager:
    """
    Simple SQLite database manager with context manager support.
    
    Examples:
        >>> with DatabaseManager('test.db') as db:
        ...     db.create_table('users', ['id INTEGER PRIMARY KEY', 'name TEXT'])
        ...     db.insert('users', {'name': 'Alice'})
    """
    
    def __init__(self, db_name):
        """
        Initialize database manager.
        
        Args:
            db_name: Name of database file
        """
        # TODO: Store db_name, initialize connection and cursor to None
        self.db_name = db_name
        self.connection = None
        self.cursor = None
    
    def __enter__(self):
        """Enter context - open connection."""
        # TODO: Open database connection, set row_factory, create cursor
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - commit and close connection."""
        # TODO: Commit if no exceptions, close connection
        pass
    
    def create_table(self, table_name, columns):
        """
        Create table if not exists.
        
        Args:
            table_name: Name of table
            columns: List of column definitions
            
        Examples:
            >>> db.create_table('users', ['id INTEGER PRIMARY KEY', 'name TEXT'])
        """
        # TODO: Execute CREATE TABLE IF NOT EXISTS
        pass
    
    def insert(self, table_name, data):
        """
        Insert record into table.
        
        Args:
            table_name: Name of table
            data: Dictionary of column: value pairs
            
        Returns:
            ID of inserted row
        """
        # TODO: Execute INSERT and return lastrowid
        pass
    
    def query(self, table_name, columns='*', where=None, order_by=None):
        """
        Query records from table.
        
        Args:
            table_name: Name of table
            columns: Columns to select (default all)
            where: Dictionary of conditions
            order_by: Column to order by
            
        Returns:
            List of records as dictionaries
            
        Examples:
            >>> db.query('users', where={'age': 30})
            [{'id': 1, 'name': 'Alice', 'age': 30}]
        """
        # TODO: Build and execute SELECT query
        pass
    
    def update(self, table_name, data, where):
        """
        Update records in table.
        
        Args:
            table_name: Name of table
            data: Dictionary of columns to update
            where: Dictionary of conditions
            
        Returns:
            Number of rows updated
        """
        # TODO: Build and execute UPDATE query
        pass
    
    def delete(self, table_name, where):
        """
        Delete records from table.
        
        Args:
            table_name: Name of table
            where: Dictionary of conditions
            
        Returns:
            Number of rows deleted
        """
        # TODO: Build and execute DELETE query
        pass


# Test
with DatabaseManager('test_users.db') as db:
    # Create table
    db.create_table('users', [
        'id INTEGER PRIMARY KEY AUTOINCREMENT',
        'name TEXT NOT NULL',
        'age INTEGER',
        'email TEXT'
    ])
    
    # Insert records
    db.insert('users', {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'})
    db.insert('users', {'name': 'Bob', 'age': 25, 'email': 'bob@example.com'})
    db.insert('users', {'name': 'Charlie', 'age': 30, 'email': 'charlie@example.com'})
    
    # Query all
    print("All users:")
    for user in db.query('users'):
        print(f"  {user}")
    
    # Query with filter
    print("\\nUsers aged 30:")
    for user in db.query('users', where={'age': 30}):
        print(f"  {user}")
    
    # Update
    updated = db.update('users', {'age': 26}, where={'name': 'Bob'})
    print(f"\\nUpdated {updated} record(s)")
    
    # Delete
    deleted = db.delete('users', where={'name': 'Charlie'})
    print(f"Deleted {deleted} record(s)")
    
    # Final query
    print("\\nFinal users:")
    for user in db.query('users', order_by='name'):
        print(f"  {user}")


# Test helper function (for automated testing)
def test_database_manager(table_name, data):
    """Test function for DatabaseManager - implement the class methods above first!"""
    try:
        with DatabaseManager('test.db') as db:
            # Create test table
            db.create_table(table_name, [
                'id INTEGER PRIMARY KEY AUTOINCREMENT',
                'name TEXT',
                'age INTEGER'
            ])
            # Insert and return the row id
            return db.insert(table_name, data)
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: ['users', { name: 'Alice', age: 30 }],
        expected: 1,
        functionName: 'test_database_manager',
      },
    ],
    solution: `import sqlite3

class DatabaseManager:
    def __init__(self, db_name):
        self.db_name = db_name
        self.connection = None
        self.cursor = None
    
    def __enter__(self):
        self.connection = sqlite3.connect(self.db_name)
        self.connection.row_factory = sqlite3.Row  # Access columns by name
        self.cursor = self.connection.cursor()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()
        return False
    
    def create_table(self, table_name, columns):
        columns_str = ', '.join(columns)
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})"
        self.cursor.execute(sql)
    
    def insert(self, table_name, data):
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        self.cursor.execute(sql, tuple(data.values()))
        return self.cursor.lastrowid
    
    def query(self, table_name, columns='*', where=None, order_by=None):
        sql = f"SELECT {columns} FROM {table_name}"
        params = []
        
        if where:
            conditions = ' AND '.join([f"{k} = ?" for k in where.keys()])
            sql += f" WHERE {conditions}"
            params.extend(where.values())
        
        if order_by:
            sql += f" ORDER BY {order_by}"
        
        self.cursor.execute(sql, params)
        return [dict(row) for row in self.cursor.fetchall()]
    
    def update(self, table_name, data, where):
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        where_clause = ' AND '.join([f"{k} = ?" for k in where.keys()])
        sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
        params = list(data.values()) + list(where.values())
        self.cursor.execute(sql, params)
        return self.cursor.rowcount
    
    def delete(self, table_name, where):
        where_clause = ' AND '.join([f"{k} = ?" for k in where.keys()])
        sql = f"DELETE FROM {table_name} WHERE {where_clause}"
        self.cursor.execute(sql, tuple(where.values()))
        return self.cursor.rowcount


# Test helper function (for automated testing)
def test_database_manager(table_name, data):
    """Test function for DatabaseManager."""
    with DatabaseManager('test.db') as db:
        db.create_table(table_name, [
            'id INTEGER PRIMARY KEY AUTOINCREMENT',
            'name TEXT',
            'age INTEGER'
        ])
        return db.insert(table_name, data)`,
    timeComplexity: 'O(n) for queries, O(1) for indexed operations',
    spaceComplexity: 'O(r) where r is number of results',
    order: 16,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-retry-decorator',
    title: 'Retry Decorator with Exponential Backoff',
    difficulty: 'Medium',
    description: `Create a decorator that retries failed function calls with exponential backoff.

**Features:**
- Retry on exception
- Exponential backoff (wait time doubles each retry)
- Maximum retry limit
- Log retry attempts
- Specify which exceptions to retry

**Exponential Backoff:** 1s, 2s, 4s, 8s...

**Example:**
\`\`\`python
@retry(max_attempts=3, delay=1, backoff=2, exceptions=(ConnectionError,))
def fetch_data():
    # Network call that might fail
    response = requests.get(url)
    return response.json()
\`\`\``,
    examples: [
      {
        input: '@retry(max_attempts=3)\\ndef flaky(): ...',
        output: 'Retries up to 3 times',
      },
    ],
    constraints: [
      'Use exponential backoff',
      'Log each retry',
      'Re-raise if max attempts exceeded',
    ],
    hints: [
      'time.sleep() for delays',
      'Multiply delay by backoff factor',
      'Use isinstance() to check exceptions',
    ],
    starterCode: `import time
import functools
import random

def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    Decorator to retry function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay (exponential backoff)
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
        
    Examples:
        >>> @retry(max_attempts=3, delay=1, backoff=2)
        ... def flaky_function():
        ...     if random.random() < 0.7:
        ...         raise ConnectionError("Failed")
        ...     return "Success"
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pass
        return wrapper
    return decorator


# Test with simulated flaky function
@retry(max_attempts=5, delay=0.5, backoff=2, exceptions=(ConnectionError, TimeoutError))
def flaky_network_call(success_rate=0.3):
    """
    Simulate a flaky network call.
    
    Args:
        success_rate: Probability of success (0-1)
        
    Returns:
        Success message
        
    Raises:
        ConnectionError: If call fails
    """
    print(f"  Attempting network call...")
    if random.random() > success_rate:
        raise ConnectionError("Network error")
    return "Data fetched successfully"


@retry(max_attempts=3, delay=1)
def divide(a, b):
    """Division with retry (will fail permanently on ZeroDivisionError)."""
    print(f"  Attempting {a} / {b}")
    return a / b


def test_retry():
    """Test function that validates retry decorator"""
    import sys
    from io import StringIO
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Test basic division (should work immediately)
        result = divide(10, 5)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify function returned correct value
        if result != 2:
            return f"FAIL: Wrong result: {result}"
        
        # Verify retry prints attempt info
        if "attempt" not in output.lower() and "retry" not in output.lower():
            return "FAIL: Decorator should log retry attempts"
        
        return 2
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"
`,
    testCases: [
      {
        input: [],
        expected: 2,
        functionName: 'test_retry',
      },
    ],
    solution: `import time
import functools
import random

def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        print(f"Max attempts ({max_attempts}) reached. Giving up.")
                        raise
                    
                    print(f"Attempt {attempt} failed: {e}")
                    print(f"Retrying in {current_delay:.1f} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # Should not reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator


@retry(max_attempts=3, delay=1)
def divide(a, b):
    """Division with retry."""
    return a / b

result = divide(10, 5)`,
    timeComplexity: 'O(2^n) for backoff delays where n is attempts',
    spaceComplexity: 'O(1)',
    order: 17,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-command-line-parser',
    title: 'Command Line Argument Parser',
    difficulty: 'Medium',
    description: `Create a command-line tool using argparse to process files.

**Features:**
- Parse command-line arguments
- Support required and optional arguments
- Provide help text
- Validate inputs
- Support subcommands

**Example CLI:**
\`\`\`bash
python tool.py process --input data.txt --output result.txt --verbose
python tool.py analyze --file data.csv --format json
\`\`\``,
    examples: [
      {
        input: 'python tool.py --input file.txt',
        output: 'Processes file with arguments',
      },
    ],
    constraints: [
      'Use argparse module',
      'Support multiple subcommands',
      'Provide good help text',
    ],
    hints: [
      'ArgumentParser for main parser',
      'add_argument() for arguments',
      'add_subparsers() for subcommands',
    ],
    starterCode: `import argparse
import sys

def create_parser():
    """
    Create and configure argument parser.
    
    Returns:
        Configured ArgumentParser
    """
    pass


def process_command(args):
    """
    Handle 'process' subcommand.
    
    Args:
        args: Parsed arguments
    """
    print(f"Processing file: {args.input}")
    print(f"Output to: {args.output}")
    
    if args.verbose:
        print("Verbose mode enabled")
    
    # Read input file
    try:
        with open(args.input, 'r') as f:
            content = f.read()
            lines = content.split('\\n')
            
            # Process based on operation
            if args.operation == 'count':
                print(f"\\nLine count: {len(lines)}")
                print(f"Word count: {len(content.split())}")
                print(f"Character count: {len(content)}")
            
            elif args.operation == 'uppercase':
                processed = content.upper()
                with open(args.output, 'w') as out:
                    out.write(processed)
                print(f"\\nConverted to uppercase and saved")
            
            elif args.operation == 'reverse':
                processed = '\\n'.join(reversed(lines))
                with open(args.output, 'w') as out:
                    out.write(processed)
                print(f"\\nReversed lines and saved")
    
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found")
        sys.exit(1)


def analyze_command(args):
    """
    Handle 'analyze' subcommand.
    
    Args:
        args: Parsed arguments
    """
    print(f"Analyzing file: {args.file}")
    print(f"Format: {args.format}")
    
    try:
        with open(args.file, 'r') as f:
            content = f.read()
            lines = [line.strip() for line in content.split('\\n') if line.strip()]
            
            stats = {
                'total_lines': len(lines),
                'total_words': len(content.split()),
                'total_chars': len(content),
                'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0
            }
            
            if args.format == 'json':
                import json
                print(json.dumps(stats, indent=2))
            else:
                for key, value in stats.items():
                    print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


# For testing purposes, we'll simulate command-line arguments
if __name__ == '__main__':
    # Create test file
    with open('test_input.txt', 'w') as f:
        f.write("Hello World\\nThis is a test\\nPython programming\\n")
    
    # Test process command
    print("Test 1: Process with count")
    sys.argv = ['tool.py', 'process', '--input', 'test_input.txt', 
                '--output', 'test_output.txt', '--operation', 'count', '--verbose']
    main()
    
    print("\\n" + "="*50 + "\\n")
    
    # Test analyze command
    print("Test 2: Analyze with JSON format")
    sys.argv = ['tool.py', 'analyze', '--file', 'test_input.txt', '--format', 'json']
    main()
`,
    testCases: [
      {
        input: ['test.txt'],
        expected: true,
      },
    ],
    solution: `import argparse
import sys

def create_parser():
    parser = argparse.ArgumentParser(
        description='File processing and analysis tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process --input data.txt --output result.txt --operation count
  %(prog)s analyze --file data.txt --format json
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process subcommand
    process_parser = subparsers.add_parser('process', help='Process a file')
    process_parser.add_argument('--input', '-i', required=True,
                               help='Input file path')
    process_parser.add_argument('--output', '-o', required=True,
                               help='Output file path')
    process_parser.add_argument('--operation', '-op',
                               choices=['count', 'uppercase', 'reverse'],
                               default='count',
                               help='Operation to perform')
    process_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Enable verbose output')
    process_parser.set_defaults(func=process_command)
    
    # Analyze subcommand
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a file')
    analyze_parser.add_argument('--file', '-f', required=True,
                               help='File to analyze')
    analyze_parser.add_argument('--format', choices=['text', 'json'],
                               default='text',
                               help='Output format')
    analyze_parser.set_defaults(func=analyze_command)
    
    return parser


def process_command(args):
    # Implementation shown in starter code
    pass


def analyze_command(args):
    # Implementation shown in starter code
    pass


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()`,
    timeComplexity: 'O(n) where n is file size',
    spaceComplexity: 'O(n)',
    order: 18,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-text-statistics',
    title: 'Advanced Text Statistics',
    difficulty: 'Medium',
    description: `Calculate comprehensive statistics for text documents.

**Statistics to Calculate:**
- Total words, characters, lines
- Unique words count
- Average word/sentence length
- Most common words
- Reading level (Flesch-Kincaid)
- Lexical diversity (unique words / total words)

**Flesch-Kincaid Reading Ease:**
\`206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)\``,
    examples: [
      {
        input: 'analyze_text("The quick brown fox...")',
        output: 'Comprehensive text statistics',
      },
    ],
    constraints: [
      'Calculate all statistics',
      'Handle edge cases',
      'Estimate syllable count',
    ],
    hints: [
      'Use collections.Counter for word frequency',
      'Split on sentence punctuation',
      'Estimate syllables by counting vowel groups',
    ],
    starterCode: `import re
from collections import Counter
import string

class TextAnalyzer:
    """
    Comprehensive text analysis tool.
    
    Examples:
        >>> analyzer = TextAnalyzer(text)
        >>> stats = analyzer.get_statistics()
        >>> print(stats['word_count'])
    """
    
    def __init__(self, text):
        """
        Initialize analyzer with text.
        
        Args:
            text: Text to analyze
        """
        self.text = text
        self.sentences = self._extract_sentences()
        self.words = self._extract_words()
    
    def _extract_sentences(self):
        """Extract sentences from text."""
        pass
    
    def _extract_words(self):
        """Extract and clean words from text."""
        pass
    
    def _count_syllables(self, word):
        """
        Estimate syllable count in word.
        
        Args:
            word: Word to analyze
            
        Returns:
            Estimated syllable count
        """
        pass
    
    def get_word_count(self):
        """Return total word count."""
        pass
    
    def get_character_count(self, include_spaces=True):
        """Return character count."""
        pass
    
    def get_unique_word_count(self):
        """Return count of unique words."""
        pass
    
    def get_average_word_length(self):
        """Return average word length."""
        pass
    
    def get_average_sentence_length(self):
        """Return average words per sentence."""
        pass
    
    def get_most_common_words(self, n=10):
        """
        Get most common words.
        
        Args:
            n: Number of words to return
            
        Returns:
            List of (word, count) tuples
        """
        pass
    
    def get_lexical_diversity(self):
        """
        Calculate lexical diversity ratio.
        
        Returns:
            Ratio of unique words to total words (0-1)
        """
        pass
    
    def get_reading_level(self):
        """
        Calculate Flesch-Kincaid reading ease score.
        
        Returns:
            Reading ease score (0-100, higher is easier)
        """
        pass
    
    def get_statistics(self):
        """
        Get all statistics as dictionary.
        
        Returns:
            Dictionary of statistics
        """
        pass


# Test
sample_text = """
The quick brown fox jumps over the lazy dog. This is a simple sentence.
Natural language processing is fascinating. It involves analyzing text data.
Python makes text analysis easy and efficient. We can calculate many statistics.
Reading level indicates text complexity. Short sentences are easier to read.
"""

analyzer = TextAnalyzer(sample_text)
stats = analyzer.get_statistics()

print("Text Statistics:")
print("=" * 50)
for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    elif isinstance(value, list):
        print(f"{key}:")
        for item in value[:5]:  # Show first 5
            print(f"  {item}")
    else:
        print(f"{key}: {value}")
`,
    testCases: [
      {
        input: ['The quick brown fox jumps over the lazy dog.'],
        expected: 9,
      },
    ],
    solution: `import re
from collections import Counter
import string

class TextAnalyzer:
    def __init__(self, text):
        self.text = text
        self.sentences = self._extract_sentences()
        self.words = self._extract_words()
    
    def _extract_sentences(self):
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', self.text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_words(self):
        # Remove punctuation and split
        text_clean = self.text.translate(str.maketrans('', '', string.punctuation))
        words = text_clean.lower().split()
        return [w for w in words if w]
    
    def _count_syllables(self, word):
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        # Ensure at least one syllable
        return max(1, syllable_count)
    
    def get_word_count(self):
        return len(self.words)
    
    def get_character_count(self, include_spaces=True):
        if include_spaces:
            return len(self.text)
        return len(self.text.replace(' ', ''))
    
    def get_unique_word_count(self):
        return len(set(self.words))
    
    def get_average_word_length(self):
        if not self.words:
            return 0
        return sum(len(word) for word in self.words) / len(self.words)
    
    def get_average_sentence_length(self):
        if not self.sentences:
            return 0
        return len(self.words) / len(self.sentences)
    
    def get_most_common_words(self, n=10):
        counter = Counter(self.words)
        return counter.most_common(n)
    
    def get_lexical_diversity(self):
        if not self.words:
            return 0
        return len(set(self.words)) / len(self.words)
    
    def get_reading_level(self):
        if not self.sentences or not self.words:
            return 0
        
        total_syllables = sum(self._count_syllables(word) for word in self.words)
        words_per_sentence = len(self.words) / len(self.sentences)
        syllables_per_word = total_syllables / len(self.words)
        
        # Flesch-Kincaid Reading Ease
        score = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        return max(0, min(100, score))  # Clamp to 0-100
    
    def get_statistics(self):
        return {
            'total_characters': self.get_character_count(),
            'total_words': self.get_word_count(),
            'total_sentences': len(self.sentences),
            'unique_words': self.get_unique_word_count(),
            'average_word_length': self.get_average_word_length(),
            'average_sentence_length': self.get_average_sentence_length(),
            'lexical_diversity': self.get_lexical_diversity(),
            'reading_ease': self.get_reading_level(),
            'most_common_words': self.get_most_common_words(10)
        }`,
    timeComplexity: 'O(n) where n is text length',
    spaceComplexity: 'O(w) where w is number of words',
    order: 19,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-markdown-parser',
    title: 'Simple Markdown to HTML Converter',
    difficulty: 'Hard',
    description: `Convert Markdown text to HTML.

**Supported Markdown:**
- Headers: # H1, ## H2, ### H3
- Bold: **text** or __text__
- Italic: *text* or _text_
- Links: [text](url)
- Lists: - item or * item
- Code: \`code\`
- Paragraphs

**Example:**
\`\`\`markdown
# Hello World

This is **bold** and this is *italic*.

- Item 1
- Item 2
\`\`\``,
    examples: [
      {
        input: '# Hello\\n\\nThis is **bold**',
        output: '<h1>Hello</h1>\\n<p>This is <strong>bold</strong></p>',
      },
    ],
    constraints: [
      'Use regex for parsing',
      'Handle nested formatting',
      'Escape HTML entities',
    ],
    hints: [
      'Process line by line',
      'Replace patterns in order',
      'Use html.escape() for safety',
    ],
    starterCode: `import re
import html

class MarkdownConverter:
    """
    Convert Markdown to HTML.
    
    Examples:
        >>> converter = MarkdownConverter()
        >>> html = converter.convert("# Hello\\n\\nWorld")
        >>> print(html)
    """
    
    def __init__(self):
        """Initialize converter with patterns."""
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self):
        """
        Compile regex patterns for Markdown elements.
        
        Returns:
            Dict of pattern name to compiled regex
        """
        # TODO: Return dict of compiled regex patterns for headers, bold, italic, etc.
        return {}  # Return empty dict for now to prevent crashes
    
    def _convert_headers(self, text):
        """Convert Markdown headers to HTML."""
        # TODO: Use regex to convert # Header to <h1>Header</h1>
        pass
    
    def _convert_bold(self, text):
        """Convert bold text to HTML."""
        # TODO: Convert **text** to <strong>text</strong>
        pass
    
    def _convert_italic(self, text):
        """Convert italic text to HTML."""
        # TODO: Convert *text* to <em>text</em>
        pass
    
    def _convert_code(self, text):
        """Convert inline code to HTML."""
        # TODO: Convert \`code\` to <code>code</code>
        pass
    
    def _convert_links(self, text):
        """Convert links to HTML."""
        # TODO: Convert [text](url) to <a href="url">text</a>
        pass
    
    def _convert_lists(self, text):
        """Convert lists to HTML."""
        # TODO: Convert - item to <ul><li>item</li></ul>
        pass
    
    def convert(self, markdown_text):
        """
        Convert Markdown text to HTML.
        
        Args:
            markdown_text: Markdown formatted text
            
        Returns:
            HTML formatted text
            
        Examples:
            >>> converter.convert("**Bold text**")
            '<p><strong>Bold text</strong></p>'
        """
        # TODO: Apply all conversion methods
        pass


# Test
converter = MarkdownConverter()

test_markdown = """
# Welcome to Markdown

This is a paragraph with **bold text** and *italic text*.

## Features

- Easy to write
- Easy to read
- Converts to HTML

Here's a [link](https://example.com) and some \`inline code\`.

### Code Example

\`\`\`
def hello():
    print("Hello, World!")
\`\`\`
"""

print("Markdown Input:")
print("=" * 50)
print(test_markdown)

print("\\n\\nHTML Output:")
print("=" * 50)
html_output = converter.convert(test_markdown)
print(html_output)


# Test helper function (for automated testing)
def test_markdown_converter(markdown_text):
    """Test function for MarkdownConverter - implement the class methods above first!"""
    try:
        converter = MarkdownConverter()
        return converter.convert(markdown_text)
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: ['# Hello'],
        expected: '<h1>Hello</h1>',
        functionName: 'test_markdown_converter',
      },
    ],
    solution: `import re
import html

class MarkdownConverter:
    def __init__(self):
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self):
        return {
            'header': re.compile(r'^(#{1,6})\\s+(.+)$'),
            'bold': re.compile(r'\\*\\*(.+?)\\*\\*|__(.+?)__'),
            'italic': re.compile(r'\\*(.+?)\\*|_(.+?)_'),
            'code': re.compile(r'\`(.+?)\`'),
            'link': re.compile(r'\\[(.+?)\\]\\((.+?)\\)'),
            'list': re.compile(r'^[-*]\\s+(.+)$')
        }
    
    def _convert_headers(self, text):
        match = self.patterns['header'].match(text)
        if match:
            level = len(match.group(1))
            content = match.group(2)
            return f"<h{level}>{content}</h{level}>"
        return None
    
    def _convert_bold(self, text):
        # Replace **text** with <strong>text</strong>
        text = re.sub(r'\\*\\*(.+?)\\*\\*', r'<strong>\\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\\1</strong>', text)
        return text
    
    def _convert_italic(self, text):
        # Replace *text* with <em>text</em>
        # Be careful not to match ** (bold)
        text = re.sub(r'(?<!\\*)\\*([^*]+?)\\*(?!\\*)', r'<em>\\1</em>', text)
        text = re.sub(r'(?<!_)_([^_]+?)_(?!_)', r'<em>\\1</em>', text)
        return text
    
    def _convert_code(self, text):
        return re.sub(r'\`(.+?)\`', r'<code>\\1</code>', text)
    
    def _convert_links(self, text):
        return re.sub(r'\\[(.+?)\\]\\((.+?)\\)', r'<a href="\\2">\\1</a>', text)
    
    def _convert_lists(self, text):
        lines = text.split('\\n')
        result = []
        in_list = False
        
        for line in lines:
            if self.patterns['list'].match(line):
                if not in_list:
                    result.append('<ul>')
                    in_list = True
                match = self.patterns['list'].match(line)
                item_text = match.group(1)
                result.append(f'  <li>{item_text}</li>')
            else:
                if in_list:
                    result.append('</ul>')
                    in_list = False
                result.append(line)
        
        if in_list:
            result.append('</ul>')
        
        return '\\n'.join(result)
    
    def convert(self, markdown_text):
        lines = markdown_text.strip().split('\\n')
        html_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check for headers
            header = self._convert_headers(line)
            if header:
                html_lines.append(header)
                i += 1
                continue
            
            # Check for list items
            if self.patterns['list'].match(line):
                # Collect all consecutive list items
                list_lines = []
                while i < len(lines) and self.patterns['list'].match(lines[i].strip()):
                    list_lines.append(lines[i].strip())
                    i += 1
                
                html_lines.append('<ul>')
                for list_line in list_lines:
                    match = self.patterns['list'].match(list_line)
                    item_text = match.group(1)
                    # Apply inline formatting
                    item_text = self._convert_links(item_text)
                    item_text = self._convert_code(item_text)
                    item_text = self._convert_bold(item_text)
                    item_text = self._convert_italic(item_text)
                    html_lines.append(f'  <li>{item_text}</li>')
                html_lines.append('</ul>')
                continue
            
            # Regular paragraph
            # Apply inline formatting
            line = self._convert_links(line)
            line = self._convert_code(line)
            line = self._convert_bold(line)
            line = self._convert_italic(line)
            html_lines.append(f'<p>{line}</p>')
            
            i += 1
        
        return '\\n'.join(html_lines)


# Test helper function (for automated testing)
def test_markdown_converter(markdown_text):
    """Test function for MarkdownConverter."""
    converter = MarkdownConverter()
    return converter.convert(markdown_text)`,
    timeComplexity: 'O(n*m) where n is lines, m is pattern matches',
    spaceComplexity: 'O(n)',
    order: 20,
    topic: 'Python Intermediate',
  },
  {
    id: 'intermediate-password-manager',
    title: 'Password Strength Validator',
    difficulty: 'Medium',
    description: `Create a comprehensive password strength validator and generator.

**Validation Criteria:**
- Minimum length (8+ characters)
- Contains uppercase and lowercase
- Contains numbers
- Contains special characters
- Not a common password
- Calculate strength score (0-100)

**Features:**
- Validate password strength
- Generate strong passwords
- Suggest improvements
- Check against common passwords`,
    examples: [
      {
        input: 'validate_password("P@ssw0rd123")',
        output: 'Strong (score: 85)',
      },
    ],
    constraints: [
      'Check all criteria',
      'Calculate numeric score',
      'Provide specific feedback',
    ],
    hints: [
      'Use regex for pattern matching',
      'Award points for each criterion',
      'Check against common password list',
    ],
    starterCode: `import re
import random
import string

class PasswordValidator:
    """
    Validate and score password strength.
    
    Examples:
        >>> validator = PasswordValidator()
        >>> result = validator.validate("MyP@ssw0rd")
        >>> print(result['score'])
    """
    
    # Common passwords to check against
    COMMON_PASSWORDS = {
        'password', '123456', '12345678', 'qwerty', 'abc123',
        'monkey', '1234567', 'letmein', 'trustno1', 'dragon'
    }
    
    def __init__(self, min_length=8):
        """
        Initialize validator.
        
        Args:
            min_length: Minimum password length
        """
        self.min_length = min_length
    
    def validate(self, password):
        """
        Validate password and return detailed results.
        
        Args:
            password: Password to validate
            
        Returns:
            Dict with score, strength, and suggestions
            
        Examples:
            >>> validator.validate("P@ssw0rd123")
            {'score': 85, 'strength': 'Strong', 'passed': True, ...}
        """
        pass
    
    def _calculate_score(self, password):
        """Calculate password strength score (0-100)."""
        pass
    
    def _get_strength_level(self, score):
        """Convert score to strength level."""
        pass
    
    def _get_suggestions(self, password):
        """Get list of improvements."""
        pass
    
    def generate_password(self, length=12, use_special=True):
        """
        Generate a strong random password.
        
        Args:
            length: Password length
            use_special: Include special characters
            
        Returns:
            Generated password string
            
        Examples:
            >>> validator.generate_password(16)
            'K9@mPz!vXcQ#w8Rt'
        """
        pass


# Test
validator = PasswordValidator()

test_passwords = [
    "weak",
    "password123",
    "MyPassword1",
    "Str0ng!P@ss",
    "C0mpl3x!P@ssw0rd#2024"
]

print("Password Strength Analysis:")
print("=" * 70)

for pwd in test_passwords:
    result = validator.validate(pwd)
    print(f"\\nPassword: {pwd}")
    print(f"Score: {result['score']}/100")
    print(f"Strength: {result['strength']}")
    print(f"Passed: {result['passed']}")
    
    if result['suggestions']:
        print("Suggestions:")
        for suggestion in result['suggestions']:
            print(f"  - {suggestion}")

# Generate strong passwords
print("\\n\\nGenerated Strong Passwords:")
print("=" * 70)
for i in range(5):
    pwd = validator.generate_password(16)
    result = validator.validate(pwd)
    print(f"{pwd} (Score: {result['score']})")


# Test helper function (for automated testing)
def test_password_validator(password):
    """Test function for PasswordValidator - implement the class methods above first!"""
    try:
        validator = PasswordValidator()
        result = validator.validate(password)
        return result['passed']
    except:
        return None  # Return None if methods not yet implemented
`,
    testCases: [
      {
        input: ['Str0ng!P@ss'],
        expected: true,
        functionName: 'test_password_validator',
      },
    ],
    solution: `import re
import random
import string

class PasswordValidator:
    COMMON_PASSWORDS = {
        'password', '123456', '12345678', 'qwerty', 'abc123',
        'monkey', '1234567', 'letmein', 'trustno1', 'dragon',
        'password123', 'password1', 'admin', 'welcome', 'login'
    }
    
    def __init__(self, min_length=8):
        self.min_length = min_length
    
    def validate(self, password):
        score = self._calculate_score(password)
        strength = self._get_strength_level(score)
        suggestions = self._get_suggestions(password)
        
        return {
            'score': score,
            'strength': strength,
            'passed': score >= 60,
            'suggestions': suggestions,
            'criteria': {
                'length': len(password) >= self.min_length,
                'uppercase': bool(re.search(r'[A-Z]', password)),
                'lowercase': bool(re.search(r'[a-z]', password)),
                'numbers': bool(re.search(r'[0-9]', password)),
                'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)),
                'not_common': password.lower() not in self.COMMON_PASSWORDS
            }
        }
    
    def _calculate_score(self, password):
        score = 0
        
        # Length (max 30 points)
        if len(password) >= self.min_length:
            score += min(30, (len(password) - self.min_length + 1) * 5)
        
        # Character variety (max 40 points)
        if re.search(r'[a-z]', password):
            score += 10
        if re.search(r'[A-Z]', password):
            score += 10
        if re.search(r'[0-9]', password):
            score += 10
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 10
        
        # Complexity bonus (max 20 points)
        unique_chars = len(set(password))
        score += min(20, unique_chars)
        
        # Penalty for common passwords
        if password.lower() in self.COMMON_PASSWORDS:
            score = min(score, 30)
        
        # Penalty for repeated characters
        if re.search(r'(.)\\1{2,}', password):
            score -= 10
        
        return max(0, min(100, score))
    
    def _get_strength_level(self, score):
        if score >= 80:
            return 'Very Strong'
        elif score >= 60:
            return 'Strong'
        elif score >= 40:
            return 'Moderate'
        elif score >= 20:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def _get_suggestions(self, password):
        suggestions = []
        
        if len(password) < self.min_length:
            suggestions.append(f"Increase length to at least {self.min_length} characters")
        
        if not re.search(r'[a-z]', password):
            suggestions.append("Add lowercase letters")
        
        if not re.search(r'[A-Z]', password):
            suggestions.append("Add uppercase letters")
        
        if not re.search(r'[0-9]', password):
            suggestions.append("Add numbers")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            suggestions.append("Add special characters (!@#$%^&*)")
        
        if password.lower() in self.COMMON_PASSWORDS:
            suggestions.append("Avoid common passwords")
        
        if re.search(r'(.)\\1{2,}', password):
            suggestions.append("Avoid repeated characters")
        
        if len(password) < 12:
            suggestions.append("Consider using 12+ characters for better security")
        
        return suggestions
    
    def generate_password(self, length=12, use_special=True):
        # Ensure we have at least one of each required type
        chars = []
        
        # Add required characters
        chars.append(random.choice(string.ascii_lowercase))
        chars.append(random.choice(string.ascii_uppercase))
        chars.append(random.choice(string.digits))
        
        if use_special:
            special_chars = '!@#$%^&*(),.?":{}|<>'
            chars.append(random.choice(special_chars))
        
        # Fill remaining with random mix
        pool = string.ascii_letters + string.digits
        if use_special:
            pool += '!@#$%^&*(),.?":{}|<>'
        
        remaining_length = length - len(chars)
        chars.extend(random.choices(pool, k=remaining_length))
        
        # Shuffle to avoid predictable pattern
        random.shuffle(chars)
        
        return ''.join(chars)


# Test helper function (for automated testing)
def test_password_validator(password):
    """Test function for PasswordValidator."""
    validator = PasswordValidator()
    result = validator.validate(password)
    return result['passed']`,
    timeComplexity: 'O(n) where n is password length',
    spaceComplexity: 'O(1)',
    order: 21,
    topic: 'Python Intermediate',
  },
  {
    id: 'group-anagrams-collections',
    title: 'Group Anagrams',
    difficulty: 'Medium',
    category: 'python-intermediate',
    description: `Given an array of strings \`strs\`, group the anagrams together using \`defaultdict\`. You can return the answer in any order.

An **anagram** is a word formed by rearranging the letters of another, using all original letters exactly once.

**Example 1:**
\`\`\`
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
\`\`\`

**Example 2:**
\`\`\`
Input: strs = [""]
Output: [[""]]
\`\`\`

**Example 3:**
\`\`\`
Input: strs = ["a"]
Output: [["a"]]
\`\`\``,
    starterCode: `from collections import defaultdict

def group_anagrams(strs):
    """
    Group strings that are anagrams of each other.
    
    Args:
        strs: List of strings
    
    Returns:
        List of lists, each containing anagrams
    """
    pass`,
    testCases: [
      {
        input: [['eat', 'tea', 'tan', 'ate', 'nat', 'bat']],
        expected: [['bat'], ['nat', 'tan'], ['ate', 'eat', 'tea']],
      },
      {
        input: [['']],
        expected: [['']],
      },
      {
        input: [['a']],
        expected: [['a']],
      },
    ],
    hints: [
      'Use defaultdict(list) to automatically handle new keys',
      'Sort characters in each word to create a signature',
      'Group words by their sorted signature',
    ],
    solution: `from collections import defaultdict

def group_anagrams(strs):
    """
    Group strings that are anagrams of each other.
    
    Args:
        strs: List of strings
    
    Returns:
        List of lists, each containing anagrams
    """
    # Use defaultdict to avoid KeyError
    anagram_groups = defaultdict(list)
    
    for word in strs:
        # Sort characters to create key
        # Anagrams will have same sorted key
        key = ''.join(sorted(word))
        anagram_groups[key].append(word)
    
    return list(anagram_groups.values())


# Alternative: Using tuple of character counts as key
def group_anagrams_alt(strs):
    from collections import Counter
    anagram_groups = defaultdict(list)
    
    for word in strs:
        # Use tuple of sorted counts as key
        key = tuple(sorted(Counter(word).items()))
        anagram_groups[key].append(word)
    
    return list(anagram_groups.values())`,
    timeComplexity:
      'O(n * k log k) where n is number of strings, k is max length',
    spaceComplexity: 'O(n * k)',
    order: 23,
    topic: 'Python Intermediate',
  },
  {
    id: 'sliding-window-maximum-deque',
    title: 'Sliding Window Maximum',
    difficulty: 'Hard',
    category: 'python-intermediate',
    description: `Given an array \`nums\` and a sliding window of size \`k\`, find the maximum element in each window as it slides from left to right.

Use \`deque\` to solve this efficiently in O(n) time.

**Example 1:**
\`\`\`
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
\`\`\`

**Example 2:**
\`\`\`
Input: nums = [1], k = 1
Output: [1]
\`\`\``,
    starterCode: `from collections import deque

def max_sliding_window(nums, k):
    """
    Find maximum in each sliding window of size k.
    
    Args:
        nums: List of integers
        k: Window size
    
    Returns:
        List of maximums for each window
    """
    pass`,
    testCases: [
      {
        input: [[1, 3, -1, -3, 5, 3, 6, 7], 3],
        expected: [3, 3, 5, 5, 6, 7],
      },
      {
        input: [[1], 1],
        expected: [1],
      },
      {
        input: [[1, -1], 1],
        expected: [1, -1],
      },
    ],
    hints: [
      'Use deque to store indices, not values',
      'Keep deque in decreasing order of values',
      'Remove indices that are out of window range',
      'Front of deque always contains maximum',
    ],
    solution: `from collections import deque

def max_sliding_window(nums, k):
    """
    Find maximum in each sliding window of size k.
    
    Args:
        nums: List of integers
        k: Window size
    
    Returns:
        List of maximums for each window
    """
    if not nums or k == 0:
        return []
    
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices that are out of current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove indices whose values are smaller than current
        # (they can never be maximum)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        # Add current index
        dq.append(i)
        
        # Add maximum to result (once window is full)
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


# Why this works:
# 1. deque maintains indices in decreasing order of their values
# 2. Front of deque is always the maximum in current window
# 3. O(1) per element since each element enters and exits deque once
# 4. Total: O(n) time, O(k) space`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(k)',
    order: 24,
    topic: 'Python Intermediate',
  },
  {
    id: 'lru-cache-implementation',
    title: 'LRU Cache Implementation',
    difficulty: 'Medium',
    category: 'python-intermediate',
    description: `Design a data structure that follows the constraints of a **Least Recently Used (LRU) cache**.

Implement the \`LRUCache\` class using \`OrderedDict\`:
- \`LRUCache(int capacity)\`: Initialize with positive capacity
- \`int get(int key)\`: Return value of key if exists, otherwise -1
- \`void put(int key, int value)\`: Update value or add new key-value pair. If cache exceeds capacity, evict the least recently used key.

Both \`get\` and \`put\` must run in O(1) average time.

**Example:**
\`\`\`python
cache = LRUCache(2)
cache.put(1, 1)  # cache: {1=1}
cache.put(2, 2)  # cache: {1=1, 2=2}
cache.get(1)     # returns 1, cache: {2=2, 1=1}
cache.put(3, 3)  # evicts key 2, cache: {1=1, 3=3}
cache.get(2)     # returns -1 (not found)
cache.put(4, 4)  # evicts key 1, cache: {3=3, 4=4}
cache.get(1)     # returns -1 (not found)
cache.get(3)     # returns 3
cache.get(4)     # returns 4
\`\`\``,
    starterCode: `from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        """Initialize LRU cache with given capacity."""
        pass
    
    def get(self, key: int) -> int:
        """Get value for key, return -1 if not found."""
        pass
    
    def put(self, key: int, value: int) -> None:
        """Add or update key-value pair."""
        pass`,
    testCases: [
      {
        input: [
          ['LRUCache', 2],
          ['put', 1, 1],
          ['put', 2, 2],
          ['get', 1],
        ],
        expected: [null, null, null, 1],
      },
      {
        input: [
          ['LRUCache', 2],
          ['put', 1, 1],
          ['put', 2, 2],
          ['put', 3, 3],
          ['get', 2],
        ],
        expected: [null, null, null, null, -1],
      },
      {
        input: [
          ['LRUCache', 2],
          ['put', 1, 1],
          ['put', 2, 2],
          ['get', 1],
          ['put', 3, 3],
          ['get', 2],
        ],
        expected: [null, null, null, 1, null, -1],
      },
    ],
    hints: [
      'OrderedDict maintains insertion order',
      'Use move_to_end() to mark items as recently used',
      'Use popitem(last=False) to remove least recently used',
    ],
    solution: `from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        """Initialize LRU cache with given capacity."""
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: int) -> int:
        """Get value for key, return -1 if not found."""
        if key not in self.cache:
            return -1
        
        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        """Add or update key-value pair."""
        if key in self.cache:
            # Update existing key and move to end
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        
        # If over capacity, remove least recently used (first item)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


# Test
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))    # 1
cache.put(3, 3)        # Evicts 2
print(cache.get(2))    # -1
cache.put(4, 4)        # Evicts 1
print(cache.get(1))    # -1
print(cache.get(3))    # 3
print(cache.get(4))    # 4`,
    timeComplexity: 'O(1) for both get and put',
    spaceComplexity: 'O(capacity)',
    order: 25,
    topic: 'Python Intermediate',
  },
  {
    id: 'task-scheduler-with-counter',
    title: 'Task Scheduler',
    difficulty: 'Medium',
    category: 'python-intermediate',
    description: `Given a characters array \`tasks\`, where each character represents a unique task and an integer \`n\` representing the cooldown period, return the minimum number of time units needed to complete all tasks.

The same task cannot be executed in two consecutive time units. During a cooldown period, you can either execute another task or stay idle.

Use \`Counter\` and smart scheduling to solve this.

**Example 1:**
\`\`\`
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B
\`\`\`

**Example 2:**
\`\`\`
Input: tasks = ["A","A","A","B","B","B"], n = 0
Output: 6
Explanation: No cooldown, so: A -> A -> A -> B -> B -> B
\`\`\`

**Example 3:**
\`\`\`
Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
Output: 16
Explanation: One optimal solution:
A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle -> idle -> A
\`\`\``,
    starterCode: `from collections import Counter

def least_interval(tasks, n):
    """
    Calculate minimum time units to complete all tasks with cooldown.
    
    Args:
        tasks: List of task characters
        n: Cooldown period
    
    Returns:
        Minimum time units needed
    """
    pass`,
    testCases: [
      {
        input: [['A', 'A', 'A', 'B', 'B', 'B'], 2],
        expected: 8,
      },
      {
        input: [['A', 'A', 'A', 'B', 'B', 'B'], 0],
        expected: 6,
      },
      {
        input: [
          ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'C', 'D', 'E', 'F', 'G'],
          2,
        ],
        expected: 16,
      },
    ],
    hints: [
      'Count task frequencies using Counter',
      'Most frequent task determines minimum time',
      'Calculate idle slots based on most frequent task',
      'Fill idle slots with other tasks',
    ],
    solution: `from collections import Counter

def least_interval(tasks, n):
    """
    Calculate minimum time units to complete all tasks with cooldown.
    
    Args:
        tasks: List of task characters
        n: Cooldown period
    
    Returns:
        Minimum time units needed
    """
    # Count task frequencies
    task_counts = Counter(tasks)
    
    # Find maximum frequency
    max_freq = max(task_counts.values())
    
    # Count how many tasks have maximum frequency
    max_freq_count = sum(1 for count in task_counts.values() if count == max_freq)
    
    # Calculate minimum intervals needed
    # For most frequent task: (max_freq - 1) complete cycles + final tasks
    # Each cycle has (n + 1) slots
    intervals = (max_freq - 1) * (n + 1) + max_freq_count
    
    # Result is max of calculated intervals or total tasks
    # (if n is small, no idle time needed)
    return max(intervals, len(tasks))


# Example walkthrough for ["A","A","A","B","B","B"], n=2:
# max_freq = 3 (both A and B appear 3 times)
# max_freq_count = 2
# intervals = (3-1) * (2+1) + 2 = 2*3 + 2 = 8
# Result: max(8, 6) = 8
#
# Schedule: A -> B -> idle -> A -> B -> idle -> A -> B`,
    timeComplexity: 'O(n) where n is number of tasks',
    spaceComplexity: 'O(1) - at most 26 unique tasks',
    order: 26,
    topic: 'Python Intermediate',
  },
  {
    id: 'design-hit-counter',
    title: 'Design Hit Counter',
    difficulty: 'Medium',
    category: 'python-intermediate',
    description: `Design a hit counter that counts the number of hits received in the past 5 minutes (300 seconds).

Implement \`HitCounter\` class using \`deque\`:
- \`HitCounter()\`: Initialize
- \`void hit(int timestamp)\`: Record a hit at given timestamp
- \`int getHits(int timestamp)\`: Return number of hits in past 5 minutes from timestamp

**Example:**
\`\`\`python
counter = HitCounter()
counter.hit(1)        # hit at timestamp 1
counter.hit(2)        # hit at timestamp 2
counter.hit(3)        # hit at timestamp 3
counter.getHits(4)    # returns 3 (hits at 1,2,3 are within 5 mins)
counter.hit(300)      # hit at timestamp 300
counter.getHits(300)  # returns 4 (hits at 1,2,3,300)
counter.getHits(301)  # returns 3 (hit at 1 is outside 5 min window)
\`\`\`

**Follow-up:** What if hit rate is very high? How would you optimize?`,
    starterCode: `from collections import deque

class HitCounter:
    def __init__(self):
        """Initialize hit counter."""
        pass
    
    def hit(self, timestamp: int) -> None:
        """Record a hit at timestamp."""
        pass
    
    def getHits(self, timestamp: int) -> int:
        """Get hits in past 300 seconds."""
        pass`,
    testCases: [
      {
        input: [
          ['HitCounter'],
          ['hit', 1],
          ['hit', 2],
          ['hit', 3],
          ['getHits', 4],
        ],
        expected: [null, null, null, null, 3],
      },
      {
        input: [
          ['HitCounter'],
          ['hit', 1],
          ['hit', 2],
          ['hit', 3],
          ['hit', 300],
          ['getHits', 300],
        ],
        expected: [null, null, null, null, null, 4],
      },
      {
        input: [
          ['HitCounter'],
          ['hit', 1],
          ['hit', 2],
          ['hit', 3],
          ['hit', 300],
          ['getHits', 301],
        ],
        expected: [null, null, null, null, null, 3],
      },
    ],
    hints: [
      'Use deque to store timestamps',
      'Remove old timestamps when checking hits',
      'Timestamps older than 300 seconds are outside window',
    ],
    solution: `from collections import deque

class HitCounter:
    def __init__(self):
        """Initialize hit counter."""
        self.hits = deque()
    
    def hit(self, timestamp: int) -> None:
        """Record a hit at timestamp."""
        self.hits.append(timestamp)
    
    def getHits(self, timestamp: int) -> int:
        """Get hits in past 300 seconds."""
        # Remove hits outside 5-minute window
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.popleft()
        
        return len(self.hits)


# Optimized version for high hit rate (using buckets)
class HitCounterOptimized:
    """
    For very high hit rates, store (timestamp, count) pairs
    instead of individual timestamps.
    """
    def __init__(self):
        self.hits = deque()  # Store (timestamp, count) tuples
    
    def hit(self, timestamp: int) -> None:
        if self.hits and self.hits[-1][0] == timestamp:
            # Increment count for current timestamp
            self.hits[-1] = (timestamp, self.hits[-1][1] + 1)
        else:
            # New timestamp
            self.hits.append((timestamp, 1))
    
    def getHits(self, timestamp: int) -> int:
        # Remove old timestamps
        while self.hits and self.hits[0][0] <= timestamp - 300:
            self.hits.popleft()
        
        # Sum all counts
        return sum(count for ts, count in self.hits)


# Test
counter = HitCounter()
counter.hit(1)
counter.hit(2)
counter.hit(3)
print(counter.getHits(4))    # 3
counter.hit(300)
print(counter.getHits(300))  # 4
print(counter.getHits(301))  # 3`,
    timeComplexity: 'O(1) for hit, O(n) for getHits where n is hits in window',
    spaceComplexity: 'O(n) where n is hits in window',
    order: 27,
    topic: 'Python Intermediate',
  },
  {
    id: 'most-common-word',
    title: 'Most Common Word',
    difficulty: 'Easy',
    category: 'python-intermediate',
    description: `Given a string \`paragraph\` and an array of banned words, return the most frequent word that is not banned. Words are case-insensitive and punctuation should be ignored.

Use \`Counter\` for efficient counting.

**Example 1:**
\`\`\`
Input: paragraph = "Bob hit a ball, the hit BALL flew far after it was hit.", banned = ["hit"]
Output: "ball"
Explanation: 
"hit" appears 3 times but is banned.
"ball" appears 2 times and is not banned.
\`\`\`

**Example 2:**
\`\`\`
Input: paragraph = "a.", banned = []
Output: "a"
\`\`\``,
    starterCode: `from collections import Counter
import re

def most_common_word(paragraph, banned):
    """
    Find most common non-banned word.
    
    Args:
        paragraph: String containing words and punctuation
        banned: List of banned words
    
    Returns:
        Most common non-banned word (lowercase)
    """
    pass`,
    testCases: [
      {
        input: [
          'Bob hit a ball, the hit BALL flew far after it was hit.',
          ['hit'],
        ],
        expected: 'ball',
      },
      {
        input: ['a.', []],
        expected: 'a',
      },
      {
        input: ['a, a, a, a, b,b,b,c, c', ['a']],
        expected: 'b',
      },
    ],
    hints: [
      'Use regex to extract words: re.findall(r"\\w+", paragraph)',
      'Convert to lowercase',
      'Use Counter.most_common() after filtering banned words',
    ],
    solution: `from collections import Counter
import re

def most_common_word(paragraph, banned):
    """
    Find most common non-banned word.
    
    Args:
        paragraph: String containing words and punctuation
        banned: List of banned words
    
    Returns:
        Most common non-banned word (lowercase)
    """
    # Extract words using regex, convert to lowercase
    words = re.findall(r'\\w+', paragraph.lower())
    
    # Create set of banned words for O(1) lookup
    banned_set = set(banned)
    
    # Count only non-banned words
    word_counts = Counter(word for word in words if word not in banned_set)
    
    # Return most common
    return word_counts.most_common(1)[0][0]


# Alternative without list comprehension
def most_common_word_alt(paragraph, banned):
    words = re.findall(r'\\w+', paragraph.lower())
    banned_set = set(banned)
    
    # Count all words first
    word_counts = Counter(words)
    
    # Remove banned words
    for word in banned_set:
        word_counts.pop(word, None)
    
    return word_counts.most_common(1)[0][0]`,
    timeComplexity: 'O(n + m) where n is paragraph length, m is banned words',
    spaceComplexity: 'O(n)',
    order: 28,
    topic: 'Python Intermediate',
  },
  // New problems (22-50)
  ...pythonIntermediateBatch1,
  ...pythonIntermediateBatch2,
  ...pythonIntermediateBatch3,
];

/**
 * Python Intermediate problems - Building practical Python skills
 */

import { Problem } from '../types';

export const pythonIntermediateProblems: Problem[] = [
  {
    id: 'intermediate-file-word-frequency',
    title: 'File Word Frequency Counter',
    difficulty: 'Medium',
    description: `Read a text file and count the frequency of each word.

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
    starterCode: `def count_word_frequency(filename):
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
    result = count_word_frequency("sample.txt")
    print(result)
except FileNotFoundError as e:
    print(f"Error: {e}")
`,
    testCases: [
      {
        input: ['test.txt'],
        expected: { the: 2, quick: 1, brown: 1 },
      },
    ],
    solution: `import string
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
    
    # Sort by frequency (descending)
    sorted_words = dict(sorted(word_count.items(), 
                               key=lambda x: x[1], 
                               reverse=True))
    
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
        pass
    
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
        pass
    
    def set(self, key, value):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        pass
    
    def save(self):
        """Save configuration to file."""
        pass


# Test
config = ConfigManager("config.json")
print(config.get("database.host", "localhost"))
config.set("database.port", 5432)
config.save()
`,
    testCases: [
      {
        input: ['config.json', 'database.host'],
        expected: 'localhost',
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
            json.dump(self.config, f, indent=2)`,
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
        
        # Try to load existing account
        self.load()
    
    def deposit(self, amount):
        """
        Deposit money into account.
        
        Args:
            amount: Amount to deposit
            
        Raises:
            ValueError: If amount is negative
        """
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
        pass
    
    def get_balance(self):
        """Get current balance."""
        return self.balance
    
    def get_transactions(self):
        """Get transaction history."""
        return self.transactions
    
    def save(self):
        """Save account state to file."""
        pass
    
    def load(self):
        """Load account state from file."""
        pass
    
    def _add_transaction(self, trans_type, amount):
        """Add transaction to history."""
        pass


# Test
account = BankAccount("12345", 1000.00)

account.deposit(500)
print(f"Balance after deposit: {account.get_balance():.2f}")

try:
    account.withdraw(200)
    print(f"Balance after withdrawal: {account.get_balance():.2f}")
except InsufficientFundsError as e:
    print(f"Error: {e}")

print("\\nTransaction History:")
for trans in account.get_transactions():
    print(f"{trans['timestamp']}: {trans['type']} {trans['amount']:.2f}")
`,
    testCases: [
      {
        input: [1000, 500],
        expected: 1500,
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
        self.transactions.append(transaction)`,
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
];

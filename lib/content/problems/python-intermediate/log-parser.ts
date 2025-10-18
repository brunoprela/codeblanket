/**
 * Log File Analyzer
 * Problem ID: intermediate-log-parser
 * Order: 7
 */

import { Problem } from '../../../types';

export const intermediate_log_parserProblem: Problem = {
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
};

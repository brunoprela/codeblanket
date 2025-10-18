/**
 * Email Address Extractor
 * Problem ID: intermediate-email-extractor
 * Order: 5
 */

import { Problem } from '../../../types';

export const intermediate_email_extractorProblem: Problem = {
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
};

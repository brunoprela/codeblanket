/**
 * URL Parser and Validator
 * Problem ID: intermediate-url-parser
 * Order: 10
 */

import { Problem } from '../../../types';

export const intermediate_url_parserProblem: Problem = {
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
         'query': {'id': '123'}, 'fragment': '}
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


def build_url(protocol, domain, path=', query_params=None, fragment='):
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


def build_url(protocol, domain, path=', query_params=None, fragment='):
    # Ensure path starts with /
    if path and not path.startswith('/'):
        path = '/' + path
    
    # Build query string
    query = '
    if query_params:
        query = urlencode(query_params)
    
    # Build URL using urlunparse
    url_parts = (protocol, domain, path, ', query, fragment)
    return urlunparse(url_parts)`,
  timeComplexity: 'O(n) where n is URL length',
  spaceComplexity: 'O(p) where p is number of query parameters',
  order: 10,
  topic: 'Python Intermediate',
};

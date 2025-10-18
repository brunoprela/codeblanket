/**
 * Design URL Shortener
 * Problem ID: url-shortener
 * Order: 12
 */

import { Problem } from '../../../types';

export const url_shortenerProblem: Problem = {
  id: 'url-shortener',
  title: 'Design URL Shortener',
  difficulty: 'Medium',
  topic: 'Design Problems',
  description: `Design a URL shortener service like bit.ly that:

1. Takes a long URL and returns a shortened URL
2. Takes a shortened URL and redirects to the original long URL
3. Tracks the number of times each short URL has been accessed

Implement the \`URLShortener\` class:

- \`URLShortener()\` Initializes the URL shortener system.
- \`String shorten(String longUrl)\` Takes a long URL and returns a shortened URL. The short URL should be unique and as short as possible.
- \`String expand(String shortUrl)\` Takes a short URL and returns the original long URL, or empty string if not found.
- \`int getClickCount(String shortUrl)\` Returns the number of times the short URL has been accessed.

The shortened URL should be in the format: \`http://short.url/{code}\` where \`{code}\` is a unique identifier.`,
  hints: [
    'Use counter for unique IDs (no collisions)',
    'Convert counter to Base62 (0-9, a-z, A-Z) for shorter codes',
    'Two HashMaps: long→short and short→long',
    'Base62: 62^7 = 3.5 trillion possible 7-character codes',
    'Track click counts in separate HashMap',
  ],
  approach: `## Intuition

URL shortener needs:
1. **Unique short codes** for each long URL
2. **Bidirectional mapping**: long ↔ short
3. **Short as possible** codes

---

## Approach: Counter + Base62 Encoding

### Why Counter?

- **Guaranteed unique**: Increment for each URL
- **No collisions**: Unlike hashing
- **Predictable length**: Know code length for given # URLs

### Why Base62?

Convert counter to base 62 (0-9, a-z, A-Z):

\`\`\`
Counter  Base62
0        0
1        1
10       a
61       Z
62       10
1000     g8
1,000,000  4c92
\`\`\`

**Comparison:**
- Base10 (decimal): 1,000,000 = "1000000" (7 chars)
- Base62: 1,000,000 = "4c92" (4 chars)

**Capacity**: 62^7 = 3.5 trillion URLs with 7 characters

---

## Data Structures:

1. \`counter\`: Global counter for unique IDs
2. \`short_to_long\`: HashMap(short_code → long_url)
3. \`long_to_short\`: HashMap(long_url → short_url) *for deduplication*
4. \`clicks\`: HashMap(short_code → click_count)

---

## Example:

\`\`\`
shorten("https://example.com/very/long/url"):
  counter = 1
  short_code = encode_base62(1) = "1"
  return "http://short.url/1"

shorten("https://another.com/long/url"):
  counter = 2
  short_code = encode_base62(2) = "2"
  return "http://short.url/2"

expand("http://short.url/1"):
  returns "https://example.com/very/long/url"
  clicks["1"] += 1
\`\`\`

---

## Time Complexity: O(1) for shorten, expand, getClickCount
## Space Complexity: O(N) where N = number of URLs`,
  testCases: [
    {
      input: [
        ['URLShortener'],
        ['shorten', 'https://leetcode.com/problems/design'],
        ['expand', 'result'],
        ['getClickCount', 'code'],
      ],
      expected: [
        null,
        'http://short.url/1',
        'https://leetcode.com/problems/design',
        1,
      ],
    },
  ],
  solution: `class URLShortener:
    def __init__(self):
        self.counter = 0  # Global counter for unique IDs
        self.short_to_long = {}  # short_code -> long_url
        self.long_to_short = {}  # long_url -> short_url (deduplication)
        self.clicks = {}  # short_code -> click count
        self.base62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.base_url = "http://short.url/"
    
    def encode_base62(self, num: int) -> str:
        """Convert number to base62 string"""
        if num == 0:
            return self.base62[0]
        
        result = []
        while num:
            result.append(self.base62[num % 62])
            num //= 62
        
        return ''.join(reversed(result))
    
    def shorten(self, long_url: str) -> str:
        """Shorten long URL - O(1)"""
        # Check if already shortened (deduplication)
        if long_url in self.long_to_short:
            return self.long_to_short[long_url]
        
        # Generate new short code
        self.counter += 1
        short_code = self.encode_base62(self.counter)
        short_url = self.base_url + short_code
        
        # Store mappings
        self.short_to_long[short_code] = long_url
        self.long_to_short[long_url] = short_url
        self.clicks[short_code] = 0
        
        return short_url
    
    def expand(self, short_url: str) -> str:
        """Expand short URL to long URL - O(1)"""
        # Extract short code from URL
        short_code = short_url.replace(self.base_url, "")
        
        if short_code in self.short_to_long:
            # Track click
            self.clicks[short_code] += 1
            return self.short_to_long[short_code]
        
        return ""  # Not found
    
    def getClickCount(self, short_url: str) -> int:
        """Get click count for short URL - O(1)"""
        short_code = short_url.replace(self.base_url, "")
        return self.clicks.get(short_code, 0)

# Example usage:
# shortener = URLShortener()
# 
# short1 = shortener.shorten("https://leetcode.com/problems/design")
# # Returns "http://short.url/1"
# 
# long1 = shortener.expand(short1)
# # Returns "https://leetcode.com/problems/design"
# # Click count incremented
# 
# clicks = shortener.getClickCount(short1)
# # Returns 1
# 
# # Duplicate URL returns same short code
# short2 = shortener.shorten("https://leetcode.com/problems/design")
# # Returns "http://short.url/1" (same as short1)`,
  timeComplexity: 'O(1) for shorten(), expand(), and getClickCount()',
  spaceComplexity: 'O(N) where N is number of unique URLs',
  patterns: ['HashMap', 'Design', 'Base62 Encoding', 'Counter'],
  companies: ['Google', 'Amazon', 'Microsoft', 'Facebook', 'Bitly'],
};

/**
 * Quiz questions for Regular Expressions section
 */

export const regexQuiz = [
  {
    id: 'pi-regex-q-1',
    question:
      'Explain the difference between re.search(), re.match(), and re.findall(). When would you use each?',
    hint: 'Think about where in the string they look and what they return.',
    sampleAnswer:
      're.match() only checks if the pattern matches at the START of the string - use for validating entire strings (like "does this string look like an email?"). re.search() finds the pattern ANYWHERE in the string, returning the first match - use when you want to find one occurrence. re.findall() returns ALL matches as a list - use when you need multiple occurrences. For validation, use match(). For finding, use search() or findall(). Most common mistake: using match() when you mean search().',
    keyPoints: [
      're.match(): checks start of string only',
      're.search(): finds first match anywhere',
      're.findall(): returns list of all matches',
      'match() for validation, search()/findall() for finding',
    ],
  },
  {
    id: 'pi-regex-q-2',
    question:
      'Why should you use raw strings (r"...") for regex patterns? What problems does it prevent?',
    hint: 'Think about backslashes and Python string escaping.',
    sampleAnswer:
      'Raw strings treat backslashes literally, preventing double-escaping issues. Without raw strings, to match a literal backslash you\'d need "\\\\\\\\", but with raw strings just r"\\\\". Common patterns like \\d (digit) or \\w (word) work as r"\\d" instead of "\\\\d". Always use raw strings for regex - it makes patterns readable and prevents bugs. Without raw strings, you\'d need to escape every backslash twice: once for Python, once for regex.',
    keyPoints: [
      'Prevents double-escaping of backslashes',
      'Makes patterns more readable',
      'r"\\d" instead of "\\\\d"',
      'Always use raw strings for regex',
    ],
  },
  {
    id: 'pi-regex-q-3',
    question:
      'When should you avoid using regex? What are better alternatives for common tasks?',
    hint: 'Consider simplicity, maintainability, and specialized tools.',
    sampleAnswer:
      'Avoid regex for: 1) Simple string operations - use str.startswith(), str.endswith(), str.split(), "substring" in string instead, 2) Parsing HTML/XML - use BeautifulSoup or lxml (regex can\'t handle nesting), 3) Complex patterns that become unreadable - break into multiple steps or use parsing libraries, 4) When string methods are clearer and faster. Regex is powerful but has a learning curve and can be hard to debug. If a simple string method works, use that. "Some people, when confronted with a problem, think \'I know, I\'ll use regular expressions.\' Now they have two problems."',
    keyPoints: [
      'Use string methods for simple operations',
      'Use parsers (BeautifulSoup) for HTML/XML',
      'Break complex patterns into simpler steps',
      'Regex is powerful but can create maintenance issues',
    ],
  },
];

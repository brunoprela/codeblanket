/**
 * Simple Markdown to HTML Converter
 * Problem ID: intermediate-markdown-parser
 * Order: 20
 */

import { Problem } from '../../../types';

export const intermediate_markdown_parserProblem: Problem = {
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
};

/**
 * Markdown & Rich Text Section
 * Module 3: File Processing & Document Understanding
 */

export const markdownrichtextSection = {
  id: 'markdown-rich-text',
  title: 'Markdown & Rich Text',
  content: `# Markdown & Rich Text

Master markdown and rich text processing for documentation and content management in LLM applications.

## Markdown Processing

\`\`\`python
# pip install markdown
import markdown
from pathlib import Path

def markdown_to_html(md_text: str) -> str:
    """Convert markdown to HTML."""
    return markdown.markdown(md_text)

# Read markdown file
def read_markdown(filepath: str) -> str:
    return Path(filepath).read_text(encoding='utf-8')

# Parse markdown with metadata
# pip install python-frontmatter
import frontmatter

def parse_markdown_with_metadata(filepath: str):
    """Parse markdown with YAML frontmatter."""
    post = frontmatter.load(filepath)
    return {
        'metadata': post.metadata,
        'content': post.content
    }
\`\`\`

## Markdown Generation

\`\`\`python
def generate_markdown_table(data: list) -> str:
    """Generate markdown table from data."""
    if not data or len(data) < 2:
        return ""
    
    # Header
    headers = data[0]
    md = "| " + " | ".join(str(h) for h in headers) + " |\\n"
    md += "| " + " | ".join("---" for _ in headers) + " |\\n"
    
    # Rows
    for row in data[1:]:
        md += "| " + " | ".join(str(cell) for cell in row) + " |\\n"
    
    return md

def create_markdown_doc(title: str, sections: list) -> str:
    """Create structured markdown document."""
    md = f"# {title}\\n\\n"
    
    for section in sections:
        if section['type'] == 'heading':
            level = '#' * section.get('level', 2)
            md += f"{level} {section['text']}\\n\\n"
        elif section['type'] == 'paragraph':
            md += f"{section['text']}\\n\\n"
        elif section['type'] == 'list':
            for item in section['items']:
                md += f"- {item}\\n"
            md += "\\n"
        elif section['type'] == 'code':
            lang = section.get('language', '')
            md += f"\`\`\`{lang}\\n{section['code']}\\n\`\`\`\\n\\n"
    
    return md
\`\`\`

## Rich Text Formats

\`\`\`python
# HTML to Markdown
# pip install html2text
import html2text

def html_to_markdown(html: str) -> str:
    """Convert HTML to markdown."""
    h = html2text.HTML2Text()
    h.ignore_links = False
    return h.handle(html)

# RTF processing
# pip install striprtf
from striprtf.striprtf import rtf_to_text

def read_rtf(filepath: str) -> str:
    """Extract text from RTF file."""
    with open(filepath, 'r') as f:
        rtf_content = f.read()
    return rtf_to_text(rtf_content)
\`\`\`

## Key Takeaways

1. **Use markdown library** for conversion
2. **Parse frontmatter** for metadata
3. **Generate markdown** programmatically
4. **Convert HTML to markdown** when needed
5. **Handle RTF** with striprtf
6. **Validate markdown** after generation
7. **Preserve formatting** in conversions
8. **Use standard markdown** for compatibility
9. **Support code blocks** with language tags
10. **Create structured documents** for clarity`,
  videoUrl: undefined,
};

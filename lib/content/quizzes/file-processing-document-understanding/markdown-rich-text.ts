/**
 * Quiz questions for Markdown & Rich Text section
 */

export const markdownrichtextQuiz = [
  {
    id: 'fpdu-markdown-q-1',
    question:
      'How would you build a documentation generator that converts Python docstrings to markdown documentation?',
    hint: 'Consider docstring parsing, markdown generation, structure, and examples.',
    sampleAnswer:
      'Documentation generator: (1) Parse Python files with AST to extract functions/classes. (2) Extract docstrings with their formatting. (3) Parse docstring format (Google, NumPy, reStructuredText). (4) Convert to markdown with proper structure (## Function Name, ### Parameters, etc.). (5) Include code examples from docstring. (6) Generate table of contents. (7) Add navigation links. (8) Create index page linking all modules.',
    keyPoints: [
      'Parse Python AST for docstrings',
      'Convert docstring format to markdown',
      'Generate structured documentation',
      'Include code examples',
      'Create navigation and index',
    ],
  },
  {
    id: 'fpdu-markdown-q-2',
    question:
      'Explain the benefits of using markdown for LLM-generated content versus HTML or plain text.',
    hint: 'Think about simplicity, structure, readability, and processing.',
    sampleAnswer:
      'Markdown benefits: (1) Simple syntax - easy for LLMs to generate correctly. (2) Human readable in raw form. (3) Structured yet lightweight. (4) Converts easily to HTML, PDF, etc. (5) Standard format across platforms. (6) Supports code blocks, tables, lists naturally. (7) Git-friendly for versioning. (8) No complex tags like HTML. For LLMs: easier to generate, fewer syntax errors, natural structure.',
    keyPoints: [
      'Simple syntax for LLMs',
      'Human readable',
      'Converts to multiple formats',
      'Standard across platforms',
      'Fewer syntax errors than HTML',
    ],
  },
  {
    id: 'fpdu-markdown-q-3',
    question:
      'How would you convert a large corpus of HTML documentation to markdown for LLM processing?',
    hint: 'Consider tools, validation, structure preservation, and batch processing.',
    sampleAnswer:
      'HTML to markdown conversion: (1) Use html2text library for conversion. (2) Clean HTML first (remove scripts, styles). (3) Configure html2text options (preserve links, ignore images if needed). (4) Batch process files maintaining directory structure. (5) Validate converted markdown renders correctly. (6) Handle special cases (tables, nested lists). (7) Preserve metadata and frontmatter. (8) Test with LLM to ensure quality.',
    keyPoints: [
      'Use html2text library',
      'Clean HTML before conversion',
      'Batch process with structure preservation',
      'Validate output',
      'Handle special cases',
      'Test with LLM',
    ],
  },
];

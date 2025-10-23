/**
 * Multiple choice questions for Markdown & Rich Text section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const markdownrichtextMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fpdu-markdown-mc-1',
    question: 'What is the markdown syntax for a heading level 2?',
    options: ['# Heading', '## Heading', '### Heading', '= Heading ='],
    correctAnswer: 1,
    explanation:
      '## creates a level 2 heading in markdown. # is level 1, ### is level 3, etc.',
  },
  {
    id: 'fpdu-markdown-mc-2',
    question: 'Which library converts markdown to HTML in Python?',
    options: ['markdown', 'md2html', 'markdown-parser', 'markdownify'],
    correctAnswer: 0,
    explanation:
      'The "markdown" library (pip install markdown) is the standard for converting markdown to HTML in Python.',
  },
  {
    id: 'fpdu-markdown-mc-3',
    question: 'What is YAML frontmatter in markdown files?',
    options: [
      'File header with metadata',
      'Markdown table format',
      'Code block syntax',
      'Image caption',
    ],
    correctAnswer: 0,
    explanation:
      'YAML frontmatter is metadata at the start of markdown files, enclosed in --- delimiters, containing key-value pairs like title, date, author.',
  },
  {
    id: 'fpdu-markdown-mc-4',
    question:
      'How do you create a code block with syntax highlighting in markdown?',
    options: [
      '`code`',
      '```language\\ncode\\n```',
      '<code>code</code>',
      'indent with 4 spaces',
    ],
    correctAnswer: 1,
    explanation:
      'Triple backticks with optional language identifier (```python) create fenced code blocks with syntax highlighting.',
  },
  {
    id: 'fpdu-markdown-mc-5',
    question: 'Which library converts HTML to markdown?',
    options: ['html2markdown', 'html2text', 'markdownify', 'Both B and C'],
    correctAnswer: 3,
    explanation:
      'Both html2text and markdownify can convert HTML to markdown. html2text is more established, markdownify offers more control.',
  },
];

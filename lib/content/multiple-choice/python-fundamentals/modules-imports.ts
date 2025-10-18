/**
 * Multiple choice questions for Modules and Imports section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const modulesimportsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does "import math" do?',
    options: [
      'Imports all Python modules',
      'Imports the math module into the current namespace',
      'Creates a new math module',
      'Deletes the math module',
    ],
    correctAnswer: 1,
    explanation:
      'import math loads the math module, making its functions available via math.function_name().',
  },
  {
    id: 'mc2',
    question:
      'What is the difference between "import math" and "from math import *"?',
    options: [
      'No difference',
      '"from math import *" imports all functions directly into namespace',
      '"import math" is faster',
      '"from math import *" is recommended',
    ],
    correctAnswer: 1,
    explanation:
      '"from math import *" imports all functions directly (e.g., sqrt() instead of math.sqrt()), but pollutes namespace and is not recommended.',
  },
  {
    id: 'mc3',
    question:
      'What is the value of __name__ when a Python file is run directly?',
    options: ['The filename', '"__main__"', '"main"', 'None'],
    correctAnswer: 1,
    explanation:
      'When a Python file is executed directly, __name__ is set to "__main__", allowing the if __name__ == "__main__": pattern.',
  },
  {
    id: 'mc4',
    question: 'What is a Python package?',
    options: [
      'A single .py file',
      'A directory containing modules and __init__.py',
      'A compressed file',
      'A function collection',
    ],
    correctAnswer: 1,
    explanation:
      'A package is a directory containing Python modules and an __init__.py file, allowing hierarchical module organization.',
  },
  {
    id: 'mc5',
    question: 'Which import style is generally recommended?',
    options: [
      'from module import *',
      'import module or from module import specific_function',
      'Always use relative imports',
      'Import in the middle of code',
    ],
    correctAnswer: 1,
    explanation:
      'Explicit imports (import module or from module import func) are recommended for clarity. Avoid "import *" and import at the top of files.',
  },
];

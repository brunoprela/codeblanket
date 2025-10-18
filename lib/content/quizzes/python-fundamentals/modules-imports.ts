/**
 * Quiz questions for Modules and Imports section
 */

export const modulesimportsQuiz = [
  {
    id: 'q1',
    question:
      'What is the difference between "import math" and "from math import sqrt"?',
    sampleAnswer:
      '"import math" imports the entire math module, and you access functions using math.sqrt(). "from math import sqrt" imports only the sqrt function, allowing you to use it directly as sqrt() without the module prefix. The first is more explicit about where functions come from, while the second is more concise but can cause naming conflicts if multiple modules have functions with the same name.',
    keyPoints: [
      'import math: use math.sqrt()',
      'from math import sqrt: use sqrt() directly',
      'First is more explicit',
      'Second is more concise',
      'Consider namespace pollution',
    ],
  },
  {
    id: 'q2',
    question: 'Why is the "if __name__ == \'__main__\':" pattern useful?',
    sampleAnswer:
      'This pattern allows a Python file to work as both an importable module and a standalone script. When the file is run directly, __name__ equals "__main__" and the code inside the if block executes. When the file is imported as a module, __name__ is the module name and the code doesn\'t run. This prevents unwanted code execution during imports and is the standard way to structure Python scripts.',
    keyPoints: [
      'File works as both module and script',
      'Code only runs when executed directly',
      'Prevents execution on import',
      'Standard Python pattern',
      '__name__ == "__main__" when run directly',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the import best practices and why should you avoid "from module import *"?',
    sampleAnswer:
      'Best practices: (1) Import at top of file for clear dependencies, (2) Use absolute imports for clarity, (3) Group imports: standard library, then third-party, then local, (4) One import per line for readability. Avoid "from module import *" because it pollutes namespace with potentially hundreds of names, makes it unclear where functions come from (reduces code readability), can cause naming conflicts if multiple modules have same function names, and makes debugging harder. For example, if you do "from math import *" and "from numpy import *", both have "sqrt" function, causing confusion. Better: "import math" and "math.sqrt()" or "from math import sqrt" for specific imports.',
    keyPoints: [
      'Import at top, use absolute imports, group by type',
      '"import *" pollutes namespace',
      'Makes code unclear (where does function come from?)',
      'Can cause naming conflicts between modules',
      'Better: explicit imports or use module prefix',
    ],
  },
];

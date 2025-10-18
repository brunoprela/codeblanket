/**
 * Quiz questions for Generators & Iterators section
 */

export const generatorsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between a generator and a list comprehension. When would you use each?',
    sampleAnswer:
      'A list comprehension creates and stores the entire list in memory immediately. A generator expression looks similar but uses parentheses and produces values lazily on-demand. Use list comprehensions when you need: (1) the entire dataset in memory, (2) random access to elements, (3) to iterate multiple times, or (4) the dataset is small. Use generators when: (1) processing large datasets, (2) values are used only once, (3) building data pipelines, or (4) working with infinite sequences. For example, [x**2 for x in range(1000000)] creates a million-element list in memory. (x**2 for x in range(1000000)) creates an iterator that computes each square on-demand.',
    keyPoints: [
      'List comp: stores entire list in memory',
      'Generator: produces values on-demand',
      'Use list comp: need random access, multiple iterations',
      'Use generator: large data, one-time use, pipelines',
      'Memory: O(n) vs O(1)',
    ],
  },
  {
    id: 'q2',
    question:
      'How do generators enable processing of datasets that do not fit in memory? Give a concrete example.',
    sampleAnswer:
      'Generators process one item at a time without storing the entire dataset. For example, processing a 50GB log file: with a list, you would read all 50GB into memory and crash. With a generator, you read and process one line at a time—memory usage stays constant regardless of file size. The key is that generators maintain state between yields but only hold the current item. This allows processing datasets larger than RAM. It is how tools like grep process terabyte files: stream processing, not batch loading.',
    keyPoints: [
      'Process one item at a time',
      'Constant memory usage',
      'Example: 50GB file read line by line',
      'Maintains state, not full dataset',
      'Enables stream processing',
    ],
  },
  {
    id: 'q3',
    question:
      'What is the advantage of chaining generators in a pipeline versus processing data in steps? How does it affect memory usage?',
    sampleAnswer:
      'Chaining generators creates a lazy pipeline where each stage processes one item at a time before passing it to the next stage. This keeps memory usage constant (O(1)) regardless of dataset size. In contrast, processing in steps requires storing intermediate results. For example: list → filter → map → list requires three full lists in memory. But generator → generator → generator processes one item through all stages before moving to the next, using minimal memory. This is how Unix pipes work (cat file | grep pattern | sort) - streaming data through transformations without materializing intermediate results.',
    keyPoints: [
      'Lazy pipeline: one item through all stages',
      'Constant O(1) memory usage',
      'No intermediate result storage',
      'Similar to Unix pipes',
      'Efficient for large datasets',
    ],
  },
];

/**
 * Quiz questions for Control Flow section
 */

export const controlflowQuiz = [
  {
    id: 'pf-control-q-1',
    question:
      'Explain the difference between break, continue, and pass in Python loops. When would you use each?',
    hint: 'Think about what happens to loop execution and when you need each control statement.',
    sampleAnswer:
      "`break` exits the entire loop immediately. Use it when you've found what you're looking for or a condition makes continuing unnecessary. `continue` skips the rest of the current iteration and moves to the next one. Use it to skip processing for certain values. `pass` does nothing - it's a placeholder for code you'll write later, or when syntax requires a statement but you don't want to do anything. Example: When searching a list, use break once found. When processing numbers, use continue to skip negatives. Use pass when defining an empty function stub.",
    keyPoints: [
      'break: exits the loop entirely',
      'continue: skips to next iteration',
      'pass: does nothing, placeholder statement',
      'All serve different purposes in flow control',
    ],
  },
  {
    id: 'pf-control-q-2',
    question:
      'When should you choose a for loop versus a while loop? Can you convert any while loop to a for loop?',
    hint: "Consider when iteration count is known vs unknown, and whether you're iterating over a sequence.",
    sampleAnswer:
      'Use `for` loops when iterating over a known sequence (list, range, string) or when the number of iterations is predetermined. Use `while` loops when iterations depend on a condition that might change unpredictably, like waiting for user input, reading until end of file, or implementing game loops. While technically any while loop can be rewritten as a for loop (using itertools or custom iterators), it often makes code less readable. For example, "while True" with conditional breaks is clearer than forcing it into a for loop structure.',
    keyPoints: [
      'for: iterate over sequences or known ranges',
      'while: condition-based iteration',
      'for loops are more Pythonic for sequences',
      'while loops better for event-driven logic',
    ],
  },
  {
    id: 'pf-control-q-3',
    question:
      'What is the purpose of the else clause in Python loops? How does it differ from putting code after the loop?',
    hint: "Think about when the else block executes and when it doesn't, especially with break statements.",
    sampleAnswer:
      "The `else` clause in loops executes only if the loop completes normally (without hitting a break statement). This is different from code after the loop, which always runs. It\'s useful for search operations: if you break when finding something, else won't run; if you don't find it, else runs to handle the \"not found\" case. For example: searching for a prime number - if you break after finding a divisor, else doesn't run; if no divisors found, else confirms it's prime. Code after the loop would run regardless of whether you broke out or not.",
    keyPoints: [
      'else runs if loop completes without break',
      'Different from code placed after loop',
      'Useful for search/validation patterns',
      'Eliminates need for flag variables',
    ],
  },
];

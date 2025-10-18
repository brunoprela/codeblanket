/**
 * Timer Context Manager
 * Problem ID: context-manager-timer
 * Order: 6
 */

import { Problem } from '../../../types';

export const context_manager_timerProblem: Problem = {
  id: 'context-manager-timer',
  title: 'Timer Context Manager',
  difficulty: 'Easy',
  description: `Implement a context manager that times code execution in a with block.

The context manager should:
- Record start time on entry
- Calculate and print elapsed time on exit
- Work even if exception occurs
- Print time with 3 decimal places

**Pattern:**
python
with Timer():
    # code to time
    time.sleep(1)
# Prints: "Elapsed time: 1.000s"
`,
  examples: [
    {
      input: 'Code block that takes 0.5 seconds',
      output: 'Prints "Elapsed time: 0.500s"',
    },
  ],
  constraints: [
    'Must implement __enter__ and __exit__',
    'Print even if exception occurs',
    'Use time.time() for measurement',
  ],
  hints: [
    'Store start time in __enter__',
    'Calculate elapsed in __exit__',
    '__exit__ receives exception info',
  ],
  starterCode: `import time

class Timer:
    """
    Context manager that times code execution.
    """
    
    def __init__(self):
        """Initialize timer."""
        # TODO: Initialize start and end times
        self.start = None
        self.end = None
    
    def __enter__(self):
        # TODO: Record start time
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: Calculate and print elapsed time
        pass


with Timer():
    time.sleep(0.5)


# Test helper function (for automated testing)
def test_timer(sleep_duration):
    """Test function for Timer - implement the class methods above first!"""
    try:
        with Timer():
            time.sleep(sleep_duration)
        return 'Elapsed'  # If it completes without error
    except:
        return None  # Return None if methods not yet implemented
`,
  testCases: [
    {
      input: [0.5], // sleep duration
      expected: 'Elapsed',
      functionName: 'test_timer',
    },
  ],
  solution: `import time

class Timer:
    def __init__(self):
        self.start = None
        self.end = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        elapsed = self.end - self.start
        print(f"Elapsed time: {elapsed:.3f}s")
        return False  # Don't suppress exceptions


# Test helper function (for automated testing)
def test_timer(sleep_duration):
    """Test function for Timer."""
    with Timer():
        time.sleep(sleep_duration)
    return 'Elapsed'`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 6,
  topic: 'Python Advanced',
};

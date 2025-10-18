/**
 * Structural Pattern Matching (Match-Case)
 * Problem ID: advanced-match-case
 * Order: 34
 */

import { Problem } from '../../../types';

export const match_caseProblem: Problem = {
  id: 'advanced-match-case',
  title: 'Structural Pattern Matching (Match-Case)',
  difficulty: 'Medium',
  description: `Use Python 3.10+ match-case for structural pattern matching.

Implement match-case for:
- Type-based dispatch
- Destructuring sequences
- Matching with guards
- Handling complex data structures

**Pattern:** More powerful and readable than if-elif chains.`,
  examples: [
    {
      input: 'match_shape(("circle", 5))',
      output: 'Circle with radius 5',
    },
  ],
  constraints: [
    'Use match-case statements',
    'Python 3.10+ required',
    'Handle all cases with default',
  ],
  hints: [
    'case pattern if guard:',
    'Use | for OR patterns',
    'Destructure with case (a, b):',
  ],
  starterCode: `def match_command(command):
    """Match command patterns.
    
    Args:
        command: Tuple of (action, *args)
        
    Returns:
        String describing action
    """
    # match command:
    #     case ("quit",):
    #         return "Quitting"
    #     case ("move", x, y):
    #         return f"Moving to {x}, {y}"
    #     case ("draw", shape, *params):
    #         return f"Drawing {shape} with {params}"
    #     case _:
    #         return "Unknown command"
    pass


def classify_point(point):
    """Classify point location.
    
    Args:
        point: Tuple of (x, y)
        
    Returns:
        Location description
    """
    # Use match with guards
    # case (0, 0): origin
    # case (0, y): on y-axis
    # case (x, 0): on x-axis
    # case (x, y) if x == y: on diagonal
    # case (x, y) if x > 0 and y > 0: quadrant 1
    pass


# Test
result = match_command(("move", 10, 20))
`,
  testCases: [
    {
      input: [],
      expected: 'Moving to 10, 20',
    },
  ],
  solution: `def match_command(command):
    match command:
        case ("quit",):
            return "Quitting"
        case ("move", x, y):
            return f"Moving to {x}, {y}"
        case ("draw", shape, *params):
            return f"Drawing {shape} with {params}"
        case _:
            return "Unknown command"


def classify_point(point):
    match point:
        case (0, 0):
            return "origin"
        case (0, y):
            return f"on y-axis at {y}"
        case (x, 0):
            return f"on x-axis at {x}"
        case (x, y) if x == y:
            return f"on diagonal at ({x}, {y})"
        case (x, y) if x > 0 and y > 0:
            return f"quadrant 1: ({x}, {y})"
        case (x, y) if x < 0 and y > 0:
            return f"quadrant 2: ({x}, {y})"
        case (x, y) if x < 0 and y < 0:
            return f"quadrant 3: ({x}, {y})"
        case (x, y):
            return f"quadrant 4: ({x}, {y})"


# Test
result = match_command(("move", 10, 20))`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 34,
  topic: 'Python Advanced',
};

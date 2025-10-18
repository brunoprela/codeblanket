/**
 * Chain of Responsibility Pattern
 * Problem ID: oop-chain-of-responsibility
 * Order: 41
 */

import { Problem } from '../../../types';

export const chain_of_responsibilityProblem: Problem = {
  id: 'oop-chain-of-responsibility',
  title: 'Chain of Responsibility Pattern',
  difficulty: 'Hard',
  description: `Implement chain of responsibility pattern.

**Pattern:**
- Chain of handlers
- Request passes along chain
- Handler processes or passes to next
- Decouples sender from receiver

This tests:
- Chain of responsibility
- Request handling
- Handler chaining`,
  examples: [
    {
      input: 'Request → Handler1 → Handler2 → Handler3',
      output: 'First capable handler processes',
    },
  ],
  constraints: ['Chain handlers', 'Pass request along chain'],
  hints: [
    'Each handler has reference to next',
    'Process or delegate',
    'Set up chain in advance',
  ],
  starterCode: `class Handler:
    """Base handler"""
    def __init__(self):
        self.next_handler = None
    
    def set_next(self, handler):
        """Set next handler in chain"""
        self.next_handler = handler
        return handler
    
    def handle(self, request):
        """Handle or pass to next"""
        if self.next_handler:
            return self.next_handler.handle(request)
        return None


class LowPriorityHandler(Handler):
    """Handles low priority (< 10)"""
    def handle(self, request):
        if request < 10:
            return f"Low priority handler: {request}"
        return super().handle(request)


class MediumPriorityHandler(Handler):
    """Handles medium priority (10-50)"""
    def handle(self, request):
        if 10 <= request < 50:
            return f"Medium priority handler: {request}"
        return super().handle(request)


class HighPriorityHandler(Handler):
    """Handles high priority (>= 50)"""
    def handle(self, request):
        if request >= 50:
            return f"High priority handler: {request}"
        return super().handle(request)


def test_chain():
    """Test chain of responsibility"""
    # Build chain
    low = LowPriorityHandler()
    medium = MediumPriorityHandler()
    high = HighPriorityHandler()
    
    low.set_next(medium).set_next(high)
    
    # Send requests
    result1 = low.handle(5)   # Low
    result2 = low.handle(25)  # Medium
    result3 = low.handle(75)  # High
    
    return len(result1) + len(result2) + len(result3)
`,
  testCases: [
    {
      input: [],
      expected: 85,
      functionName: 'test_chain',
    },
  ],
  solution: `class Handler:
    def __init__(self):
        self.next_handler = None
    
    def set_next(self, handler):
        self.next_handler = handler
        return handler
    
    def handle(self, request):
        if self.next_handler:
            return self.next_handler.handle(request)
        return None


class LowPriorityHandler(Handler):
    def handle(self, request):
        if request < 10:
            return f"Low priority handler: {request}"
        return super().handle(request)


class MediumPriorityHandler(Handler):
    def handle(self, request):
        if 10 <= request < 50:
            return f"Medium priority handler: {request}"
        return super().handle(request)


class HighPriorityHandler(Handler):
    def handle(self, request):
        if request >= 50:
            return f"High priority handler: {request}"
        return super().handle(request)


def test_chain():
    low = LowPriorityHandler()
    medium = MediumPriorityHandler()
    high = HighPriorityHandler()
    
    low.set_next(medium).set_next(high)
    
    result1 = low.handle(5)
    result2 = low.handle(25)
    result3 = low.handle(75)
    
    return len(result1) + len(result2) + len(result3)`,
  timeComplexity: 'O(n) where n is chain length',
  spaceComplexity: 'O(1)',
  order: 41,
  topic: 'Python Object-Oriented Programming',
};

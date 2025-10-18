/**
 * State Pattern
 * Problem ID: oop-state-pattern
 * Order: 35
 */

import { Problem } from '../../../types';

export const state_patternProblem: Problem = {
  id: 'oop-state-pattern',
  title: 'State Pattern',
  difficulty: 'Hard',
  description: `Implement state pattern to change behavior based on internal state.

**Pattern:**
- Object behavior depends on state
- State encapsulated in separate classes
- State transitions
- Delegates to state object

This tests:
- State pattern
- Behavior switching
- State machines`,
  examples: [
    {
      input: 'Connection state: disconnected → connecting → connected',
      output: 'Different behavior per state',
    },
  ],
  constraints: ['Separate state classes', 'Delegate to state'],
  hints: [
    'State interface',
    'Concrete state classes',
    'Context delegates to state',
  ],
  starterCode: `class State:
    """Base state"""
    def handle(self, context):
        raise NotImplementedError


class OffState(State):
    """Off state"""
    def handle(self, context):
        context.state = OnState()
        return "Turning on..."


class OnState(State):
    """On state"""
    def handle(self, context):
        context.state = OffState()
        return "Turning off..."


class Switch:
    """Context with state"""
    def __init__(self):
        self.state = OffState()
    
    def press(self):
        """Delegate to current state"""
        return self.state.handle(self)


def test_state():
    """Test state pattern"""
    switch = Switch()
    
    # Press 1: off -> on
    result1 = switch.press()
    
    # Press 2: on -> off
    result2 = switch.press()
    
    # Press 3: off -> on
    result3 = switch.press()
    
    return len(result1 + result2 + result3)
`,
  testCases: [
    {
      input: [],
      expected: 39,
      functionName: 'test_state',
    },
  ],
  solution: `class State:
    def handle(self, context):
        raise NotImplementedError


class OffState(State):
    def handle(self, context):
        context.state = OnState()
        return "Turning on..."


class OnState(State):
    def handle(self, context):
        context.state = OffState()
        return "Turning off..."


class Switch:
    def __init__(self):
        self.state = OffState()
    
    def press(self):
        return self.state.handle(self)


def test_state():
    switch = Switch()
    result1 = switch.press()
    result2 = switch.press()
    result3 = switch.press()
    return len(result1 + result2 + result3)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 35,
  topic: 'Python Object-Oriented Programming',
};

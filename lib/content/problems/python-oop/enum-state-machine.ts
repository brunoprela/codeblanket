/**
 * State Machine with Enum
 * Problem ID: oop-enum-state-machine
 * Order: 13
 */

import { Problem } from '../../../types';

export const enum_state_machineProblem: Problem = {
  id: 'oop-enum-state-machine',
  title: 'State Machine with Enum',
  difficulty: 'Medium',
  description: `Implement a state machine using Python's \`Enum\` for a traffic light system.

Create:
- \`TrafficLightState\` Enum with RED, YELLOW, GREEN
- \`TrafficLight\` class that manages state transitions
- Transition rules: RED → GREEN → YELLOW → RED
- Time duration for each state
- \`next_state()\` method with validation

**Pattern:** Enums provide type-safe state management.`,
  examples: [
    {
      input: 'light.next_state() from RED',
      output: 'Changes to GREEN',
    },
  ],
  constraints: [
    'Use enum.Enum',
    'Enforce valid transitions only',
    'Each state has a duration',
  ],
  hints: [
    'Import Enum from enum',
    'Use dictionary for transition rules',
    'Store current_state in traffic light',
  ],
  starterCode: `from enum import Enum

class TrafficLightState(Enum):
    """Traffic light states."""
    RED = 1
    YELLOW = 2
    GREEN = 3


class TrafficLight:
    """Traffic light with state machine."""
    
    # State transitions
    TRANSITIONS = {
        TrafficLightState.RED: TrafficLightState.GREEN,
        TrafficLightState.GREEN: TrafficLightState.YELLOW,
        TrafficLightState.YELLOW: TrafficLightState.RED,
    }
    
    # Duration for each state (seconds)
    DURATIONS = {
        TrafficLightState.RED: 30,
        TrafficLightState.YELLOW: 5,
        TrafficLightState.GREEN: 25,
    }
    
    def __init__(self):
        self.current_state = TrafficLightState.RED
    
    def next_state(self):
        """Transition to next state."""
        pass
    
    def get_duration(self):
        """Get duration for current state."""
        pass
    
    def can_transition_to(self, state):
        """Check if transition to state is valid."""
        pass
    
    def __str__(self):
        return f"Traffic Light: {self.current_state.name}"


# Test
light = TrafficLight()
print(light)  # RED
print(f"Duration: {light.get_duration()}s")

light.next_state()
print(light)  # GREEN

light.next_state()
print(light)  # YELLOW

light.next_state()
print(light)  # RED again


def test_state_machine(initial_state):
    """Test function for State Machine with Enum."""
    from enum import Enum
    
    class TrafficLightState(Enum):
        RED = 1
        YELLOW = 2
        GREEN = 3
    
    light = TrafficLight()
    # Set initial state if specified
    if initial_state == 'RED':
        light.current_state = TrafficLightState.RED
    elif initial_state == 'YELLOW':
        light.current_state = TrafficLightState.YELLOW
    elif initial_state == 'GREEN':
        light.current_state = TrafficLightState.GREEN
    
    light.next_state()
    return light.current_state.name
`,
  testCases: [
    {
      input: ['RED'],
      expected: 'GREEN',
      functionName: 'test_state_machine',
    },
  ],
  solution: `from enum import Enum

class TrafficLightState(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3


class TrafficLight:
    TRANSITIONS = {
        TrafficLightState.RED: TrafficLightState.GREEN,
        TrafficLightState.GREEN: TrafficLightState.YELLOW,
        TrafficLightState.YELLOW: TrafficLightState.RED,
    }
    
    DURATIONS = {
        TrafficLightState.RED: 30,
        TrafficLightState.YELLOW: 5,
        TrafficLightState.GREEN: 25,
    }
    
    def __init__(self):
        self.current_state = TrafficLightState.RED
    
    def next_state(self):
        self.current_state = self.TRANSITIONS[self.current_state]
        return self.current_state
    
    def get_duration(self):
        return self.DURATIONS[self.current_state]
    
    def can_transition_to(self, state):
        return self.TRANSITIONS[self.current_state] == state
    
    def force_state(self, state):
        """Force transition to any state (e.g., for maintenance)."""
        if not isinstance(state, TrafficLightState):
            raise TypeError("State must be TrafficLightState")
        self.current_state = state
    
    def __str__(self):
        return f"Traffic Light: {self.current_state.name}"


def test_state_machine(initial_state):
    """Test function for State Machine with Enum."""
    from enum import Enum
    
    class TrafficLightState(Enum):
        RED = 1
        YELLOW = 2
        GREEN = 3
    
    light = TrafficLight()
    # Set initial state if specified
    if initial_state == 'RED':
        light.current_state = TrafficLightState.RED
    elif initial_state == 'YELLOW':
        light.current_state = TrafficLightState.YELLOW
    elif initial_state == 'GREEN':
        light.current_state = TrafficLightState.GREEN
    
    light.next_state()
    return light.current_state.name`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 13,
  topic: 'Python Object-Oriented Programming',
};

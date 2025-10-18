/**
 * Facade Pattern
 * Problem ID: oop-facade-pattern
 * Order: 39
 */

import { Problem } from '../../../types';

export const facade_patternProblem: Problem = {
  id: 'oop-facade-pattern',
  title: 'Facade Pattern',
  difficulty: 'Easy',
  description: `Implement facade pattern to provide simplified interface to complex system.

**Pattern:**
- Unified interface to subsystems
- Hides complexity
- Easier to use
- Decouples client from subsystems

This tests:
- Facade pattern
- Interface simplification
- System decoupling`,
  examples: [
    {
      input: 'Complex subsystems → Simple facade',
      output: 'One method instead of many',
    },
  ],
  constraints: ['Simplify complex interface', 'Hide subsystem details'],
  hints: [
    'Facade delegates to subsystems',
    'Provides high-level operations',
    'Clients use facade only',
  ],
  starterCode: `class CPU:
    """Subsystem"""
    def freeze(self):
        return "CPU frozen"
    
    def execute(self):
        return "CPU executing"


class Memory:
    """Subsystem"""
    def load(self):
        return "Memory loaded"


class HardDrive:
    """Subsystem"""
    def read(self):
        return "Data read from disk"


class ComputerFacade:
    """Facade for complex computer startup"""
    def __init__(self):
        self.cpu = CPU()
        self.memory = Memory()
        self.hard_drive = HardDrive()
    
    def start(self):
        """Simple interface to complex operation"""
        results = []
        results.append(self.cpu.freeze())
        results.append(self.memory.load())
        results.append(self.hard_drive.read())
        results.append(self.cpu.execute())
        return " → ".join(results)


def test_facade():
    """Test facade pattern"""
    # Instead of calling multiple subsystems,
    # client uses simple facade
    computer = ComputerFacade()
    result = computer.start()
    
    return len(result)
`,
  testCases: [
    {
      input: [],
      expected: 64,
      functionName: 'test_facade',
    },
  ],
  solution: `class CPU:
    def freeze(self):
        return "CPU frozen"
    
    def execute(self):
        return "CPU executing"


class Memory:
    def load(self):
        return "Memory loaded"


class HardDrive:
    def read(self):
        return "Data read from disk"


class ComputerFacade:
    def __init__(self):
        self.cpu = CPU()
        self.memory = Memory()
        self.hard_drive = HardDrive()
    
    def start(self):
        results = []
        results.append(self.cpu.freeze())
        results.append(self.memory.load())
        results.append(self.hard_drive.read())
        results.append(self.cpu.execute())
        return " → ".join(results)


def test_facade():
    computer = ComputerFacade()
    result = computer.start()
    return len(result)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 39,
  topic: 'Python Object-Oriented Programming',
};

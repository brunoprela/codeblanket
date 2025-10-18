/**
 * Command Pattern
 * Problem ID: oop-command-pattern
 * Order: 38
 */

import { Problem } from '../../../types';

export const command_patternProblem: Problem = {
  id: 'oop-command-pattern',
  title: 'Command Pattern',
  difficulty: 'Medium',
  description: `Implement command pattern to encapsulate requests as objects.

**Pattern:**
- Request as object
- Parameterize clients
- Queue/log requests
- Support undo

This tests:
- Command pattern
- Request encapsulation
- Undo/redo support`,
  examples: [
    {
      input: 'Command objects execute and undo',
      output: 'Requests as first-class objects',
    },
  ],
  constraints: ['Encapsulate as objects', 'Support execute and undo'],
  hints: [
    'Command interface',
    'execute() and undo() methods',
    'Store state for undo',
  ],
  starterCode: `class Command:
    """Base command"""
    def execute(self):
        raise NotImplementedError
    
    def undo(self):
        raise NotImplementedError


class Light:
    """Receiver"""
    def __init__(self):
        self.is_on = False
    
    def turn_on(self):
        self.is_on = True
        return "Light on"
    
    def turn_off(self):
        self.is_on = False
        return "Light off"


class LightOnCommand(Command):
    """Concrete command"""
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        return self.light.turn_on()
    
    def undo(self):
        return self.light.turn_off()


class LightOffCommand(Command):
    """Concrete command"""
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        return self.light.turn_off()
    
    def undo(self):
        return self.light.turn_on()


class RemoteControl:
    """Invoker"""
    def __init__(self):
        self.history = []
    
    def execute_command(self, command):
        result = command.execute()
        self.history.append(command)
        return result
    
    def undo_last(self):
        if self.history:
            command = self.history.pop()
            return command.undo()


def test_command():
    """Test command pattern"""
    light = Light()
    remote = RemoteControl()
    
    # Turn on
    on_cmd = LightOnCommand(light)
    remote.execute_command(on_cmd)
    
    # Turn off
    off_cmd = LightOffCommand(light)
    remote.execute_command(off_cmd)
    
    # Undo (turns back on)
    remote.undo_last()
    
    return 1 if light.is_on else 0
`,
  testCases: [
    {
      input: [],
      expected: 1,
      functionName: 'test_command',
    },
  ],
  solution: `class Command:
    def execute(self):
        raise NotImplementedError
    
    def undo(self):
        raise NotImplementedError


class Light:
    def __init__(self):
        self.is_on = False
    
    def turn_on(self):
        self.is_on = True
        return "Light on"
    
    def turn_off(self):
        self.is_on = False
        return "Light off"


class LightOnCommand(Command):
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        return self.light.turn_on()
    
    def undo(self):
        return self.light.turn_off()


class LightOffCommand(Command):
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        return self.light.turn_off()
    
    def undo(self):
        return self.light.turn_on()


class RemoteControl:
    def __init__(self):
        self.history = []
    
    def execute_command(self, command):
        result = command.execute()
        self.history.append(command)
        return result
    
    def undo_last(self):
        if self.history:
            command = self.history.pop()
            return command.undo()


def test_command():
    light = Light()
    remote = RemoteControl()
    
    on_cmd = LightOnCommand(light)
    remote.execute_command(on_cmd)
    
    off_cmd = LightOffCommand(light)
    remote.execute_command(off_cmd)
    
    remote.undo_last()
    
    return 1 if light.is_on else 0`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(n) for history',
  order: 38,
  topic: 'Python Object-Oriented Programming',
};

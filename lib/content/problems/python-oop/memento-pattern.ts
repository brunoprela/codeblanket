/**
 * Memento Pattern
 * Problem ID: oop-memento-pattern
 * Order: 43
 */

import { Problem } from '../../../types';

export const memento_patternProblem: Problem = {
  id: 'oop-memento-pattern',
  title: 'Memento Pattern',
  difficulty: 'Hard',
  description: `Implement memento pattern for undo functionality.

**Pattern:**
- Save and restore object state
- Memento stores state
- Originator creates/restores from memento
- Caretaker manages mementos

This tests:
- Memento pattern
- State saving
- Undo/redo`,
  examples: [
    {
      input: 'Save state, modify, restore',
      output: 'Object returns to saved state',
    },
  ],
  constraints: [
    'Save state without exposing internals',
    'Restore from memento',
  ],
  hints: [
    'Memento holds state',
    'Originator creates memento',
    'Caretaker stores mementos',
  ],
  starterCode: `class Memento:
    """Stores state"""
    def __init__(self, state):
        self._state = state
    
    def get_state(self):
        return self._state


class TextEditor:
    """Originator"""
    def __init__(self):
        self._content = ""
    
    def write(self, text):
        """Modify state"""
        self._content += text
    
    def get_content(self):
        return self._content
    
    def save(self):
        """Create memento"""
        return Memento(self._content)
    
    def restore(self, memento):
        """Restore from memento"""
        self._content = memento.get_state()


class History:
    """Caretaker"""
    def __init__(self):
        self._mementos = []
    
    def save(self, memento):
        self._mementos.append(memento)
    
    def undo(self):
        if self._mementos:
            return self._mementos.pop()
        return None


def test_memento():
    """Test memento pattern"""
    editor = TextEditor()
    history = History()
    
    # Write and save
    editor.write("Hello ")
    history.save(editor.save())
    
    editor.write("World")
    history.save(editor.save())
    
    editor.write("!")
    
    # Current: "Hello World!"
    current_len = len(editor.get_content())
    
    # Undo to "Hello World"
    memento = history.undo()
    editor.restore(memento)
    
    after_undo_len = len(editor.get_content())
    
    return current_len + after_undo_len
`,
  testCases: [
    {
      input: [],
      expected: 23,
      functionName: 'test_memento',
    },
  ],
  solution: `class Memento:
    def __init__(self, state):
        self._state = state
    
    def get_state(self):
        return self._state


class TextEditor:
    def __init__(self):
        self._content = ""
    
    def write(self, text):
        self._content += text
    
    def get_content(self):
        return self._content
    
    def save(self):
        return Memento(self._content)
    
    def restore(self, memento):
        self._content = memento.get_state()


class History:
    def __init__(self):
        self._mementos = []
    
    def save(self, memento):
        self._mementos.append(memento)
    
    def undo(self):
        if self._mementos:
            return self._mementos.pop()
        return None


def test_memento():
    editor = TextEditor()
    history = History()
    
    editor.write("Hello ")
    history.save(editor.save())
    
    editor.write("World")
    history.save(editor.save())
    
    editor.write("!")
    
    current_len = len(editor.get_content())
    
    memento = history.undo()
    editor.restore(memento)
    
    after_undo_len = len(editor.get_content())
    
    return current_len + after_undo_len`,
  timeComplexity: 'O(1) for save/restore',
  spaceComplexity: 'O(n) for n saves',
  order: 43,
  topic: 'Python Object-Oriented Programming',
};

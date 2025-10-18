/**
 * Observer Pattern
 * Problem ID: oop-observer-pattern-subject
 * Order: 16
 */

import { Problem } from '../../../types';

export const observer_pattern_subjectProblem: Problem = {
  id: 'oop-observer-pattern-subject',
  title: 'Observer Pattern',
  difficulty: 'Hard',
  description: `Implement observer pattern for event notification.

**Pattern:**
- Subject maintains list of observers
- Notifies observers of changes
- Observers update themselves

**Use Case:** Event systems, MVC

This tests:
- Observer pattern
- Event notification
- Loose coupling`,
  examples: [
    {
      input: 'subject.attach(observer)',
      output: 'Observer gets notified of changes',
    },
  ],
  constraints: ['Subject notifies observers', 'Observers register themselves'],
  hints: [
    'Maintain observer list',
    'notify() calls update on each',
    'Observers implement update()',
  ],
  starterCode: `class Subject:
    """Subject being observed"""
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        """Add observer"""
        self._observers.append(observer)
    
    def detach(self, observer):
        """Remove observer"""
        self._observers.remove(observer)
    
    def notify(self):
        """Notify all observers"""
        for observer in self._observers:
            observer.update(self._state)
    
    def set_state(self, state):
        """Change state and notify"""
        self._state = state
        self.notify()


class Observer:
    """Observer base class"""
    def __init__(self, name):
        self.name = name
        self.state = None
    
    def update(self, state):
        """Receive update from subject"""
        self.state = state


def test_observer():
    """Test observer pattern"""
    subject = Subject()
    
    # Create observers
    obs1 = Observer("Observer1")
    obs2 = Observer("Observer2")
    
    # Attach observers
    subject.attach(obs1)
    subject.attach(obs2)
    
    # Change state
    subject.set_state(42)
    
    # Both observers should have new state
    return obs1.state + obs2.state
`,
  testCases: [
    {
      input: [],
      expected: 84,
      functionName: 'test_observer',
    },
  ],
  solution: `class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self._state)
    
    def set_state(self, state):
        self._state = state
        self.notify()


class Observer:
    def __init__(self, name):
        self.name = name
        self.state = None
    
    def update(self, state):
        self.state = state


def test_observer():
    subject = Subject()
    obs1 = Observer("Observer1")
    obs2 = Observer("Observer2")
    
    subject.attach(obs1)
    subject.attach(obs2)
    subject.set_state(42)
    
    return obs1.state + obs2.state`,
  timeComplexity: 'O(n) for notify',
  spaceComplexity: 'O(n) for observers',
  order: 16,
  topic: 'Python Object-Oriented Programming',
};

/**
 * Observer Pattern Implementation
 * Problem ID: oop-observer-pattern
 * Order: 5
 */

import { Problem } from '../../../types';

export const observer_patternProblem: Problem = {
  id: 'oop-observer-pattern',
  title: 'Observer Pattern Implementation',
  difficulty: 'Hard',
  description: `Implement the Observer pattern where subjects notify observers of state changes.

Create:
- \`Subject\` class that maintains list of observers
- Methods: \`attach(observer)\`, \`detach(observer)\`, \`notify()\`
- \`Observer\` interface with \`update(subject)\` method
- Concrete observer that reacts to subject changes

**Use Case:** Event systems, MVC patterns, pub-sub systems.`,
  examples: [
    {
      input: 'subject.state = 10; subject.notify()',
      output: 'All observers receive update',
    },
  ],
  constraints: [
    'Subject maintains observer list',
    'Observers implement update method',
    'Support multiple observers',
  ],
  hints: [
    'Store observers in a list',
    'Loop through observers in notify()',
    'Pass self to observer.update()',
  ],
  starterCode: `from abc import ABC, abstractmethod

class Observer(ABC):
    """Observer interface."""
    
    @abstractmethod
    def update(self, subject):
        """Called when subject changes."""
        pass


class Subject:
    """Subject that observers watch."""
    
    def __init__(self):
        self._observers = []
        self._state = None
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        self._state = value
        self.notify()
    
    def attach(self, observer):
        """Add an observer."""
        pass
    
    def detach(self, observer):
        """Remove an observer."""
        pass
    
    def notify(self):
        """Notify all observers of state change."""
        pass


class ConcreteObserver(Observer):
    """Concrete observer implementation."""
    
    def __init__(self, name):
        self.name = name
    
    def update(self, subject):
        """React to subject state change."""
        pass


# Test
subject = Subject()
observer1 = ConcreteObserver("Observer1")
observer2 = ConcreteObserver("Observer2")

subject.attach(observer1)
subject.attach(observer2)
subject.state = 10  # Should notify both observers


def test_observer_pattern(state, num_observers):
    """Test function for Observer pattern."""
    subject = Subject()
    for i in range(num_observers):
        observer = ConcreteObserver(f"Observer{i+1}")
        subject.attach(observer)
    subject.state = state
    return 'notified'
`,
  testCases: [
    {
      input: [10, 2], // state, num observers
      expected: 'notified',
      functionName: 'test_observer_pattern',
    },
  ],
  solution: `from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass


class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        self._state = value
        self.notify()
    
    def attach(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer):
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)


class ConcreteObserver(Observer):
    def __init__(self, name):
        self.name = name
    
    def update(self, subject):
        print(f"{self.name} received update: state = {subject.state}")


def test_observer_pattern(state, num_observers):
    """Test function for Observer pattern."""
    subject = Subject()
    for i in range(num_observers):
        observer = ConcreteObserver(f"Observer{i+1}")
        subject.attach(observer)
    subject.state = state
    return 'notified'`,
  timeComplexity: 'O(n) for notify where n is number of observers',
  spaceComplexity: 'O(n) to store observers',
  order: 5,
  topic: 'Python Object-Oriented Programming',
};

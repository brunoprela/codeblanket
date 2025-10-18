/**
 * Template Method Pattern
 * Problem ID: oop-template-method-processor
 * Order: 33
 */

import { Problem } from '../../../types';

export const template_method_processorProblem: Problem = {
  id: 'oop-template-method-processor',
  title: 'Template Method Pattern',
  difficulty: 'Medium',
  description: `Implement template method pattern with algorithm skeleton.

**Pattern:**
- Base class defines algorithm structure
- Subclasses fill in details
- Template method not overridden
- Hook methods can be overridden

This tests:
- Template method pattern
- Inheritance
- Algorithm structure`,
  examples: [
    {
      input: 'Base defines steps, subclass implements',
      output: 'Consistent algorithm flow',
    },
  ],
  constraints: ['Define template in base class', 'Override steps in subclass'],
  hints: [
    'Template method calls steps',
    'Steps are abstract or have defaults',
    'Subclasses override steps',
  ],
  starterCode: `from abc import ABC, abstractmethod

class DataProcessor(ABC):
    """Template for data processing"""
    def process(self):
        """Template method - defines algorithm structure"""
        data = self.read_data()
        processed = self.process_data(data)
        self.save_data(processed)
        return processed
    
    @abstractmethod
    def read_data(self):
        """Step 1: Read data"""
        pass
    
    @abstractmethod
    def process_data(self, data):
        """Step 2: Process data"""
        pass
    
    def save_data(self, data):
        """Step 3: Save (has default implementation)"""
        pass


class CSVProcessor(DataProcessor):
    """Concrete processor for CSV"""
    def __init__(self, data):
        self.data = data
    
    def read_data(self):
        """Read CSV data"""
        return self.data
    
    def process_data(self, data):
        """Process: uppercase strings"""
        return [item.upper() if isinstance(item, str) else item for item in data]


def test_template():
    """Test template method"""
    processor = CSVProcessor(["hello", "world"])
    result = processor.process()
    
    return len(result[0])
`,
  testCases: [
    {
      input: [],
      expected: 5,
      functionName: 'test_template',
    },
  ],
  solution: `from abc import ABC, abstractmethod

class DataProcessor(ABC):
    def process(self):
        data = self.read_data()
        processed = self.process_data(data)
        self.save_data(processed)
        return processed
    
    @abstractmethod
    def read_data(self):
        pass
    
    @abstractmethod
    def process_data(self, data):
        pass
    
    def save_data(self, data):
        pass


class CSVProcessor(DataProcessor):
    def __init__(self, data):
        self.data = data
    
    def read_data(self):
        return self.data
    
    def process_data(self, data):
        return [item.upper() if isinstance(item, str) else item for item in data]


def test_template():
    processor = CSVProcessor(["hello", "world"])
    result = processor.process()
    return len(result[0])`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 33,
  topic: 'Python Object-Oriented Programming',
};

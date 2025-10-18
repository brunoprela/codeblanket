/**
 * Template Method Pattern
 * Problem ID: oop-template-method
 * Order: 9
 */

import { Problem } from '../../../types';

export const template_methodProblem: Problem = {
  id: 'oop-template-method',
  title: 'Template Method Pattern',
  difficulty: 'Medium',
  description: `Implement the Template Method pattern for a data processing pipeline.

Create:
- Abstract \`DataProcessor\` class with template method \`process()\`
- Template method calls: \`load_data()\`, \`parse_data()\`, \`analyze_data()\`, \`save_results()\`
- Subclasses: \`CSVProcessor\` and \`JSONProcessor\` that implement specific steps
- Template method defines the algorithm structure

**Pattern:** Template method defines skeleton, subclasses fill in steps.`,
  examples: [
    {
      input: 'CSVProcessor().process()',
      output: 'Executes CSV-specific steps in defined order',
    },
  ],
  constraints: [
    'Template method is concrete in base class',
    'Hook methods are abstract',
    "Subclasses don't override template method",
  ],
  hints: [
    'Template method calls other methods',
    'Make step methods abstract',
    'Each processor implements steps differently',
  ],
  starterCode: `from abc import ABC, abstractmethod

class DataProcessor(ABC):
    """Template for data processing pipeline."""
    
    def process(self):
        """Template method defining the algorithm structure."""
        # Define the steps here
        pass
    
    @abstractmethod
    def load_data(self):
        """Load data from source."""
        pass
    
    @abstractmethod
    def parse_data(self, raw_data):
        """Parse raw data."""
        pass
    
    @abstractmethod
    def analyze_data(self, parsed_data):
        """Analyze parsed data."""
        pass
    
    @abstractmethod
    def save_results(self, results):
        """Save analysis results."""
        pass


class CSVProcessor(DataProcessor):
    """Process CSV data."""
    
    def load_data(self):
        # Simulate loading CSV
        pass
    
    def parse_data(self, raw_data):
        # Parse CSV format
        pass
    
    def analyze_data(self, parsed_data):
        # Analyze data
        pass
    
    def save_results(self, results):
        # Save results
        pass


class JSONProcessor(DataProcessor):
    """Process JSON data."""
    
    def load_data(self):
        pass
    
    def parse_data(self, raw_data):
        pass
    
    def analyze_data(self, parsed_data):
        pass
    
    def save_results(self, results):
        pass


# Test
csv_processor = CSVProcessor()
csv_processor.process()


def test_template_method(processor_type):
    """Test function for Template Method pattern."""
    if processor_type == 'CSV':
        processor = CSVProcessor()
    elif processor_type == 'JSON':
        processor = JSONProcessor()
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")
    processor.process()
    return 'processed'
`,
  testCases: [
    {
      input: ['CSV'],
      expected: 'processed',
      functionName: 'test_template_method',
    },
  ],
  solution: `from abc import ABC, abstractmethod

class DataProcessor(ABC):
    def process(self):
        """Template method defining the algorithm structure."""
        print("Starting processing pipeline...")
        raw_data = self.load_data()
        parsed_data = self.parse_data(raw_data)
        results = self.analyze_data(parsed_data)
        self.save_results(results)
        print("Processing complete!")
        return results
    
    @abstractmethod
    def load_data(self):
        pass
    
    @abstractmethod
    def parse_data(self, raw_data):
        pass
    
    @abstractmethod
    def analyze_data(self, parsed_data):
        pass
    
    @abstractmethod
    def save_results(self, results):
        pass


class CSVProcessor(DataProcessor):
    def load_data(self):
        print("Loading CSV data...")
        return "name,age\\nAlice,30\\nBob,25"
    
    def parse_data(self, raw_data):
        print("Parsing CSV...")
        lines = raw_data.split('\\n')
        header = lines[0].split(',')
        data = [dict(zip(header, line.split(','))) for line in lines[1:]]
        return data
    
    def analyze_data(self, parsed_data):
        print("Analyzing CSV data...")
        avg_age = sum(int(row['age']) for row in parsed_data) / len(parsed_data)
        return {'average_age': avg_age}
    
    def save_results(self, results):
        print(f"Saving results: {results}")


class JSONProcessor(DataProcessor):
    def load_data(self):
        print("Loading JSON data...")
        return '{"users": [{"name": "Alice", "age": 30}]}'
    
    def parse_data(self, raw_data):
        print("Parsing JSON...")
        import json
        return json.loads(raw_data)
    
    def analyze_data(self, parsed_data):
        print("Analyzing JSON data...")
        users = parsed_data['users']
        return {'user_count': len(users)}
    
    def save_results(self, results):
        print(f"Saving results: {results}")


def test_template_method(processor_type):
    """Test function for Template Method pattern."""
    if processor_type == 'CSV':
        processor = CSVProcessor()
    elif processor_type == 'JSON':
        processor = JSONProcessor()
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")
    processor.process()
    return 'processed'`,
  timeComplexity: 'O(n) where n is data size',
  spaceComplexity: 'O(n)',
  order: 9,
  topic: 'Python Object-Oriented Programming',
};

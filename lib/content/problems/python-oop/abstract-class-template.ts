/**
 * Abstract Class with Template Methods
 * Problem ID: oop-abstract-class-template
 * Order: 46
 */

import { Problem } from '../../../types';

export const abstract_class_templateProblem: Problem = {
  id: 'oop-abstract-class-template',
  title: 'Abstract Class with Template Methods',
  difficulty: 'Medium',
  description: `Combine abstract base class with template method pattern.

**Pattern:**
- Abstract methods force implementation
- Template method defines workflow
- Best of both patterns

This tests:
- ABC with template method
- Abstract + concrete methods
- Design patterns combination`,
  examples: [
    {
      input: 'Abstract methods + template workflow',
      output: 'Enforced implementation with structure',
    },
  ],
  constraints: ['Use ABC', 'Template method calls abstract methods'],
  hints: [
    'ABC for abstract methods',
    'Template method for workflow',
    'Subclass implements abstracts',
  ],
  starterCode: `from abc import ABC, abstractmethod

class DataImporter(ABC):
    """Abstract importer with template"""
    def import_data(self, source):
        """Template method"""
        # Step 1
        data = self.read_source(source)
        
        # Step 2
        validated = self.validate(data)
        
        # Step 3
        transformed = self.transform(validated)
        
        # Step 4
        self.save(transformed)
        
        return len(transformed)
    
    @abstractmethod
    def read_source(self, source):
        """Must implement: read data"""
        pass
    
    def validate(self, data):
        """Optional hook: validate data"""
        return data
    
    @abstractmethod
    def transform(self, data):
        """Must implement: transform data"""
        pass
    
    def save(self, data):
        """Optional hook: save data"""
        pass


class CSVImporter(DataImporter):
    """Concrete importer"""
    def read_source(self, source):
        """Read CSV"""
        return source.split(',')
    
    def transform(self, data):
        """Transform: uppercase"""
        return [item.upper() for item in data]


def test_abstract_template():
    """Test abstract template pattern"""
    importer = CSVImporter()
    result = importer.import_data("apple,banana,cherry")
    
    return result
`,
  testCases: [
    {
      input: [],
      expected: 3,
      functionName: 'test_abstract_template',
    },
  ],
  solution: `from abc import ABC, abstractmethod

class DataImporter(ABC):
    def import_data(self, source):
        data = self.read_source(source)
        validated = self.validate(data)
        transformed = self.transform(validated)
        self.save(transformed)
        return len(transformed)
    
    @abstractmethod
    def read_source(self, source):
        pass
    
    def validate(self, data):
        return data
    
    @abstractmethod
    def transform(self, data):
        pass
    
    def save(self, data):
        pass


class CSVImporter(DataImporter):
    def read_source(self, source):
        return source.split(',')
    
    def transform(self, data):
        return [item.upper() for item in data]


def test_abstract_template():
    importer = CSVImporter()
    result = importer.import_data("apple,banana,cherry")
    return result`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 46,
  topic: 'Python Object-Oriented Programming',
};

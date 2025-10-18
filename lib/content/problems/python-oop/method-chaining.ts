/**
 * Method Chaining (Fluent Interface)
 * Problem ID: oop-method-chaining
 * Order: 22
 */

import { Problem } from '../../../types';

export const method_chainingProblem: Problem = {
  id: 'oop-method-chaining',
  title: 'Method Chaining (Fluent Interface)',
  difficulty: 'Easy',
  description: `Implement method chaining by returning self.

**Pattern:**
\`\`\`python
obj.method1().method2().method3()
\`\`\`

Each method returns self to enable chaining.

This tests:
- Fluent interface
- Method design
- Return self pattern`,
  examples: [
    {
      input: 'builder.set_x(1).set_y(2).build()',
      output: 'Chained method calls',
    },
  ],
  constraints: ['Return self from methods', 'Enable chaining'],
  hints: [
    'Return self from each method',
    'Allows chaining',
    'Common in builders',
  ],
  starterCode: `class QueryBuilder:
    """SQL query builder with chaining"""
    def __init__(self):
        self._select = []
        self._where = []
        self._limit = None
    
    def select(self, *fields):
        """Add fields to SELECT"""
        self._select.extend(fields)
        return self
    
    def where(self, condition):
        """Add WHERE condition"""
        self._where.append(condition)
        return self
    
    def limit(self, n):
        """Set LIMIT"""
        self._limit = n
        return self
    
    def build(self):
        """Build query string"""
        query = f"SELECT {', '.join(self._select)}"
        if self._where:
            query += f" WHERE {' AND '.join(self._where)}"
        if self._limit:
            query += f" LIMIT {self._limit}"
        return query


def test_chaining():
    """Test method chaining"""
    query = (QueryBuilder()
             .select('name', 'age')
             .where('age > 18')
             .where('city = "NYC"')
             .limit(10)
             .build())
    
    return len(query)
`,
  testCases: [
    {
      input: [],
      expected: 67,
      functionName: 'test_chaining',
    },
  ],
  solution: `class QueryBuilder:
    def __init__(self):
        self._select = []
        self._where = []
        self._limit = None
    
    def select(self, *fields):
        self._select.extend(fields)
        return self
    
    def where(self, condition):
        self._where.append(condition)
        return self
    
    def limit(self, n):
        self._limit = n
        return self
    
    def build(self):
        query = f"SELECT {', '.join(self._select)}"
        if self._where:
            query += f" WHERE {' AND '.join(self._where)}"
        if self._limit:
            query += f" LIMIT {self._limit}"
        return query


def test_chaining():
    query = (QueryBuilder()
             .select('name', 'age')
             .where('age > 18')
             .where('city = "NYC"')
             .limit(10)
             .build())
    
    return len(query)`,
  timeComplexity: 'O(1) per method',
  spaceComplexity: 'O(n) for query parts',
  order: 22,
  topic: 'Python Object-Oriented Programming',
};

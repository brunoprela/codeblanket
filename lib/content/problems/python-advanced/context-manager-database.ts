/**
 * Database Transaction Context Manager
 * Problem ID: context-manager-database
 * Order: 7
 */

import { Problem } from '../../../types';

export const context_manager_databaseProblem: Problem = {
  id: 'context-manager-database',
  title: 'Database Transaction Context Manager',
  difficulty: 'Medium',
  description: `Create a context manager that simulates database transaction management.

The context manager should:
- Begin transaction on entry
- Commit if no exception occurs
- Rollback if exception occurs
- Close connection in both cases

**Pattern:**
python
with Transaction(db):
    db.execute("INSERT ...")
    db.execute("UPDATE ...")
# Commits if successful, rolls back if error
`,
  examples: [
    {
      input: 'Successful operations',
      output: 'Commit called',
    },
    {
      input: 'Operation raises exception',
      output: 'Rollback called',
    },
  ],
  constraints: [
    'Commit only if no exception',
    'Always rollback on exception',
    'Connection must close regardless',
  ],
  hints: [
    'Check exc_type in __exit__',
    'exc_type is None if no exception',
    'Use try/finally for cleanup',
  ],
  starterCode: `class Transaction:
    """
    Context manager for database transactions.
    """
    
    def __init__(self, connection):
        self.conn = connection
    
    def __enter__(self):
        # Your code here
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Your code here
        pass


# Mock database connection
class MockDB:
    def begin(self):
        print("Transaction started")
    
    def commit(self):
        print("Transaction committed")
    
    def rollback(self):
        print("Transaction rolled back")
    
    def close(self):
        print("Connection closed")

db = MockDB()
with Transaction(db):
    print("Doing work...")


# Test helper function (for automated testing)
def test_transaction(should_succeed):
    """Test function for Transaction - implement the class methods above first!"""
    try:
        mock_db = MockDB()
        status = []
        
        # Override methods to capture what was called
        original_commit = mock_db.commit
        original_rollback = mock_db.rollback
        
        def capture_commit():
            status.append('committed')
            original_commit()
        
        def capture_rollback():
            status.append('rolled back')
            original_rollback()
        
        mock_db.commit = capture_commit
        mock_db.rollback = capture_rollback
        
        try:
            with Transaction(mock_db):
                if not should_succeed:
                    raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected for failure case
        
        return status[0] if status else None
    except:
        return None  # Return None if methods not yet implemented
`,
  testCases: [
    {
      input: [true], // success case
      expected: 'committed',
      functionName: 'test_transaction',
    },
    {
      input: [false], // error case
      expected: 'rolled back',
      functionName: 'test_transaction',
    },
  ],
  solution: `class Transaction:
    def __init__(self, connection):
        self.conn = connection
    
    def __enter__(self):
        self.conn.begin()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.conn.close()
        return False  # Don't suppress exceptions


class MockDB:
    def begin(self):
        print("Transaction started")
    
    def commit(self):
        print("Transaction committed")
    
    def rollback(self):
        print("Transaction rolled back")
    
    def close(self):
        print("Connection closed")


# Test helper function (for automated testing)
def test_transaction(should_succeed):
    """Test function for Transaction."""
    mock_db = MockDB()
    status = []
    
    original_commit = mock_db.commit
    original_rollback = mock_db.rollback
    
    def capture_commit():
        status.append('committed')
        original_commit()
    
    def capture_rollback():
        status.append('rolled back')
        original_rollback()
    
    mock_db.commit = capture_commit
    mock_db.rollback = capture_rollback
    
    try:
        with Transaction(mock_db):
            if not should_succeed:
                raise ValueError("Simulated error")
    except ValueError:
        pass
    
    return status[0] if status else None`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 7,
  topic: 'Python Advanced',
};

/**
 * SQLite Database Manager
 * Problem ID: intermediate-sqlite-manager
 * Order: 16
 */

import { Problem } from '../../../types';

export const intermediate_sqlite_managerProblem: Problem = {
  id: 'intermediate-sqlite-manager',
  title: 'SQLite Database Manager',
  difficulty: 'Hard',
  description: `Create a simple database manager for SQLite operations.

**Features:**
- Create tables
- Insert, update, delete records
- Query with filters
- Use context manager for connections
- Handle transactions

**Example:**
\`\`\`python
with DatabaseManager('users.db') as db:
    db.create_table('users', ['id INTEGER PRIMARY KEY', 'name TEXT', 'age INTEGER'])
    db.insert('users', {'name': 'Alice', 'age': 30})
    users = db.query('users', where={'age': 30})
\`\`\``,
  examples: [
    {
      input: "db.insert('users', {'name': 'Bob', 'age': 25})",
      output: 'Inserts record into database',
    },
  ],
  constraints: [
    'Use sqlite3 module',
    'Implement context manager',
    'Handle SQL injection safely',
  ],
  hints: [
    'Use parameterized queries (? placeholders)',
    'Implement __enter__ and __exit__',
    'Commit transactions in __exit__',
  ],
  starterCode: `import sqlite3

class DatabaseManager:
    """
    Simple SQLite database manager with context manager support.
    
    Examples:
        >>> with DatabaseManager('test.db') as db:
        ...     db.create_table('users', ['id INTEGER PRIMARY KEY', 'name TEXT'])
        ...     db.insert('users', {'name': 'Alice'})
    """
    
    def __init__(self, db_name):
        """
        Initialize database manager.
        
        Args:
            db_name: Name of database file
        """
        # TODO: Store db_name, initialize connection and cursor to None
        self.db_name = db_name
        self.connection = None
        self.cursor = None
    
    def __enter__(self):
        """Enter context - open connection."""
        # TODO: Open database connection, set row_factory, create cursor
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - commit and close connection."""
        # TODO: Commit if no exceptions, close connection
        pass
    
    def create_table(self, table_name, columns):
        """
        Create table if not exists.
        
        Args:
            table_name: Name of table
            columns: List of column definitions
            
        Examples:
            >>> db.create_table('users', ['id INTEGER PRIMARY KEY', 'name TEXT'])
        """
        # TODO: Execute CREATE TABLE IF NOT EXISTS
        pass
    
    def insert(self, table_name, data):
        """
        Insert record into table.
        
        Args:
            table_name: Name of table
            data: Dictionary of column: value pairs
            
        Returns:
            ID of inserted row
        """
        # TODO: Execute INSERT and return lastrowid
        pass
    
    def query(self, table_name, columns='*', where=None, order_by=None):
        """
        Query records from table.
        
        Args:
            table_name: Name of table
            columns: Columns to select (default all)
            where: Dictionary of conditions
            order_by: Column to order by
            
        Returns:
            List of records as dictionaries
            
        Examples:
            >>> db.query('users', where={'age': 30})
            [{'id': 1, 'name': 'Alice', 'age': 30}]
        """
        # TODO: Build and execute SELECT query
        pass
    
    def update(self, table_name, data, where):
        """
        Update records in table.
        
        Args:
            table_name: Name of table
            data: Dictionary of columns to update
            where: Dictionary of conditions
            
        Returns:
            Number of rows updated
        """
        # TODO: Build and execute UPDATE query
        pass
    
    def delete(self, table_name, where):
        """
        Delete records from table.
        
        Args:
            table_name: Name of table
            where: Dictionary of conditions
            
        Returns:
            Number of rows deleted
        """
        # TODO: Build and execute DELETE query
        pass


# Test
with DatabaseManager('test_users.db') as db:
    # Create table
    db.create_table('users', [
        'id INTEGER PRIMARY KEY AUTOINCREMENT',
        'name TEXT NOT NULL',
        'age INTEGER',
        'email TEXT'
    ])
    
    # Insert records
    db.insert('users', {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'})
    db.insert('users', {'name': 'Bob', 'age': 25, 'email': 'bob@example.com'})
    db.insert('users', {'name': 'Charlie', 'age': 30, 'email': 'charlie@example.com'})
    
    # Query all
    print("All users:")
    for user in db.query('users'):
        print(f"  {user}")
    
    # Query with filter
    print("\\nUsers aged 30:")
    for user in db.query('users', where={'age': 30}):
        print(f"  {user}")
    
    # Update
    updated = db.update('users', {'age': 26}, where={'name': 'Bob'})
    print(f"\\nUpdated {updated} record(s)")
    
    # Delete
    deleted = db.delete('users', where={'name': 'Charlie'})
    print(f"Deleted {deleted} record(s)")
    
    # Final query
    print("\\nFinal users:")
    for user in db.query('users', order_by='name'):
        print(f"  {user}")


# Test helper function (for automated testing)
def test_database_manager(table_name, data):
    """Test function for DatabaseManager - implement the class methods above first!"""
    try:
        with DatabaseManager('test.db') as db:
            # Create test table
            db.create_table(table_name, [
                'id INTEGER PRIMARY KEY AUTOINCREMENT',
                'name TEXT',
                'age INTEGER'
            ])
            # Insert and return the row id
            return db.insert(table_name, data)
    except:
        return None  # Return None if methods not yet implemented
`,
  testCases: [
    {
      input: ['users', { name: 'Alice', age: 30 }],
      expected: 1,
      functionName: 'test_database_manager',
    },
  ],
  solution: `import sqlite3

class DatabaseManager:
    def __init__(self, db_name):
        self.db_name = db_name
        self.connection = None
        self.cursor = None
    
    def __enter__(self):
        self.connection = sqlite3.connect(self.db_name)
        self.connection.row_factory = sqlite3.Row  # Access columns by name
        self.cursor = self.connection.cursor()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()
        return False
    
    def create_table(self, table_name, columns):
        columns_str = ', '.join(columns)
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})"
        self.cursor.execute(sql)
    
    def insert(self, table_name, data):
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        self.cursor.execute(sql, tuple(data.values()))
        return self.cursor.lastrowid
    
    def query(self, table_name, columns='*', where=None, order_by=None):
        sql = f"SELECT {columns} FROM {table_name}"
        params = []
        
        if where:
            conditions = ' AND '.join([f"{k} = ?" for k in where.keys()])
            sql += f" WHERE {conditions}"
            params.extend(where.values())
        
        if order_by:
            sql += f" ORDER BY {order_by}"
        
        self.cursor.execute(sql, params)
        return [dict(row) for row in self.cursor.fetchall()]
    
    def update(self, table_name, data, where):
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        where_clause = ' AND '.join([f"{k} = ?" for k in where.keys()])
        sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
        params = list(data.values()) + list(where.values())
        self.cursor.execute(sql, params)
        return self.cursor.rowcount
    
    def delete(self, table_name, where):
        where_clause = ' AND '.join([f"{k} = ?" for k in where.keys()])
        sql = f"DELETE FROM {table_name} WHERE {where_clause}"
        self.cursor.execute(sql, tuple(where.values()))
        return self.cursor.rowcount


# Test helper function (for automated testing)
def test_database_manager(table_name, data):
    """Test function for DatabaseManager."""
    with DatabaseManager('test.db') as db:
        db.create_table(table_name, [
            'id INTEGER PRIMARY KEY AUTOINCREMENT',
            'name TEXT',
            'age INTEGER'
        ])
        return db.insert(table_name, data)`,
  timeComplexity: 'O(n) for queries, O(1) for indexed operations',
  spaceComplexity: 'O(r) where r is number of results',
  order: 16,
  topic: 'Python Intermediate',
};

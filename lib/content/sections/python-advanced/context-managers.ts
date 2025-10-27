/**
 * Context Managers & Resource Management Section
 */

export const contextmanagersSection = {
  id: 'context-managers',
  title: 'Context Managers & Resource Management',
  content: `**What are Context Managers?**
Context managers handle resource setup and cleanup automatically using the with statement.

**Basic Pattern:**
\`\`\`python
class MyContextManager:
    def __enter__(self):
        # Setup code
        print("Entering context")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup code (always runs)
        print("Exiting context")
        return False  # Don't suppress exceptions

with MyContextManager() as manager:
    print("Inside context")
\`\`\`

**Common Use Cases:**1. **File Handling:**
\`\`\`python
# Without context manager (error-prone)
f = open('file.txt')
try:
    data = f.read()
finally:
    f.close()  # Must remember to close

# With context manager (automatic cleanup)
with open('file.txt') as f:
    data = f.read()
# File automatically closed, even if exception occurs
\`\`\`

2. **Database Connections:**
\`\`\`python
class DatabaseConnection:
    def __enter__(self):
        self.conn = database.connect()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()

with DatabaseConnection() as conn:
    conn.execute("INSERT ...")
\`\`\`

3. **Locks and Threading:**
\`\`\`python
import threading

lock = threading.Lock()

with lock:
    # Critical section
    # Lock automatically released
    pass
\`\`\`

**Using contextlib:**
\`\`\`python
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"Time: {end - start:.2f}s")

with timer():
    # Code to time
    time.sleep(1)
\`\`\`

**Exception Handling:**
\`\`\`python
class FileHandler:
    def __enter__(self):
        self.file = open('data.txt', 'w')
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if exc_type is IOError:
            print("IO Error occurred")
            return True  # Suppress the exception
        return False  # Don't suppress
\`\`\`

**Multiple Context Managers:**
\`\`\`python
# Python 3.10+
with (
    open('input.txt') as infile,
    open('output.txt', 'w') as outfile
):
    outfile.write (infile.read())
\`\`\`

**Best Practices:**
- Always use context managers for resources that need cleanup
- __exit__ is always called, even with exceptions
- Return True from __exit__ to suppress exceptions (use carefully)
- Use contextlib for simpler context managers`,
};

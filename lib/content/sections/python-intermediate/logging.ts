/**
 * Logging and Debugging Section
 */

export const loggingSection = {
  id: 'logging',
  title: 'Logging and Debugging',
  content: `# Logging and Debugging

## Why Use Logging?

Logging is better than \`print()\` for production code because:
- **Levels**: Control verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Formatting**: Timestamps, filenames, line numbers automatically
- **Output**: Can log to files, console, network, etc.
- **Performance**: Can be turned off without code changes
- **Thread-safe**: Safe for multi-threaded applications

## Basic Logging

\`\`\`python
import logging

# Basic configuration (do this once at start)
logging.basicConfig(level=logging.DEBUG)

# Log at different levels
logging.debug("Detailed information for debugging")
logging.info("General informational messages")
logging.warning("Warning messages")
logging.error("Error messages")
logging.critical("Critical errors that may cause termination")

# By default, only WARNING and above are shown
\`\`\`

## Configuring Logging

\`\`\`python
import logging

# Configure format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
logger.info("This is an info message")
# Output: 2024-01-15 10:30:45 - __main__ - INFO - This is an info message
\`\`\`

## Logging to Files

\`\`\`python
import logging

# Log to file
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("This goes to the file")
logging.error("So does this error")

# Append mode (default) vs write mode
logging.basicConfig(filename='app.log', filemode='w')  # Overwrites
\`\`\`

## Multiple Handlers

\`\`\`python
import logging

# Create logger
logger = logging.getLogger('my_app')
logger.setLevel(logging.DEBUG)

# Console handler (INFO and above)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# File handler (DEBUG and above)
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Log messages
logger.debug("Detailed debug info")  # Only in file
logger.info("Important info")       # Console and file
logger.error("An error occurred")   # Console and file
\`\`\`

## Logging with Variables

\`\`\`python
import logging

logger = logging.getLogger(__name__)

# Old way (string formatting happens even if not logged)
logging.debug("User: " + username + ", Age: " + str(age))

# Better way (lazy evaluation)
logging.debug("User: %s, Age: %d", username, age)

# f-strings work but aren't lazy
logging.debug(f"User: {username}, Age: {age}")

# Extra context with exc_info
try:
    result = 10 / 0
except Exception as e:
    logging.error("Division failed", exc_info=True)  # Includes traceback
    # Or use logging.exception() which does this automatically
    logging.exception("Division failed")
\`\`\`

## Logging in Modules

\`\`\`python
# my_module.py
import logging

# Use __name__ so log shows which module logged it
logger = logging.getLogger(__name__)

def process_data(data):
    logger.info("Processing %d items", len(data))
    try:
        # Process data
        result = do_something(data)
        logger.debug("Result: %s", result)
        return result
    except Exception:
        logger.exception("Failed to process data")
        raise

# main.py
import logging
import my_module

logging.basicConfig(level=logging.INFO)
my_module.process_data([1, 2, 3])
# Output: my_module - INFO - Processing 3 items
\`\`\`

## Rotating Log Files

\`\`\`python
import logging
from logging.handlers import RotatingFileHandler

# Create logger
logger = logging.getLogger('my_app')
logger.setLevel(logging.DEBUG)

# Rotating file handler (5 MB per file, keep 3 backups)
handler = RotatingFileHandler(
    'app.log',
    maxBytes=5*1024*1024,  # 5 MB
    backupCount=3
)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Files: app.log, app.log.1, app.log.2, app.log.3
\`\`\`

## Time-Based Rotation

\`\`\`python
import logging
from logging.handlers import TimedRotatingFileHandler

# Rotate daily at midnight, keep 7 days
handler = TimedRotatingFileHandler(
    'app.log',
    when='midnight',
    interval=1,
    backupCount=7
)

# Other options for 'when':
# 'S' - Seconds
# 'M' - Minutes
# 'H' - Hours
# 'D' - Days
# 'midnight' - Roll over at midnight
# 'W0'-'W6' - Weekday (0=Monday)
\`\`\`

## Structured Logging

\`\`\`python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

# Use JSON formatter
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger('app')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info("User login", extra={'user_id': 123, 'ip': '192.168.1.1'})
# Output: {"timestamp": "2024-01-15 10:30:00", "level": "INFO", ...}
\`\`\`

## Logging Best Practices

**1. Choose the Right Level:**
- **DEBUG**: Detailed diagnostic info (algorithm steps, variable values)
- **INFO**: Confirmation things are working (server started, job completed)
- **WARNING**: Something unexpected but not breaking (deprecated feature, fallback used)
- **ERROR**: Serious problem, function failed
- **CRITICAL**: Program may crash, data corruption

**2. Use Lazy Evaluation:**
\`\`\`python
# Good - string only built if logged
logging.debug("Processing %s with %d items", name, len(items))

# Bad - string always built
logging.debug(f"Processing {name} with {len(items)} items")
\`\`\`

**3. Log Exceptions Properly:**
\`\`\`python
try:
    risky_operation()
except Exception:
    logging.exception("Operation failed")  # Includes traceback
\`\`\`

**4. Don't Log Sensitive Data:**
\`\`\`python
# Bad
logging.info(f"User logged in: {username}, password: {password}")

# Good
logging.info(f"User logged in: {username}")
\`\`\`

**5. Use Module-Level Loggers:**
\`\`\`python
# Good - shows which module logged
logger = logging.getLogger(__name__)

# Bad - all logs show same name
logger = logging.getLogger('app')
\`\`\`

## Common Patterns

**Configuration File:**
\`\`\`python
import logging
import logging.config

# logging.ini
''
[loggers]
keys=root

[handlers]
keys=console,file

[formatters]
keys=simple,detailed

[logger_root]
level=DEBUG
handlers=console,file

[handler_console]
class=StreamHandler
level=INFO
formatter=simple
args=(sys.stdout,)

[handler_file]
class=FileHandler
level=DEBUG
formatter=detailed
args=('app.log', 'a')

[formatter_simple]
format=%(levelname)s - %(message)s

[formatter_detailed]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
''

# Load configuration
logging.config.fileConfig('logging.ini')
logger = logging.getLogger()
\`\`\`

**Testing/Development vs Production:**
\`\`\`python
import logging
import os

# Set level based on environment
level = logging.DEBUG if os.getenv('ENV') == 'development' else logging.INFO

logging.basicConfig(
    level=level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
\`\`\``,
};

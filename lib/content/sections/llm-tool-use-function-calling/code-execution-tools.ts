export const codeExecutionTools = {
  title: 'Code Execution Tools',
  id: 'code-execution-tools',
  description:
    'Implement safe code execution tools like ChatGPT Code Interpreter, with sandboxing, resource limits, and security measures.',
  content: `

# Code Execution Tools

## Introduction

Code execution tools enable LLMs to write and run code dynamically - one of the most powerful capabilities in modern AI systems. ChatGPT's Code Interpreter, Cursor\'s terminal execution, and similar tools allow LLMs to perform calculations, analyze data, generate visualizations, and much more.

However, executing arbitrary code is inherently dangerous. We need robust sandboxing, resource limits, timeouts, and security measures to prevent malicious or accidental damage. In this section, we'll learn how to build production-grade code execution tools safely.

## The Security Challenge

Running user-generated or LLM-generated code poses serious risks:

**Security Risks:**
- File system access (reading/writing sensitive files)
- Network access (data exfiltration, DDoS attacks)
- System calls (shutting down the system, killing processes)
- Resource exhaustion (infinite loops, memory bombs)
- Privilege escalation
- Code injection

**Our Goals:**
1. Isolate code execution from the host system
2. Limit resources (CPU, memory, disk, time)
3. Restrict dangerous operations
4. Monitor execution
5. Capture outputs safely
6. Handle errors gracefully

## Sandboxing Approaches

### 1. Docker Containers (Recommended)

Most robust approach - complete isolation:

\`\`\`python
import docker
import tempfile
import os
import json
from typing import Dict, Any, Optional

class DockerCodeExecutor:
    """
    Execute code in isolated Docker containers.
    """
    def __init__(self, image: str = "python:3.11-slim"):
        self.client = docker.from_env()
        self.image = image
        self._ensure_image()
    
    def _ensure_image (self):
        """Ensure Docker image is available."""
        try:
            self.client.images.get (self.image)
        except docker.errors.ImageNotFound:
            print(f"Pulling image {self.image}...")
            self.client.images.pull (self.image)
    
    def execute_python (self, 
                      code: str, 
                      timeout: int = 30,
                      memory_limit: str = "128m",
                      cpu_quota: int = 50000) -> Dict[str, Any]:
        """
        Execute Python code in a Docker container.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            memory_limit: Maximum memory (e.g., "128m", "512m")
            cpu_quota: CPU quota (100000 = 1 CPU core)
        
        Returns:
            Dictionary with stdout, stderr, exit_code, and any errors
        """
        # Create temporary file with code
        with tempfile.NamedTemporaryFile (mode='w', suffix='.py', delete=False) as f:
            f.write (code)
            code_file = f.name
        
        try:
            # Run container
            container = self.client.containers.run(
                image=self.image,
                command=["python", "/code/script.py"],
                volumes={
                    code_file: {'bind': '/code/script.py', 'mode': 'ro'}
                },
                mem_limit=memory_limit,
                cpu_quota=cpu_quota,
                network_disabled=True,  # No network access
                remove=True,            # Auto-remove after execution
                detach=True
            )
            
            # Wait for completion with timeout
            try:
                result = container.wait (timeout=timeout)
                logs = container.logs()
                
                return {
                    "success": True,
                    "stdout": logs.decode('utf-8', errors='replace'),
                    "stderr": "",
                    "exit_code": result['StatusCode']
                }
            
            except Exception as e:
                # Timeout or other error
                try:
                    container.kill()
                except:
                    pass
                
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Execution error: {str (e)}",
                    "exit_code": -1
                }
        
        finally:
            # Clean up temporary file
            try:
                os.unlink (code_file)
            except:
                pass

# Usage
executor = DockerCodeExecutor()

result = executor.execute_python("""
import math

def calculate_pi (n):
    pi = 0
    for i in range (n):
        pi += ((-1)**i) / (2*i + 1)
    return 4 * pi

print(f"Pi approximation: {calculate_pi(10000)}")
print(f"Actual pi: {math.pi}")
""", timeout=10, memory_limit="64m")

print(result["stdout"])
\`\`\`

### 2. RestrictedPython (Lightweight)

For simpler cases without full Docker:

\`\`\`python
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safe_builtins
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

class RestrictedPythonExecutor:
    """
    Execute Python code with restricted capabilities.
    """
    def __init__(self):
        # Build safe globals
        self.safe_globals = {
            '__builtins__': safe_builtins,
            '_getitem_': default_guarded_getitem,
            '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
            '__name__': 'restricted_module',
            '__metaclass__': type,
        }
        
        # Add safe modules
        self.safe_globals['math'] = __import__('math')
        self.safe_globals['json'] = __import__('json')
    
    def execute (self, code: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Execute code with restrictions.
        """
        # Compile with restrictions
        byte_code = compile_restricted (code, '<string>', 'exec')
        
        if byte_code.errors:
            return {
                "success": False,
                "stdout": "",
                "stderr": "\\n".join (byte_code.errors),
                "exit_code": 1
            }
        
        # Capture output
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        try:
            with redirect_stdout (stdout), redirect_stderr (stderr):
                # Execute with timeout
                import signal
                
                def timeout_handler (signum, frame):
                    raise TimeoutError("Execution timeout")
                
                signal.signal (signal.SIGALRM, timeout_handler)
                signal.alarm (timeout)
                
                try:
                    exec (byte_code.code, self.safe_globals)
                finally:
                    signal.alarm(0)
            
            return {
                "success": True,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "exit_code": 0
            }
        
        except TimeoutError:
            return {
                "success": False,
                "stdout": stdout.getvalue(),
                "stderr": "Execution timed out",
                "exit_code": -1
            }
        
        except Exception as e:
            return {
                "success": False,
                "stdout": stdout.getvalue(),
                "stderr": f"Error: {str (e)}",
                "exit_code": 1
            }

# Usage
executor = RestrictedPythonExecutor()

result = executor.execute("""
import math

result = math.sqrt(16) + math.pi
print(f"Result: {result}")
""")

print(result["stdout"])
\`\`\`

### 3. PyPy Sandbox (Historical Reference)

PyPy had a sandboxing mode, but it's deprecated. Mentioned for completeness.

## Tool Implementation: Python Code Interpreter

Complete tool for LLM code execution:

\`\`\`python
from tools import tool, ToolCategory
import logging

logger = logging.getLogger(__name__)

class CodeInterpreter:
    """
    Code interpreter similar to ChatGPT Code Interpreter.
    """
    def __init__(self):
        self.executor = DockerCodeExecutor()
        self.execution_count = 0
    
    def execute_python (self, code: str) -> Dict[str, Any]:
        """
        Execute Python code safely.
        
        Security measures:
        - Docker isolation
        - No network access
        - Memory limit: 256MB
        - CPU limit: 50% of one core
        - Timeout: 30 seconds
        """
        self.execution_count += 1
        logger.info (f"Executing code (execution #{self.execution_count})")
        
        # Execute in Docker
        result = self.executor.execute_python(
            code=code,
            timeout=30,
            memory_limit="256m",
            cpu_quota=50000
        )
        
        # Log result
        if result["success"]:
            logger.info (f"Execution successful")
        else:
            logger.warning (f"Execution failed: {result['stderr']}")
        
        return result

code_interpreter = CodeInterpreter()

@tool(
    description="""Execute Python code in a secure sandbox environment.
    
    Capabilities:
    - Mathematical calculations
    - Data analysis (pandas, numpy available)
    - Plotting (matplotlib available)
    - JSON/CSV processing
    
    Limitations:
    - No network access
    - No file system access (except /tmp)
    - 256MB memory limit
    - 30 second timeout
    
    Use this when you need to:
    - Perform calculations
    - Analyze data
    - Generate visualizations
    - Process structured data
    """,
    category=ToolCategory.COMPUTATION,
    requires_auth=True
)
def execute_python_code (code: str) -> dict:
    """
    Execute Python code safely in a sandbox.
    
    Args:
        code: Python code to execute
    
    Returns:
        Execution results including stdout, stderr, and exit code
    """
    result = code_interpreter.execute_python (code)
    
    return {
        "success": result["success"],
        "output": result["stdout"],
        "error": result["stderr"] if result["stderr"] else None,
        "exit_code": result["exit_code"]
    }

# Example usage with LLM
user_message = "Calculate the fibonacci sequence up to the 20th number"

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": user_message}],
    functions=[{
        "name": "execute_python_code",
        "description": execute_python_code._tool_metadata["description"],
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }]
)

if response.choices[0].message.function_call:
    args = json.loads (response.choices[0].message.function_call.arguments)
    result = execute_python_code(**args)
    print(result["output"])
\`\`\`

## Enhanced Docker Setup for Data Science

Create a Docker image with common data science libraries:

\`\`\`dockerfile
# Dockerfile
FROM python:3.11-slim

# Install common libraries
RUN pip install --no-cache-dir \\
    numpy \\
    pandas \\
    matplotlib \\
    scipy \\
    scikit-learn \\
    seaborn

# Create working directory
WORKDIR /code

# Set non-root user for security
RUN useradd -m -u 1000 sandbox
USER sandbox

# Disable pip for security
RUN echo "alias pip='echo pip is disabled'" >> ~/.bashrc
\`\`\`

Build the image:
\`\`\`bash
docker build -t python-sandbox:latest .
\`\`\`

Use it:
\`\`\`python
executor = DockerCodeExecutor (image="python-sandbox:latest")
\`\`\`

## Handling Files and Plots

Allow code to generate files (like plots):

\`\`\`python
import tempfile
import os
import base64

class CodeInterpreterWithFiles:
    """Code interpreter that can handle file outputs."""
    
    def __init__(self):
        self.client = docker.from_env()
        self.image = "python-sandbox:latest"
    
    def execute_python (self, code: str) -> Dict[str, Any]:
        """
        Execute code and retrieve generated files.
        """
        # Create temp directory for outputs
        output_dir = tempfile.mkdtemp()
        code_file = os.path.join (output_dir, "script.py")
        
        # Write code to file
        with open (code_file, 'w') as f:
            f.write (code)
        
        try:
            # Run container with output directory mounted
            container = self.client.containers.run(
                image=self.image,
                command=["python", "/code/script.py"],
                volumes={
                    output_dir: {'bind': '/code', 'mode': 'rw'}
                },
                mem_limit="256m",
                network_disabled=True,
                remove=True,
                detach=True
            )
            
            # Wait for completion
            result = container.wait (timeout=30)
            logs = container.logs()
            
            # Collect generated files
            files = []
            for filename in os.listdir (output_dir):
                if filename != 'script.py':
                    filepath = os.path.join (output_dir, filename)
                    with open (filepath, 'rb') as f:
                        content = f.read()
                        files.append({
                            "filename": filename,
                            "content": base64.b64encode (content).decode('utf-8'),
                            "size": len (content)
                        })
            
            return {
                "success": True,
                "stdout": logs.decode('utf-8', errors='replace'),
                "stderr": "",
                "exit_code": result['StatusCode'],
                "files": files
            }
        
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str (e),
                "exit_code": -1,
                "files": []
            }
        
        finally:
            # Clean up
            import shutil
            shutil.rmtree (output_dir, ignore_errors=True)

# Usage
interpreter = CodeInterpreterWithFiles()

result = interpreter.execute_python("""
import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin (x)

# Create plot
plt.figure (figsize=(10, 6))
plt.plot (x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin (x)')
plt.grid(True)

# Save plot
plt.savefig('plot.png')
print("Plot saved to plot.png")
""")

print(f"Output: {result['stdout']}")
print(f"Files generated: {[f['filename'] for f in result['files']]}")

# Decode and save plot
if result['files']:
    plot_data = base64.b64decode (result['files'][0]['content'])
    with open('output_plot.png', 'wb') as f:
        f.write (plot_data)
\`\`\`

## Resource Monitoring

Monitor resource usage during execution:

\`\`\`python
import psutil
import threading
import time

class ResourceMonitor:
    """Monitor resource usage of code execution."""
    
    def __init__(self):
        self.metrics = {
            "cpu_percent": [],
            "memory_mb": [],
            "peak_memory_mb": 0
        }
        self.monitoring = False
    
    def start_monitoring (self, container):
        """Start monitoring container resources."""
        self.monitoring = True
        self.metrics = {
            "cpu_percent": [],
            "memory_mb": [],
            "peak_memory_mb": 0
        }
        
        def monitor():
            while self.monitoring:
                try:
                    stats = container.stats (stream=False)
                    
                    # CPU usage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \\
                                stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \\
                                   stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * 100.0
                    
                    # Memory usage
                    memory_mb = stats['memory_stats']['usage'] / (1024 * 1024)
                    
                    self.metrics['cpu_percent'].append (cpu_percent)
                    self.metrics['memory_mb'].append (memory_mb)
                    self.metrics['peak_memory_mb'] = max(
                        self.metrics['peak_memory_mb'],
                        memory_mb
                    )
                    
                    time.sleep(0.1)
                
                except:
                    break
        
        thread = threading.Thread (target=monitor, daemon=True)
        thread.start()
    
    def stop_monitoring (self):
        """Stop monitoring."""
        self.monitoring = False
    
    def get_summary (self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.metrics['cpu_percent']:
            return {}
        
        return {
            "avg_cpu_percent": sum (self.metrics['cpu_percent']) / len (self.metrics['cpu_percent']),
            "max_cpu_percent": max (self.metrics['cpu_percent']),
            "avg_memory_mb": sum (self.metrics['memory_mb']) / len (self.metrics['memory_mb']),
            "peak_memory_mb": self.metrics['peak_memory_mb']
        }
\`\`\`

## Security Best Practices

### 1. Input Validation

\`\`\`python
import re

def validate_python_code (code: str) -> tuple[bool, str]:
    """
    Validate Python code for dangerous patterns.
    
    Returns: (is_valid, error_message)
    """
    # Check for dangerous imports
    dangerous_imports = [
        'os', 'subprocess', 'socket', 'urllib',
        'requests', 'http', 'ftplib', 'smtplib'
    ]
    
    for module in dangerous_imports:
        if re.search (rf'\\bimport\\s+{module}\\b', code):
            return False, f"Import of '{module}' is not allowed"
        if re.search (rf'\\bfrom\\s+{module}\\b', code):
            return False, f"Import from '{module}' is not allowed"
    
    # Check for dangerous functions
    dangerous_functions = [
        'eval', 'exec', 'compile', '__import__',
        'open', 'file', 'input', 'raw_input'
    ]
    
    for func in dangerous_functions:
        if re.search (rf'\\b{func}\\s*\\(', code):
            return False, f"Use of '{func}()' is not allowed"
    
    # Check code length
    if len (code) > 10000:
        return False, "Code is too long (max 10000 characters)"
    
    return True, ""

@tool (description="Execute Python code")
def execute_python_code_safe (code: str) -> dict:
    """Execute Python code with validation."""
    # Validate first
    is_valid, error = validate_python_code (code)
    if not is_valid:
        return {
            "success": False,
            "error": f"Code validation failed: {error}"
        }
    
    # Execute
    return code_interpreter.execute_python (code)
\`\`\`

### 2. Rate Limiting

\`\`\`python
from collections import defaultdict
from datetime import datetime, timedelta

class ExecutionRateLimiter:
    """Rate limit code executions per user."""
    
    def __init__(self, max_executions: int = 10, time_window: int = 60):
        self.max_executions = max_executions
        self.time_window = time_window
        self.executions = defaultdict (list)
    
    def can_execute (self, user_id: str) -> bool:
        """Check if user can execute code."""
        now = datetime.now()
        cutoff = now - timedelta (seconds=self.time_window)
        
        # Remove old executions
        self.executions[user_id] = [
            t for t in self.executions[user_id] if t > cutoff
        ]
        
        # Check limit
        if len (self.executions[user_id]) >= self.max_executions:
            return False
        
        # Record execution
        self.executions[user_id].append (now)
        return True

rate_limiter = ExecutionRateLimiter (max_executions=10, time_window=60)

@tool (description="Execute Python code")
def execute_python_code_rate_limited (code: str, user_id: str) -> dict:
    """Execute code with rate limiting."""
    if not rate_limiter.can_execute (user_id):
        return {
            "success": False,
            "error": "Rate limit exceeded. Please wait before executing more code."
        }
    
    return code_interpreter.execute_python (code)
\`\`\`

### 3. Audit Logging

\`\`\`python
import logging
import json
from datetime import datetime

class CodeExecutionAuditor:
    """Audit all code executions."""
    
    def __init__(self, log_file: str):
        self.logger = logging.getLogger('code_execution_audit')
        handler = logging.FileHandler (log_file)
        handler.setFormatter (logging.Formatter('%(message)s'))
        self.logger.addHandler (handler)
        self.logger.setLevel (logging.INFO)
    
    def log_execution (self, user_id: str, code: str, result: Dict[str, Any]):
        """Log a code execution."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "code": code,
            "success": result["success"],
            "exit_code": result.get("exit_code"),
            "output_length": len (result.get("stdout", "")),
            "error": result.get("stderr", "")
        }
        
        self.logger.info (json.dumps (log_entry))

auditor = CodeExecutionAuditor('code_executions.log')

@tool (description="Execute Python code")
def execute_python_code_audited (code: str, user_id: str) -> dict:
    """Execute code with auditing."""
    result = code_interpreter.execute_python (code)
    auditor.log_execution (user_id, code, result)
    return result
\`\`\`

## Testing Code Execution Tools

\`\`\`python
import pytest

def test_basic_execution():
    """Test basic code execution."""
    result = execute_python_code("print('Hello, World!')")
    assert result["success"] is True
    assert "Hello, World!" in result["output"]

def test_timeout():
    """Test that infinite loops timeout."""
    code = """
import time
while True:
    time.sleep(1)
"""
    result = execute_python_code (code)
    assert result["success"] is False
    assert "timeout" in result["error"].lower()

def test_memory_limit():
    """Test memory limits."""
    code = """
# Try to allocate large amount of memory
data = bytearray(500 * 1024 * 1024)  # 500 MB
"""
    result = execute_python_code (code)
    # Should fail due to 256MB limit
    assert result["success"] is False

def test_network_access_blocked():
    """Test that network access is blocked."""
    code = """
import urllib.request
urllib.request.urlopen('http://google.com')
"""
    result = execute_python_code (code)
    assert result["success"] is False

def test_file_operations():
    """Test file operations in /tmp."""
    code = """
with open('/tmp/test.txt', 'w') as f:
    f.write('test')

with open('/tmp/test.txt', 'r') as f:
    print(f.read())
"""
    result = execute_python_code (code)
    assert result["success"] is True
    assert "test" in result["output"]
\`\`\`

## Summary

Building safe code execution tools requires:

1. **Strong isolation** - Docker containers or equivalent
2. **Resource limits** - CPU, memory, time
3. **Network restrictions** - Disable network access
4. **Input validation** - Block dangerous patterns
5. **Rate limiting** - Prevent abuse
6. **Audit logging** - Track all executions
7. **Error handling** - Graceful failure modes
8. **Monitoring** - Track resource usage
9. **File handling** - Support for generated files
10. **Testing** - Comprehensive security tests

Code execution is powerful but dangerous. Always prioritize security over convenience.

Next, we'll explore prompt engineering techniques specifically for tool-using agents.
`,
};

/**
 * Code Execution & Validation Section
 * Module 5: Building Code Generation Systems
 */

export const codeexecutionvalidationSection = {
  id: 'code-execution-validation',
  title: 'Code Execution & Validation',
  content: `# Code Execution & Validation

Master safely executing and validating generated code - essential for ensuring quality and correctness.

## Overview: Why Execute Generated Code?

Executing generated code allows you to:
- Verify it actually works
- Run tests automatically
- Capture outputs
- Detect runtime errors
- Measure performance
- Validate behavior

### The Risk

**Never run untrusted code directly!**
- Could delete files
- Access network
- Read sensitive data
- Install malware
- Consume resources

**Solution**: Sandboxing

## Docker-Based Sandboxing

### Secure Code Execution

\`\`\`python
import docker
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    
    def __bool__(self):
        return self.success

class DockerSandbox:
    """Execute code safely in Docker container."""
    
    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit: str = "256m",
        cpu_quota: int = 50000,  # 50% of one CPU
        network_disabled: bool = True
    ):
        """
        Initialize sandbox.
        
        Args:
            image: Docker image to use
            memory_limit: Memory limit (e.g., "256m")
            cpu_quota: CPU quota (100000 = 1 CPU)
            network_disabled: Disable network access
        """
        self.client = docker.from_env()
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.network_disabled = network_disabled
        
        # Ensure image exists
        try:
            self.client.images.get(image)
        except docker.errors.ImageNotFound:
            print(f"Pulling image {image}...")
            self.client.images.pull(image)
    
    def execute(
        self,
        code: str,
        timeout: int = 5,
        stdin: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute code in sandbox.
        
        Args:
            code: Code to execute
            timeout: Max execution time in seconds
            stdin: Standard input to provide
        
        Returns:
            ExecutionResult with output and status
        """
        import time
        
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file
            code_path = Path(tmpdir) / "script.py"
            code_path.write_text(code)
            
            # Write stdin if provided
            if stdin:
                stdin_path = Path(tmpdir) / "input.txt"
                stdin_path.write_text(stdin)
                stdin_redirect = " < /sandbox/input.txt"
            else:
                stdin_redirect = ""
            
            try:
                # Run container
                container = self.client.containers.run(
                    self.image,
                    command=f"sh -c 'python /sandbox/script.py{stdin_redirect}'",
                    volumes={tmpdir: {'bind': '/sandbox', 'mode': 'ro'}},
                    mem_limit=self.memory_limit,
                    cpu_quota=self.cpu_quota,
                    network_disabled=self.network_disabled,
                    detach=True,
                    remove=True,
                    # Security options
                    security_opt=["no-new-privileges"],
                    cap_drop=["ALL"]
                )
                
                # Wait for completion with timeout
                try:
                    result = container.wait(timeout=timeout)
                    exit_code = result['StatusCode']
                    
                    # Get output
                    stdout = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
                    stderr = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')
                    
                    execution_time = time.time() - start_time
                    
                    return ExecutionResult(
                        success=(exit_code == 0),
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=exit_code,
                        execution_time=execution_time
                    )
                
                except docker.errors.ContainerError as e:
                    return ExecutionResult(
                        success=False,
                        stdout="",
                        stderr=str(e),
                        exit_code=1,
                        execution_time=time.time() - start_time
                    )
            
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Sandbox error: {str(e)}",
                    exit_code=1,
                    execution_time=time.time() - start_time
                )

# Usage
sandbox = DockerSandbox(
    memory_limit="128m",
    cpu_quota=25000,  # 25% CPU
    network_disabled=True
)

code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
print(factorial(10))
"""

result = sandbox.execute(code, timeout=3)

if result:
    print("✓ Execution successful")
    print(f"Output: {result.stdout}")
    print(f"Time: {result.execution_time:.2f}s")
else:
    print("✗ Execution failed")
    print(f"Error: {result.stderr}")
\`\`\`

## Test Execution

### Run Generated Tests

\`\`\`python
class TestExecutor:
    """Execute tests for generated code."""
    
    def __init__(self, sandbox: DockerSandbox):
        self.sandbox = sandbox
    
    def execute_tests(
        self,
        code: str,
        tests: str,
        framework: str = "pytest"
    ) -> Tuple[bool, str]:
        """
        Execute tests for code.
        
        Args:
            code: Code to test
            tests: Test code
            framework: Test framework ("pytest", "unittest")
        
        Returns:
            (all_passed, output)
        """
        # Combine code and tests
        full_code = f"""
{code}

{tests}
"""
        
        # Add test runner
        if framework == "pytest":
            full_code += """
if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
"""
        elif framework == "unittest":
            full_code += """
if __name__ == "__main__":
    import unittest
    unittest.main()
"""
        
        # Execute
        result = self.sandbox.execute(full_code, timeout=10)
        
        return result.success, result.stdout + result.stderr
    
    def execute_with_coverage(
        self,
        code: str,
        tests: str
    ) -> Dict[str, Any]:
        """Execute tests and measure coverage."""
        
        # Add coverage measurement
        full_code = f"""
import coverage

cov = coverage.Coverage()
cov.start()

{code}

{tests}

cov.stop()
cov.save()

# Print coverage report
print("\\n=== Coverage Report ===")
cov.report()
"""
        
        result = self.sandbox.execute(full_code, timeout=10)
        
        # Parse coverage from output
        coverage_percent = self._parse_coverage(result.stdout)
        
        return {
            "success": result.success,
            "output": result.stdout,
            "coverage": coverage_percent
        }
    
    def _parse_coverage(self, output: str) -> float:
        """Parse coverage percentage from output."""
        # Look for "TOTAL" line in coverage report
        for line in output.split("\\n"):
            if "TOTAL" in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        return float(parts[-1].rstrip("%"))
                    except ValueError:
                        pass
        return 0.0

# Usage
sandbox = DockerSandbox()
executor = TestExecutor(sandbox)

code = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""

tests = """
def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 1) == -1
"""

all_passed, output = executor.execute_tests(code, tests)

if all_passed:
    print("✓ All tests passed")
else:
    print("✗ Some tests failed")

print(output)
\`\`\`

## Runtime Validation

### Validate Code Behavior

\`\`\`python
class RuntimeValidator:
    """Validate code behavior at runtime."""
    
    def __init__(self, sandbox: DockerSandbox):
        self.sandbox = sandbox
    
    def validate_with_examples(
        self,
        code: str,
        function_name: str,
        examples: List[Tuple[Tuple, Any]]  # (inputs, expected_output)
    ) -> Tuple[bool, List[str]]:
        """
        Validate function with example inputs/outputs.
        
        Returns:
            (all_passed, error_messages)
        """
        errors = []
        
        # Build validation code
        validation_code = f"""
{code}

# Test cases
test_cases = {examples}

for i, (inputs, expected) in enumerate(test_cases):
    try:
        result = {function_name}(*inputs)
        if result != expected:
            print(f"FAIL {{i}}: Expected {{expected}}, got {{result}}")
        else:
            print(f"PASS {{i}}")
    except Exception as e:
        print(f"ERROR {{i}}: {{e}}")
"""
        
        result = self.sandbox.execute(validation_code, timeout=5)
        
        # Parse results
        for line in result.stdout.split("\\n"):
            if line.startswith("FAIL") or line.startswith("ERROR"):
                errors.append(line)
        
        return len(errors) == 0, errors
    
    def validate_no_errors(
        self,
        code: str,
        test_inputs: List[Tuple]
    ) -> Tuple[bool, str]:
        """
        Validate that code doesn't raise errors for given inputs.
        
        Returns:
            (no_errors, error_message)
        """
        validation_code = f"""
{code}

test_inputs = {test_inputs}

for i, inputs in enumerate(test_inputs):
    try:
        result = process(*inputs)  # Assumes function named 'process'
        print(f"OK {{i}}: {{result}}")
    except Exception as e:
        print(f"ERROR {{i}}: {{type(e).__name__}}: {{e}}")
        import traceback
        traceback.print_exc()
"""
        
        result = self.sandbox.execute(validation_code, timeout=5)
        
        # Check for errors
        has_errors = "ERROR" in result.stdout or result.stderr
        
        return not has_errors, result.stdout + result.stderr

# Usage
sandbox = DockerSandbox()
validator = RuntimeValidator(sandbox)

code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

examples = [
    ((0,), 1),
    ((1,), 1),
    ((5,), 120),
    ((10,), 3628800),
]

all_passed, errors = validator.validate_with_examples(
    code,
    "factorial",
    examples
)

if all_passed:
    print("✓ All examples passed")
else:
    print("✗ Validation failed:")
    for error in errors:
        print(f"  {error}")
\`\`\`

## Performance Measurement

### Measure Code Performance

\`\`\`python
class PerformanceMeasurer:
    """Measure code performance."""
    
    def __init__(self, sandbox: DockerSandbox):
        self.sandbox = sandbox
    
    def measure_execution_time(
        self,
        code: str,
        function_name: str,
        test_input: Tuple,
        iterations: int = 100
    ) -> Dict[str, float]:
        """Measure average execution time."""
        
        measurement_code = f"""
import time

{code}

# Warm up
for _ in range(10):
    {function_name}{test_input}

# Measure
times = []
for _ in range({iterations}):
    start = time.perf_counter()
    {function_name}{test_input}
    end = time.perf_counter()
    times.append(end - start)

# Statistics
avg_time = sum(times) / len(times)
min_time = min(times)
max_time = max(times)

print(f"Average: {{avg_time*1000:.4f}}ms")
print(f"Min: {{min_time*1000:.4f}}ms")
print(f"Max: {{max_time*1000:.4f}}ms")
"""
        
        result = self.sandbox.execute(measurement_code, timeout=30)
        
        # Parse results
        stats = {}
        for line in result.stdout.split("\\n"):
            if "Average:" in line:
                stats["avg_ms"] = float(line.split(":")[1].replace("ms", ""))
            elif "Min:" in line:
                stats["min_ms"] = float(line.split(":")[1].replace("ms", ""))
            elif "Max:" in line:
                stats["max_ms"] = float(line.split(":")[1].replace("ms", ""))
        
        return stats
    
    def measure_memory_usage(
        self,
        code: str,
        function_name: str,
        test_input: Tuple
    ) -> Dict[str, float]:
        """Measure memory usage."""
        
        measurement_code = f"""
import tracemalloc

{code}

# Measure memory
tracemalloc.start()

{function_name}{test_input}

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Current: {{current / 1024 / 1024:.2f}} MB")
print(f"Peak: {{peak / 1024 / 1024:.2f}} MB")
"""
        
        result = self.sandbox.execute(measurement_code, timeout=10)
        
        # Parse results
        stats = {}
        for line in result.stdout.split("\\n"):
            if "Current:" in line:
                stats["current_mb"] = float(line.split(":")[1].replace("MB", ""))
            elif "Peak:" in line:
                stats["peak_mb"] = float(line.split(":")[1].replace("MB", ""))
        
        return stats

# Usage
sandbox = DockerSandbox()
measurer = PerformanceMeasurer(sandbox)

code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""

# Measure time
time_stats = measurer.measure_execution_time(
    code,
    "fibonacci",
    (20,),
    iterations=10
)

print(f"Average execution time: {time_stats['avg_ms']:.2f}ms")

# Measure memory
memory_stats = measurer.measure_memory_usage(
    code,
    "fibonacci",
    (30,)
)

print(f"Peak memory usage: {memory_stats['peak_mb']:.2f}MB")
\`\`\`

## Complete Validation Pipeline

### End-to-End Validation

\`\`\`python
class CodeValidator:
    """Complete code validation pipeline."""
    
    def __init__(self):
        self.sandbox = DockerSandbox()
        self.test_executor = TestExecutor(self.sandbox)
        self.runtime_validator = RuntimeValidator(self.sandbox)
        self.perf_measurer = PerformanceMeasurer(self.sandbox)
    
    def validate_generated_code(
        self,
        code: str,
        tests: Optional[str] = None,
        examples: Optional[List[Tuple]] = None,
        performance_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run complete validation pipeline.
        
        Returns:
            Validation report
        """
        report = {
            "syntax_valid": False,
            "tests_passed": False,
            "examples_valid": False,
            "performance_ok": False,
            "errors": []
        }
        
        # 1. Syntax validation
        try:
            import ast
            ast.parse(code)
            report["syntax_valid"] = True
        except SyntaxError as e:
            report["errors"].append(f"Syntax error: {e}")
            return report  # Can't continue if syntax is invalid
        
        # 2. Execute tests if provided
        if tests:
            tests_passed, output = self.test_executor.execute_tests(code, tests)
            report["tests_passed"] = tests_passed
            report["test_output"] = output
            
            if not tests_passed:
                report["errors"].append("Some tests failed")
        
        # 3. Validate with examples if provided
        if examples:
            # Extract function name from code
            tree = ast.parse(code)
            func_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    break
            
            if func_name:
                examples_valid, errors = self.runtime_validator.validate_with_examples(
                    code, func_name, examples
                )
                report["examples_valid"] = examples_valid
                
                if not examples_valid:
                    report["errors"].extend(errors)
        
        # 4. Performance check if threshold provided
        if performance_threshold:
            # Measure on first example if available
            if examples and examples[0]:
                func_name = None
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        break
                
                if func_name:
                    stats = self.perf_measurer.measure_execution_time(
                        code, func_name, examples[0][0], iterations=10
                    )
                    
                    report["performance_ms"] = stats.get("avg_ms", 0)
                    report["performance_ok"] = stats.get("avg_ms", float('inf')) <= performance_threshold
                    
                    if not report["performance_ok"]:
                        report["errors"].append(
                            f"Performance exceeded threshold: "
                            f"{stats.get('avg_ms')}ms > {performance_threshold}ms"
                        )
        
        # Overall success
        report["valid"] = (
            report["syntax_valid"] and
            (not tests or report["tests_passed"]) and
            (not examples or report["examples_valid"]) and
            (performance_threshold is None or report["performance_ok"])
        )
        
        return report

# Usage
validator = CodeValidator()

code = """
def calculate_primes(n):
    '''Return list of prime numbers up to n.'''
    primes = []
    for num in range(2, n + 1):
        is_prime = True
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes
"""

tests = """
def test_calculate_primes():
    assert calculate_primes(10) == [2, 3, 5, 7]
    assert calculate_primes(2) == [2]
    assert calculate_primes(1) == []
"""

examples = [
    ((10,), [2, 3, 5, 7]),
    ((20,), [2, 3, 5, 7, 11, 13, 17, 19]),
]

report = validator.validate_generated_code(
    code,
    tests=tests,
    examples=examples,
    performance_threshold=100.0  # 100ms max
)

print(f"Valid: {report['valid']}")
print(f"Syntax: {'✓' if report['syntax_valid'] else '✗'}")
print(f"Tests: {'✓' if report['tests_passed'] else '✗'}")
print(f"Examples: {'✓' if report['examples_valid'] else '✗'}")

if report['errors']:
    print("\\nErrors:")
    for error in report['errors']:
        print(f"  - {error}")
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Always use sandboxing** for code execution
2. **Set resource limits** (memory, CPU, time)
3. **Disable network** by default
4. **Run tests automatically**
5. **Validate with examples**
6. **Measure performance**
7. **Check for errors** before deploying
8. **Use minimal container images**

### ❌ DON'T:
1. **Execute untrusted code directly**
2. **Skip resource limits**
3. **Allow network access** without reason
4. **Trust generated code** without validation
5. **Skip error checking**
6. **Ignore performance**
7. **Run indefinitely** without timeouts
8. **Give too many privileges**

## Next Steps

You've mastered code execution and validation! Next:
- Language-specific generation
- Building complete code editors

Remember: **Sandbox Everything + Validate Always = Safe Code Generation**
`,
};

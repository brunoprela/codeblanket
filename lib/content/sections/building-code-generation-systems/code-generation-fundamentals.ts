/**
 * Code Generation Fundamentals Section
 * Module 5: Building Code Generation Systems
 */

export const codegenerationfundamentalsSection = {
  id: 'code-generation-fundamentals',
  title: 'Code Generation Fundamentals',
  content: `# Code Generation Fundamentals

Master the foundational concepts of generating code with LLMs and understand why it's uniquely challenging.

## Overview: Why Code Generation is Hard

Generating code with LLMs is fundamentally different from generating prose. Code must be:
- **Syntactically correct**: One missing bracket breaks everything
- **Semantically valid**: It must do what's intended
- **Contextually appropriate**: Fits with existing codebase patterns
- **Executable**: Actually runs without errors
- **Maintainable**: Other humans must understand it

### The Unique Challenges

**1. Precision Requirements**

Unlike creative writing where "close enough" is fine, code demands exactness:

\`\`\`python
# This essay is 99% correct - still readable
"Ths sentance has erors but you can still read it"

# This code is 99% correct - completely broken
def calculate_total(items:
    return sum([item.price for item in items)
#          ^ Missing bracket breaks everything
\`\`\`

**2. Context Dependencies**

Code rarely exists in isolation. It depends on:
- Imports and libraries
- Function signatures
- Type definitions
- File structure
- Project conventions

**3. Testing Requirements**

Generated code must be validated:

\`\`\`python
# This looks right but is wrong
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

# What if numbers is empty? Runtime error!
calculate_average([])  # ZeroDivisionError

# Correct version
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
\`\`\`

## Common Failure Modes

Understanding how code generation fails is crucial for building robust systems.

### 1. Syntax Errors

LLMs can generate invalid syntax, especially with:
- Bracket matching
- Indentation (Python)
- String escaping
- Edge cases in grammar

\`\`\`python
# Example: Mismatched brackets (common LLM error)
def process_data(items):
    result = []
    for item in items:
        if item['status'] == 'active':
            result.append({
                'id': item['id'],
                'name': item['name']
            )  # Missing closing brace
    return result

# Detection strategy:
import ast

def validate_python_syntax(code: str) -> tuple[bool, str]:
    """Check if Python code is syntactically valid."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"

# Usage
is_valid, error = validate_python_syntax(generated_code)
if not is_valid:
    print(f"Generated code has syntax error: {error}")
    # Trigger regeneration or correction
\`\`\`

### 2. Import Hallucination

LLMs often invent non-existent imports or functions:

\`\`\`python
# LLM might generate:
from sklearn.preprocessing import OneHotEncoder
from sklearn.magic import AutoML  # This doesn't exist!

def train_model(data):
    model = AutoML()  # Hallucinated
    model.fit(data)
    return model

# Detection strategy:
import importlib
import sys

def validate_imports(code: str) -> list[str]:
    """Extract and validate all imports in code."""
    tree = ast.parse(code)
    invalid_imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                try:
                    importlib.import_module(name.name)
                except ImportError:
                    invalid_imports.append(name.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                try:
                    module = importlib.import_module(node.module)
                    # Check if attributes exist
                    for name in node.names:
                        if not hasattr(module, name.name):
                            invalid_imports.append(
                                f"{node.module}.{name.name}"
                            )
                except ImportError:
                    invalid_imports.append(node.module)
    
    return invalid_imports

# Usage
invalid = validate_imports(generated_code)
if invalid:
    print(f"Invalid imports detected: {invalid}")
\`\`\`

### 3. Logic Errors

Code that's syntactically valid but semantically wrong:

\`\`\`python
# Task: "Write a function to find the maximum value in a list"

# LLM generates (looks right, but has bugs):
def find_max(numbers):
    max_val = 0  # Bug: what if all numbers are negative?
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val

# Also fails on empty list
find_max([])  # Returns 0, should raise error or return None

# Correct version:
def find_max(numbers):
    if not numbers:
        raise ValueError("Cannot find max of empty list")
    return max(numbers)

# Better: Test-driven approach
def test_find_max():
    assert find_max([1, 2, 3]) == 3
    assert find_max([-5, -1, -10]) == -1  # Catches the 0 bug!
    assert find_max([42]) == 42
    
    try:
        find_max([])
        assert False, "Should raise error"
    except ValueError:
        pass  # Expected
\`\`\`

### 4. Type Mismatches

Even with type hints, LLMs can generate type-incorrect code:

\`\`\`python
from typing import List, Dict

# Task: "Parse user data"
# LLM generates:
def parse_user_data(data: str) -> Dict[str, str]:
    # Forgot to handle JSON parsing
    return data  # Returns str, not dict!

# With type checking:
from typing import cast
import json

def parse_user_data(data: str) -> Dict[str, str]:
    parsed = json.loads(data)
    return cast(Dict[str, str], parsed)

# Runtime validation:
def validate_return_type(func, expected_type):
    """Decorator to validate return types at runtime."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not isinstance(result, expected_type):
            raise TypeError(
                f"{func.__name__} returned {type(result)}, "
                f"expected {expected_type}"
            )
        return result
    return wrapper

@validate_return_type(dict)
def parse_user_data_safe(data: str) -> Dict[str, str]:
    return json.loads(data)
\`\`\`

## Validation Strategies

Building a robust validation pipeline is essential for production code generation.

### Comprehensive Validation Pipeline

\`\`\`python
from dataclasses import dataclass
from typing import Optional, List
import ast
import subprocess
import tempfile
import os

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self):
        return self.is_valid

class CodeValidator:
    """Comprehensive code validation pipeline."""
    
    def __init__(self, language: str = "python"):
        self.language = language
        self.checks = [
            self.check_syntax,
            self.check_imports,
            self.check_style,
            self.check_security,
        ]
    
    def validate(self, code: str) -> ValidationResult:
        """Run all validation checks."""
        errors = []
        warnings = []
        
        for check in self.checks:
            result = check(code)
            errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def check_syntax(self, code: str) -> ValidationResult:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return ValidationResult(True, [], [])
        except SyntaxError as e:
            return ValidationResult(
                False,
                [f"Syntax error at line {e.lineno}: {e.msg}"],
                []
            )
    
    def check_imports(self, code: str) -> ValidationResult:
        """Validate all imports exist."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ValidationResult(True, [], [])  # Syntax check handles this
        
        errors = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    try:
                        __import__(name.name)
                    except ImportError:
                        errors.append(f"Import not found: {name.name}")
        
        return ValidationResult(len(errors) == 0, errors, [])
    
    def check_style(self, code: str) -> ValidationResult:
        """Check code style with pylint."""
        warnings = []
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['pylint', temp_path, '--output-format=json'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Parse pylint output
            if result.stdout:
                import json
                issues = json.loads(result.stdout)
                for issue in issues:
                    if issue['type'] == 'error':
                        warnings.append(f"Style error: {issue['message']}")
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # pylint not available or timeout
        finally:
            os.unlink(temp_path)
        
        return ValidationResult(True, [], warnings)
    
    def check_security(self, code: str) -> ValidationResult:
        """Check for basic security issues."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ValidationResult(True, [], [])
        
        warnings = []
        dangerous_functions = {'eval', 'exec', '__import__'}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_functions:
                        warnings.append(
                            f"Dangerous function used: {node.func.id}"
                        )
        
        return ValidationResult(True, [], warnings)

# Usage
validator = CodeValidator()
result = validator.validate(generated_code)

if result:
    print("✓ Code is valid")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
else:
    print(f"✗ Validation failed: {result.errors}")
\`\`\`

## Sandboxing Generated Code

Never run untrusted generated code directly in your process!

### Docker-Based Sandbox

\`\`\`python
import docker
import tempfile
import os

class CodeSandbox:
    """Execute code safely in Docker container."""
    
    def __init__(self, image: str = "python:3.11-slim"):
        self.client = docker.from_env()
        self.image = image
    
    def execute(
        self,
        code: str,
        timeout: int = 5,
        memory_limit: str = "128m"
    ) -> tuple[bool, str, str]:
        """
        Execute code in sandbox.
        
        Returns:
            (success, stdout, stderr)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file
            code_path = os.path.join(tmpdir, "script.py")
            with open(code_path, 'w') as f:
                f.write(code)
            
            try:
                # Run container
                container = self.client.containers.run(
                    self.image,
                    command=["python", "/code/script.py"],
                    volumes={tmpdir: {'bind': '/code', 'mode': 'ro'}},
                    mem_limit=memory_limit,
                    network_disabled=True,
                    detach=True,
                    remove=True
                )
                
                # Wait for completion
                result = container.wait(timeout=timeout)
                
                # Get output
                stdout = container.logs(stdout=True, stderr=False).decode()
                stderr = container.logs(stdout=False, stderr=True).decode()
                
                success = result['StatusCode'] == 0
                return success, stdout, stderr
            
            except docker.errors.ContainerError as e:
                return False, "", str(e)
            except Exception as e:
                return False, "", f"Sandbox error: {e}"

# Usage
sandbox = CodeSandbox()

generated_code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
'''

success, stdout, stderr = sandbox.execute(generated_code)

if success:
    print(f"Output: {stdout}")
else:
    print(f"Error: {stderr}")
\`\`\`

## Hallucination Handling

LLMs hallucinate function names, APIs, and entire libraries.

### Detection and Mitigation

\`\`\`python
import ast
from typing import Set

class HallucinationDetector:
    """Detect when LLMs hallucinate code elements."""
    
    def __init__(self):
        # Known valid standard library modules
        self.known_modules = {
            'os', 'sys', 'json', 'math', 're', 'datetime',
            'collections', 'itertools', 'functools', 'typing',
            # ... add more
        }
        
        # Track project-specific valid imports
        self.project_imports: Set[str] = set()
    
    def check_code(self, code: str) -> List[str]:
        """Return list of potentially hallucinated elements."""
        hallucinations = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ["Code has syntax errors"]
        
        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module.split('.')[0]
                    if (base_module not in self.known_modules and
                        base_module not in self.project_imports):
                        hallucinations.append(
                            f"Unknown module: {node.module}"
                        )
            
            # Check for AI-sounding function names (often hallucinated)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Functions like .auto_optimize() or .magic_fix()
                    func_name = node.func.attr
                    suspicious = ['auto_', 'magic_', 'smart_', 'ai_']
                    if any(func_name.startswith(s) for s in suspicious):
                        hallucinations.append(
                            f"Suspicious function name: {func_name}"
                        )
        
        return hallucinations
    
    def add_valid_import(self, module: str):
        """Register a project-specific valid import."""
        self.project_imports.add(module)

# Usage
detector = HallucinationDetector()
detector.add_valid_import('myproject')

hallucinations = detector.check_code(generated_code)
if hallucinations:
    print("⚠️  Potential hallucinations detected:")
    for h in hallucinations:
        print(f"  - {h}")
    
    # Retry with explicit instruction
    prompt += "\\nIMPORTANT: Only use standard library imports."
\`\`\`

## Testing Generated Code

Always test generated code before accepting it.

### Automated Testing Pipeline

\`\`\`python
import unittest
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

class GeneratedCodeTester:
    """Test generated code against specifications."""
    
    def test_with_examples(
        self,
        code: str,
        test_cases: List[tuple]
    ) -> tuple[bool, str]:
        """
        Test code with input/output examples.
        
        Args:
            code: Generated code (must define a 'solution' function)
            test_cases: List of (input, expected_output) tuples
        
        Returns:
            (all_passed, error_message)
        """
        # Create namespace for execution
        namespace = {}
        
        try:
            exec(code, namespace)
        except Exception as e:
            return False, f"Execution error: {e}"
        
        if 'solution' not in namespace:
            return False, "No 'solution' function found"
        
        solution = namespace['solution']
        
        # Test each case
        for i, (input_val, expected) in enumerate(test_cases):
            try:
                result = solution(input_val)
                if result != expected:
                    return False, (
                        f"Test case {i+1} failed: "
                        f"input={input_val}, "
                        f"expected={expected}, "
                        f"got={result}"
                    )
            except Exception as e:
                return False, f"Test case {i+1} raised error: {e}"
        
        return True, ""

# Usage
tester = GeneratedCodeTester()

prompt = "Write a function that returns the sum of two numbers"
generated_code = """
def solution(x, y):
    return x + y
"""

test_cases = [
    ((2, 3), 5),
    ((0, 0), 0),
    ((-1, 1), 0),
    ((100, 200), 300),
]

passed, error = tester.test_with_examples(generated_code, test_cases)

if passed:
    print("✓ All tests passed!")
else:
    print(f"✗ Tests failed: {error}")
    # Trigger regeneration with test failure info
\`\`\`

## Production Safeguards

Essential safeguards for production code generation systems.

### Complete Safety System

\`\`\`python
from dataclasses import dataclass
from typing import Optional
import logging

@dataclass
class GenerationConfig:
    """Configuration for safe code generation."""
    max_retries: int = 3
    validate_syntax: bool = True
    validate_imports: bool = True
    sandbox_execution: bool = True
    require_tests: bool = True
    max_execution_time: int = 5
    allow_network: bool = False
    allow_file_io: bool = False

class SafeCodeGenerator:
    """Production-grade code generator with all safeguards."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.validator = CodeValidator()
        self.sandbox = CodeSandbox()
        self.detector = HallucinationDetector()
        self.logger = logging.getLogger(__name__)
    
    def generate(
        self,
        prompt: str,
        test_cases: Optional[List[tuple]] = None
    ) -> Optional[str]:
        """
        Generate code with full validation pipeline.
        
        Returns:
            Valid code or None if generation failed
        """
        for attempt in range(self.config.max_retries):
            self.logger.info(f"Generation attempt {attempt + 1}")
            
            # Generate code (using your LLM)
            code = self._call_llm(prompt)
            
            # Validate
            validation_result = self._validate(code)
            if not validation_result.is_valid:
                self.logger.warning(
                    f"Validation failed: {validation_result.errors}"
                )
                prompt = self._add_error_context(
                    prompt, validation_result.errors
                )
                continue
            
            # Test if test cases provided
            if test_cases and self.config.require_tests:
                tester = GeneratedCodeTester()
                passed, error = tester.test_with_examples(code, test_cases)
                if not passed:
                    self.logger.warning(f"Tests failed: {error}")
                    prompt = self._add_test_failure_context(prompt, error)
                    continue
            
            # If we get here, code is valid
            self.logger.info("✓ Code generation successful")
            return code
        
        self.logger.error("Max retries exceeded")
        return None
    
    def _validate(self, code: str) -> ValidationResult:
        """Run all validation checks."""
        # Syntax and imports
        result = self.validator.validate(code)
        if not result.is_valid:
            return result
        
        # Hallucinations
        hallucinations = self.detector.check_code(code)
        if hallucinations:
            return ValidationResult(
                False,
                [f"Potential hallucination: {h}" for h in hallucinations],
                []
            )
        
        return result
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate code."""
        # This would be your actual LLM call
        # For now, placeholder
        from openai import OpenAI
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a code generator. "
                              "Only output valid Python code."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2  # Lower for more deterministic code
        )
        
        return response.choices[0].message.content
    
    def _add_error_context(
        self, prompt: str, errors: List[str]
    ) -> str:
        """Add error context to prompt for retry."""
        error_str = "\\n".join(errors)
        return f"""{prompt}

IMPORTANT: Previous attempt had these errors:
{error_str}

Please fix these issues in the new code."""
    
    def _add_test_failure_context(
        self, prompt: str, error: str
    ) -> str:
        """Add test failure context to prompt."""
        return f"""{prompt}

IMPORTANT: Previous attempt failed tests:
{error}

Please fix the logic to pass all tests."""

# Usage
config = GenerationConfig(
    max_retries=3,
    require_tests=True,
    sandbox_execution=True
)

generator = SafeCodeGenerator(config)

code = generator.generate(
    prompt="Write a function to calculate factorial",
    test_cases=[
        (0, 1),
        (1, 1),
        (5, 120),
        (10, 3628800),
    ]
)

if code:
    print("Generated valid code:")
    print(code)
else:
    print("Failed to generate valid code")
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Always validate syntax** before accepting generated code
2. **Verify imports exist** to catch hallucinations
3. **Use sandboxing** for execution
4. **Test with examples** when possible
5. **Set timeouts** on all operations
6. **Log all attempts** for debugging
7. **Implement retry logic** with context
8. **Use lower temperature** (0.2-0.3) for code generation

### ❌ DON'T:
1. **Never exec() untrusted code** directly
2. **Don't assume syntax is valid**
3. **Don't skip import validation**
4. **Don't generate without retries**
5. **Don't ignore test failures**
6. **Don't allow unlimited execution time**
7. **Don't trust hallucinated APIs**
8. **Don't use high temperature** for code

## Next Steps

Now that you understand the fundamentals, you'll learn:
- Prompt engineering specifically for code generation
- Generating complete files vs edits
- Multi-file code generation
- Building production code generation systems

The key insight: **Code generation requires validation at every step**. Unlike prose, there's no "close enough" - code either works or it doesn't.
`,
};

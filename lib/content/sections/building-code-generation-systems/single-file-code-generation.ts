/**
 * Single File Code Generation Section
 * Module 5: Building Code Generation Systems
 */

export const singlefilecodegenerationSection = {
  id: 'single-file-code-generation',
  title: 'Single File Code Generation',
  content: `# Single File Code Generation

Master generating complete files, functions, and classes from natural language descriptions.

## Overview: Complete File Generation

Single file generation is the foundation of code generation systems. You'll learn to:
- Generate complete Python/JavaScript/etc. files
- Create functions with proper signatures
- Generate classes with methods
- Add boilerplate code
- Fill in templates
- Maintain consistency

### When to Generate Complete Files

**✅ Good Use Cases:**
- Creating new files from scratch
- Generating boilerplate (models, routes, tests)
- Template-based generation
- Prototyping new features
- Converting specifications to code

**❌ Poor Use Cases:**
- Modifying existing files (use diffs instead)
- Large files (>500 lines)
- Files with complex dependencies
- When you need precise control

## Complete File Generation System

### Basic File Generator

\`\`\`python
from dataclasses import dataclass
from typing import Optional, List
from openai import OpenAI

@dataclass
class FileSpec:
    """Specification for file generation."""
    filename: str
    language: str
    description: str
    requirements: List[str]
    examples: Optional[str] = None

class FileGenerator:
    """Generate complete code files."""
    
    def __init__(self):
        self.client = OpenAI()
        self.language_templates = {
            'python': self._python_template,
            'javascript': self._javascript_template,
            'typescript': self._typescript_template,
        }
    
    def generate (self, spec: FileSpec) -> str:
        """Generate a complete file from specification."""
        
        # Get language-specific template
        template_fn = self.language_templates.get(
            spec.language,
            self._generic_template
        )
        
        # Build prompt
        prompt = template_fn (spec)
        
        # Generate
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert {spec.language} programmer. "
                              "Generate complete, production-ready code files."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        code = response.choices[0].message.content
        
        # Clean the code
        code = self._extract_code (code)
        
        return code
    
    def _python_template (self, spec: FileSpec) -> str:
        """Python-specific prompt template."""
        requirements = "\\n".join (f"- {r}" for r in spec.requirements)
        
        prompt = f"""Generate a Python file: {spec.filename}

Description:
{spec.description}

Requirements:
{requirements}

Follow these Python conventions:
- Use type hints for all functions
- Add docstrings (Google style)
- Follow PEP 8
- Include proper imports
- Add error handling
- Use meaningful variable names
"""
        
        if spec.examples:
            prompt += f"""

Examples of similar code:
{spec.examples}
"""
        
        return prompt
    
    def _javascript_template (self, spec: FileSpec) -> str:
        """JavaScript-specific prompt template."""
        requirements = "\\n".join (f"- {r}" for r in spec.requirements)
        
        return f"""Generate a JavaScript file: {spec.filename}

Description:
{spec.description}

Requirements:
{requirements}

Follow these JavaScript conventions:
- Use modern ES6+ syntax
- Use const/let (not var)
- Add JSDoc comments
- Use arrow functions
- Proper error handling
- Export functions properly
"""
    
    def _typescript_template (self, spec: FileSpec) -> str:
        """TypeScript-specific prompt template."""
        requirements = "\\n".join (f"- {r}" for r in spec.requirements)
        
        return f"""Generate a TypeScript file: {spec.filename}

Description:
{spec.description}

Requirements:
{requirements}

Follow these TypeScript conventions:
- Explicit types for all parameters and returns
- Use interfaces for object shapes
- Use proper access modifiers
- Add TSDoc comments
- Use generics where appropriate
- Strict mode compliance
"""
    
    def _generic_template (self, spec: FileSpec) -> str:
        """Generic template for other languages."""
        requirements = "\\n".join (f"- {r}" for r in spec.requirements)
        
        return f"""Generate a {spec.language} file: {spec.filename}

Description:
{spec.description}

Requirements:
{requirements}
"""
    
    def _extract_code (self, response: str) -> str:
        """Extract code from markdown or text response."""
        # Check if wrapped in markdown code block
        if "\`\`\`" in response:
            # Extract content between code fences
            parts = response.split("\`\`\`")
            if len (parts) >= 3:
                # Get the code block (skip language identifier line)
                code = parts[1]
                # Remove language identifier if present
                lines = code.split("\\n")
                if lines and lines[0].strip() in {
                    'python', 'javascript', 'typescript', 'js', 'ts'
                }:
                    code = "\\n".join (lines[1:])
                return code.strip()
        
        return response.strip()

# Usage
generator = FileGenerator()

spec = FileSpec(
    filename="user_service.py",
    language="python",
    description="A service for managing user accounts",
    requirements=[
        "CRUD operations for users",
        "Email validation",
        "Password hashing",
        "User authentication",
        "Database integration using SQLAlchemy"
    ]
)

code = generator.generate (spec)
print(code)
\`\`\`

## Function Generation

### Focused Function Generator

\`\`\`python
class FunctionGenerator:
    """Generate individual functions with precision."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_function(
        self,
        name: str,
        description: str,
        parameters: List[tuple[str, str]],  # (name, type) pairs
        return_type: str,
        language: str = "python",
        include_tests: bool = False
    ) -> str:
        """Generate a single function."""
        
        # Build parameter string
        if language == "python":
            params = ", ".join (f"{name}: {type_}" for name, type_ in parameters)
            signature = f"def {name}({params}) -> {return_type}:"
        elif language in ["javascript", "typescript"]:
            params = ", ".join (f"{name}: {type_}" for name, type_ in parameters)
            signature = f"function {name}({params}): {return_type}"
        
        prompt = f"""Generate a {language} function with this signature:
{signature}

Description:
{description}

Requirements:
- Include comprehensive docstring/comments
- Add input validation
- Handle edge cases
- Include error handling
- Use best practices for {language}
"""
        
        if include_tests:
            prompt += "\\nAlso generate unit tests for this function."
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert {language} programmer."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code (response.choices[0].message.content)
    
    def _extract_code (self, response: str) -> str:
        """Extract code from response."""
        if "\`\`\`" in response:
            parts = response.split("\`\`\`")
            if len (parts) >= 3:
                return parts[1].strip()
        return response.strip()

# Usage
func_gen = FunctionGenerator()

code = func_gen.generate_function(
    name="validate_email",
    description="Validate an email address format",
    parameters=[("email", "str")],
    return_type="bool",
    language="python",
    include_tests=True
)

print(code)
\`\`\`

## Class Generation

### Complete Class Generator

\`\`\`python
@dataclass
class ClassSpec:
    """Specification for class generation."""
    name: str
    description: str
    attributes: List[tuple[str, str]]  # (name, type) pairs
    methods: List[tuple[str, str]]     # (name, description) pairs
    base_classes: List[str] = None
    decorators: List[str] = None

class ClassGenerator:
    """Generate complete classes."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_class(
        self,
        spec: ClassSpec,
        language: str = "python"
    ) -> str:
        """Generate a complete class."""
        
        if language == "python":
            return self._generate_python_class (spec)
        elif language == "typescript":
            return self._generate_typescript_class (spec)
        else:
            return self._generate_generic_class (spec, language)
    
    def _generate_python_class (self, spec: ClassSpec) -> str:
        """Generate Python class."""
        # Build attributes string
        attrs = "\\n".join(
            f"  - {name}: {type_}" 
            for name, type_ in spec.attributes
        )
        
        # Build methods string
        methods = "\\n".join(
            f"  - {name}(): {desc}"
            for name, desc in spec.methods
        )
        
        # Build base classes
        bases = ""
        if spec.base_classes:
            bases = f"Inherits from: {', '.join (spec.base_classes)}"
        
        # Build decorators
        decorators = ""
        if spec.decorators:
            decorators = f"Decorators: {', '.join (spec.decorators)}"
        
        prompt = f"""Generate a Python class: {spec.name}

Description:
{spec.description}

{bases}
{decorators}

Attributes:
{attrs}

Methods:
{methods}

Requirements:
- Use type hints throughout
- Add comprehensive docstrings (Google style)
- Include __init__ method
- Add __repr__ and __str__ if appropriate
- Include property decorators where appropriate
- Add input validation
- Follow Python best practices
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python programmer."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code (response.choices[0].message.content)
    
    def _extract_code (self, response: str) -> str:
        """Extract code from response."""
        if "\`\`\`" in response:
            parts = response.split("\`\`\`")
            if len (parts) >= 3:
                code = parts[1]
                lines = code.split("\\n")
                if lines and lines[0].strip() in {'python', 'py'}:
                    code = "\\n".join (lines[1:])
                return code.strip()
        return response.strip()

# Usage
class_spec = ClassSpec(
    name="User",
    description="Represents a user in the system",
    attributes=[
        ("id", "int"),
        ("username", "str"),
        ("email", "str"),
        ("created_at", "datetime"),
        ("is_active", "bool")
    ],
    methods=[
        ("validate_email", "Validate the user's email address"),
        ("set_password", "Set a new password (hashed)"),
        ("check_password", "Verify a password"),
        ("deactivate", "Deactivate the user account")
    ],
    base_classes=["BaseModel"],
    decorators=["dataclass"]
)

class_gen = ClassGenerator()
code = class_gen.generate_class (class_spec)
print(code)
\`\`\`

## Boilerplate Generation

### Template-Based Generation

\`\`\`python
class BoilerplateGenerator:
    """Generate common boilerplate code."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_api_route(
        self,
        endpoint: str,
        method: str,
        description: str,
        framework: str = "fastapi"
    ) -> str:
        """Generate API route boilerplate."""
        
        prompt = f"""Generate a {framework} API route:

Endpoint: {endpoint}
HTTP Method: {method}
Description: {description}

Include:
- Proper route decorator
- Request/response models (Pydantic)
- Input validation
- Error handling
- Docstring
- Type hints
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert in {framework}."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code (response.choices[0].message.content)
    
    def generate_database_model(
        self,
        table_name: str,
        fields: List[tuple[str, str, bool]],  # (name, type, required)
        orm: str = "sqlalchemy"
    ) -> str:
        """Generate ORM model boilerplate."""
        
        fields_str = "\\n".join(
            f"- {name}: {type_} ({'required' if req else 'optional'})"
            for name, type_, req in fields
        )
        
        prompt = f"""Generate a {orm} model for table: {table_name}

Fields:
{fields_str}

Include:
- Proper table definition
- All field definitions with types
- Relationships if appropriate
- Constraints (primary key, unique, etc.)
- __repr__ method
- Type hints
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert in {orm}."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code (response.choices[0].message.content)
    
    def generate_test_file(
        self,
        module_name: str,
        functions_to_test: List[str],
        framework: str = "pytest"
    ) -> str:
        """Generate test file boilerplate."""
        
        functions_str = "\\n".join (f"- {func}" for func in functions_to_test)
        
        prompt = f"""Generate a {framework} test file for module: {module_name}

Functions to test:
{functions_str}

Include:
- Proper test class structure
- Setup/teardown if needed
- Test for normal cases
- Test for edge cases
- Test for error cases
- Clear test names
- Assertions
- Mock objects if needed
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert in {framework} testing."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code (response.choices[0].message.content)
    
    def _extract_code (self, response: str) -> str:
        """Extract code from response."""
        if "\`\`\`" in response:
            parts = response.split("\`\`\`")
            if len (parts) >= 3:
                return parts[1].strip()
        return response.strip()

# Usage
boilerplate = BoilerplateGenerator()

# Generate API route
route_code = boilerplate.generate_api_route(
    endpoint="/api/users",
    method="POST",
    description="Create a new user",
    framework="fastapi"
)

# Generate database model
model_code = boilerplate.generate_database_model(
    table_name="users",
    fields=[
        ("id", "Integer", True),
        ("username", "String", True),
        ("email", "String", True),
        ("created_at", "DateTime", False)
    ],
    orm="sqlalchemy"
)

# Generate test file
test_code = boilerplate.generate_test_file(
    module_name="user_service",
    functions_to_test=["create_user", "get_user", "update_user", "delete_user"],
    framework="pytest"
)
\`\`\`

## Output Validation

### Validate Generated Files

\`\`\`python
import ast
import tempfile
import subprocess
from typing import Optional

class GeneratedFileValidator:
    """Validate generated files before accepting."""
    
    def validate_python_file (self, code: str) -> tuple[bool, List[str]]:
        """Validate Python file."""
        errors = []
        
        # 1. Syntax check
        try:
            ast.parse (code)
        except SyntaxError as e:
            errors.append (f"Syntax error at line {e.lineno}: {e.msg}")
            return False, errors
        
        # 2. Check imports
        tree = ast.parse (code)
        for node in ast.walk (tree):
            if isinstance (node, ast.Import):
                for name in node.names:
                    try:
                        __import__(name.name)
                    except ImportError:
                        errors.append (f"Import not found: {name.name}")
        
        # 3. Style check with black (if available)
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False
            ) as f:
                f.write (code)
                temp_path = f.name
            
            result = subprocess.run(
                ['black', '--check', temp_path],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode != 0:
                errors.append("Code doesn't follow Black style")
        
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # black not available
        finally:
            import os
            if 'temp_path' in locals():
                os.unlink (temp_path)
        
        return len (errors) == 0, errors
    
    def validate_javascript_file (self, code: str) -> tuple[bool, List[str]]:
        """Validate JavaScript file."""
        errors = []
        
        # Write to temp file and check with node
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.js', delete=False
        ) as f:
            f.write (code)
            temp_path = f.name
        
        try:
            # Check syntax with node
            result = subprocess.run(
                ['node', '--check', temp_path],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode != 0:
                errors.append (f"Syntax error: {result.stderr.decode()}")
        
        except FileNotFoundError:
            errors.append("Node.js not available for validation")
        except subprocess.TimeoutExpired:
            errors.append("Validation timeout")
        finally:
            import os
            os.unlink (temp_path)
        
        return len (errors) == 0, errors

# Usage
validator = GeneratedFileValidator()

generated_code = """
def calculate_sum (numbers: List[int]) -> int:
    ''Calculate sum of numbers.''
    return sum (numbers)
"""

is_valid, errors = validator.validate_python_file (generated_code)

if is_valid:
    print("✓ Generated file is valid")
else:
    print("✗ Validation errors:")
    for error in errors:
        print(f"  - {error}")
\`\`\`

## Complete Generation Pipeline

### End-to-End File Generation

\`\`\`python
class ProductionFileGenerator:
    """Production-ready file generation pipeline."""
    
    def __init__(self):
        self.generator = FileGenerator()
        self.validator = GeneratedFileValidator()
        self.max_retries = 3
    
    def generate_file(
        self,
        spec: FileSpec,
        output_path: Optional[str] = None,
        auto_fix: bool = True
    ) -> Optional[str]:
        """Generate and validate file with retries."""
        
        for attempt in range (self.max_retries):
            print(f"Generation attempt {attempt + 1}/{self.max_retries}")
            
            # Generate
            code = self.generator.generate (spec)
            
            # Validate
            if spec.language == "python":
                is_valid, errors = self.validator.validate_python_file (code)
            elif spec.language in ["javascript", "typescript"]:
                is_valid, errors = self.validator.validate_javascript_file (code)
            else:
                is_valid, errors = True, []
            
            if is_valid:
                # Success!
                if output_path:
                    with open (output_path, 'w') as f:
                        f.write (code)
                    print(f"✓ File written to {output_path}")
                
                return code
            
            # Failed validation
            print(f"✗ Validation failed: {errors}")
            
            if auto_fix and attempt < self.max_retries - 1:
                # Add error context for retry
                spec.requirements.append(
                    f"Fix these issues from previous attempt: {', '.join (errors)}"
                )
        
        print("✗ Failed to generate valid file after max retries")
        return None

# Usage
prod_gen = ProductionFileGenerator()

spec = FileSpec(
    filename="payment_processor.py",
    language="python",
    description="Process payment transactions",
    requirements=[
        "Support multiple payment methods (card, paypal)",
        "Validate payment details",
        "Handle transaction failures",
        "Log all transactions",
        "Return transaction status"
    ]
)

code = prod_gen.generate_file(
    spec,
    output_path="payment_processor.py",
    auto_fix=True
)

if code:
    print("Successfully generated file!")
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Provide detailed specifications** with requirements
2. **Use language-specific templates** for better results
3. **Include examples** of similar code
4. **Validate generated code** before accepting
5. **Implement retry logic** with error context
6. **Extract code cleanly** from markdown
7. **Test generated functions** automatically
8. **Use lower temperature** (0.2-0.3)

### ❌ DON'T:
1. **Generate very large files** (>500 lines)
2. **Skip validation**
3. **Ignore language conventions**
4. **Accept first generation** without testing
5. **Generate without specifications**
6. **Use high temperature**
7. **Forget error handling**
8. **Skip type hints/annotations**

## Next Steps

You've mastered single file generation. Next:
- Code editing and diff generation
- Multi-file generation
- Building complete code generation systems

Remember: **Specification → Generation → Validation → Iteration**
`,
};

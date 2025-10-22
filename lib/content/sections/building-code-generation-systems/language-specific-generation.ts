/**
 * Language-Specific Generation Section  
 * Module 5: Building Code Generation Systems
 */

export const languagespecificgenerationSection = {
    id: 'language-specific-generation',
    title: 'Language-Specific Generation',
    content: `# Language-Specific Generation

Master generating code for different programming languages with language-specific patterns, conventions, and best practices.

## Overview: Language-Specific Challenges

Each language has unique characteristics:
- **Python**: Indentation, dynamic typing, decorators
- **JavaScript/TypeScript**: Async patterns, promises, prototypes
- **Java**: Verbosity, strong typing, OOP patterns
- **C++**: Memory management, pointers, templates
- **Rust**: Ownership, borrowing, lifetimes

### Why Language-Specific Matters

Generic code generation produces mediocre results. Language-specific generation produces idiomatic, high-quality code.

## Python-Specific Generation

### Python Code Generator

\`\`\`python
from openai import OpenAI
from typing import Dict, Any

class PythonCodeGenerator:
    """Generate idiomatic Python code."""
    
    PYTHON_CONVENTIONS = """
Python Best Practices:
1. Use snake_case for functions and variables
2. Use PascalCase for classes
3. Add type hints to all functions
4. Use docstrings (Google or NumPy style)
5. Follow PEP 8 style guide
6. Use f-strings for formatting
7. Prefer list comprehensions for simple iterations
8. Use context managers (with statements)
9. Use @property for getters/setters
10. Add error handling with specific exceptions
"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_python_function(
        self,
        description: str,
        context: Optional[str] = None
    ) -> str:
        """Generate Python function."""
        
        prompt = f"""{self.PYTHON_CONVENTIONS}

Generate a Python function:
{description}

{f"Context: {context}" if context else ""}

Include:
- Type hints
- Docstring (Google style)
- Error handling
- Input validation
- Proper naming conventions
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python programmer. "
                              "Generate idiomatic, PEP 8 compliant code."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code(response.choices[0].message.content)
    
    def generate_python_class(
        self,
        description: str,
        attributes: List[str],
        methods: List[str]
    ) -> str:
        """Generate Python class."""
        
        prompt = f"""{self.PYTHON_CONVENTIONS}

Generate a Python class:
{description}

Attributes: {', '.join(attributes)}
Methods: {', '.join(methods)}

Include:
- __init__ method with type hints
- __repr__ and __str__ methods
- @property decorators where appropriate
- Docstrings for class and methods
- Type hints throughout
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
        
        return self._extract_code(response.choices[0].message.content)
    
    def add_type_hints(self, code: str) -> str:
        """Add comprehensive type hints to Python code."""
        
        prompt = f"""{self.PYTHON_CONVENTIONS}

Add comprehensive type hints to this Python code:

\`\`\`python
{code}
\`\`\`

Add:
- Parameter type hints
- Return type hints
- Variable type hints where helpful
- Use typing module (List, Dict, Optional, Union, etc.)
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Python type hints."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code(response.choices[0].message.content)
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        if "\`\`\`" in response:
            parts = response.split("\`\`\`")
            if len(parts) >= 3:
                return parts[1].strip()
        return response.strip()

# Usage
py_gen = PythonCodeGenerator()

# Generate function
func = py_gen.generate_python_function(
    "A function that validates email addresses and returns detailed validation results",
    context="Part of a user registration system"
)
print(func)

# Output will be idiomatic Python with type hints, docstrings, etc.
\`\`\`

## JavaScript/TypeScript Generation

### TypeScript Code Generator

\`\`\`python
class TypeScriptCodeGenerator:
    """Generate idiomatic TypeScript code."""
    
    TYPESCRIPT_CONVENTIONS = """
TypeScript Best Practices:
1. Use camelCase for functions and variables
2. Use PascalCase for classes and interfaces
3. Explicit types for all parameters and returns
4. Use interfaces for object shapes
5. Use async/await over raw promises
6. Prefer const over let, never var
7. Use template literals for strings
8. Use arrow functions
9. Add JSDoc comments
10. Use proper access modifiers (private, public, protected)
"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_typescript_function(
        self,
        description: str,
        is_async: bool = False
    ) -> str:
        """Generate TypeScript function."""
        
        prompt = f"""{self.TYPESCRIPT_CONVENTIONS}

Generate a TypeScript function:
{description}

{"Make it async using async/await" if is_async else ""}

Include:
- Explicit type annotations
- JSDoc comment
- Error handling
- Proper TypeScript syntax
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert TypeScript programmer."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code(response.choices[0].message.content)
    
    def generate_typescript_interface(
        self,
        name: str,
        properties: Dict[str, str]  # property_name: type
    ) -> str:
        """Generate TypeScript interface."""
        
        properties_str = "\\n".join(
            f"  {prop}: {type_}"
            for prop, type_ in properties.items()
        )
        
        prompt = f"""{self.TYPESCRIPT_CONVENTIONS}

Generate a TypeScript interface:

interface {name} {{
{properties_str}
}}

Add:
- JSDoc comment explaining the interface
- Optional properties where appropriate (?)
- Readonly properties where appropriate
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert TypeScript programmer."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code(response.choices[0].message.content)
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        if "\`\`\`" in response:
            parts = response.split("\`\`\`")
            if len(parts) >= 3:
                return parts[1].strip()
        return response.strip()

# Usage
ts_gen = TypeScriptCodeGenerator()

# Generate async function
func = ts_gen.generate_typescript_function(
    "A function that fetches user data from an API",
    is_async=True
)
print(func)

# Generate interface
interface = ts_gen.generate_typescript_interface(
    "UserProfile",
    {
        "id": "string",
        "email": "string",
        "name": "string",
        "createdAt": "Date",
        "settings": "UserSettings"
    }
)
print(interface)
\`\`\`

## Multi-Language Generator

### Universal Code Generator

\`\`\`python
class MultiLanguageGenerator:
    """Generate code for any language."""
    
    LANGUAGE_CONFIGS = {
        "python": {
            "conventions": PythonCodeGenerator.PYTHON_CONVENTIONS,
            "file_extension": ".py",
            "comment_style": "#"
        },
        "typescript": {
            "conventions": TypeScriptCodeGenerator.TYPESCRIPT_CONVENTIONS,
            "file_extension": ".ts",
            "comment_style": "//"
        },
        "java": {
            "conventions": """
Java Best Practices:
1. PascalCase for classes
2. camelCase for methods and variables
3. UPPER_SNAKE_CASE for constants
4. Use meaningful names
5. Add Javadoc comments
6. Use interfaces for contracts
7. Proper exception handling
8. Follow SOLID principles
""",
            "file_extension": ".java",
            "comment_style": "//"
        },
        "rust": {
            "conventions": """
Rust Best Practices:
1. snake_case for functions and variables
2. PascalCase for types and traits
3. SCREAMING_SNAKE_CASE for constants
4. Use Result<T, E> for error handling
5. Proper lifetime annotations
6. Use Option<T> instead of null
7. Add doc comments (///)
8. Follow ownership rules
""",
            "file_extension": ".rs",
            "comment_style": "//"
        }
    }
    
    def __init__(self):
        self.client = OpenAI()
        self.generators = {
            "python": PythonCodeGenerator(),
            "typescript": TypeScriptCodeGenerator()
        }
    
    def generate(
        self,
        language: str,
        description: str,
        code_type: str = "function"  # "function", "class", "interface"
    ) -> str:
        """Generate code in any language."""
        
        # Use specialized generator if available
        if language in self.generators:
            generator = self.generators[language]
            
            if code_type == "function":
                if hasattr(generator, f"generate_{language}_function"):
                    return getattr(generator, f"generate_{language}_function")(description)
        
        # Fall back to generic generation
        return self._generic_generate(language, description, code_type)
    
    def _generic_generate(
        self,
        language: str,
        description: str,
        code_type: str
    ) -> str:
        """Generic code generation for any language."""
        
        config = self.LANGUAGE_CONFIGS.get(
            language,
            {"conventions": "Follow language best practices", "comment_style": "//"}
        )
        
        prompt = f"""Generate {language} code:

{config['conventions']}

Type: {code_type}
Description: {description}

Generate clean, idiomatic {language} code following best practices.
"""
        
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
        
        return self._extract_code(response.choices[0].message.content)
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        if "\`\`\`" in response:
            parts = response.split("\`\`\`")
            if len(parts) >= 3:
                return parts[1].strip()
        return response.strip()
    
    def convert_between_languages(
        self,
        code: str,
        from_language: str,
        to_language: str
    ) -> str:
        """Convert code from one language to another."""
        
        from_config = self.LANGUAGE_CONFIGS.get(from_language, {})
        to_config = self.LANGUAGE_CONFIGS.get(to_language, {})
        
        prompt = f"""Convert this {from_language} code to {to_language}:

{from_language} code:
\`\`\`{from_language}
{code}
\`\`\`

{to_language} Conventions:
{to_config.get('conventions', 'Follow best practices')}

Generate equivalent {to_language} code that:
1. Maintains the same functionality
2. Follows {to_language} idioms and conventions
3. Uses appropriate {to_language} patterns
4. Is idiomatic and clean
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert at {from_language} and {to_language}."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code(response.choices[0].message.content)

# Usage
multi_gen = MultiLanguageGenerator()

# Generate in different languages
python_code = multi_gen.generate(
    "python",
    "A function to calculate Fibonacci numbers with memoization",
    "function"
)

java_code = multi_gen.generate(
    "java",
    "A class representing a binary search tree with insert and search methods",
    "class"
)

rust_code = multi_gen.generate(
    "rust",
    "A function to parse JSON with proper error handling",
    "function"
)

# Convert between languages
python_func = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""

typescript_func = multi_gen.convert_between_languages(
    python_func,
    "python",
    "typescript"
)

print("TypeScript version:")
print(typescript_func)
\`\`\`

## Language Detection

### Detect Programming Language

\`\`\`python
class LanguageDetector:
    """Detect programming language from code."""
    
    LANGUAGE_PATTERNS = {
        "python": [
            "def ", "import ", "from ", "class ", "self", ":", "\\n    "
        ],
        "javascript": [
            "function ", "const ", "let ", "var ", "=>", "console.log"
        ],
        "typescript": [
            "interface ", ": string", ": number", "type ", "async "
        ],
        "java": [
            "public class", "private ", "System.out", "void ", "String[]"
        ],
        "cpp": [
            "#include", "std::", "cout", "int main", "namespace"
        ],
        "rust": [
            "fn ", "let mut", "impl ", "pub ", "Result<", "Option<"
        ]
    }
    
    def detect(self, code: str) -> str:
        """Detect language from code sample."""
        scores = {lang: 0 for lang in self.LANGUAGE_PATTERNS}
        
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                if pattern in code:
                    scores[lang] += 1
        
        # Return language with highest score
        return max(scores, key=scores.get)

# Usage
detector = LanguageDetector()

code = """
function calculateSum(numbers: number[]): number {
    return numbers.reduce((sum, num) => sum + num, 0);
}
"""

language = detector.detect(code)
print(f"Detected language: {language}")  # typescript
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Use language-specific conventions** for each language
2. **Generate idiomatic code** following best practices
3. **Add appropriate documentation** (docstrings, JSDoc, etc.)
4. **Include type information** (hints, annotations, interfaces)
5. **Follow naming conventions** for each language
6. **Use language-specific patterns** (decorators in Python, async in JS)
7. **Add proper error handling** for each language
8. **Detect language** when converting or editing

### ❌ DON'T:
1. **Use generic patterns** for all languages
2. **Ignore language conventions**
3. **Skip type information**
4. **Use wrong naming styles**
5. **Forget language-specific features**
6. **Generate non-idiomatic code**
7. **Skip documentation**
8. **Mix language patterns**

## Next Steps

You've mastered language-specific generation! Next:
- Building a complete code editor
- Putting everything together

Remember: **Idiomatic Code = Quality Code**
`,
};


/**
 * Code Comment & Documentation Generation Section
 * Module 5: Building Code Generation Systems
 */

export const codecommentdocumentationgenerationSection = {
    id: 'code-comment-documentation-generation',
    title: 'Code Comment & Documentation Generation',
    content: `# Code Comment & Documentation Generation

Master generating comprehensive documentation, docstrings, and inline comments automatically.

## Overview: Automated Documentation

Good documentation is crucial but time-consuming. LLMs can:
- Generate docstrings for functions/classes
- Add inline comments
- Create README files
- Generate API documentation
- Explain complex code
- Add type hints

### Why Generate Documentation?

**Manual Documentation Issues:**
- Time-consuming
- Often outdated
- Inconsistent format
- Skipped under deadline pressure

**Generated Documentation:**
- Fast and comprehensive
- Consistent format
- Always up-to-date
- Covers all functions

## Docstring Generation

### Generate Function Docstrings

\`\`\`python
from openai import OpenAI
import ast
from typing import Optional

class DocstringGenerator:
    """Generate docstrings for Python functions."""
    
    def __init__(self, style: str = "google"):
        """
        Initialize docstring generator.
        
        Args:
            style: Docstring style ('google', 'numpy', 'sphinx')
        """
        self.client = OpenAI()
        self.style = style
        self.style_templates = {
            "google": self._google_template,
            "numpy": self._numpy_template,
            "sphinx": self._sphinx_template
        }
    
    def generate_docstring(
        self,
        function_code: str,
        include_examples: bool = True
    ) -> str:
        """Generate docstring for a function."""
        
        # Analyze function
        func_info = self._analyze_function(function_code)
        
        # Build prompt
        template = self.style_templates.get(
            self.style,
            self._google_template
        )
        
        prompt = template(func_info, include_examples)
        
        # Generate
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert at writing {self.style} "
                              "style docstrings."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
    
    def _analyze_function(self, function_code: str) -> dict:
        """Extract function metadata."""
        tree = ast.parse(function_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get parameters with types
                params = []
                for arg in node.args.args:
                    param_info = {"name": arg.arg}
                    if arg.annotation:
                        param_info["type"] = ast.unparse(arg.annotation)
                    params.append(param_info)
                
                # Get return type
                return_type = None
                if node.returns:
                    return_type = ast.unparse(node.returns)
                
                # Get function body to understand what it does
                body_lines = []
                for stmt in node.body:
                    if not isinstance(stmt, ast.Expr):  # Skip docstring
                        body_lines.append(ast.unparse(stmt))
                
                return {
                    "name": node.name,
                    "params": params,
                    "return_type": return_type,
                    "body": "\\n".join(body_lines),
                    "code": function_code
                }
        
        return {}
    
    def _google_template(
        self,
        func_info: dict,
        include_examples: bool
    ) -> str:
        """Google style docstring template."""
        params_str = "\\n".join(
            f"    {p['name']}: {p.get('type', 'Type needed')}"
            for p in func_info['params']
        )
        
        return f"""Generate a Google-style docstring for this function:

\`\`\`python
{func_info['code']}
\`\`\`

Function does: [Analyze the function body to understand what it does]

Format:
\"\"\"
Brief description (one line).

Detailed description (multiple lines if needed).

Args:
{params_str}

Returns:
    {func_info.get('return_type', 'Return type needed')}: Description

Raises:
    ExceptionType: When this exception is raised

{"Examples:" if include_examples else ""}
{"    >>> example usage" if include_examples else ""}
\"\"\"

Generate the complete docstring.
"""
    
    def _numpy_template(self, func_info: dict, include_examples: bool) -> str:
        """NumPy style docstring template."""
        return f"""Generate a NumPy-style docstring for this function:

\`\`\`python
{func_info['code']}
\`\`\`

Use NumPy docstring format with Parameters, Returns, Examples sections.
"""
    
    def _sphinx_template(self, func_info: dict, include_examples: bool) -> str:
        """Sphinx style docstring template."""
        return f"""Generate a Sphinx-style docstring for this function:

\`\`\`python
{func_info['code']}
\`\`\`

Use Sphinx format with :param, :return, :raises tags.
"""
    
    def add_docstring_to_function(
        self,
        function_code: str,
        docstring: str
    ) -> str:
        """Add docstring to function code."""
        lines = function_code.split("\\n")
        
        # Find first line after function definition
        insert_line = 1
        for i, line in enumerate(lines):
            if line.strip().endswith(":"):
                insert_line = i + 1
                break
        
        # Get indentation
        indent = self._get_indentation(lines[insert_line] if insert_line < len(lines) else "    ")
        
        # Format docstring with proper indentation
        docstring_lines = [indent + '"""']
        for line in docstring.strip().split("\\n"):
            docstring_lines.append(indent + line)
        docstring_lines.append(indent + '"""')
        
        # Insert docstring
        result = lines[:insert_line] + docstring_lines + lines[insert_line:]
        
        return "\\n".join(result)
    
    def _get_indentation(self, line: str) -> str:
        """Get indentation of a line."""
        return line[:len(line) - len(line.lstrip())]

# Usage
doc_gen = DocstringGenerator(style="google")

function_code = """
def calculate_statistics(data: List[float], trim_outliers: bool = False) -> Dict[str, float]:
    if trim_outliers:
        data = remove_outliers(data)
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    
    return {
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev
    }
"""

docstring = doc_gen.generate_docstring(function_code, include_examples=True)
print("Generated docstring:")
print(docstring)

# Add to function
complete_function = doc_gen.add_docstring_to_function(function_code, docstring)
print("\\nComplete function:")
print(complete_function)
\`\`\`

## Inline Comment Generation

### Add Explanatory Comments

\`\`\`python
class InlineCommentGenerator:
    """Generate inline comments for complex code."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def add_comments(
        self,
        code: str,
        comment_style: str = "explanatory"  # "explanatory", "summary", "intent"
    ) -> str:
        """Add inline comments to code."""
        
        prompt = f"""Add {comment_style} comments to this code:

\`\`\`python
{code}
\`\`\`

Guidelines:
- Add comments for complex logic
- Explain WHY, not just WHAT
- Keep comments concise
- Don't over-comment obvious code
- Use proper Python comment style (#)

Output the code with comments added.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at code documentation."
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
comment_gen = InlineCommentGenerator()

code = """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)
"""

commented = comment_gen.add_comments(code, comment_style="explanatory")
print(commented)

# Output will have comments like:
# def quick_sort(arr):
#     # Base case: arrays with 0 or 1 element are already sorted
#     if len(arr) <= 1:
#         return arr
#     
#     # Choose middle element as pivot for balanced partitioning
#     pivot = arr[len(arr) // 2]
#     
#     # Partition array into three groups around pivot
#     left = [x for x in arr if x < pivot]      # Elements less than pivot
#     middle = [x for x in arr if x == pivot]   # Elements equal to pivot
#     right = [x for x in arr if x > pivot]     # Elements greater than pivot
#     
#     # Recursively sort left and right, concatenate with middle
#     return quick_sort(left) + middle + quick_sort(right)
\`\`\`

## Class Documentation

### Generate Class Documentation

\`\`\`python
class ClassDocGenerator:
    """Generate comprehensive class documentation."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def document_class(
        self,
        class_code: str,
        include_examples: bool = True
    ) -> str:
        """Generate complete documentation for a class."""
        
        # Analyze class
        class_info = self._analyze_class(class_code)
        
        prompt = f"""Generate comprehensive documentation for this class:

\`\`\`python
{class_code}
\`\`\`

Generate:
1. Class-level docstring explaining purpose
2. Docstrings for __init__ and all methods
3. Document attributes
{"4. Usage examples" if include_examples else ""}

Use Google-style docstrings.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at class documentation."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code(response.choices[0].message.content)
    
    def _analyze_class(self, class_code: str) -> dict:
        """Analyze class structure."""
        tree = ast.parse(class_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                attributes = []
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
                    elif isinstance(item, ast.AnnAssign):
                        if isinstance(item.target, ast.Name):
                            attributes.append(item.target.id)
                
                return {
                    "name": node.name,
                    "methods": methods,
                    "attributes": attributes
                }
        
        return {}
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        if "\`\`\`" in response:
            parts = response.split("\`\`\`")
            if len(parts) >= 3:
                return parts[1].strip()
        return response.strip()

# Usage
class_doc_gen = ClassDocGenerator()

class_code = """
class UserManager:
    def __init__(self, database):
        self.database = database
        self.cache = {}
    
    def create_user(self, username, email):
        user = User(username=username, email=email)
        self.database.save(user)
        self.cache[user.id] = user
        return user
    
    def get_user(self, user_id):
        if user_id in self.cache:
            return self.cache[user_id]
        
        user = self.database.get(user_id)
        if user:
            self.cache[user_id] = user
        return user
    
    def clear_cache(self):
        self.cache.clear()
"""

documented_class = class_doc_gen.document_class(class_code, include_examples=True)
print(documented_class)
\`\`\`

## README Generation

### Generate Project README

\`\`\`python
class ReadmeGenerator:
    """Generate README.md files for projects."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_readme(
        self,
        project_name: str,
        description: str,
        main_files: List[str],
        dependencies: List[str]
    ) -> str:
        """Generate comprehensive README."""
        
        prompt = f"""Generate a README.md for this project:

Project Name: {project_name}
Description: {description}

Main Files:
{self._format_files(main_files)}

Dependencies:
{self._format_dependencies(dependencies)}

Include:
1. Project title and description
2. Features
3. Installation instructions
4. Usage examples
5. API documentation (if applicable)
6. Contributing guidelines
7. License information

Use proper Markdown formatting.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at technical writing."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def _format_files(self, files: List[str]) -> str:
        """Format file list."""
        return "\\n".join(f"- {file}" for file in files)
    
    def _format_dependencies(self, deps: List[str]) -> str:
        """Format dependency list."""
        return "\\n".join(f"- {dep}" for dep in deps)

# Usage
readme_gen = ReadmeGenerator()

readme = readme_gen.generate_readme(
    project_name="Code Analyzer",
    description="A tool for analyzing Python code quality and complexity",
    main_files=[
        "analyzer.py - Main analysis engine",
        "metrics.py - Code metrics calculator",
        "reporter.py - Generate analysis reports"
    ],
    dependencies=[
        "ast (standard library)",
        "pylint",
        "radon",
        "click"
    ]
)

print(readme)
\`\`\`

## API Documentation

### Generate API Docs

\`\`\`python
class APIDocGenerator:
    """Generate API documentation."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_api_docs(
        self,
        api_routes: List[dict]  # List of route information
    ) -> str:
        """Generate API documentation from routes."""
        
        routes_info = self._format_routes(api_routes)
        
        prompt = f"""Generate API documentation for these routes:

{routes_info}

For each endpoint, document:
1. HTTP method and path
2. Description
3. Request parameters
4. Request body schema (if applicable)
5. Response schema
6. Example request/response
7. Error codes

Use Markdown format suitable for API docs.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at API documentation."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    def _format_routes(self, routes: List[dict]) -> str:
        """Format routes information."""
        lines = []
        for route in routes:
            lines.append(f"- {route['method']} {route['path']}")
            if 'function' in route:
                lines.append(f"  Handler: {route['function']}")
        return "\\n".join(lines)

# Usage
api_doc_gen = APIDocGenerator()

routes = [
    {
        "method": "POST",
        "path": "/api/users",
        "function": "create_user"
    },
    {
        "method": "GET",
        "path": "/api/users/{id}",
        "function": "get_user"
    },
    {
        "method": "PUT",
        "path": "/api/users/{id}",
        "function": "update_user"
    }
]

api_docs = api_doc_gen.generate_api_docs(routes)
print(api_docs)
\`\`\`

## Type Hint Addition

### Add Type Hints to Code

\`\`\`python
class TypeHintAdder:
    """Add type hints to Python code."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def add_type_hints(
        self,
        code: str,
        use_typing: bool = True
    ) -> str:
        """Add comprehensive type hints to code."""
        
        prompt = f"""Add comprehensive type hints to this Python code:

\`\`\`python
{code}
\`\`\`

Requirements:
- Add type hints to all function parameters
- Add return type hints
- {"Use typing module (List, Dict, Optional, etc.)" if use_typing else "Use built-in types where possible"}
- Add type hints to class attributes
- Use Union for multiple types
- Use Optional for nullable values

Output the code with type hints added.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at Python type hints."
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
type_hint_adder = TypeHintAdder()

code = """
def process_orders(orders, discount_rate=0.1):
    results = []
    for order in orders:
        total = sum(item['price'] * item['quantity'] for item in order['items'])
        discounted = total * (1 - discount_rate)
        results.append({
            'order_id': order['id'],
            'total': discounted
        })
    return results
"""

typed_code = type_hint_adder.add_type_hints(code)
print(typed_code)

# Output:
# from typing import List, Dict, Any
#
# def process_orders(
#     orders: List[Dict[str, Any]],
#     discount_rate: float = 0.1
# ) -> List[Dict[str, Any]]:
#     results: List[Dict[str, Any]] = []
#     for order in orders:
#         total: float = sum(
#             item['price'] * item['quantity']
#             for item in order['items']
#         )
#         discounted: float = total * (1 - discount_rate)
#         results.append({
#             'order_id': order['id'],
#             'total': discounted
#         })
#     return results
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Use consistent style** (Google, NumPy, or Sphinx)
2. **Explain WHY**, not just what
3. **Add examples** to docstrings
4. **Document all parameters** and return values
5. **Include type hints** in documentation
6. **Keep comments up-to-date**
7. **Document exceptions** that can be raised
8. **Generate README** for all projects

### ❌ DON'T:
1. **Over-comment obvious code**
2. **Write vague descriptions**
3. **Skip parameter documentation**
4. **Forget return value documentation**
5. **Use inconsistent styles**
6. **Generate outdated information**
7. **Skip usage examples**
8. **Forget to document exceptions**

## Next Steps

You've mastered documentation generation! Next:
- Code review and bug detection
- Building complete code generation systems

Remember: **Good Documentation = Maintainable Code**
`,
};


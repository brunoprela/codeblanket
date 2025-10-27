/**
 * Code Refactoring with LLMs Section
 * Module 5: Building Code Generation Systems
 */

export const coderefactoringllmsSection = {
  id: 'code-refactoring-llms',
  title: 'Code Refactoring with LLMs',
  content: `# Code Refactoring with LLMs

Master automated code refactoring using LLMs - rename variables, extract functions, and restructure code safely.

## Overview: Automated Refactoring

Refactoring is transforming code structure without changing behavior. LLMs can:
- Rename variables/functions across files
- Extract functions from code blocks
- Inline small functions
- Move methods between classes
- Update all references automatically

### Why LLM Refactoring is Powerful

**Traditional IDEs**: Rule-based, limited to known patterns
**LLM Refactoring**: Understands context, handles edge cases, maintains style

## Rename Refactoring

### Intelligent Rename Across Files

\`\`\`python
import ast
from typing import List, Dict, Set, Tuple
from pathlib import Path

class RenameRefactorer:
    """Rename variables, functions, or classes across codebase."""
    
    def __init__(self, project_root: str):
        self.project_root = Path (project_root)
        self.analyzer = ProjectAnalyzer (project_root)
    
    def rename_symbol(
        self,
        symbol_name: str,
        new_name: str,
        file_path: Optional[str] = None,
        scope: str = "project"  # "file", "class", "function", "project"
    ) -> Dict[str, List[SearchReplace]]:
        """
        Rename a symbol across appropriate scope.
        
        Returns:
            Dict mapping file paths to list of edits
        """
        # Find all occurrences
        occurrences = self._find_symbol_occurrences(
            symbol_name,
            file_path,
            scope
        )
        
        # Group by file
        edits_by_file = {}
        for occurrence in occurrences:
            file = occurrence['file']
            if file not in edits_by_file:
                edits_by_file[file] = []
            
            # Create search/replace edit
            edit = SearchReplace(
                search=occurrence['context'],
                replace=occurrence['context'].replace (symbol_name, new_name)
            )
            edits_by_file[file].append (edit)
        
        return edits_by_file
    
    def _find_symbol_occurrences(
        self,
        symbol_name: str,
        file_path: Optional[str],
        scope: str
    ) -> List[Dict]:
        """Find all occurrences of symbol."""
        occurrences = []
        
        if scope == "file" and file_path:
            # Search only in one file
            files_to_search = [file_path]
        elif scope == "project":
            # Search all Python files
            files_to_search = [
                str (p.relative_to (self.project_root))
                for p in self.project_root.rglob("*.py")
                if "__pycache__" not in str (p)
            ]
        else:
            files_to_search = []
        
        for file in files_to_search:
            full_path = self.project_root / file
            try:
                with open (full_path) as f:
                    content = f.read()
                
                tree = ast.parse (content)
                
                # Find all Name nodes with this identifier
                for node in ast.walk (tree):
                    if isinstance (node, ast.Name) and node.id == symbol_name:
                        # Get context (the line it's on)
                        lines = content.split("\\n")
                        if hasattr (node, 'lineno'):
                            line = lines[node.lineno - 1]
                            
                            occurrences.append({
                                'file': file,
                                'line': node.lineno,
                                'context': line,
                                'col': node.col_offset
                            })
            
            except (IOError, SyntaxError):
                continue
        
        return occurrences

# Usage
refactorer = RenameRefactorer("/path/to/project")

edits = refactorer.rename_symbol(
    symbol_name="old_name",
    new_name="new_name",
    scope="project"
)

for file_path, file_edits in edits.items():
    print(f"\\n{file_path}:")
    for edit in file_edits:
        print(f"  {edit}")
\`\`\`

## Extract Function

### LLM-Powered Function Extraction

\`\`\`python
from openai import OpenAI

class FunctionExtractor:
    """Extract code blocks into separate functions."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def extract_function(
        self,
        file_content: str,
        start_line: int,
        end_line: int,
        new_function_name: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Extract lines into a new function.
        
        Returns:
            (new_file_content, extracted_function_code)
        """
        lines = file_content.split("\\n")
        
        # Get code to extract
        code_to_extract = "\\n".join (lines[start_line-1:end_line])
        
        # Get surrounding context
        context_before = "\\n".join (lines[max(0, start_line-10):start_line-1])
        context_after = "\\n".join (lines[end_line:min (len (lines), end_line+10)])
        
        # Generate function
        prompt = self._build_extraction_prompt(
            code_to_extract,
            context_before,
            context_after,
            new_function_name
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at refactoring code."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        # Parse response
        result = self._parse_extraction_result(
            response.choices[0].message.content
        )
        
        # Build new file content
        new_content = self._replace_with_function_call(
            lines,
            start_line,
            end_line,
            result['function_call'],
            result['function_definition']
        )
        
        return new_content, result['function_definition']
    
    def _build_extraction_prompt(
        self,
        code_to_extract: str,
        context_before: str,
        context_after: str,
        function_name: Optional[str]
    ) -> str:
        """Build prompt for extraction."""
        name_instruction = f"Name it: {function_name}" if function_name else \
                          "Choose an appropriate name"
        
        return f"""Extract this code into a function:

CODE TO EXTRACT:
{code_to_extract}

CONTEXT BEFORE:
{context_before}

CONTEXT AFTER:
{context_after}

Requirements:
1. {name_instruction}
2. Identify all variables from context needed as parameters
3. Determine what should be returned
4. Add type hints
5. Add docstring
6. Provide the function definition
7. Provide the function call to replace extracted code

Output as JSON:
{{
    "function_name": "...",
    "function_definition": "def ...",
    "function_call": "result = function_name(...)",
    "parameters": ["param1", "param2"],
    "return_value": "what is returned"
}}
"""
    
    def _parse_extraction_result (self, response: str) -> Dict:
        """Parse LLM response."""
        import json
        
        # Extract JSON from response
        if "\`\`\`json" in response:
            parts = response.split("\`\`\`json")
            if len (parts) > 1:
                json_str = parts[1].split("\`\`\`")[0]
                return json.loads (json_str.strip())
        
        # Try to parse as JSON directly
        return json.loads (response)
    
    def _replace_with_function_call(
        self,
        lines: List[str],
        start_line: int,
        end_line: int,
        function_call: str,
        function_definition: str
    ) -> str:
        """Replace extracted code with function call."""
        # Remove extracted lines
        before = lines[:start_line-1]
        after = lines[end_line:]
        
        # Add function call
        indent = self._get_indentation (lines[start_line-1])
        indented_call = "\\n".join(
            indent + line for line in function_call.split("\\n")
        )
        
        middle = [indented_call]
        
        # Add function definition at appropriate location
        # (Usually before the function containing the extracted code)
        insertion_point = self._find_function_insertion_point (before)
        
        before.insert (insertion_point, "")
        before.insert (insertion_point + 1, function_definition)
        before.insert (insertion_point + 2, "")
        
        return "\\n".join (before + middle + after)
    
    def _get_indentation (self, line: str) -> str:
        """Get indentation of a line."""
        return line[:len (line) - len (line.lstrip())]
    
    def _find_function_insertion_point (self, lines: List[str]) -> int:
        """Find where to insert new function definition."""
        # Find last function definition
        for i in range (len (lines) - 1, -1, -1):
            if lines[i].strip().startswith("def "):
                # Find end of that function
                for j in range (i + 1, len (lines)):
                    if lines[j].strip() and not lines[j][0].isspace():
                        return j
                return len (lines)
        
        return 0  # Insert at top if no functions found

# Usage
extractor = FunctionExtractor()

file_content = """
def process_data (items):
    results = []
    for item in items:
        # Complex processing logic (lines to extract)
        cleaned = item.strip().lower()
        parts = cleaned.split(',')
        validated = [p for p in parts if len (p) > 3]
        results.extend (validated)
    return results
"""

new_content, extracted_func = extractor.extract_function(
    file_content,
    start_line=5,  # Start of extraction
    end_line=8,    # End of extraction
    new_function_name="clean_and_validate_item"
)

print("Extracted function:")
print(extracted_func)
print("\\nNew file content:")
print(new_content)
\`\`\`

## Inline Function

### Inline Small Functions

\`\`\`python
class FunctionInliner:
    """Inline function calls with their definitions."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def inline_function(
        self,
        file_content: str,
        function_name: str
    ) -> str:
        """Inline all calls to a function."""
        # Parse to find function definition
        tree = ast.parse (file_content)
        
        func_def = None
        for node in ast.walk (tree):
            if isinstance (node, ast.FunctionDef) and node.name == function_name:
                func_def = node
                break
        
        if not func_def:
            raise ValueError (f"Function {function_name} not found")
        
        # Get function body
        func_body = ast.unparse (func_def)
        
        # Find all calls to this function
        calls = self._find_function_calls (tree, function_name)
        
        # For each call, inline it
        new_content = file_content
        for call in calls:
            inlined = self._inline_call (func_body, call)
            new_content = new_content.replace (call['original'], inlined)
        
        # Remove function definition
        new_content = self._remove_function_definition (new_content, function_name)
        
        return new_content
    
    def _find_function_calls(
        self,
        tree: ast.AST,
        function_name: str
    ) -> List[Dict]:
        """Find all calls to function."""
        calls = []
        
        for node in ast.walk (tree):
            if isinstance (node, ast.Call):
                if isinstance (node.func, ast.Name) and node.func.id == function_name:
                    calls.append({
                        'node': node,
                        'original': ast.unparse (node),
                        'args': [ast.unparse (arg) for arg in node.args]
                    })
        
        return calls
    
    def _inline_call (self, func_body: str, call: Dict) -> str:
        """Inline a specific function call."""
        # Use LLM to intelligently inline
        prompt = f"""Inline this function call:

Function definition:
{func_body}

Function call:
{call['original']}

Replace the function call with its body, substituting arguments appropriately.
Output only the inlined code.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at code refactoring."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
    
    def _remove_function_definition(
        self,
        content: str,
        function_name: str
    ) -> str:
        """Remove function definition from code."""
        lines = content.split("\\n")
        tree = ast.parse (content)
        
        # Find function line range
        for node in ast.walk (tree):
            if isinstance (node, ast.FunctionDef) and node.name == function_name:
                start_line = node.lineno - 1
                end_line = node.end_lineno
                
                # Remove those lines
                del lines[start_line:end_line]
                break
        
        return "\\n".join (lines)

# Usage
inliner = FunctionInliner()

file_content = """
def add_one (x):
    return x + 1

def process():
    value = add_one(5)
    result = add_one (value)
    return result
"""

inlined = inliner.inline_function (file_content, "add_one")
print(inlined)
\`\`\`

## Move Method

### Move Methods Between Classes

\`\`\`python
class MethodMover:
    """Move methods between classes."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def move_method(
        self,
        file_content: str,
        method_name: str,
        from_class: str,
        to_class: str
    ) -> str:
        """Move a method from one class to another."""
        tree = ast.parse (file_content)
        
        # Find source method
        source_method = self._find_method (tree, from_class, method_name)
        if not source_method:
            raise ValueError (f"Method {method_name} not found in {from_class}")
        
        # Find target class
        target_class = self._find_class (tree, to_class)
        if not target_class:
            raise ValueError (f"Class {to_class} not found")
        
        # Adapt method for new class
        adapted_method = self._adapt_method(
            ast.unparse (source_method),
            from_class,
            to_class,
            file_content
        )
        
        # Remove from source class
        content = self._remove_method (file_content, from_class, method_name)
        
        # Add to target class
        content = self._add_method (content, to_class, adapted_method)
        
        return content
    
    def _find_method(
        self,
        tree: ast.AST,
        class_name: str,
        method_name: str
    ) -> Optional[ast.FunctionDef]:
        """Find method in class."""
        for node in ast.walk (tree):
            if isinstance (node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance (item, ast.FunctionDef) and item.name == method_name:
                        return item
        return None
    
    def _find_class(
        self,
        tree: ast.AST,
        class_name: str
    ) -> Optional[ast.ClassDef]:
        """Find class definition."""
        for node in ast.walk (tree):
            if isinstance (node, ast.ClassDef) and node.name == class_name:
                return node
        return None
    
    def _adapt_method(
        self,
        method_code: str,
        from_class: str,
        to_class: str,
        context: str
    ) -> str:
        """Adapt method for new class using LLM."""
        prompt = f"""Move this method from {from_class} to {to_class}:

Method:
{method_code}

Context (full file):
{context}

Adapt the method:
1. Update self references if needed
2. Add parameters for any references to old class
3. Update docstring
4. Maintain functionality

Output the adapted method code.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at refactoring code."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
    
    def _remove_method(
        self,
        content: str,
        class_name: str,
        method_name: str
    ) -> str:
        """Remove method from class."""
        tree = ast.parse (content)
        method = self._find_method (tree, class_name, method_name)
        
        if method:
            lines = content.split("\\n")
            start = method.lineno - 1
            end = method.end_lineno
            del lines[start:end]
            return "\\n".join (lines)
        
        return content
    
    def _add_method(
        self,
        content: str,
        class_name: str,
        method_code: str
    ) -> str:
        """Add method to class."""
        lines = content.split("\\n")
        tree = ast.parse (content)
        
        target_class = self._find_class (tree, class_name)
        if target_class:
            # Find insertion point (end of class)
            insert_line = target_class.end_lineno - 1
            
            # Add method with proper indentation
            method_lines = method_code.split("\\n")
            indented = ["    " + line for line in method_lines]
            
            lines = lines[:insert_line] + [""] + indented + lines[insert_line:]
        
        return "\\n".join (lines)

# Usage
mover = MethodMover()

moved = mover.move_method(
    file_content,
    method_name="validate",
    from_class="User",
    to_class="UserValidator"
)
\`\`\`

## Change Signature

### Update Function Signatures

\`\`\`python
class SignatureChanger:
    """Change function signatures and update all calls."""
    
    def __init__(self, analyzer: ProjectAnalyzer):
        self.analyzer = analyzer
        self.client = OpenAI()
    
    def change_signature(
        self,
        function_name: str,
        file_path: str,
        new_parameters: List[Tuple[str, str]],  # (name, type) pairs
        new_return_type: Optional[str] = None
    ) -> Dict[str, List[SearchReplace]]:
        """
        Change function signature and update all calls.
        
        Returns:
            Dict mapping files to edits
        """
        # Find function definition
        with open (self.analyzer.project_root / file_path) as f:
            content = f.read()
        
        tree = ast.parse (content)
        func_def = self._find_function (tree, function_name)
        
        if not func_def:
            raise ValueError (f"Function {function_name} not found")
        
        # Generate new signature
        old_signature = ast.unparse (func_def)
        new_signature = self._generate_new_signature(
            function_name,
            new_parameters,
            new_return_type
        )
        
        edits = {}
        
        # Update definition
        edits[file_path] = [SearchReplace(
            search=old_signature.split(":\\n")[0] + ":",  # Just the signature line
            replace=new_signature + ":"
        )]
        
        # Find and update all calls
        call_updates = self._update_all_calls(
            function_name,
            func_def,
            new_parameters
        )
        
        for file, file_edits in call_updates.items():
            if file in edits:
                edits[file].extend (file_edits)
            else:
                edits[file] = file_edits
        
        return edits
    
    def _find_function(
        self,
        tree: ast.AST,
        function_name: str
    ) -> Optional[ast.FunctionDef]:
        """Find function definition."""
        for node in ast.walk (tree):
            if isinstance (node, ast.FunctionDef) and node.name == function_name:
                return node
        return None
    
    def _generate_new_signature(
        self,
        name: str,
        parameters: List[Tuple[str, str]],
        return_type: Optional[str]
    ) -> str:
        """Generate new function signature."""
        params = ", ".join (f"{name}: {type_}" for name, type_ in parameters)
        sig = f"def {name}({params})"
        
        if return_type:
            sig += f" -> {return_type}"
        
        return sig
    
    def _update_all_calls(
        self,
        function_name: str,
        old_func_def: ast.FunctionDef,
        new_parameters: List[Tuple[str, str]]
    ) -> Dict[str, List[SearchReplace]]:
        """Find and update all calls to function."""
        edits = {}
        
        # Find all files that might call this function
        for py_file in self.analyzer.project_root.rglob("*.py"):
            if "__pycache__" in str (py_file):
                continue
            
            try:
                with open (py_file) as f:
                    content = f.read()
                
                tree = ast.parse (content)
                
                # Find calls
                for node in ast.walk (tree):
                    if isinstance (node, ast.Call):
                        if isinstance (node.func, ast.Name) and \
                           node.func.id == function_name:
                            # Found a call - update it
                            old_call = ast.unparse (node)
                            new_call = self._adapt_call(
                                old_call,
                                old_func_def,
                                new_parameters
                            )
                            
                            relative_path = str (py_file.relative_to(
                                self.analyzer.project_root
                            ))
                            
                            if relative_path not in edits:
                                edits[relative_path] = []
                            
                            edits[relative_path].append(SearchReplace(
                                search=old_call,
                                replace=new_call
                            ))
            
            except (IOError, SyntaxError):
                continue
        
        return edits
    
    def _adapt_call(
        self,
        old_call: str,
        old_func_def: ast.FunctionDef,
        new_parameters: List[Tuple[str, str]]
    ) -> str:
        """Adapt a function call to new signature."""
        # Use LLM for complex adaptations
        old_params = [arg.arg for arg in old_func_def.args.args]
        new_params = [name for name, _ in new_parameters]
        
        prompt = f"""Update this function call for new signature:

Old parameters: {old_params}
New parameters: {new_params}

Old call:
{old_call}

Generate the updated call.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at refactoring code."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()

# Usage
analyzer = ProjectAnalyzer("/path/to/project")
changer = SignatureChanger (analyzer)

edits = changer.change_signature(
    function_name="process_user",
    file_path="app/services/user.py",
    new_parameters=[
        ("user_id", "int"),
        ("options", "Dict[str, Any]")
    ],
    new_return_type="UserResult"
)

for file, file_edits in edits.items():
    print(f"\\n{file}:")
    for edit in file_edits:
        print(f"  {edit}")
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Analyze before refactoring** - understand dependencies
2. **Update all references** automatically
3. **Preserve behavior** - only change structure
4. **Test after refactoring**5. **Use transactional application** with rollback
6. **Generate descriptive names** for extracted functions
7. **Maintain code style** throughout
8. **Update imports** as needed

### ❌ DON'T:
1. **Refactor without understanding context**2. **Break existing functionality**3. **Rename without updating all references**4. **Extract without proper parameter analysis**5. **Skip validation** after changes
6. **Ignore type information**7. **Make changes without backups**8. **Forget to update documentation**

## Next Steps

You've mastered refactoring! Next:
- Test generation
- Code review systems
- Building complete code generation platforms

Remember: **Structure Changes + Behavior Preservation = Safe Refactoring**
`,
};

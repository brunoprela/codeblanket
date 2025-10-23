/**
 * Multi-File Code Generation Section
 * Module 5: Building Code Generation Systems
 */

export const multifilecodegenerationSection = {
  id: 'multi-file-code-generation',
  title: 'Multi-File Code Generation',
  content: `# Multi-File Code Generation

Master generating and coordinating changes across multiple files - essential for real-world code generation systems.

## Overview: Why Multi-File Generation is Hard

Real software isn't single files. It's interconnected systems where:
- Files import from each other
- Changes propagate across boundaries
- Consistency must be maintained
- Order of changes matters

### The Challenges

**Dependency Management**
- Files depend on each other
- Changes in one file affect others
- Import statements must be updated
- Type definitions must stay consistent

**Consistency**
- Function signatures across files
- Shared types and interfaces
- Naming conventions
- Style consistency

**Order of Operations**
- What gets created first?
- What gets modified in what sequence?
- How to handle circular dependencies?

## Multi-File Architecture

### Project-Aware Generator

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from pathlib import Path
import ast

@dataclass
class FileChange:
    """Represents a change to a file."""
    path: str
    operation: str  # "create", "modify", "delete"
    content: Optional[str] = None
    edits: Optional[List[SearchReplace]] = None

@dataclass
class MultiFileSpec:
    """Specification for multi-file generation."""
    description: str
    files: List[FileChange]
    project_context: Optional[str] = None

class ProjectAnalyzer:
    """Analyze project structure and dependencies."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.file_tree = self._build_file_tree()
        self.import_graph = self._build_import_graph()
    
    def _build_file_tree(self) -> Dict[str, List[str]]:
        """Build project file tree."""
        tree = {}
        
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            relative = py_file.relative_to(self.project_root)
            dir_path = str(relative.parent)
            
            if dir_path not in tree:
                tree[dir_path] = []
            tree[dir_path].append(py_file.name)
        
        return tree
    
    def _build_import_graph(self) -> Dict[str, Set[str]]:
        """Build import dependency graph."""
        graph = {}
        
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            try:
                with open(py_file) as f:
                    tree = ast.parse(f.read())
                
                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.add(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module)
                
                relative = str(py_file.relative_to(self.project_root))
                graph[relative] = imports
            
            except (SyntaxError, IOError):
                continue
        
        return graph
    
    def get_dependent_files(self, file_path: str) -> List[str]:
        """Get files that depend on the given file."""
        dependents = []
        
        # Convert file path to module path
        module_path = file_path.replace("/", ".").replace(".py", "")
        
        for file, imports in self.import_graph.items():
            if module_path in imports:
                dependents.append(file)
        
        return dependents
    
    def get_required_imports(self, file_path: str) -> Set[str]:
        """Get imports required by a file."""
        return self.import_graph.get(file_path, set())
    
    def find_related_files(
        self,
        file_path: str,
        max_depth: int = 2
    ) -> List[str]:
        """Find files related to given file."""
        related = set()
        to_visit = [(file_path, 0)]
        visited = set()
        
        while to_visit:
            current, depth = to_visit.pop(0)
            
            if current in visited or depth > max_depth:
                continue
            
            visited.add(current)
            
            # Add files this imports
            imports = self.get_required_imports(current)
            for imp in imports:
                imp_file = imp.replace(".", "/") + ".py"
                if self.project_root.joinpath(imp_file).exists():
                    related.add(imp_file)
                    to_visit.append((imp_file, depth + 1))
            
            # Add files that import this
            dependents = self.get_dependent_files(current)
            related.update(dependents)
        
        return list(related)

# Usage
analyzer = ProjectAnalyzer("/path/to/project")

# Find what depends on a file
dependents = analyzer.get_dependent_files("app/models/user.py")
print(f"Files that depend on user.py: {dependents}")

# Find related files
related = analyzer.find_related_files("app/routes/auth.py")
print(f"Related files: {related}")
\`\`\`

## Cross-File Context Building

### Build Multi-File Context

\`\`\`python
class MultiFileContextBuilder:
    """Build context spanning multiple files."""
    
    def __init__(self, analyzer: ProjectAnalyzer):
        self.analyzer = analyzer
    
    def build_context(
        self,
        target_files: List[str],
        include_related: bool = True
    ) -> str:
        """Build comprehensive context for multiple files."""
        context_parts = []
        
        # Project structure
        context_parts.append("# Project Structure")
        context_parts.append(self._format_file_tree())
        context_parts.append("")
        
        # Target files
        context_parts.append("# Target Files")
        for file_path in target_files:
            content = self._read_file(file_path)
            context_parts.append(f"## {file_path}")
            context_parts.append(content)
            context_parts.append("")
        
        # Related files (if requested)
        if include_related:
            related_files = set()
            for file_path in target_files:
                related = self.analyzer.find_related_files(file_path, max_depth=1)
                related_files.update(related)
            
            # Remove target files from related
            related_files -= set(target_files)
            
            if related_files:
                context_parts.append("# Related Files (for context)")
                for file_path in related_files:
                    # Include just signatures, not full content
                    signatures = self._extract_signatures(file_path)
                    context_parts.append(f"## {file_path}")
                    context_parts.append(signatures)
                    context_parts.append("")
        
        return "\\n".join(context_parts)
    
    def _format_file_tree(self) -> str:
        """Format project file tree."""
        tree_lines = []
        for dir_path, files in sorted(self.analyzer.file_tree.items()):
            tree_lines.append(f"{dir_path}/")
            for file in sorted(files):
                tree_lines.append(f"  {file}")
        return "\\n".join(tree_lines)
    
    def _read_file(self, file_path: str) -> str:
        """Read file content with line numbers."""
        full_path = self.analyzer.project_root / file_path
        try:
            with open(full_path) as f:
                lines = f.readlines()
            
            numbered = [
                f"{i+1:4d} | {line.rstrip()}"
                for i, line in enumerate(lines)
            ]
            return "\\n".join(numbered)
        except IOError:
            return "[File not found]"
    
    def _extract_signatures(self, file_path: str) -> str:
        """Extract function and class signatures."""
        full_path = self.analyzer.project_root / file_path
        try:
            with open(full_path) as f:
                tree = ast.parse(f.read())
            
            signatures = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = [arg.arg for arg in node.args.args]
                    sig = f"def {node.name}({', '.join(args)})"
                    if node.returns:
                        sig += f" -> {ast.unparse(node.returns)}"
                    signatures.append(sig)
                
                elif isinstance(node, ast.ClassDef):
                    signatures.append(f"class {node.name}")
            
            return "\\n".join(signatures)
        except (IOError, SyntaxError):
            return "[Could not extract signatures]"

# Usage
analyzer = ProjectAnalyzer("/path/to/project")
builder = MultiFileContextBuilder(analyzer)

context = builder.build_context(
    target_files=["app/models/user.py", "app/routes/auth.py"],
    include_related=True
)

print(context)
\`\`\`

## Multi-File Generation Strategy

### Plan and Execute Multi-File Changes

\`\`\`python
from openai import OpenAI
from typing import Tuple

class MultiFileGenerator:
    """Generate changes across multiple files."""
    
    def __init__(self, analyzer: ProjectAnalyzer):
        self.analyzer = analyzer
        self.context_builder = MultiFileContextBuilder(analyzer)
        self.client = OpenAI()
    
    def generate_multi_file_changes(
        self,
        description: str,
        target_files: List[str]
    ) -> List[FileChange]:
        """Generate coordinated changes across files."""
        
        # Step 1: Build context
        context = self.context_builder.build_context(
            target_files,
            include_related=True
        )
        
        # Step 2: Generate change plan
        plan = self._generate_plan(description, context, target_files)
        
        # Step 3: Generate changes for each file
        changes = []
        for file_change in plan:
            if file_change.operation == "create":
                content = self._generate_new_file(
                    file_change.path,
                    description,
                    context
                )
                changes.append(FileChange(
                    path=file_change.path,
                    operation="create",
                    content=content
                ))
            
            elif file_change.operation == "modify":
                edits = self._generate_edits(
                    file_change.path,
                    description,
                    context
                )
                changes.append(FileChange(
                    path=file_change.path,
                    operation="modify",
                    edits=edits
                ))
        
        return changes
    
    def _generate_plan(
        self,
        description: str,
        context: str,
        target_files: List[str]
    ) -> List[FileChange]:
        """Generate a plan for multi-file changes."""
        
        prompt = f"""You are planning multi-file code changes.

{context}

Task: {description}

Target files: {', '.join(target_files)}

Generate a plan listing:
1. Files to create (if any)
2. Files to modify (with brief description of changes)
3. Files to delete (if any)

Format as JSON:
{{
    "changes": [
        {{"path": "file.py", "operation": "create", "reason": "..."}},
        {{"path": "other.py", "operation": "modify", "reason": "..."}}
    ]
}}
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a code architect planning multi-file changes."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        import json
        plan_data = json.loads(response.choices[0].message.content)
        
        changes = []
        for change in plan_data.get("changes", []):
            changes.append(FileChange(
                path=change["path"],
                operation=change["operation"]
            ))
        
        return changes
    
    def _generate_new_file(
        self,
        file_path: str,
        description: str,
        context: str
    ) -> str:
        """Generate a new file."""
        prompt = f"""Create a new Python file: {file_path}

Project context:
{context}

Task: {description}

Generate complete file content following project conventions.
Include all necessary imports.
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
    
    def _generate_edits(
        self,
        file_path: str,
        description: str,
        context: str
    ) -> List[SearchReplace]:
        """Generate edits for existing file."""
        prompt = f"""Edit file: {file_path}

{context}

Task: {description}

Generate SEARCH/REPLACE blocks for necessary changes.
Format:
<<<<<<< SEARCH
[exact text]
=======
[new text]
>>>>>>> REPLACE
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise code editor."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        parser = SearchReplaceParser()
        return parser.parse(response.choices[0].message.content)
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        if "\`\`\`" in response:
            parts = response.split("\`\`\`")
            if len(parts) >= 3:
                return parts[1].strip()
        return response.strip()

# Usage
analyzer = ProjectAnalyzer("/path/to/project")
generator = MultiFileGenerator(analyzer)

changes = generator.generate_multi_file_changes(
    description="Add user authentication with JWT tokens",
    target_files=[
        "app/models/user.py",
        "app/routes/auth.py",
        "app/utils/jwt.py"
    ]
)

for change in changes:
    print(f"{change.operation}: {change.path}")
\`\`\`

## Applying Multi-File Changes

### Transactional Multi-File Application

\`\`\`python
class MultiFileApplicator:
    """Apply changes across multiple files transactionally."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.applicator = SafeEditApplicator()
    
    def apply_changes(
        self,
        changes: List[FileChange],
        dry_run: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Apply multiple file changes transactionally.
        
        If any change fails, all changes are rolled back.
        
        Returns:
            (success, error_messages)
        """
        backups = {}
        applied_files = []
        errors = []
        
        try:
            for change in changes:
                file_path = self.project_root / change.path
                
                if change.operation == "create":
                    if file_path.exists():
                        errors.append(f"File already exists: {change.path}")
                        raise Exception("Create failed")
                    
                    if not dry_run:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(change.content)
                    
                    applied_files.append(change.path)
                    print(f"✓ Created: {change.path}")
                
                elif change.operation == "modify":
                    if not file_path.exists():
                        errors.append(f"File not found: {change.path}")
                        raise Exception("Modify failed")
                    
                    # Backup original
                    original_content = file_path.read_text()
                    backups[change.path] = original_content
                    
                    # Apply edits
                    success, new_content, edit_errors = \
                        self.applicator.apply_multiple_edits(
                            original_content,
                            change.edits,
                            fuzzy=True
                        )
                    
                    if not success:
                        errors.extend(edit_errors)
                        raise Exception("Edit failed")
                    
                    if not dry_run:
                        file_path.write_text(new_content)
                    
                    applied_files.append(change.path)
                    print(f"✓ Modified: {change.path}")
                
                elif change.operation == "delete":
                    if not file_path.exists():
                        errors.append(f"File not found: {change.path}")
                        raise Exception("Delete failed")
                    
                    # Backup before delete
                    backups[change.path] = file_path.read_text()
                    
                    if not dry_run:
                        file_path.unlink()
                    
                    applied_files.append(change.path)
                    print(f"✓ Deleted: {change.path}")
            
            # All changes successful
            if dry_run:
                print("\\n[DRY RUN] No changes actually applied")
            
            return True, []
        
        except Exception as e:
            # Rollback changes
            print(f"\\n✗ Error occurred: {e}")
            print("Rolling back changes...")
            
            if not dry_run:
                self._rollback(backups, applied_files)
            
            return False, errors
    
    def _rollback(
        self,
        backups: Dict[str, str],
        applied_files: List[str]
    ):
        """Rollback applied changes."""
        for file_path in applied_files:
            full_path = self.project_root / file_path
            
            if file_path in backups:
                # Restore backup
                full_path.write_text(backups[file_path])
                print(f"  Restored: {file_path}")
            else:
                # Was created, delete it
                if full_path.exists():
                    full_path.unlink()
                    print(f"  Removed: {file_path}")

# Usage
applicator = MultiFileApplicator("/path/to/project")

# Apply changes with dry run first
success, errors = applicator.apply_changes(changes, dry_run=True)

if success:
    # Looks good, apply for real
    user_input = input("Apply changes? (yes/no): ")
    if user_input.lower() == "yes":
        success, errors = applicator.apply_changes(changes, dry_run=False)
        
        if success:
            print("\\n✓ All changes applied successfully!")
        else:
            print("\\n✗ Changes rolled back")
            for error in errors:
                print(f"  - {error}")
\`\`\`

## Import Management

### Update Imports Across Files

\`\`\`python
class ImportManager:
    """Manage imports across multiple files."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
    
    def add_import(
        self,
        file_path: str,
        module: str,
        names: Optional[List[str]] = None
    ) -> str:
        """Add import to file."""
        full_path = self.project_root / file_path
        content = full_path.read_text()
        
        tree = ast.parse(content)
        
        # Check if import already exists
        if self._import_exists(tree, module, names):
            return content  # Already imported
        
        # Find where to insert import
        insert_line = self._find_import_position(tree)
        
        # Create import statement
        if names:
            import_stmt = f"from {module} import {', '.join(names)}"
        else:
            import_stmt = f"import {module}"
        
        # Insert import
        lines = content.split("\\n")
        lines.insert(insert_line, import_stmt)
        
        return "\\n".join(lines)
    
    def _import_exists(
        self,
        tree: ast.AST,
        module: str,
        names: Optional[List[str]]
    ) -> bool:
        """Check if import already exists."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == module:
                    if names is None:
                        return True
                    imported_names = {n.name for n in node.names}
                    if all(name in imported_names for name in names):
                        return True
            
            elif isinstance(node, ast.Import) and names is None:
                imported = {n.name for n in node.names}
                if module in imported:
                    return True
        
        return False
    
    def _find_import_position(self, tree: ast.AST) -> int:
        """Find appropriate line to insert import."""
        # Find last import statement
        last_import_line = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, 'lineno'):
                    last_import_line = max(last_import_line, node.lineno)
        
        # Insert after last import, or at top if no imports
        return last_import_line if last_import_line > 0 else 0
    
    def update_imports_for_new_file(
        self,
        new_file_path: str,
        dependent_files: List[str]
    ) -> Dict[str, str]:
        """Update imports in dependent files for new file."""
        updates = {}
        
        # Convert file path to module path
        module_path = new_file_path.replace("/", ".").replace(".py", "")
        
        # For each dependent file, add import if needed
        for dep_file in dependent_files:
            updated_content = self.add_import(
                dep_file,
                module_path
            )
            updates[dep_file] = updated_content
        
        return updates

# Usage
import_mgr = ImportManager("/path/to/project")

# Add import to file
updated = import_mgr.add_import(
    "app/routes/auth.py",
    "app.models.user",
    ["User", "UserRole"]
)

# Update imports for new file
updates = import_mgr.update_imports_for_new_file(
    "app/utils/jwt.py",
    ["app/routes/auth.py", "app/middleware/auth.py"]
)
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Analyze project structure** before generating
2. **Build comprehensive context** with related files
3. **Generate a plan** before making changes
4. **Apply changes transactionally** with rollback
5. **Update imports** automatically
6. **Validate each file** after changes
7. **Use dry-run mode** first
8. **Maintain consistency** across files

### ❌ DON'T:
1. **Generate files in isolation**
2. **Ignore dependencies** between files
3. **Apply changes without backups**
4. **Forget to update imports**
5. **Skip validation** after changes
6. **Make changes without a plan**
7. **Ignore circular dependencies**
8. **Break existing functionality**

## Next Steps

You've mastered multi-file generation! Next:
- Code refactoring systems
- Interactive editing
- Building complete code generation platforms

Remember: **Plan → Generate → Validate → Apply Transactionally**
`,
};

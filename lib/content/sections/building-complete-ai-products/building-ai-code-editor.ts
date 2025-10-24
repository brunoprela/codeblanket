export const buildingAICodeEditor = {
  title: 'Building an AI Code Editor (Cursor Clone)',
  id: 'building-ai-code-editor',
  content: `
# Building an AI Code Editor (Cursor Clone)

## Introduction

**Cursor** has revolutionized how developers write code by integrating AI directly into the development environment. In this section, we'll reverse-engineer and build a Cursor-like AI code editor from scratch—understanding every component from file system integration to diff generation and real-time code assistance.

By the end, you'll have built a functional AI code editor that can:
- Understand codebases by parsing file structures
- Provide context-aware code suggestions
- Generate and apply code edits from natural language
- Handle multi-file changes
- Stream responses in real-time
- Validate changes before applying

### What Makes Cursor Special?

Traditional code completion (like GitHub Copilot) provides autocomplete suggestions. **Cursor goes further**:

1. **Context Awareness**: Understands entire project structure, not just current file
2. **Natural Language Edits**: "Add error handling" → generates complete try/catch blocks
3. **Multi-File Changes**: Can modify multiple related files simultaneously
4. **Diff-Based Edits**: Shows exactly what will change before applying
5. **Chat Interface**: Conversational code assistance with history
6. **Code Understanding**: Answers questions about your codebase

### Architecture Overview

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                   Cursor Clone                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │    Client    │──────│  WebSocket   │               │
│  │  (Browser/   │      │   Server     │               │
│  │   Terminal)  │      └──────┬───────┘               │
│  └──────────────┘             │                        │
│                               │                        │
│        ┌──────────────────────┼──────────────────┐    │
│        │                      │                  │    │
│        ▼                      ▼                  ▼    │
│  ┌──────────┐        ┌──────────────┐    ┌─────────┐ │
│  │   File   │        │   Context    │    │   LLM   │ │
│  │  System  │        │   Manager    │    │ Service │ │
│  │  Watcher │        └──────────────┘    └─────────┘ │
│  └──────────┘               │                  │      │
│        │                    │                  │      │
│        ▼                    ▼                  ▼      │
│  ┌──────────┐        ┌──────────────┐    ┌─────────┐ │
│  │   AST    │        │   Prompt     │    │  Cache  │ │
│  │  Parser  │        │   Builder    │    │  Layer  │ │
│  └──────────┘        └──────────────┘    └─────────┘ │
│        │                    │                  │      │
│        └────────────────────┼──────────────────┘      │
│                             │                         │
│                             ▼                         │
│                     ┌──────────────┐                  │
│                     │     Diff     │                  │
│                     │  Generator   │                  │
│                     └──────────────┘                  │
│                             │                         │
│                             ▼                         │
│                     ┌──────────────┐                  │
│                     │   Applicator │                  │
│                     │  (with undo) │                  │
│                     └──────────────┘                  │
└─────────────────────────────────────────────────────────┘
\`\`\`

---

## File System Integration

### Watching for Changes

A code editor needs to detect file changes instantly:

\`\`\`python
"""
File System Watcher with Change Detection
"""

from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Callable, Set
import asyncio
import logging

logger = logging.getLogger(__name__)

class CodebaseWatcher(FileSystemEventHandler):
    """
    Watches codebase for file changes
    """
    
    def __init__(
        self,
        root_path: Path,
        on_change: Callable,
        ignored_patterns: Set[str] = None
    ):
        self.root_path = root_path
        self.on_change = on_change
        self.ignored_patterns = ignored_patterns or {
            '__pycache__', 'node_modules', '.git', 
            '.venv', 'venv', 'dist', 'build'
        }
        
    def should_ignore(self, path: str) -> bool:
        """Check if path should be ignored"""
        path_obj = Path(path)
        return any(
            ignored in path_obj.parts 
            for ignored in self.ignored_patterns
        )
    
    def on_modified(self, event):
        """Handle file modification"""
        if event.is_directory or self.should_ignore(event.src_path):
            return
        
        logger.info(f"File modified: {event.src_path}")
        asyncio.create_task(self.on_change(event.src_path, "modified"))
    
    def on_created(self, event):
        """Handle file creation"""
        if event.is_directory or self.should_ignore(event.src_path):
            return
        
        logger.info(f"File created: {event.src_path}")
        asyncio.create_task(self.on_change(event.src_path, "created"))
    
    def on_deleted(self, event):
        """Handle file deletion"""
        if event.is_directory or self.should_ignore(event.src_path):
            return
        
        logger.info(f"File deleted: {event.src_path}")
        asyncio.create_task(self.on_change(event.src_path, "deleted"))


class FileSystemManager:
    """
    Manages file system operations for code editor
    """
    
    def __init__(self, root_path: Path):
        self.root_path = Path(root_path)
        self.observer = None
        self.file_cache = {}
        
    async def start_watching(self, on_change: Callable):
        """Start watching for file changes"""
        self.observer = Observer()
        handler = CodebaseWatcher(self.root_path, on_change)
        self.observer.schedule(handler, str(self.root_path), recursive=True)
        self.observer.start()
        logger.info(f"Started watching: {self.root_path}")
    
    def stop_watching(self):
        """Stop watching for file changes"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
    
    async def read_file(self, file_path: Path) -> str:
        """Read file contents with caching"""
        file_path = self.root_path / file_path
        
        # Check cache
        if file_path in self.file_cache:
            return self.file_cache[file_path]
        
        # Read from disk
        try:
            content = file_path.read_text(encoding='utf-8')
            self.file_cache[file_path] = content
            return content
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise
    
    async def write_file(self, file_path: Path, content: str):
        """Write file contents"""
        file_path = self.root_path / file_path
        
        # Create directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        file_path.write_text(content, encoding='utf-8')
        
        # Update cache
        self.file_cache[file_path] = content
        
        logger.info(f"Wrote file: {file_path}")
    
    async def list_files(self, pattern: str = "**/*.py") -> list[Path]:
        """List files matching pattern"""
        files = []
        for file_path in self.root_path.glob(pattern):
            if file_path.is_file():
                rel_path = file_path.relative_to(self.root_path)
                files.append(rel_path)
        return files
    
    def get_file_tree(self) -> dict:
        """
        Get hierarchical file tree structure
        """
        def build_tree(path: Path) -> dict:
            tree = {
                "name": path.name,
                "path": str(path.relative_to(self.root_path)),
                "type": "directory" if path.is_dir() else "file"
            }
            
            if path.is_dir():
                children = []
                try:
                    for child in sorted(path.iterdir()):
                        # Skip ignored directories
                        if child.name.startswith('.') or child.name in {'node_modules', '__pycache__'}:
                            continue
                        children.append(build_tree(child))
                    tree["children"] = children
                except PermissionError:
                    pass
            
            return tree
        
        return build_tree(self.root_path)


# Usage Example
async def on_file_change(file_path: str, event_type: str):
    """Handle file change events"""
    print(f"File {event_type}: {file_path}")
    # Re-index file, update context, etc.

fs_manager = FileSystemManager(Path("/project/root"))
await fs_manager.start_watching(on_file_change)

# Get file tree for UI
file_tree = fs_manager.get_file_tree()
print(file_tree)
\`\`\`

### Code Parsing with AST

Understanding code structure is essential for context:

\`\`\`python
"""
Code Parser using AST (Abstract Syntax Tree)
"""

import ast
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class FunctionInfo:
    """Information about a function"""
    name: str
    line_start: int
    line_end: int
    docstring: Optional[str]
    parameters: List[str]
    return_type: Optional[str]
    decorators: List[str]

@dataclass
class ClassInfo:
    """Information about a class"""
    name: str
    line_start: int
    line_end: int
    docstring: Optional[str]
    methods: List[FunctionInfo]
    bases: List[str]

@dataclass
class ImportInfo:
    """Information about imports"""
    module: str
    names: List[str]
    is_from_import: bool


class CodeParser:
    """
    Parse Python code and extract structure
    """
    
    @staticmethod
    def parse_file(file_path: Path) -> dict:
        """
        Parse Python file and extract all information
        """
        content = file_path.read_text(encoding='utf-8')
        return CodeParser.parse_code(content)
    
    @staticmethod
    def parse_code(code: str) -> dict:
        """
        Parse Python code string
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"error": str(e), "valid": False}
        
        imports = CodeParser._extract_imports(tree)
        functions = CodeParser._extract_functions(tree)
        classes = CodeParser._extract_classes(tree)
        
        return {
            "valid": True,
            "imports": imports,
            "functions": functions,
            "classes": classes,
            "module_docstring": ast.get_docstring(tree)
        }
    
    @staticmethod
    def _extract_imports(tree: ast.AST) -> List[ImportInfo]:
        """Extract all imports"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[alias.asname or alias.name],
                        is_from_import=False
                    ))
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append(ImportInfo(
                    module=module,
                    names=names,
                    is_from_import=True
                ))
        
        return imports
    
    @staticmethod
    def _extract_functions(
        tree: ast.AST,
        class_methods: bool = False
    ) -> List[FunctionInfo]:
        """Extract function definitions"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip methods if not requested
                if not class_methods and CodeParser._is_class_method(node, tree):
                    continue
                
                functions.append(FunctionInfo(
                    name=node.name,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    docstring=ast.get_docstring(node),
                    parameters=[arg.arg for arg in node.args.args],
                    return_type=CodeParser._get_return_annotation(node),
                    decorators=[CodeParser._decorator_name(d) for d in node.decorator_list]
                ))
        
        return functions
    
    @staticmethod
    def _extract_classes(tree: ast.AST) -> List[ClassInfo]:
        """Extract class definitions"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(FunctionInfo(
                            name=item.name,
                            line_start=item.lineno,
                            line_end=item.end_lineno or item.lineno,
                            docstring=ast.get_docstring(item),
                            parameters=[arg.arg for arg in item.args.args],
                            return_type=CodeParser._get_return_annotation(item),
                            decorators=[CodeParser._decorator_name(d) for d in item.decorator_list]
                        ))
                
                classes.append(ClassInfo(
                    name=node.name,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    docstring=ast.get_docstring(node),
                    methods=methods,
                    bases=[CodeParser._get_base_name(base) for base in node.bases]
                ))
        
        return classes
    
    @staticmethod
    def _is_class_method(func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is a class method"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return True
        return False
    
    @staticmethod
    def _get_return_annotation(node: ast.FunctionDef) -> Optional[str]:
        """Get return type annotation"""
        if node.returns:
            return ast.unparse(node.returns)
        return None
    
    @staticmethod
    def _decorator_name(decorator: ast.expr) -> str:
        """Get decorator name"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            return ast.unparse(decorator.func)
        return ast.unparse(decorator)
    
    @staticmethod
    def _get_base_name(base: ast.expr) -> str:
        """Get base class name"""
        return ast.unparse(base)


# Usage
code = ''
from typing import List, Optional

class UserManager:
    """Manages user operations"""
    
    def __init__(self, db):
        self.db = db
    
    def get_user(self, user_id: int) -> Optional[dict]:
        """Get user by ID"""
        return self.db.get(user_id)
    
    def list_users(self) -> List[dict]:
        """List all users"""
        return self.db.all()
''

parsed = CodeParser.parse_code(code)
print(f"Classes: {[c.name for c in parsed['classes']]}")
print(f"Methods: {[m.name for m in parsed['classes'][0].methods]}")
print(f"Imports: {[i.module for i in parsed['imports']]}")
\`\`\`

---

## Context Management

The key to Cursor's intelligence is sending **relevant context** to the LLM, not the entire codebase.

### Smart Context Selection

\`\`\`python
"""
Context Manager - Intelligently selects relevant code context
"""

from dataclasses import dataclass
from typing import List, Set
from pathlib import Path
import tiktoken

@dataclass
class FileContext:
    """Context from a single file"""
    path: Path
    content: str
    relevance_score: float
    tokens: int

class ContextManager:
    """
    Manages code context for LLM prompts
    """
    
    def __init__(
        self,
        fs_manager: FileSystemManager,
        max_tokens: int = 8000
    ):
        self.fs_manager = fs_manager
        self.max_tokens = max_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    async def get_context_for_query(
        self,
        query: str,
        current_file: Path,
        cursor_position: int = None
    ) -> List[FileContext]:
        """
        Get relevant context for a user query
        """
        # Start with current file (highest priority)
        contexts = await self._get_current_file_context(current_file, cursor_position)
        
        # Add imported files
        imported_contexts = await self._get_imported_files(current_file)
        contexts.extend(imported_contexts)
        
        # Add related files (same directory, similar names)
        related_contexts = await self._get_related_files(current_file)
        contexts.extend(related_contexts)
        
        # Rank by relevance to query
        ranked_contexts = self._rank_by_relevance(query, contexts)
        
        # Select contexts within token budget
        selected = self._select_within_budget(ranked_contexts, self.max_tokens)
        
        return selected
    
    async def _get_current_file_context(
        self,
        file_path: Path,
        cursor_position: int = None
    ) -> List[FileContext]:
        """
        Get context from current file
        """
        content = await self.fs_manager.read_file(file_path)
        
        # If cursor position provided, prioritize surrounding code
        if cursor_position:
            content = self._get_surrounding_code(content, cursor_position)
        
        tokens = len(self.encoder.encode(content))
        
        return [FileContext(
            path=file_path,
            content=content,
            relevance_score=1.0,  # Current file always most relevant
            tokens=tokens
        )]
    
    async def _get_imported_files(self, file_path: Path) -> List[FileContext]:
        """
        Get context from imported files
        """
        content = await self.fs_manager.read_file(file_path)
        parsed = CodeParser.parse_code(content)
        
        contexts = []
        for import_info in parsed.get('imports', []):
            # Convert import to file path
            import_path = self._import_to_path(import_info.module)
            if import_path and import_path.exists():
                import_content = await self.fs_manager.read_file(import_path)
                
                # Extract only imported functions/classes
                relevant_content = self._extract_relevant_imports(
                    import_content,
                    import_info.names
                )
                
                tokens = len(self.encoder.encode(relevant_content))
                contexts.append(FileContext(
                    path=import_path,
                    content=relevant_content,
                    relevance_score=0.8,
                    tokens=tokens
                ))
        
        return contexts
    
    async def _get_related_files(self, file_path: Path) -> List[FileContext]:
        """
        Get context from related files (same directory, similar names)
        """
        contexts = []
        
        # Same directory
        directory = file_path.parent
        for sibling in directory.glob("*.py"):
            if sibling == file_path:
                continue
            
            content = await self.fs_manager.read_file(sibling)
            tokens = len(self.encoder.encode(content))
            
            contexts.append(FileContext(
                path=sibling,
                content=content,
                relevance_score=0.5,
                tokens=tokens
            ))
        
        # Test file
        test_file = self._get_test_file(file_path)
        if test_file and test_file.exists():
            content = await self.fs_manager.read_file(test_file)
            tokens = len(self.encoder.encode(content))
            
            contexts.append(FileContext(
                path=test_file,
                content=content,
                relevance_score=0.7,
                tokens=tokens
            ))
        
        return contexts
    
    def _rank_by_relevance(
        self,
        query: str,
        contexts: List[FileContext]
    ) -> List[FileContext]:
        """
        Rank contexts by relevance to query
        """
        # Simple keyword matching (in production, use embeddings)
        query_words = set(query.lower().split())
        
        for context in contexts:
            content_words = set(context.content.lower().split())
            
            # Calculate overlap
            overlap = len(query_words & content_words) / len(query_words) if query_words else 0
            
            # Adjust relevance score
            context.relevance_score *= (1 + overlap)
        
        # Sort by relevance
        return sorted(contexts, key=lambda x: x.relevance_score, reverse=True)
    
    def _select_within_budget(
        self,
        contexts: List[FileContext],
        max_tokens: int
    ) -> List[FileContext]:
        """
        Select contexts that fit within token budget
        """
        selected = []
        total_tokens = 0
        
        for context in contexts:
            if total_tokens + context.tokens <= max_tokens:
                selected.append(context)
                total_tokens += context.tokens
            else:
                # Try to include partial content
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 500:  # Minimum useful amount
                    truncated_content = self._truncate_to_tokens(
                        context.content,
                        remaining_tokens
                    )
                    selected.append(FileContext(
                        path=context.path,
                        content=truncated_content,
                        relevance_score=context.relevance_score,
                        tokens=remaining_tokens
                    ))
                break
        
        return selected
    
    def _get_surrounding_code(
        self,
        content: str,
        cursor_position: int,
        lines_before: int = 20,
        lines_after: int = 20
    ) -> str:
        """
        Extract code surrounding cursor position
        """
        lines = content.split('\\n')
        cursor_line = content[:cursor_position].count('\\n')
        
        start = max(0, cursor_line - lines_before)
        end = min(len(lines), cursor_line + lines_after + 1)
        
        return '\\n'.join(lines[start:end])
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoder.decode(truncated_tokens)
    
    def _import_to_path(self, module: str) -> Optional[Path]:
        """Convert import module to file path"""
        # Simple conversion: my.module -> my/module.py
        return self.fs_manager.root_path / f"{module.replace('.', '/')}.py"
    
    def _get_test_file(self, file_path: Path) -> Optional[Path]:
        """Get corresponding test file"""
        # Common patterns: test_file.py, file_test.py, tests/file.py
        patterns = [
            file_path.parent / f"test_{file_path.name}",
            file_path.parent / f"{file_path.stem}_test.py",
            file_path.parent / "tests" / file_path.name
        ]
        
        for pattern in patterns:
            if pattern.exists():
                return pattern
        
        return None
    
    def _extract_relevant_imports(
        self,
        content: str,
        imported_names: List[str]
    ) -> str:
        """
        Extract only the imported functions/classes from content
        """
        parsed = CodeParser.parse_code(content)
        relevant_lines = []
        
        lines = content.split('\\n')
        
        # Extract functions
        for func in parsed.get('functions', []):
            if func.name in imported_names:
                relevant_lines.extend(lines[func.line_start-1:func.line_end])
        
        # Extract classes
        for cls in parsed.get('classes', []):
            if cls.name in imported_names:
                relevant_lines.extend(lines[cls.line_start-1:cls.line_end])
        
        return '\\n'.join(relevant_lines) if relevant_lines else content[:1000]
\`\`\`

---

## Prompt Builder

Building effective prompts for code generation:

\`\`\`python
"""
Prompt Builder for Code Generation
"""

from typing import List, Optional
from dataclasses import dataclass

@dataclass
class CodeEdit:
    """Represents a code edit request"""
    file_path: Path
    instruction: str
    current_content: str
    contexts: List[FileContext]

class PromptBuilder:
    """
    Builds prompts for code generation
    """
    
    SYSTEM_PROMPT = """You are an expert software engineer helping to edit code.

Your task is to generate precise code edits based on user instructions.

Important rules:
1. Maintain existing code style and conventions
2. Only modify what's necessary to fulfill the instruction
3. Preserve comments and docstrings unless asked to change them
4. Keep formatting consistent with the existing code
5. Return changes in a structured format that can be automatically applied

When making changes:
- For small edits: return a search/replace block
- For new functions: specify exactly where to insert
- For deletions: clearly mark what to remove
- For multi-line changes: use proper indentation

Response format:
\`\`\`python
# FILE: path/ to / file.py
# ACTION: REPLACE
# SEARCH:
old code here
# REPLACE:
new code here
\`\`\`
"""
    
    def build_edit_prompt(self, edit: CodeEdit) -> str:
        """
        Build prompt for code editing
        """
        prompt_parts = []
        
        # System instruction
        prompt_parts.append(self.SYSTEM_PROMPT)
        
        # Current file
        prompt_parts.append(f"\\n## Current File: {edit.file_path}\\n")
        prompt_parts.append("\`\`\`python")
prompt_parts.append(edit.current_content)
prompt_parts.append("\`\`\`")
        
        # Context files
if edit.contexts:
    prompt_parts.append("\\n## Related Files (for context):\\n")
for ctx in edit.contexts:
    prompt_parts.append(f"### {ctx.path}")
prompt_parts.append("\`\`\`python")
prompt_parts.append(ctx.content)
prompt_parts.append("\`\`\`")
        
        # User instruction
prompt_parts.append(f"\\n## Instruction:\\n{edit.instruction}")
        
        # Request structured output
prompt_parts.append("\\nProvide the edit in the format specified above.")

return "\\n".join(prompt_parts)
    
    def build_chat_prompt(
    self,
    query: str,
    contexts: List[FileContext],
    conversation_history: List[dict] = None
) -> List[dict]:
"""
        Build chat prompt with context
        """
messages = [{ "role": "system", "content": self.SYSTEM_PROMPT }]
        
        # Add conversation history
if conversation_history:
    messages.extend(conversation_history)
        
        # Build user message with context
        user_message = []
        
        # Add context
if contexts:
    user_message.append("## Relevant Code Context:\\n")
for ctx in contexts:
    user_message.append(f"### {ctx.path}")
user_message.append(f"\`\`\`python\\n{ctx.content}\\n\`\`\`\\n")
        
        # Add query
user_message.append(f"## Question:\\n{query}")

messages.append({
    "role": "user",
    "content": "\\n".join(user_message)
})

return messages
    
    def build_function_generation_prompt(
    self,
    function_description: str,
    file_context: str,
    type_hints: bool = True
) -> str:
"""
        Build prompt for generating new function
        """
prompt = f"""Generate a Python function based on this description:

{ function_description }

Requirements:
- Include docstring with description, parameters, and return value
    - Add type hints: { type_hints }
- Include error handling where appropriate
    - Follow PEP 8 style guidelines

Context(existing code in file):
\`\`\`python
{file_context}
\`\`\`

Generate only the function, maintaining consistency with the existing code style.
"""
return prompt


# Usage
prompt_builder = PromptBuilder()

edit = CodeEdit(
    file_path = Path("app/users.py"),
    instruction = "Add error handling to get_user function",
    current_content = "def get_user(id):\\n    return db.query(User).get(id)",
    contexts = []
)

prompt = prompt_builder.build_edit_prompt(edit)
print(prompt)
\`\`\`

---

## Diff Generation and Application

The core of Cursor's editing functionality:

\`\`\`python
"""
Diff Generator and Applicator
"""

import difflib
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class DiffHunk:
    """Represents a single diff hunk"""
    file_path: Path
    old_start: int
    old_lines: List[str]
    new_start: int
    new_lines: List[str]
    
    def to_unified_diff(self) -> str:
        """Convert to unified diff format"""
        lines = []
        lines.append(f"@@ -{self.old_start},{len(self.old_lines)} +{self.new_start},{len(self.new_lines)} @@")
        
        for line in self.old_lines:
            lines.append(f"-{line}")
        
        for line in self.new_lines:
            lines.append(f"+{line}")
        
        return "\\n".join(lines)

class DiffGenerator:
    """
    Generates diffs between code versions
    """
    
    @staticmethod
    def generate_diff(
        old_content: str,
        new_content: str,
        file_path: Path
    ) -> str:
        """
        Generate unified diff
        """
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=""
        )
        
        return "".join(diff)
    
    @staticmethod
    def generate_minimal_diff(
        old_content: str,
        new_content: str
    ) -> List[DiffHunk]:
        """
        Generate minimal diff hunks
        """
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        hunks = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                hunks.append(DiffHunk(
                    file_path=None,  # Set by caller
                    old_start=i1 + 1,
                    old_lines=old_lines[i1:i2],
                    new_start=j1 + 1,
                    new_lines=new_lines[j1:j2]
                ))
        
        return hunks
    
    @staticmethod
    def apply_diff(
        original_content: str,
        diff_hunks: List[DiffHunk]
    ) -> str:
        """
        Apply diff hunks to content
        """
        lines = original_content.splitlines()
        
        # Apply hunks in reverse order to maintain line numbers
        for hunk in sorted(diff_hunks, key=lambda h: h.old_start, reverse=True):
            # Remove old lines
            del lines[hunk.old_start-1:hunk.old_start-1+len(hunk.old_lines)]
            
            # Insert new lines
            for i, new_line in enumerate(hunk.new_lines):
                lines.insert(hunk.old_start-1+i, new_line)
        
        return "\\n".join(lines)


class SearchReplaceApplicator:
    """
    Applies search/replace style edits (Cursor's format)
    """
    
    @staticmethod
    def parse_edit_response(response: str) -> List[dict]:
        """
        Parse LLM response into structured edits
        
        Expected format:
        # FILE: path/to/file.py
        # ACTION: REPLACE
        # SEARCH:
        old code
        # REPLACE:
        new code
        """
        edits = []
        current_edit = {}
        section = None
        content_lines = []
        
        for line in response.splitlines():
            if line.startswith("# FILE:"):
                if current_edit:
                    edits.append(current_edit)
                current_edit = {"file": line.split(":", 1)[1].strip()}
                content_lines = []
                section = None
            
            elif line.startswith("# ACTION:"):
                current_edit["action"] = line.split(":", 1)[1].strip()
            
            elif line.startswith("# SEARCH:"):
                if content_lines:
                    current_edit[section] = "\\n".join(content_lines)
                section = "search"
                content_lines = []
            
            elif line.startswith("# REPLACE:"):
                if content_lines:
                    current_edit[section] = "\\n".join(content_lines)
                section = "replace"
                content_lines = []
            
            elif line.startswith("# INSERT_AFTER:"):
                if content_lines:
                    current_edit[section] = "\\n".join(content_lines)
                section = "insert_after"
                content_lines = []
            
            elif line.startswith("# INSERT_BEFORE:"):
                if content_lines:
                    current_edit[section] = "\\n".join(content_lines)
                section = "insert_before"
                content_lines = []
            
            elif line.startswith("# CONTENT:"):
                section = "content"
                content_lines = []
            
            elif not line.startswith("#"):
                if section:
                    content_lines.append(line)
        
        # Don't forget last edit
        if content_lines and section:
            current_edit[section] = "\\n".join(content_lines)
        if current_edit:
            edits.append(current_edit)
        
        return edits
    
    @staticmethod
    def apply_edit(content: str, edit: dict) -> Tuple[str, bool]:
        """
        Apply a single edit to content
        
        Returns: (new_content, success)
        """
        action = edit.get("action", "REPLACE")
        
        if action == "REPLACE":
            search = edit.get("search", "")
            replace = edit.get("replace", "")
            
            if search in content:
                new_content = content.replace(search, replace, 1)
                return new_content, True
            else:
                return content, False
        
        elif action == "INSERT_AFTER":
            search = edit.get("insert_after", "")
            content_to_insert = edit.get("content", "")
            
            if search in content:
                new_content = content.replace(
                    search,
                    search + "\\n" + content_to_insert,
                    1
                )
                return new_content, True
            else:
                return content, False
        
        elif action == "INSERT_BEFORE":
            search = edit.get("insert_before", "")
            content_to_insert = edit.get("content", "")
            
            if search in content:
                new_content = content.replace(
                    search,
                    content_to_insert + "\\n" + search,
                    1
                )
                return new_content, True
            else:
                return content, False
        
        elif action == "DELETE":
            search = edit.get("search", "")
            
            if search in content:
                new_content = content.replace(search, "", 1)
                return new_content, True
            else:
                return content, False
        
        return content, False
    
    @staticmethod
    def apply_all_edits(
        fs_manager: FileSystemManager,
        edits: List[dict]
    ) -> dict:
        """
        Apply all edits from LLM response
        
        Returns: {file_path: {"old": str, "new": str, "success": bool}}
        """
        results = {}
        
        for edit in edits:
            file_path = Path(edit["file"])
            
            # Read current content
            try:
                content = await fs_manager.read_file(file_path)
            except Exception as e:
                results[str(file_path)] = {
                    "old": None,
                    "new": None,
                    "success": False,
                    "error": str(e)
                }
                continue
            
            # Apply edit
            new_content, success = SearchReplaceApplicator.apply_edit(
                content,
                edit
            )
            
            results[str(file_path)] = {
                "old": content,
                "new": new_content,
                "success": success,
                "diff": DiffGenerator.generate_diff(content, new_content, file_path) if success else None
            }
        
        return results


# Usage
llm_response = """
# FILE: app/users.py
# ACTION: REPLACE
# SEARCH:
def get_user(id):
    return db.query(User).get(id)
# REPLACE:
def get_user(id):
    try:
        user = db.query(User).get(id)
        if user is None:
            raise ValueError(f"User {id} not found")
        return user
    except Exception as e:
        logger.error(f"Error fetching user {id}: {e}")
        raise
"""

edits = SearchReplaceApplicator.parse_edit_response(llm_response)
results = await SearchReplaceApplicator.apply_all_edits(fs_manager, edits)

for file_path, result in results.items():
    if result["success"]:
        print(f"✓ Applied edit to {file_path}")
        print(result["diff"])
    else:
        print(f"✗ Failed to apply edit to {file_path}")
\`\`\`

---

## Putting It All Together

Here's the complete code editor engine:

\`\`\`python
"""
Complete AI Code Editor Engine
"""

import asyncio
from typing import Optional
from openai import AsyncOpenAI

class AICodeEditor:
    """
    Complete AI-powered code editor (Cursor clone)
    """
    
    def __init__(
        self,
        root_path: Path,
        openai_api_key: str
    ):
        self.fs_manager = FileSystemManager(root_path)
        self.context_manager = ContextManager(self.fs_manager)
        self.prompt_builder = PromptBuilder()
        self.llm_client = AsyncOpenAI(api_key=openai_api_key)
        self.conversation_history = []
    
    async def start(self):
        """Start the editor"""
        await self.fs_manager.start_watching(self._on_file_change)
        print(f"AI Code Editor started. Watching: {self.fs_manager.root_path}")
    
    async def edit_code(
        self,
        file_path: Path,
        instruction: str,
        cursor_position: Optional[int] = None
    ) -> dict:
        """
        Edit code based on natural language instruction
        """
        # Get context
        contexts = await self.context_manager.get_context_for_query(
            query=instruction,
            current_file=file_path,
            cursor_position=cursor_position
        )
        
        # Read current file
        current_content = await self.fs_manager.read_file(file_path)
        
        # Build prompt
        edit = CodeEdit(
            file_path=file_path,
            instruction=instruction,
            current_content=current_content,
            contexts=contexts
        )
        prompt = self.prompt_builder.build_edit_prompt(edit)
        
        # Call LLM
        response = await self.llm_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        llm_output = response.choices[0].message.content
        
        # Parse and apply edits
        edits = SearchReplaceApplicator.parse_edit_response(llm_output)
        results = await SearchReplaceApplicator.apply_all_edits(
            self.fs_manager,
            edits
        )
        
        # Write successful edits
        for file_path, result in results.items():
            if result["success"]:
                await self.fs_manager.write_file(
                    Path(file_path),
                    result["new"]
                )
        
        return results
    
    async def chat(
        self,
        query: str,
        current_file: Optional[Path] = None
    ) -> str:
        """
        Chat about code
        """
        # Get context
        contexts = []
        if current_file:
            contexts = await self.context_manager.get_context_for_query(
                query=query,
                current_file=current_file
            )
        
        # Build messages
        messages = self.prompt_builder.build_chat_prompt(
            query=query,
            contexts=contexts,
            conversation_history=self.conversation_history
        )
        
        # Call LLM
        response = await self.llm_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            stream=True
        )
        
        # Stream response
        full_response = []
        async for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response.append(content)
                print(content, end="", flush=True)
        
        print()  # New line after streaming
        
        response_text = "".join(full_response)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        return response_text
    
    async def _on_file_change(self, file_path: str, event_type: str):
        """Handle file changes"""
        # Invalidate cache, re-parse, etc.
        if Path(file_path) in self.fs_manager.file_cache:
            del self.fs_manager.file_cache[Path(file_path)]


# Example usage
async def main():
    editor = AICodeEditor(
        root_path=Path("/path/to/project"),
        openai_api_key="sk-..."
    )
    
    await editor.start()
    
    # Edit code
    results = await editor.edit_code(
        file_path=Path("app/users.py"),
        instruction="Add error handling to all database queries"
    )
    
    for file_path, result in results.items():
        if result["success"]:
            print(f"✓ Successfully edited {file_path}")
            print(result["diff"])
    
    # Chat about code
    response = await editor.chat(
        query="How does the user authentication work?",
        current_file=Path("app/auth.py")
    )

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

---

## Conclusion

You now have a complete understanding of how to build an AI code editor like Cursor. Key takeaways:

1. **File System Integration**: Watch for changes, maintain cache
2. **Code Parsing**: Use AST to understand code structure
3. **Smart Context**: Select relevant files, not entire codebase
4. **Effective Prompts**: Provide context and clear instructions
5. **Diff Generation**: Show precise changes before applying
6. **Search/Replace**: Structured edit format for reliability

This foundation enables building production-ready AI code assistance tools.
`,
};

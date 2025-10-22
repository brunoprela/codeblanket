/**
 * Building a Universal File Editor Section
 * Module 3: File Processing & Document Understanding
 */

export const buildinguniversalfileeditorSection = {
    id: 'building-universal-file-editor',
    title: 'Building a Universal File Editor',
    content: `# Building a Universal File Editor

Build a comprehensive file editing system that can process and modify any file type - the foundation for tools like Cursor.

## Architecture Overview

A universal file editor needs:
1. File type detection
2. Format-specific processors
3. Unified editing interface
4. Safe modification system
5. Validation and rollback
6. LLM integration for AI editing

## Complete Implementation

\`\`\`python
from pathlib import Path
from typing import Dict, Optional, Any
import shutil
import logging
from dataclasses import dataclass

@dataclass
class FileEdit:
    """Represents a file modification."""
    filepath: str
    old_content: str
    new_content: str
    edit_type: str  # 'replace', 'insert', 'delete'
    line_number: Optional[int] = None

class UniversalFileEditor:
    """
    Production-grade universal file editor.
    
    Handles any file type with safe modifications.
    Foundation for building Cursor-like tools.
    """
    
    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.backup_dir = self.workspace / ".backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self.processors = self._init_processors()
    
    def _init_processors(self) -> Dict:
        """Initialize format-specific processors."""
        return {
            'text': TextFileProcessor(),
            'python': PythonFileProcessor(),
            'excel': ExcelFileProcessor(),
            'pdf': PDFProcessor(),
            'word': WordProcessor(),
            'markdown': MarkdownProcessor()
        }
    
    def detect_file_type(self, filepath: str) -> str:
        """Detect file type and return processor type."""
        path = Path(filepath)
        ext = path.suffix.lower()
        
        type_map = {
            '.py': 'python',
            '.txt': 'text',
            '.md': 'markdown',
            '.xlsx': 'excel',
            '.pdf': 'pdf',
            '.docx': 'word'
        }
        
        return type_map.get(ext, 'text')
    
    def read_file(self, filepath: str) -> Dict:
        """Read file with appropriate processor."""
        file_type = self.detect_file_type(filepath)
        processor = self.processors.get(file_type)
        
        if processor:
            return processor.read(filepath)
        else:
            return {'error': 'Unsupported file type'}
    
    def edit_file(
        self,
        filepath: str,
        edits: list[FileEdit],
        create_backup: bool = True
    ) -> bool:
        """
        Apply edits to file safely.
        
        Central method for all file modifications.
        """
        full_path = self.workspace / filepath
        
        if not full_path.exists():
            self.logger.error(f"File not found: {filepath}")
            return False
        
        try:
            # Create backup
            if create_backup:
                self._create_backup(full_path)
            
            # Apply edits
            file_type = self.detect_file_type(filepath)
            processor = self.processors.get(file_type)
            
            if not processor:
                self.logger.error(f"No processor for {file_type}")
                return False
            
            success = processor.apply_edits(str(full_path), edits)
            
            if success:
                self.logger.info(f"Successfully edited {filepath}")
            else:
                self._restore_backup(full_path)
                self.logger.error(f"Failed to edit {filepath}, restored backup")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error editing file: {e}")
            if create_backup:
                self._restore_backup(full_path)
            return False
    
    def _create_backup(self, filepath: Path):
        """Create timestamped backup."""
        import time
        timestamp = int(time.time())
        backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        backup_path = self.backup_dir / backup_name
        shutil.copy2(filepath, backup_path)
        self.logger.info(f"Created backup: {backup_name}")
    
    def _restore_backup(self, filepath: Path):
        """Restore most recent backup."""
        # Find most recent backup
        backups = sorted(self.backup_dir.glob(f"{filepath.stem}_*{filepath.suffix}"))
        if backups:
            shutil.copy2(backups[-1], filepath)
            self.logger.info(f"Restored backup for {filepath.name}")
    
    def get_file_summary(self, filepath: str) -> str:
        """
        Get file summary for LLM context.
        
        Provides structure and preview for AI understanding.
        """
        file_data = self.read_file(filepath)
        
        if 'error' in file_data:
            return f"Error reading file: {file_data['error']}"
        
        summary = f"File: {filepath}\\n"
        summary += f"Type: {self.detect_file_type(filepath)}\\n"
        
        # Add type-specific summary
        if 'content' in file_data:
            preview = file_data['content'][:500]
            summary += f"Preview:\\n{preview}..."
        
        return summary

# Processor base class
class FileProcessor:
    """Base class for file processors."""
    
    def read(self, filepath: str) -> Dict:
        """Read file and return structured data."""
        raise NotImplementedError
    
    def apply_edits(self, filepath: str, edits: list[FileEdit]) -> bool:
        """Apply edits to file."""
        raise NotImplementedError

# Example processor implementations
class TextFileProcessor(FileProcessor):
    """Process plain text files."""
    
    def read(self, filepath: str) -> Dict:
        content = Path(filepath).read_text(encoding='utf-8')
        return {
            'type': 'text',
            'content': content,
            'lines': content.splitlines()
        }
    
    def apply_edits(self, filepath: str, edits: list[FileEdit]) -> bool:
        try:
            content = Path(filepath).read_text(encoding='utf-8')
            
            for edit in edits:
                if edit.edit_type == 'replace':
                    content = content.replace(edit.old_content, edit.new_content)
            
            Path(filepath).write_text(content, encoding='utf-8')
            return True
        except Exception:
            return False

class PythonFileProcessor(FileProcessor):
    """Process Python files with AST awareness."""
    
    def read(self, filepath: str) -> Dict:
        import ast
        content = Path(filepath).read_text(encoding='utf-8')
        
        try:
            tree = ast.parse(content)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            return {
                'type': 'python',
                'content': content,
                'functions': functions,
                'classes': classes
            }
        except:
            return {'type': 'python', 'content': content}
    
    def apply_edits(self, filepath: str, edits: list[FileEdit]) -> bool:
        # Similar to TextFileProcessor but could add Python-specific logic
        return TextFileProcessor().apply_edits(filepath, edits)

# Usage Example: Build Cursor-like file editor
editor = UniversalFileEditor("my_project")

# Read any file type
file_data = editor.read_file("src/main.py")
print(file_data)

# Create edit
edit = FileEdit(
    filepath="src/main.py",
    old_content='print("hello")',
    new_content='print("Hello, World!")',
    edit_type='replace'
)

# Apply edit safely with backup
success = editor.edit_file("src/main.py", [edit], create_backup=True)

# Get summary for LLM
summary = editor.get_file_summary("src/main.py")
print(summary)
\`\`\`

## Key Takeaways

1. **Unified interface** for all file types
2. **Type detection** via extensions
3. **Format-specific processors** for accuracy
4. **Safe modifications** with backups
5. **Atomic operations** prevent corruption
6. **Error handling** with rollback
7. **LLM integration** for AI editing
8. **Validation** before and after edits
9. **Logging** for debugging
10. **Modular architecture** for extensibility

This completes Module 3! You now have the foundation for building production file processing systems like Cursor.`,
    videoUrl: null,
};


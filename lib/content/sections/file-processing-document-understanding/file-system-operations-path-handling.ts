/**
 * File System Operations & Path Handling Section
 * Module 3: File Processing & Document Understanding
 */

export const filesystemoperationspathhandlingSection = {
  id: 'file-system-operations-path-handling',
  title: 'File System Operations & Path Handling',
  content: `# File System Operations & Path Handling

Master safe, reliable, and cross-platform file system operations for production LLM applications.

## Overview: Why File Operations Matter for LLM Apps

When building AI applications that process documents, generate code, or manipulate files, robust file system operations are foundational. Whether you're building a Cursor-like code editor or a document processing pipeline, you need to handle files safely, efficiently, and reliably across different operating systems.

**Key Challenges:**
- Cross-platform path compatibility (Windows vs Unix)
- Race conditions and concurrent access
- Permissions and security
- Atomic operations
- Error handling and recovery
- Memory efficiency with large files

## The pathlib Module: Modern Path Handling

Python\'s \`pathlib\` module provides an object-oriented interface for file paths that works consistently across platforms.

### Why pathlib Over os.path?

\`\`\`python
# ❌ Old way: os.path (string manipulation)
import os
path = os.path.join("/", "Users", "data", "file.txt")
dirname = os.path.dirname (path)
basename = os.path.basename (path)
exists = os.path.exists (path)

# ✅ New way: pathlib (object-oriented)
from pathlib import Path

path = Path("/") / "Users" / "data" / "file.txt"
dirname = path.parent
basename = path.name
exists = path.exists()

# pathlib is cleaner, chainable, and cross-platform
\`\`\`

### Creating and Working with Paths

\`\`\`python
from pathlib import Path

# Create paths
home = Path.home()  # /Users/username
cwd = Path.cwd()    # Current working directory

# Build paths with / operator (works on all platforms!)
data_dir = Path.home() / "Documents" / "data"
config_file = data_dir / "config.json"

# Access path components
print(config_file.name)       # config.json
print(config_file.stem)       # config
print(config_file.suffix)     # .json
print(config_file.parent)     # /Users/username/Documents/data
print(config_file.parts)      # ('/', 'Users', 'username', 'Documents', 'data', 'config.json')

# Convert to string when needed
path_str = str (config_file)

# Resolve to absolute path
absolute = config_file.resolve()
\`\`\`

### Path Manipulation and Inspection

\`\`\`python
from pathlib import Path

path = Path("documents/reports/2024/report.pdf")

# Change components
new_path = path.with_name("summary.pdf")  # documents/reports/2024/summary.pdf
backup = path.with_suffix(".pdf.bak")     # documents/reports/2024/report.pdf.bak
text_version = path.with_suffix(".txt")   # documents/reports/2024/report.txt

# Check path properties
print(path.is_absolute())    # False
print(path.is_relative_to(Path("documents")))  # True

# Get relative paths
rel_path = path.relative_to("documents")  # reports/2024/report.pdf
\`\`\`

## Reading Files Safely

### Basic File Reading

\`\`\`python
from pathlib import Path

def read_file_simple (filepath: str) -> str:
    """Simple file reading with automatic cleanup."""
    path = Path (filepath)
    
    # Context manager ensures file is closed
    with path.open("r", encoding="utf-8") as f:
        content = f.read()
    
    return content

# Even simpler with pathlib
def read_file_pathlib (filepath: str) -> str:
    """One-liner file reading."""
    return Path (filepath).read_text (encoding="utf-8")

# For binary files
def read_binary (filepath: str) -> bytes:
    """Read binary files."""
    return Path (filepath).read_bytes()
\`\`\`

### Production-Grade File Reading with Error Handling

\`\`\`python
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def read_file_safe(
    filepath: str,
    encoding: str = "utf-8",
    fallback_encoding: str = "latin-1"
) -> Optional[str]:
    """
    Production-grade file reading with comprehensive error handling.
    
    Args:
        filepath: Path to file
        encoding: Primary encoding to try
        fallback_encoding: Fallback if primary fails
        
    Returns:
        File content or None if reading fails
    """
    path = Path (filepath)
    
    # Check if file exists
    if not path.exists():
        logger.error (f"File not found: {filepath}")
        return None
    
    # Check if it's a file (not directory)
    if not path.is_file():
        logger.error (f"Path is not a file: {filepath}")
        return None
    
    # Check file size (avoid loading huge files into memory)
    max_size = 100 * 1024 * 1024  # 100MB
    if path.stat().st_size > max_size:
        logger.error (f"File too large: {filepath} ({path.stat().st_size} bytes)")
        return None
    
    # Try reading with primary encoding
    try:
        return path.read_text (encoding=encoding)
    except UnicodeDecodeError:
        logger.warning (f"Failed to decode with {encoding}, trying {fallback_encoding}")
        try:
            return path.read_text (encoding=fallback_encoding)
        except UnicodeDecodeError:
            logger.error (f"Failed to decode file with both encodings: {filepath}")
            return None
    except PermissionError:
        logger.error (f"Permission denied: {filepath}")
        return None
    except Exception as e:
        logger.error (f"Unexpected error reading {filepath}: {e}")
        return None
\`\`\`

### Reading Large Files Efficiently

\`\`\`python
from pathlib import Path
from typing import Iterator

def read_file_lines (filepath: str, chunk_size: int = 8192) -> Iterator[str]:
    """
    Read large files line by line without loading into memory.
    
    Perfect for processing large log files or datasets.
    """
    path = Path (filepath)
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\\n")

def read_file_chunks (filepath: str, chunk_size: int = 8192) -> Iterator[str]:
    """
    Read file in chunks for processing large files.
    
    Useful when you need to process file content but can't load it all.
    """
    path = Path (filepath)
    
    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read (chunk_size)
            if not chunk:
                break
            yield chunk

# Usage example
def count_lines_efficiently (filepath: str) -> int:
    """Count lines in a large file without loading into memory."""
    return sum(1 for _ in read_file_lines (filepath))

def search_in_large_file (filepath: str, search_term: str) -> list[str]:
    """Search for a term in a large file."""
    matches = []
    for line_num, line in enumerate (read_file_lines (filepath), 1):
        if search_term in line:
            matches.append (f"Line {line_num}: {line}")
    return matches
\`\`\`

## Writing Files Safely

### Basic File Writing

\`\`\`python
from pathlib import Path

def write_file_simple (filepath: str, content: str) -> None:
    """Simple file writing."""
    path = Path (filepath)
    
    # Create parent directories if they don't exist
    path.parent.mkdir (parents=True, exist_ok=True)
    
    # Write file
    path.write_text (content, encoding="utf-8")

def write_binary (filepath: str, content: bytes) -> None:
    """Write binary files."""
    path = Path (filepath)
    path.parent.mkdir (parents=True, exist_ok=True)
    path.write_bytes (content)
\`\`\`

### Atomic File Writing (Production Pattern)

\`\`\`python
from pathlib import Path
import tempfile
import shutil
import os

def write_file_atomic (filepath: str, content: str) -> bool:
    """
    Write file atomically to prevent corruption if process is interrupted.
    
    How it works:
    1. Write to temporary file
    2. If successful, rename (atomic operation) to target
    3. If fails, temporary file is cleaned up
    
    This ensures you never have a partially-written file.
    """
    path = Path (filepath)
    path.parent.mkdir (parents=True, exist_ok=True)
    
    # Create temporary file in same directory (ensures same filesystem)
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )
    
    try:
        # Write to temporary file
        with os.fdopen (fd, "w", encoding="utf-8") as f:
            f.write (content)
            f.flush()
            os.fsync (f.fileno())  # Ensure data is written to disk
        
        # Atomic rename (or as atomic as possible)
        shutil.move (temp_path, path)
        return True
        
    except Exception as e:
        # Clean up temporary file if something went wrong
        try:
            os.unlink (temp_path)
        except:
            pass
        raise e

# Usage in a code editor like Cursor
def save_code_file_safe (filepath: str, code: str) -> bool:
    """
    Save code file atomically with backup.
    Critical for code editors to prevent data loss.
    """
    path = Path (filepath)
    
    # Create backup if file exists
    if path.exists():
        backup_path = path.with_suffix (path.suffix + ".bak")
        shutil.copy2(path, backup_path)
    
    try:
        write_file_atomic (filepath, code)
        return True
    except Exception as e:
        print(f"Failed to save file: {e}")
        # Restore from backup if available
        if backup_path.exists():
            shutil.copy2(backup_path, path)
        return False
\`\`\`

## Directory Operations

### Creating Directories

\`\`\`python
from pathlib import Path

# Create single directory
path = Path("output/results")
path.mkdir (exist_ok=True)  # Don't error if exists

# Create nested directories
path = Path("output/data/processed/2024")
path.mkdir (parents=True, exist_ok=True)  # Create all parent dirs

# Check if directory was created
assert path.is_dir()
\`\`\`

### Listing Directory Contents

\`\`\`python
from pathlib import Path

def list_directory (dir_path: str) -> dict:
    """List all files and directories with metadata."""
    path = Path (dir_path)
    
    contents = {
        "files": [],
        "directories": [],
        "total_size": 0
    }
    
    for item in path.iterdir():
        if item.is_file():
            stat = item.stat()
            contents["files"].append({
                "name": item.name,
                "path": str (item),
                "size": stat.st_size,
                "modified": stat.st_mtime
            })
            contents["total_size"] += stat.st_size
        elif item.is_dir():
            contents["directories"].append({
                "name": item.name,
                "path": str (item)
            })
    
    return contents

def find_files_by_extension (dir_path: str, extension: str) -> list[Path]:
    """Find all files with given extension in directory tree."""
    path = Path (dir_path)
    
    # Use glob for pattern matching
    return list (path.rglob (f"*.{extension}"))

# Examples
python_files = find_files_by_extension("src", "py")
json_files = find_files_by_extension("config", "json")

# More complex patterns
def find_code_files (dir_path: str) -> dict[str, list[Path]]:
    """Find all code files by language."""
    path = Path (dir_path)
    
    return {
        "python": list (path.rglob("*.py")),
        "javascript": list (path.rglob("*.js")) + list (path.rglob("*.jsx")),
        "typescript": list (path.rglob("*.ts")) + list (path.rglob("*.tsx")),
        "java": list (path.rglob("*.java")),
        "cpp": list (path.rglob("*.cpp")) + list (path.rglob("*.hpp"))
    }
\`\`\`

### Traversing Directory Trees

\`\`\`python
from pathlib import Path
from typing import Callable

def walk_directory(
    dir_path: str,
    file_callback: Callable[[Path], None],
    dir_callback: Optional[Callable[[Path], None]] = None,
    max_depth: Optional[int] = None
) -> None:
    """
    Walk directory tree with callbacks for files and directories.
    
    Similar to os.walk but with pathlib and callbacks.
    """
    def _walk (path: Path, depth: int = 0):
        if max_depth is not None and depth > max_depth:
            return
        
        for item in path.iterdir():
            if item.is_file():
                file_callback (item)
            elif item.is_dir():
                if dir_callback:
                    dir_callback (item)
                _walk (item, depth + 1)
    
    _walk(Path (dir_path))

# Usage: Build file tree for LLM context (like Cursor does)
def build_file_tree (dir_path: str, ignore_patterns: list[str] = None) -> dict:
    """
    Build a file tree structure for providing context to LLMs.
    """
    ignore_patterns = ignore_patterns or [
        ".git", "__pycache__", "node_modules", ".venv", "venv"
    ]
    
    def should_ignore (path: Path) -> bool:
        return any (pattern in str (path) for pattern in ignore_patterns)
    
    tree = {"name": Path (dir_path).name, "type": "directory", "children": []}
    
    def add_file (file_path: Path):
        if not should_ignore (file_path):
            tree["children"].append({
                "name": file_path.name,
                "type": "file",
                "path": str (file_path.relative_to (dir_path))
            })
    
    walk_directory (dir_path, add_file, max_depth=3)
    return tree
\`\`\`

## File Permissions and Metadata

### Checking Permissions

\`\`\`python
from pathlib import Path
import os
import stat

def check_file_permissions (filepath: str) -> dict:
    """Check what operations are allowed on a file."""
    path = Path (filepath)
    
    if not path.exists():
        return {"exists": False}
    
    return {
        "exists": True,
        "readable": os.access (path, os.R_OK),
        "writable": os.access (path, os.W_OK),
        "executable": os.access (path, os.X_OK),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "is_symlink": path.is_symlink()
    }

def get_file_metadata (filepath: str) -> dict:
    """Get comprehensive file metadata."""
    path = Path (filepath)
    stat_info = path.stat()
    
    return {
        "size": stat_info.st_size,
        "created": stat_info.st_ctime,
        "modified": stat_info.st_mtime,
        "accessed": stat_info.st_atime,
        "owner": stat_info.st_uid,
        "group": stat_info.st_gid,
        "permissions": oct (stat_info.st_mode)[-3:],
        "is_readonly": not os.access (path, os.W_OK)
    }
\`\`\`

### Setting Permissions

\`\`\`python
from pathlib import Path
import stat

def make_executable (filepath: str) -> None:
    """Make a file executable (useful for scripts)."""
    path = Path (filepath)
    current = path.stat().st_mode
    path.chmod (current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

def make_readonly (filepath: str) -> None:
    """Make a file read-only."""
    path = Path (filepath)
    path.chmod (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

def make_writable (filepath: str) -> None:
    """Make a file writable."""
    path = Path (filepath)
    current = path.stat().st_mode
    path.chmod (current | stat.S_IWUSR)
\`\`\`

## Temporary Files and Directories

### Using tempfile Module

\`\`\`python
import tempfile
from pathlib import Path

# Temporary file that's automatically deleted
def process_with_temp_file():
    with tempfile.NamedTemporaryFile (mode="w", suffix=".txt", delete=True) as f:
        f.write("Temporary content")
        f.flush()
        
        # Use the file
        temp_path = Path (f.name)
        print(f"Temp file: {temp_path}")
        # File is automatically deleted when context exits

# Temporary directory
def process_with_temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path (temp_dir)
        
        # Create files in temp directory
        (temp_path / "data.txt").write_text("Some data")
        (temp_path / "config.json").write_text('{"key": "value"}')
        
        # Process files...
        
        # Directory and all contents automatically deleted

# Manual temporary file management
def create_temp_file_manual() -> str:
    """Create temp file that persists after function returns."""
    fd, path = tempfile.mkstemp (suffix=".txt", prefix="llm_output_")
    
    # Write to file
    with os.fdopen (fd, "w") as f:
        f.write("Content")
    
    return path  # Caller is responsible for cleanup
\`\`\`

## File Watching and Monitoring

### Watching for File Changes

\`\`\`python
from pathlib import Path
import time
from typing import Callable

class FileWatcher:
    """
    Watch a file for changes (like auto-reload in development).
    
    Similar to how Cursor watches files for changes.
    """
    
    def __init__(self, filepath: str, callback: Callable[[Path], None]):
        self.path = Path (filepath)
        self.callback = callback
        self.last_modified = self.path.stat().st_mtime if self.path.exists() else 0
    
    def check_for_changes (self) -> bool:
        """Check if file has been modified."""
        if not self.path.exists():
            return False
        
        current_mtime = self.path.stat().st_mtime
        
        if current_mtime > self.last_modified:
            self.last_modified = current_mtime
            self.callback (self.path)
            return True
        
        return False
    
    def watch (self, interval: float = 1.0):
        """Watch file continuously."""
        try:
            while True:
                self.check_for_changes()
                time.sleep (interval)
        except KeyboardInterrupt:
            print("Stopped watching")

# Usage
def on_file_change (path: Path):
    print(f"File changed: {path}")
    content = path.read_text()
    # Process the updated content...

watcher = FileWatcher("config.json", on_file_change)
# watcher.watch()  # Blocks and watches continuously
\`\`\`

### Production File Watcher with watchdog

\`\`\`python
# pip install watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path

class CodeFileHandler(FileSystemEventHandler):
    """
    Watch directory for code file changes.
    Similar to how Cursor watches your codebase.
    """
    
    def __init__(self, on_change: Callable[[str], None]):
        self.on_change = on_change
        self.code_extensions = {".py", ".js", ".ts", ".java", ".cpp"}
    
    def on_modified (self, event):
        if event.is_directory:
            return
        
        path = Path (event.src_path)
        if path.suffix in self.code_extensions:
            self.on_change (str (path))

def watch_directory (dir_path: str, callback: Callable[[str], None]):
    """Watch directory for code changes."""
    event_handler = CodeFileHandler (callback)
    observer = Observer()
    observer.schedule (event_handler, dir_path, recursive=True)
    observer.start()
    
    try:
        observer.join()
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

# Usage
def on_code_change (filepath: str):
    print(f"Code file changed: {filepath}")
    # Re-analyze file, update embeddings, etc.

# watch_directory("src", on_code_change)
\`\`\`

## Cross-Platform Considerations

### Path Separators

\`\`\`python
from pathlib import Path
import os

# ❌ Don't hardcode separators
wrong_path = "C:\\\\Users\\\\data\\\\file.txt"  # Breaks on Unix

# ✅ Use pathlib - automatically handles separators
correct_path = Path("C:/") / "Users" / "data" / "file.txt"

# Or use os.path.join for strings
correct_str = os.path.join("C:", "Users", "data", "file.txt")

# Current platform separator
print(os.sep)  # '/' on Unix, '\\' on Windows
\`\`\`

### Home Directory and Environment

\`\`\`python
from pathlib import Path
import os

# Get home directory (cross-platform)
home = Path.home()  # Works on all platforms

# Get environment variables
data_dir = Path (os.getenv("DATA_DIR", str(Path.home() / "data")))

# Cross-platform config directory
def get_config_dir() -> Path:
    """Get appropriate config directory for platform."""
    if os.name == "nt":  # Windows
        return Path (os.getenv("APPDATA")) / "MyApp"
    else:  # Unix-like
        return Path.home() / ".config" / "myapp"
\`\`\`

## Common Pitfalls and Solutions

### Pitfall 1: Not Handling Encoding

\`\`\`python
# ❌ Default encoding might not be UTF-8
with open("file.txt") as f:
    content = f.read()  # May fail on non-ASCII

# ✅ Always specify encoding
with open("file.txt", encoding="utf-8") as f:
    content = f.read()
\`\`\`

### Pitfall 2: Not Creating Parent Directories

\`\`\`python
from pathlib import Path

# ❌ Will fail if 'output' doesn't exist
# Path("output/results/data.txt").write_text("data")

# ✅ Create parent directories first
path = Path("output/results/data.txt")
path.parent.mkdir (parents=True, exist_ok=True)
path.write_text("data")
\`\`\`

### Pitfall 3: Resource Leaks

\`\`\`python
# ❌ File might not be closed if error occurs
f = open("file.txt")
content = f.read()
f.close()

# ✅ Use context manager (always closes)
with open("file.txt") as f:
    content = f.read()
\`\`\`

### Pitfall 4: Race Conditions

\`\`\`python
from pathlib import Path

# ❌ Race condition: file could be deleted between check and open
path = Path("file.txt")
if path.exists():
    content = path.read_text()  # Might fail if deleted meanwhile

# ✅ Use EAFP (Easier to Ask Forgiveness than Permission)
try:
    content = Path("file.txt").read_text()
except FileNotFoundError:
    print("File doesn't exist")
\`\`\`

## Production Checklist

When implementing file operations in production LLM applications:

✅ **Always specify encoding** (\`encoding="utf-8"\`)  
✅ **Use pathlib** for cross-platform compatibility  
✅ **Create parent directories** before writing  
✅ **Use context managers** (\`with\` statements)  
✅ **Handle errors gracefully** (try/except blocks)  
✅ **Check file sizes** before loading into memory  
✅ **Use atomic operations** for critical files  
✅ **Create backups** before modifying existing files  
✅ **Implement proper logging** for debugging  
✅ **Test on multiple platforms** if deploying across OS  
✅ **Validate file permissions** before operations  
✅ **Use temporary files** for intermediate results  

## Real-World Example: Cursor-Like File Manager

\`\`\`python
from pathlib import Path
from typing import Optional, List
import shutil
import logging

class CodeFileManager:
    """
    File manager for a code editor (like Cursor).
    Handles safe file operations with backups and rollbacks.
    """
    
    def __init__(self, workspace_dir: str):
        self.workspace = Path (workspace_dir)
        self.backup_dir = self.workspace / ".backups"
        self.backup_dir.mkdir (exist_ok=True)
        
        logging.basicConfig (level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def read_file (self, filepath: str) -> Optional[str]:
        """Read file safely with error handling."""
        try:
            path = self.workspace / filepath
            return path.read_text (encoding="utf-8")
        except Exception as e:
            self.logger.error (f"Failed to read {filepath}: {e}")
            return None
    
    def write_file (self, filepath: str, content: str, create_backup: bool = True) -> bool:
        """Write file atomically with optional backup."""
        try:
            path = self.workspace / filepath
            
            # Create backup if file exists
            if create_backup and path.exists():
                self._create_backup (path)
            
            # Create parent directories
            path.parent.mkdir (parents=True, exist_ok=True)
            
            # Atomic write
            temp_path = path.with_suffix (path.suffix + ".tmp")
            temp_path.write_text (content, encoding="utf-8")
            shutil.move (str (temp_path), str (path))
            
            self.logger.info (f"Successfully wrote {filepath}")
            return True
            
        except Exception as e:
            self.logger.error (f"Failed to write {filepath}: {e}")
            return False
    
    def _create_backup (self, path: Path) -> None:
        """Create timestamped backup of file."""
        import time
        timestamp = int (time.time())
        backup_name = f"{path.stem}_{timestamp}{path.suffix}"
        backup_path = self.backup_dir / backup_name
        shutil.copy2(path, backup_path)
        self.logger.info (f"Created backup: {backup_name}")
    
    def list_files (self, pattern: str = "*.py") -> List[Path]:
        """List all files matching pattern."""
        return list (self.workspace.rglob (pattern))
    
    def get_file_tree (self) -> dict:
        """Get file tree for LLM context."""
        return self._build_tree (self.workspace)
    
    def _build_tree (self, path: Path, max_depth: int = 3, depth: int = 0) -> dict:
        """Recursively build file tree."""
        if depth > max_depth:
            return None
        
        tree = {"name": path.name, "type": "directory", "children": []}
        
        try:
            for item in sorted (path.iterdir()):
                if item.name.startswith('.'):
                    continue
                
                if item.is_file():
                    tree["children"].append({
                        "name": item.name,
                        "type": "file"
                    })
                elif item.is_dir():
                    subtree = self._build_tree (item, max_depth, depth + 1)
                    if subtree:
                        tree["children"].append (subtree)
        except PermissionError:
            pass
        
        return tree

# Usage
manager = CodeFileManager("my_project")

# Read file
content = manager.read_file("src/main.py")

# Write with automatic backup
manager.write_file("src/main.py", "# Updated code", create_backup=True)

# Get file tree for LLM context
tree = manager.get_file_tree()
\`\`\`

## Key Takeaways

1. **Use pathlib** for modern, cross-platform path handling
2. **Always specify encoding** when reading/writing text files
3. **Use context managers** to ensure resources are cleaned up
4. **Implement atomic writes** for critical files to prevent corruption
5. **Create backups** before modifying existing files
6. **Handle errors gracefully** with proper try/except blocks
7. **Check file sizes** before loading into memory
8. **Use temporary files** for intermediate processing
9. **Implement file watching** for reactive applications
10. **Test cross-platform** if deploying on multiple OS

These patterns form the foundation for building robust file processing in LLM applications like Cursor, where reliability and data safety are paramount.`,
  videoUrl: undefined,
};

/**
 * Text File Processing Section
 * Module 3: File Processing & Document Understanding
 */

export const textfileprocessingSection = {
  id: 'text-file-processing',
  title: 'Text File Processing',
  content: `# Text File Processing

Master efficient text file processing for building production LLM applications that handle code, documents, and data files.

## Overview: Text Files in LLM Applications

Text file processing is at the heart of LLM applications. Whether you're building a code editor like Cursor, processing documentation, or analyzing logs, you need to handle text files efficiently and correctly. This section covers everything from basic reading to advanced diff generation and memory-efficient processing of massive files.

**Common Use Cases:**
- Code editors: Reading, editing, and saving code files
- Document processing: Parsing markdown, text reports, configs
- Log analysis: Processing application logs for debugging
- Data pipelines: ETL operations on text data
- Diff generation: Creating patches for code changes

## Character Encoding Deep Dive

### Understanding Encoding Issues

\`\`\`python
# The most common source of bugs in text processing!

# ❌ This can fail with non-ASCII characters
try:
    with open("file.txt") as f:  # Uses system default encoding
        content = f.read()
except UnicodeDecodeError:
    print("Failed to decode file!")

# ✅ Always specify encoding explicitly
with open("file.txt", encoding="utf-8") as f:
    content = f.read()
\`\`\`

### Common Encodings and When to Use Them

\`\`\`python
from pathlib import Path
from typing import Optional

def read_text_with_encoding_detection(filepath: str) -> tuple[str, str]:
    """
    Try multiple encodings to read a file.
    Returns content and detected encoding.
    """
    path = Path(filepath)
    
    # Common encodings to try
    encodings = [
        "utf-8",          # Modern standard, try first
        "utf-8-sig",      # UTF-8 with BOM (Windows)
        "latin-1",        # Western European (ISO-8859-1)
        "cp1252",         # Windows Western European
        "ascii",          # Basic ASCII
    ]
    
    for encoding in encodings:
        try:
            content = path.read_text(encoding=encoding)
            return content, encoding
        except (UnicodeDecodeError, LookupError):
            continue
    
    # If all fail, read as binary and decode with errors='ignore'
    content = path.read_text(encoding="utf-8", errors="ignore")
    return content, "utf-8 (with errors ignored)"

# Usage
content, encoding = read_text_with_encoding_detection("unknown_file.txt")
print(f"Successfully read file with encoding: {encoding}")
\`\`\`

### Production-Grade Encoding Handler

\`\`\`python
# pip install chardet
import chardet
from pathlib import Path

def read_file_auto_encoding(filepath: str, sample_size: int = 10000) -> str:
    """
    Automatically detect encoding using chardet library.
    
    Used by tools like VS Code for automatic encoding detection.
    """
    path = Path(filepath)
    
    # Read sample bytes for detection
    with path.open("rb") as f:
        raw_data = f.read(sample_size)
    
    # Detect encoding
    detection = chardet.detect(raw_data)
    encoding = detection["encoding"]
    confidence = detection["confidence"]
    
    print(f"Detected encoding: {encoding} (confidence: {confidence:.2%})")
    
    # Read with detected encoding
    if encoding and confidence > 0.7:
        return path.read_text(encoding=encoding)
    else:
        # Fall back to UTF-8 with error handling
        return path.read_text(encoding="utf-8", errors="replace")
\`\`\`

## Line-by-Line Processing

### Basic Line Iteration

\`\`\`python
from pathlib import Path

def process_file_lines_basic(filepath: str) -> None:
    """Process file line by line - memory efficient."""
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            # line includes newline character
            process_line(line_num, line)

def process_line(line_num: int, line: str) -> None:
    """Process a single line."""
    # Strip whitespace and newline
    clean_line = line.rstrip("\\n\\r")
    
    if not clean_line:  # Skip empty lines
        return
    
    print(f"Line {line_num}: {clean_line}")
\`\`\`

### Advanced Line Processing with Context

\`\`\`python
from pathlib import Path
from typing import Iterator, Tuple
from collections import deque

def read_lines_with_context(
    filepath: str,
    context_before: int = 2,
    context_after: int = 2
) -> Iterator[Tuple[int, str, list[str], list[str]]]:
    """
    Read file with context lines (like grep -C).
    
    Yields: (line_num, line, before_lines, after_lines)
    Useful for showing context around matching lines.
    """
    with open(filepath, encoding="utf-8") as f:
        lines = [line.rstrip("\\n") for line in f]
    
    for i, line in enumerate(lines):
        before = lines[max(0, i - context_before):i]
        after = lines[i + 1:min(len(lines), i + 1 + context_after)]
        yield (i + 1, line, before, after)

# Usage: Find TODO comments with context (like Cursor does)
def find_todos_with_context(filepath: str) -> None:
    """Find TODO comments and show surrounding code."""
    for line_num, line, before, after in read_lines_with_context(filepath):
        if "TODO" in line:
            print(f"\\n=== TODO at line {line_num} ===")
            for b in before:
                print(f"  {b}")
            print(f"> {line}")  # Highlight the TODO line
            for a in after:
                print(f"  {a}")
\`\`\`

### Streaming Large Files

\`\`\`python
from pathlib import Path
from typing import Iterator, Callable
import time

class LargeFileProcessor:
    """
    Process large files efficiently with progress tracking.
    
    Similar to how Cursor processes large codebases.
    """
    
    def __init__(self, filepath: str, chunk_size: int = 8192):
        self.path = Path(filepath)
        self.chunk_size = chunk_size
        self.file_size = self.path.stat().st_size
    
    def process_lines(
        self,
        processor: Callable[[int, str], None],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> dict:
        """
        Process file line-by-line with progress tracking.
        
        Args:
            processor: Function to process each line
            progress_callback: Called with progress percentage (0-100)
        
        Returns:
            Statistics about processing
        """
        start_time = time.time()
        lines_processed = 0
        bytes_processed = 0
        
        with self.path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                # Process line
                processor(line_num, line)
                
                # Update progress
                lines_processed += 1
                bytes_processed += len(line.encode("utf-8"))
                
                # Report progress every 1000 lines
                if progress_callback and lines_processed % 1000 == 0:
                    progress = (bytes_processed / self.file_size) * 100
                    progress_callback(progress)
        
        elapsed = time.time() - start_time
        
        return {
            "lines_processed": lines_processed,
            "bytes_processed": bytes_processed,
            "elapsed_seconds": elapsed,
            "lines_per_second": lines_processed / elapsed if elapsed > 0 else 0
        }

# Usage
def analyze_large_log_file(filepath: str):
    """Analyze a large log file efficiently."""
    processor = LargeFileProcessor(filepath)
    
    error_count = 0
    warning_count = 0
    
    def count_errors(line_num: int, line: str):
        nonlocal error_count, warning_count
        if "ERROR" in line:
            error_count += 1
        elif "WARNING" in line:
            warning_count += 1
    
    def show_progress(progress: float):
        print(f"\\rProgress: {progress:.1f}%", end="", flush=True)
    
    stats = processor.process_lines(count_errors, show_progress)
    
    print(f"\\nFound {error_count} errors and {warning_count} warnings")
    print(f"Processed {stats['lines_processed']:,} lines in {stats['elapsed_seconds']:.2f}s")
    print(f"Speed: {stats['lines_per_second']:,.0f} lines/second")
\`\`\`

## Chunking Strategies for LLMs

### Fixed-Size Chunking

\`\`\`python
from pathlib import Path
from typing import Iterator

def chunk_text_by_chars(text: str, chunk_size: int = 1000, overlap: int = 200) -> Iterator[str]:
    """
    Split text into overlapping chunks by character count.
    
    Overlap ensures context isn't lost between chunks.
    Useful for processing with LLMs that have token limits.
    """
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        yield chunk
        
        # Move start forward by chunk_size minus overlap
        start = end - overlap
        
        if start >= text_len:
            break

# Usage: Process large document with LLM
def process_large_document_with_llm(filepath: str):
    """Process document in chunks to stay under token limits."""
    content = Path(filepath).read_text(encoding="utf-8")
    
    summaries = []
    for i, chunk in enumerate(chunk_text_by_chars(content, chunk_size=2000), 1):
        print(f"Processing chunk {i}...")
        # summary = call_llm_to_summarize(chunk)
        # summaries.append(summary)
    
    # Combine summaries
    # final_summary = call_llm_to_combine(summaries)
\`\`\`

### Semantic Chunking (Paragraph-Based)

\`\`\`python
from pathlib import Path
from typing import Iterator

def chunk_text_by_paragraphs(text: str, max_chunk_size: int = 1000) -> Iterator[str]:
    """
    Split text by paragraphs, combining until reaching max_chunk_size.
    
    Better for LLMs as it respects document structure.
    """
    paragraphs = text.split("\\n\\n")
    
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_size = len(para)
        
        # If single paragraph exceeds max size, yield it anyway
        if para_size > max_chunk_size:
            if current_chunk:
                yield "\\n\\n".join(current_chunk)
                current_chunk = []
                current_size = 0
            yield para
            continue
        
        # If adding this paragraph would exceed max size
        if current_size + para_size > max_chunk_size:
            yield "\\n\\n".join(current_chunk)
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size + 2  # +2 for \\n\\n
    
    # Yield remaining chunk
    if current_chunk:
        yield "\\n\\n".join(current_chunk)

# Usage
text = Path("document.txt").read_text()
chunks = list(chunk_text_by_paragraphs(text, max_chunk_size=2000))
print(f"Split document into {len(chunks)} chunks")
\`\`\`

### Sentence-Based Chunking

\`\`\`python
import re
from typing import Iterator

def chunk_text_by_sentences(text: str, sentences_per_chunk: int = 5) -> Iterator[str]:
    """
    Split text by sentences, grouping N sentences per chunk.
    
    Most natural for reading comprehension tasks.
    """
    # Simple sentence splitter (for production, use NLTK or spaCy)
    sentences = re.split(r'(?<=[.!?])\\s+', text)
    
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        yield chunk.strip()

# Better sentence splitting with NLTK
try:
    import nltk
    nltk.download('punkt', quiet=True)
    
    def chunk_text_by_sentences_nltk(text: str, sentences_per_chunk: int = 5) -> Iterator[str]:
        """Better sentence splitting using NLTK."""
        sentences = nltk.sent_tokenize(text)
        
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i:i + sentences_per_chunk])
            yield chunk.strip()
except ImportError:
    pass
\`\`\`

## Diff Generation and Comparison

### Basic Diff Generation

\`\`\`python
import difflib
from pathlib import Path

def generate_unified_diff(
    file1: str,
    file2: str,
    context_lines: int = 3
) -> str:
    """
    Generate unified diff between two files.
    
    This is the format used by git diff and patch tools.
    Similar to how Cursor shows code changes.
    """
    path1 = Path(file1)
    path2 = Path(file2)
    
    with path1.open() as f1, path2.open() as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    
    diff = difflib.unified_diff(
        lines1,
        lines2,
        fromfile=str(path1),
        tofile=str(path2),
        lineterm="",
        n=context_lines
    )
    
    return "\\n".join(diff)

# Usage
diff = generate_unified_diff("old_code.py", "new_code.py")
print(diff)
\`\`\`

### Computing Edit Distance

\`\`\`python
def compute_edit_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein distance (edit distance) between two strings.
    
    Useful for finding similar files or detecting changes.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1]   # Replace
                )
    
    return dp[m][n]

def similarity_ratio(s1: str, s2: str) -> float:
    """
    Compute similarity ratio (0.0 to 1.0) between two strings.
    
    1.0 = identical, 0.0 = completely different
    """
    distance = compute_edit_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    if max_len == 0:
        return 1.0
    
    return 1.0 - (distance / max_len)

# Usage
old_code = "def hello():\\n    print('hi')"
new_code = "def hello():\\n    print('hello')"
similarity = similarity_ratio(old_code, new_code)
print(f"Similarity: {similarity:.2%}")
\`\`\`

### Finding Changed Lines

\`\`\`python
import difflib
from typing import List, Tuple
from enum import Enum

class ChangeType(Enum):
    ADDED = "+"
    DELETED = "-"
    MODIFIED = "~"
    UNCHANGED = " "

def find_changed_lines(
    old_content: str,
    new_content: str
) -> List[Tuple[int, ChangeType, str]]:
    """
    Find all changed lines between two versions.
    
    Returns: List of (line_number, change_type, line_content)
    
    Used by code editors to highlight changes.
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    changes = []
    
    # Generate sequence matcher
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i, line in enumerate(new_lines[j1:j2]):
                changes.append((j1 + i + 1, ChangeType.UNCHANGED, line))
        elif tag == 'delete':
            for i, line in enumerate(old_lines[i1:i2]):
                changes.append((i1 + i + 1, ChangeType.DELETED, line))
        elif tag == 'insert':
            for i, line in enumerate(new_lines[j1:j2]):
                changes.append((j1 + i + 1, ChangeType.ADDED, line))
        elif tag == 'replace':
            for i, line in enumerate(old_lines[i1:i2]):
                changes.append((i1 + i + 1, ChangeType.DELETED, line))
            for i, line in enumerate(new_lines[j1:j2]):
                changes.append((j1 + i + 1, ChangeType.ADDED, line))
    
    return changes

# Usage: Display changes like Cursor
def display_diff(old_content: str, new_content: str):
    """Display diff with color indicators."""
    changes = find_changed_lines(old_content, new_content)
    
    for line_num, change_type, line in changes:
        prefix = change_type.value
        print(f"{prefix} {line_num:4d} | {line.rstrip()}")
\`\`\`

## Patch Application

### Applying Diffs

\`\`\`python
import difflib
from pathlib import Path
from typing import Optional

def apply_patch(
    original_content: str,
    patch_content: str
) -> Optional[str]:
    """
    Apply a unified diff patch to content.
    
    Returns: Patched content or None if patch fails
    
    Similar to how Cursor applies LLM-generated code changes.
    """
    original_lines = original_content.splitlines(keepends=True)
    patch_lines = patch_content.splitlines(keepends=True)
    
    # Parse patch and apply
    # This is a simplified version - production code would use
    # a proper patch library
    
    result_lines = original_lines.copy()
    
    # For production, use libraries like:
    # - whatthepatch
    # - unidiff
    # Or call external tools like patch command
    
    return "".join(result_lines)

# Better approach: Use whatthepatch library
try:
    import whatthepatch
    
    def apply_patch_proper(
        original_content: str,
        patch_content: str
    ) -> Optional[str]:
        """Apply patch using proper parsing library."""
        patches = whatthepatch.parse_patch(patch_content)
        
        for patch in patches:
            # Apply diff changes
            original_lines = original_content.splitlines(keepends=True)
            
            for diff in patch.changes:
                line_num, old_line, new_line = diff
                if old_line is None:  # Addition
                    original_lines.insert(line_num, new_line)
                elif new_line is None:  # Deletion
                    del original_lines[line_num]
                else:  # Modification
                    original_lines[line_num] = new_line
            
            return "".join(original_lines)
        
        return None
except ImportError:
    pass
\`\`\`

## Memory-Mapped Files

### Using mmap for Large Files

\`\`\`python
import mmap
from pathlib import Path

def search_large_file_mmap(filepath: str, search_term: str) -> List[int]:
    """
    Search for term in large file using memory mapping.
    
    Much faster than reading line-by-line for large files.
    """
    path = Path(filepath)
    matches = []
    
    with path.open("r+b") as f:
        # Memory-map the file
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
            # Convert to bytes
            search_bytes = search_term.encode("utf-8")
            
            # Find all occurrences
            position = 0
            while True:
                position = mmapped.find(search_bytes, position)
                if position == -1:
                    break
                matches.append(position)
                position += 1
    
    return matches

# Usage
matches = search_large_file_mmap("large_log.txt", "ERROR")
print(f"Found {len(matches)} occurrences")
\`\`\`

## Text Normalization

### Cleaning Text for LLM Processing

\`\`\`python
import re
from typing import Optional

def normalize_text(text: str) -> str:
    """
    Normalize text for consistent LLM processing.
    
    Handles common issues like:
    - Multiple consecutive spaces
    - Mixed line endings
    - Extra whitespace
    """
    # Normalize line endings to \\n
    text = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
    
    # Remove multiple consecutive blank lines
    text = re.sub(r"\\n\\n\\n+", "\\n\\n", text)
    
    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in text.split("\\n")]
    text = "\\n".join(lines)
    
    # Normalize multiple spaces (but preserve indentation)
    # text = re.sub(r" +", " ", text)
    
    # Remove leading/trailing whitespace from entire text
    text = text.strip()
    
    return text

def remove_comments(code: str, language: str = "python") -> str:
    """
    Remove comments from code.
    Useful for focusing LLM on actual code logic.
    """
    if language == "python":
        # Remove single-line comments
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        # Remove docstrings (simplified)
        code = re.sub(r'"""[\\s\\S]*?"""', "", code)
        code = re.sub(r"'''[\\s\\S]*?'''", "", code)
    elif language in ["javascript", "typescript", "java", "cpp"]:
        # Remove single-line comments
        code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
        # Remove multi-line comments
        code = re.sub(r"/\\*[\\s\\S]*?\\*/", "", code)
    
    return normalize_text(code)
\`\`\`

## Production Example: File Processor for Code Editor

\`\`\`python
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import difflib
import logging

@dataclass
class FileChange:
    """Represents a change to a file."""
    line_number: int
    change_type: str  # 'add', 'delete', 'modify'
    old_content: Optional[str]
    new_content: Optional[str]

class CodeFileProcessor:
    """
    Production-grade file processor for code editors.
    
    Similar to how Cursor processes and modifies code files.
    """
    
    def __init__(self, workspace: str):
        self.workspace = Path(workspace)
        self.logger = logging.getLogger(__name__)
    
    def read_file(self, filepath: str) -> Optional[str]:
        """Read file with encoding detection and error handling."""
        full_path = self.workspace / filepath
        
        try:
            # Try UTF-8 first
            return full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with fallback encoding
            try:
                return full_path.read_text(encoding="latin-1")
            except Exception as e:
                self.logger.error(f"Failed to read {filepath}: {e}")
                return None
    
    def generate_diff(
        self,
        filepath: str,
        new_content: str
    ) -> Tuple[str, List[FileChange]]:
        """Generate diff between current file and new content."""
        current = self.read_file(filepath)
        if current is None:
            return "", []
        
        # Generate unified diff
        current_lines = current.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            current_lines,
            new_lines,
            fromfile=filepath,
            tofile=filepath,
            lineterm=""
        )
        
        diff_text = "\\n".join(diff)
        
        # Extract changes
        changes = self._extract_changes(current, new_content)
        
        return diff_text, changes
    
    def _extract_changes(
        self,
        old_content: str,
        new_content: str
    ) -> List[FileChange]:
        """Extract individual changes between versions."""
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        
        changes = []
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'delete':
                for i in range(i1, i2):
                    changes.append(FileChange(
                        line_number=i + 1,
                        change_type='delete',
                        old_content=old_lines[i],
                        new_content=None
                    ))
            elif tag == 'insert':
                for j in range(j1, j2):
                    changes.append(FileChange(
                        line_number=j + 1,
                        change_type='add',
                        old_content=None,
                        new_content=new_lines[j]
                    ))
            elif tag == 'replace':
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    changes.append(FileChange(
                        line_number=i + 1,
                        change_type='modify',
                        old_content=old_lines[i],
                        new_content=new_lines[j]
                    ))
        
        return changes
    
    def apply_changes(
        self,
        filepath: str,
        new_content: str,
        create_backup: bool = True
    ) -> bool:
        """Apply changes to file with backup."""
        full_path = self.workspace / filepath
        
        try:
            # Create backup
            if create_backup and full_path.exists():
                backup_path = full_path.with_suffix(full_path.suffix + ".bak")
                import shutil
                shutil.copy2(full_path, backup_path)
            
            # Write new content atomically
            temp_path = full_path.with_suffix(full_path.suffix + ".tmp")
            temp_path.write_text(new_content, encoding="utf-8")
            temp_path.replace(full_path)
            
            self.logger.info(f"Successfully updated {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update {filepath}: {e}")
            return False
    
    def chunk_file_for_llm(
        self,
        filepath: str,
        max_chunk_size: int = 2000
    ) -> List[str]:
        """Chunk file for LLM processing."""
        content = self.read_file(filepath)
        if not content:
            return []
        
        return list(chunk_text_by_paragraphs(content, max_chunk_size))

# Usage
processor = CodeFileProcessor("my_project")

# Read file
content = processor.read_file("src/main.py")

# Generate diff
new_code = "# Updated code\\nprint('Hello')"
diff, changes = processor.generate_diff("src/main.py", new_code)

print(f"Diff:\\n{diff}")
print(f"\\nChanges: {len(changes)}")
for change in changes:
    print(f"  Line {change.line_number}: {change.change_type}")

# Apply changes
processor.apply_changes("src/main.py", new_code, create_backup=True)
\`\`\`

## Key Takeaways

1. **Always specify encoding** (UTF-8 is the standard)
2. **Process large files line-by-line** to avoid memory issues
3. **Use streaming** for files > 100MB
4. **Chunk intelligently** for LLM processing (paragraphs > sentences > characters)
5. **Generate diffs** to show changes clearly
6. **Handle encoding errors** with fallback strategies
7. **Use memory mapping** for searching large files
8. **Normalize text** before LLM processing
9. **Track changes** for undo/redo functionality
10. **Create backups** before modifying files

These patterns are essential for building production-grade text processing in LLM applications, especially code editors like Cursor where data integrity and performance are critical.`,
  videoUrl: undefined,
};

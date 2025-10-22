/**
 * Code Editing & Diff Generation Section
 * Module 5: Building Code Generation Systems
 */

export const codeeditingdiffgenerationSection = {
    id: 'code-editing-diff-generation',
    title: 'Code Editing & Diff Generation',
    content: `# Code Editing & Diff Generation

Master generating precise code edits and diffs like Cursor does - the most important skill for production code generation.

## Overview: Why Diffs Matter

Generating complete files works for new code, but editing existing code requires a different approach:

**Generating Full Files ❌**
- Loses formatting and comments
- Breaks unrelated code
- Harder to review
- Error-prone

**Generating Diffs ✅**
- Precise changes
- Preserves existing code
- Easy to review
- Safer to apply

### How Cursor Generates Edits

When you ask Cursor to "add error handling", it doesn't regenerate the entire file. Instead:

1. **Identifies** the exact lines to change
2. **Generates** minimal modifications
3. **Produces** a diff or replace instruction
4. **Applies** changes surgically

## Diff Formats

### 1. Unified Diff Format

The standard format used by git:

\`\`\`python
def generate_unified_diff(
    original: str,
    modified: str,
    filename: str = "file.py"
) -> str:
    """Generate a unified diff."""
    import difflib
    from datetime import datetime
    
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)
    
    timestamp = datetime.now().isoformat()
    
    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"{filename} (original)",
        tofile=f"{filename} (modified)",
        lineterm=""
    )
    
    return "".join(diff)

# Example
original = """
def calculate(x, y):
    return x + y
"""

modified = """
def calculate(x: int, y: int) -> int:
    '''Add two numbers.'''
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError("Arguments must be integers")
    return x + y
"""

diff = generate_unified_diff(original, modified)
print(diff)

# Output:
# --- file.py (original)
# +++ file.py (modified)
# @@ -1,2 +1,5 @@
# -def calculate(x, y):
# +def calculate(x: int, y: int) -> int:
# +    '''Add two numbers.'''
# +    if not isinstance(x, int) or not isinstance(y, int):
# +        raise TypeError("Arguments must be integers")
#      return x + y
\`\`\`

### 2. Search/Replace Format

Cursor's preferred format - more LLM-friendly:

\`\`\`python
from dataclasses import dataclass
from typing import List

@dataclass
class SearchReplace:
    """Represents a search/replace edit."""
    search: str  # Exact text to find
    replace: str  # Text to replace with
    
    def __str__(self):
        return f"""<<<<<<< SEARCH
{self.search}
=======
{self.replace}
>>>>>>> REPLACE
"""

class SearchReplaceParser:
    """Parse search/replace blocks from LLM output."""
    
    def parse(self, text: str) -> List[SearchReplace]:
        """Extract all search/replace blocks."""
        edits = []
        
        # Split by SEARCH markers
        blocks = text.split("<<<<<<< SEARCH")
        
        for block in blocks[1:]:  # Skip first (before any SEARCH)
            if "=======" in block and ">>>>>>> REPLACE" in block:
                parts = block.split("=======")
                search = parts[0].strip()
                
                replace_parts = parts[1].split(">>>>>>> REPLACE")
                replace = replace_parts[0].strip()
                
                edits.append(SearchReplace(search, replace))
        
        return edits

# Usage
llm_output = """Here are the changes:

<<<<<<< SEARCH
def calculate(x, y):
    return x + y
=======
def calculate(x: int, y: int) -> int:
    '''Add two numbers.'''
    return x + y
>>>>>>> REPLACE

<<<<<<< SEARCH
result = calculate(5, 3)
=======
result = calculate(x=5, y=3)
>>>>>>> REPLACE
"""

parser = SearchReplaceParser()
edits = parser.parse(llm_output)

for edit in edits:
    print(edit)
\`\`\`

### 3. Line-Based Edits

Simple format for small changes:

\`\`\`python
@dataclass
class LineEdit:
    """Represents a line-based edit."""
    line_number: int
    old_content: str
    new_content: str
    operation: str  # "replace", "insert", "delete"

class LineEditor:
    """Apply line-based edits."""
    
    def apply_edits(
        self,
        content: str,
        edits: List[LineEdit]
    ) -> str:
        """Apply line edits to content."""
        lines = content.split("\\n")
        
        # Sort edits by line number (descending) to avoid offset issues
        sorted_edits = sorted(edits, key=lambda e: e.line_number, reverse=True)
        
        for edit in sorted_edits:
            line_idx = edit.line_number - 1  # Convert to 0-indexed
            
            if edit.operation == "replace":
                if line_idx < len(lines):
                    lines[line_idx] = edit.new_content
            
            elif edit.operation == "insert":
                lines.insert(line_idx, edit.new_content)
            
            elif edit.operation == "delete":
                if line_idx < len(lines):
                    del lines[line_idx]
        
        return "\\n".join(lines)

# Usage
original = """def hello():
    print("world")
    return True
"""

edits = [
    LineEdit(1, "def hello():", "def hello(name: str):", "replace"),
    LineEdit(2, 'print("world")', 'print(f"Hello, {name}")', "replace")
]

editor = LineEditor()
result = editor.apply_edits(original, edits)
print(result)
\`\`\`

## Prompting for Edits

### Edit-Focused Prompts

\`\`\`python
class EditPromptBuilder:
    """Build prompts optimized for code editing."""
    
    def build_edit_prompt(
        self,
        file_content: str,
        instruction: str,
        format: str = "search_replace"
    ) -> str:
        """Build prompt for generating edits."""
        
        format_instructions = {
            "search_replace": """
Output your changes in SEARCH/REPLACE blocks:

<<<<<<< SEARCH
[exact lines to find]
=======
[new lines]
>>>>>>> REPLACE

You can have multiple SEARCH/REPLACE blocks.
IMPORTANT: The SEARCH block must match EXACTLY (including whitespace).
""",
            "unified_diff": """
Output your changes as a unified diff format.
""",
            "line_edit": """
Output your changes as line-by-line edits in this JSON format:
{
    "edits": [
        {"line": 5, "operation": "replace", "content": "new content"},
        {"line": 10, "operation": "insert", "content": "new line"}
    ]
}
"""
        }
        
        format_instr = format_instructions.get(
            format,
            format_instructions["search_replace"]
        )
        
        # Number the lines for reference
        lines = file_content.split("\\n")
        numbered = "\\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
        
        prompt = f"""You are editing this file:

{numbered}

TASK: {instruction}

{format_instr}

Generate ONLY the necessary changes. Do not regenerate unchanged code.
"""
        
        return prompt

# Usage
builder = EditPromptBuilder()

file_content = """def process_user(user):
    name = user.get('name')
    email = user.get('email')
    return name, email
"""

prompt = builder.build_edit_prompt(
    file_content,
    "Add type hints and input validation",
    format="search_replace"
)

print(prompt)
\`\`\`

## Applying Edits Safely

### Safe Edit Applicator

\`\`\`python
from typing import Optional
import difflib

class SafeEditApplicator:
    """Apply edits with validation and rollback."""
    
    def apply_search_replace(
        self,
        content: str,
        search: str,
        replace: str,
        fuzzy: bool = False
    ) -> tuple[bool, str, str]:
        """
        Apply a search/replace edit.
        
        Returns:
            (success, new_content, error_message)
        """
        # Try exact match first
        if search in content:
            new_content = content.replace(search, replace, 1)
            return True, new_content, ""
        
        # If fuzzy matching enabled, try approximate match
        if fuzzy:
            best_match = self._find_fuzzy_match(content, search)
            if best_match:
                new_content = content.replace(best_match, replace, 1)
                return True, new_content, f"Used fuzzy match: {best_match[:50]}..."
        
        return False, content, f"Search text not found: {search[:50]}..."
    
    def _find_fuzzy_match(
        self,
        content: str,
        search: str,
        threshold: float = 0.8
    ) -> Optional[str]:
        """Find approximate match using difflib."""
        lines = content.split("\\n")
        search_lines = search.split("\\n")
        search_len = len(search_lines)
        
        best_match = None
        best_ratio = 0.0
        
        # Try all possible positions
        for i in range(len(lines) - search_len + 1):
            candidate = "\\n".join(lines[i:i+search_len])
            ratio = difflib.SequenceMatcher(
                None, search, candidate
            ).ratio()
            
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = candidate
        
        return best_match
    
    def apply_multiple_edits(
        self,
        content: str,
        edits: List[SearchReplace],
        fuzzy: bool = False
    ) -> tuple[bool, str, List[str]]:
        """
        Apply multiple edits in sequence.
        
        Returns:
            (all_succeeded, final_content, error_messages)
        """
        current_content = content
        errors = []
        
        for i, edit in enumerate(edits):
            success, new_content, error = self.apply_search_replace(
                current_content,
                edit.search,
                edit.replace,
                fuzzy=fuzzy
            )
            
            if success:
                current_content = new_content
            else:
                errors.append(f"Edit {i+1} failed: {error}")
        
        return len(errors) == 0, current_content, errors

# Usage
applicator = SafeEditApplicator()

original = """def calculate(x, y):
    result = x + y
    return result
"""

success, new_content, error = applicator.apply_search_replace(
    original,
    "def calculate(x, y):",
    "def calculate(x: int, y: int) -> int:"
)

if success:
    print("✓ Edit applied successfully")
    print(new_content)
else:
    print(f"✗ Edit failed: {error}")
\`\`\`

## Minimal Diff Generation

### Generate Smallest Possible Diff

\`\`\`python
class MinimalDiffGenerator:
    """Generate minimal diffs between code versions."""
    
    def generate_minimal_diff(
        self,
        original: str,
        modified: str
    ) -> List[SearchReplace]:
        """Generate minimal search/replace edits."""
        import difflib
        
        original_lines = original.split("\\n")
        modified_lines = modified.split("\\n")
        
        # Get diff operations
        matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)
        edits = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Lines were changed
                search = "\\n".join(original_lines[i1:i2])
                replace = "\\n".join(modified_lines[j1:j2])
                edits.append(SearchReplace(search, replace))
            
            elif tag == 'delete':
                # Lines were deleted
                search = "\\n".join(original_lines[i1:i2])
                replace = ""
                edits.append(SearchReplace(search, replace))
            
            elif tag == 'insert':
                # Lines were inserted - need context
                # Find surrounding context for stable anchor
                context_before = original_lines[i1-1] if i1 > 0 else ""
                context_after = original_lines[i1] if i1 < len(original_lines) else ""
                
                search = f"{context_before}\\n{context_after}"
                new_lines = "\\n".join(modified_lines[j1:j2])
                replace = f"{context_before}\\n{new_lines}\\n{context_after}"
                edits.append(SearchReplace(search, replace))
        
        return edits

# Usage
original = """def process(data):
    result = transform(data)
    return result
"""

modified = """def process(data: dict) -> dict:
    '''Process input data.'''
    if not data:
        raise ValueError("Data cannot be empty")
    result = transform(data)
    validate(result)
    return result
"""

generator = MinimalDiffGenerator()
edits = generator.generate_minimal_diff(original, modified)

for edit in edits:
    print(edit)
    print()
\`\`\`

## Cursor-Style Edit Generation

### Replicate Cursor's Edit Approach

\`\`\`python
from openai import OpenAI

class CursorStyleEditor:
    """Generate edits like Cursor does."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_edit(
        self,
        file_content: str,
        instruction: str,
        context: Optional[str] = None
    ) -> List[SearchReplace]:
        """Generate edits for file content."""
        
        # Build Cursor-style prompt
        prompt = self._build_cursor_prompt(
            file_content,
            instruction,
            context
        )
        
        # Generate edits
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": """You are a precise code editor. Generate minimal, 
surgical edits using SEARCH/REPLACE blocks. 

Rules:
1. SEARCH must match EXACTLY (including indentation)
2. Only change what's necessary
3. Preserve code style
4. Don't break functionality
5. Multiple small edits are better than one large edit"""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        # Parse edits
        parser = SearchReplaceParser()
        edits = parser.parse(response.choices[0].message.content)
        
        return edits
    
    def _build_cursor_prompt(
        self,
        file_content: str,
        instruction: str,
        context: Optional[str]
    ) -> str:
        """Build Cursor-style prompt."""
        lines = file_content.split("\\n")
        numbered = "\\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
        
        prompt = f"""File content:
{numbered}

"""
        
        if context:
            prompt += f"""Additional context:
{context}

"""
        
        prompt += f"""Instruction: {instruction}

Generate SEARCH/REPLACE blocks for the necessary changes.

Format:
<<<<<<< SEARCH
[exact lines to replace, with perfect indentation]
=======
[new lines]
>>>>>>> REPLACE

Important:
- SEARCH must match EXACTLY
- Include enough context for unique matching
- Make minimal changes
- Preserve indentation and style
"""
        
        return prompt
    
    def apply_edits(
        self,
        file_content: str,
        instruction: str,
        validate: bool = True
    ) -> tuple[bool, str, List[str]]:
        """
        Generate and apply edits.
        
        Returns:
            (success, new_content, errors)
        """
        # Generate edits
        edits = self.generate_edit(file_content, instruction)
        
        if not edits:
            return False, file_content, ["No edits generated"]
        
        # Apply edits
        applicator = SafeEditApplicator()
        success, new_content, errors = applicator.apply_multiple_edits(
            file_content,
            edits,
            fuzzy=True  # Allow small whitespace differences
        )
        
        # Validate if requested
        if success and validate:
            validator = GeneratedFileValidator()
            is_valid, val_errors = validator.validate_python_file(new_content)
            
            if not is_valid:
                return False, file_content, val_errors
        
        return success, new_content, errors

# Usage
editor = CursorStyleEditor()

file_content = """
def process_users(users):
    results = []
    for user in users:
        if user['active']:
            results.append(user['name'])
    return results
"""

success, new_content, errors = editor.apply_edits(
    file_content,
    "Add type hints and docstring"
)

if success:
    print("✓ Edits applied successfully:")
    print(new_content)
else:
    print(f"✗ Failed to apply edits:")
    for error in errors:
        print(f"  - {error}")
\`\`\`

## Three-Way Merging

### Handle Concurrent Edits

\`\`\`python
class ThreeWayMerger:
    """Merge changes when base has been modified."""
    
    def merge(
        self,
        base: str,
        ours: str,
        theirs: str
    ) -> tuple[bool, str, List[str]]:
        """
        Three-way merge.
        
        Args:
            base: Original content
            ours: Our modifications
            theirs: Their modifications
        
        Returns:
            (success, merged_content, conflicts)
        """
        import difflib
        
        base_lines = base.split("\\n")
        ours_lines = ours.split("\\n")
        theirs_lines = theirs.split("\\n")
        
        # Get changes from base to ours
        our_changes = difflib.SequenceMatcher(
            None, base_lines, ours_lines
        ).get_opcodes()
        
        # Get changes from base to theirs
        their_changes = difflib.SequenceMatcher(
            None, base_lines, theirs_lines
        ).get_opcodes()
        
        # Detect conflicts
        conflicts = self._find_conflicts(our_changes, their_changes)
        
        if conflicts:
            # Has conflicts - need manual resolution
            merged = self._create_conflict_markers(
                base_lines, ours_lines, theirs_lines, conflicts
            )
            return False, "\\n".join(merged), conflicts
        
        # No conflicts - merge automatically
        merged = self._auto_merge(
            base_lines, our_changes, their_changes
        )
        return True, "\\n".join(merged), []
    
    def _find_conflicts(self, our_changes, their_changes):
        """Identify conflicting changes."""
        conflicts = []
        
        for our_tag, our_i1, our_i2, _, _ in our_changes:
            if our_tag == 'equal':
                continue
            
            for their_tag, their_i1, their_i2, _, _ in their_changes:
                if their_tag == 'equal':
                    continue
                
                # Check if ranges overlap
                if not (our_i2 <= their_i1 or their_i2 <= our_i1):
                    conflicts.append((our_i1, our_i2, their_i1, their_i2))
        
        return conflicts
    
    def _create_conflict_markers(
        self, base_lines, ours_lines, theirs_lines, conflicts
    ):
        """Create conflict markers for manual resolution."""
        # Simplified version - just mark the conflicts
        result = base_lines.copy()
        
        for conflict in conflicts:
            our_i1, our_i2, their_i1, their_i2 = conflict
            
            result.insert(our_i1, "<<<<<<< OURS")
            result.insert(our_i2 + 1, "=======")
            result.insert(their_i2 + 2, ">>>>>>> THEIRS")
        
        return result

# Usage
merger = ThreeWayMerger()

base = """def calculate(x, y):
    return x + y
"""

ours = """def calculate(x: int, y: int):
    return x + y
"""

theirs = """def calculate(x, y):
    '''Add two numbers.'''
    return x + y
"""

success, merged, conflicts = merger.merge(base, ours, theirs)

if success:
    print("✓ Merged successfully:")
    print(merged)
else:
    print("✗ Conflicts detected:")
    for conflict in conflicts:
        print(f"  Line {conflict}")
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Generate minimal diffs** - only change what's needed
2. **Use SEARCH/REPLACE format** - more reliable than full file gen
3. **Include context** in search blocks for unique matching
4. **Validate edits** before applying
5. **Support fuzzy matching** for whitespace differences
6. **Preserve code style** and formatting
7. **Test edits** with rollback capability
8. **Use line numbers** for reference in prompts

### ❌ DON'T:
1. **Regenerate entire files** for small changes
2. **Make search blocks too short** (ambiguous matching)
3. **Skip validation** after applying edits
4. **Ignore whitespace** in search blocks
5. **Apply edits without backups**
6. **Use large edit blocks** (split into smaller ones)
7. **Forget to handle conflicts**
8. **Skip fuzzy matching** entirely

## Next Steps

You've mastered diff generation! Next:
- Multi-file code generation
- Refactoring systems
- Building complete editors

Remember: **Minimal, Surgical Edits > Complete File Regeneration**
`,
};


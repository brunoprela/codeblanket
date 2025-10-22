/**
 * Diff Generation & Patch Application Section
 * Module 3: File Processing & Document Understanding
 */

export const diffgenerationpatchapplicationSection = {
    id: 'diff-generation-patch-application',
    title: 'Diff Generation & Patch Application',
    content: `# Diff Generation & Patch Application

Master diff generation and patch application for building code editors and version control tools like Cursor.

## Understanding Diffs

Diffs show changes between two versions of text. Essential for:
- Code editors showing changes
- Version control systems
- Collaborative editing
- Change tracking

## Generating Diffs with difflib

\`\`\`python
import difflib

def generate_unified_diff(old_text: str, new_text: str):
    """Generate unified diff (git-style)."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile='old.txt',
        tofile='new.txt',
        lineterm=''
    )
    
    return '\\n'.join(diff)

# Usage - similar to how Cursor shows changes
old_code = '''def hello():
    print("hi")'''

new_code = '''def hello():
    print("Hello, World!")'''

diff = generate_unified_diff(old_code, new_code)
print(diff)
\`\`\`

## Finding Changes

\`\`\`python
def find_changes(old_text: str, new_text: str):
    """Find all changes between texts."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    
    changes = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            changes.append({
                'type': tag,  # replace, delete, insert
                'old_start': i1,
                'old_end': i2,
                'new_start': j1,
                'new_end': j2,
                'old_lines': old_lines[i1:i2],
                'new_lines': new_lines[j1:j2]
            })
    
    return changes
\`\`\`

## Applying Patches

\`\`\`python
# For production, use libraries like:
# pip install whatthepatch
import whatthepatch

def apply_patch(original_text: str, patch_text: str):
    """Apply unified diff patch to text."""
    patches = list(whatthepatch.parse_patch(patch_text))
    
    if not patches:
        return original_text
    
    lines = original_text.splitlines()
    
    for patch in patches:
        for change in patch.changes:
            old_line_no, new_line_no, text = change
            # Apply change logic here
    
    return '\\n'.join(lines)
\`\`\`

## Key Takeaways

1. **Use difflib** for diff generation
2. **Unified diff** is standard format
3. **SequenceMatcher** for detailed changes
4. **Patch application** requires careful handling
5. **Line-based** diffs for text files
6. **Character-based** diffs for precision
7. **Three-way merge** for conflicts
8. **Test patches** before applying
9. **Preserve line endings** in diffs
10. **Handle edge cases** (empty files, etc.)`,
    videoUrl: null,
};


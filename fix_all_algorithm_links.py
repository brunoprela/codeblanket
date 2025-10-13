#!/usr/bin/env python3
"""
Add missing LeetCode/YouTube links to algorithm problems.
Focuses only on algorithm problems, skipping Python educational content.
"""
import re

# Comprehensive mapping of ALL algorithm problems that need links
# Format: file_path: {problem_id: (leetcode_url, youtube_url)}

LINKS_TO_ADD = {
    'lib/problems/time-space-complexity.ts': {
        'valid-number': (
            'https://leetcode.com/problems/valid-number/',
            'https://www.youtube.com/watch?v=QfRSeibcugw'
        ),
    },
    'lib/problems/sorting.ts': {
        'sort-array-parity': (
            'https://leetcode.com/problems/sort-array-by-parity/',
            'https://www.youtube.com/watch?v=6YZn-z5jkrg'
        ),
        'insertion-sort-list': (
            'https://leetcode.com/problems/insertion-sort-list/',
            'https://www.youtube.com/watch?v=Kk6mXAzQ3zs'
        ),
        'sort-list': (
            'https://leetcode.com/problems/sort-list/',
            'https://www.youtube.com/watch?v=TGveA1oFhrc'
        ),
        'count-smaller': (
            'https://leetcode.com/problems/count-of-smaller-numbers-after-self/',
            'https://www.youtube.com/watch?v=2SVLYsq5W8M'
        ),
    },
    'lib/problems/two-pointers.ts': {
        'valid-palindrome': (
            'https://leetcode.com/problems/valid-palindrome/',
            'https://www.youtube.com/watch?v=jJXJ16kPFWg'
        ),
    },
    'lib/problems/trees.ts': {
        'invert-binary-tree': (
            'https://leetcode.com/problems/invert-binary-tree/',
            'https://www.youtube.com/watch?v=OnSn2XEQ4MY'
        ),
        'validate-bst': (
            'https://leetcode.com/problems/validate-binary-search-tree/',
            'https://www.youtube.com/watch?v=s6ATEkipzow'
        ),
        'binary-tree-max-path-sum': (
            'https://leetcode.com/problems/binary-tree-maximum-path-sum/',
            'https://www.youtube.com/watch?v=Hr5cWUld4vU'
        ),
    },
    'lib/problems/tries.ts': {
        'implement-trie': (
            'https://leetcode.com/problems/implement-trie-prefix-tree/',
            'https://www.youtube.com/watch?v=oobqoCJlHA0'
        ),
        'add-and-search-word': (
            'https://leetcode.com/problems/design-add-and-search-words-data-structure/',
            'https://www.youtube.com/watch?v=BTf05gs_8iU'
        ),
        'word-search-ii': (
            'https://leetcode.com/problems/word-search-ii/',
            'https://www.youtube.com/watch?v=asbcE9mZz_U'
        ),
    },
    'lib/problems/stack.ts': {
        'valid-parentheses': (
            'https://leetcode.com/problems/valid-parentheses/',
            'https://www.youtube.com/watch?v=WTzjTskDFMg'
        ),
        'min-stack': (
            'https://leetcode.com/problems/min-stack/',
            'https://www.youtube.com/watch?v=qkLl7nAwDPo'
        ),
        'largest-rectangle': (
            'https://leetcode.com/problems/largest-rectangle-in-histogram/',
            'https://www.youtube.com/watch?v=zx5Sw9130L0'
        ),
    },
    'lib/problems/sliding-window.ts': {
        'best-time-to-buy-sell-stock': (
            'https://leetcode.com/problems/best-time-to-buy-and-sell-stock/',
            'https://www.youtube.com/watch?v=1pkOgXD63yU'
        ),
        'longest-substring-without-repeating': (
            'https://leetcode.com/problems/longest-substring-without-repeating-characters/',
            'https://www.youtube.com/watch?v=wiGpQwVHdE0'
        ),
        'minimum-window-substring': (
            'https://leetcode.com/problems/minimum-window-substring/',
            'https://www.youtube.com/watch?v=jSto0O4AJbM'
        ),
    },
}

def add_links_to_problem(content, problem_id, leetcode_url, youtube_url):
    """Add missing LeetCode and YouTube links to a problem."""
    
    # Find the problem block - look for the problem ID and find where to insert links
    # Pattern: Find problem with this ID, then find the end (before order: or },)
    
    # Try to find where links should go - typically before order: or at the end before },
    pattern = rf"(id:\s*['\"]" + re.escape(problem_id) + r"['\'].*?)((?:order:|leetcodeUrl:|youtubeUrl:))"
    
    def replacement(match):
        before = match.group(1)
        next_field = match.group(2)
        
        # Check what's already there
        has_leetcode = 'leetcodeUrl:' in before
        has_youtube = 'youtubeUrl:' in before
        
        result = before
        
        # Add missing links
        if not has_leetcode:
            result += f"\n    leetcodeUrl: '{leetcode_url}',"
        if not has_youtube:
            result += f"\n    youtubeUrl: '{youtube_url}',"
        
        return result + "\n    " + next_field
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.MULTILINE)
    return new_content

# Process each file
updated_files = []
for filepath, problems in LINKS_TO_ADD.items():
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for problem_id, (leetcode, youtube) in problems.items():
            content = add_links_to_problem(content, problem_id, leetcode, youtube)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            updated_files.append(filepath)
            print(f"✓ Updated {filepath}")
    
    except FileNotFoundError:
        print(f"⚠ File not found: {filepath}")
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")

if updated_files:
    print(f"\n✅ Successfully updated {len(updated_files)} files!")
else:
    print("\n⚠ No files needed updates")


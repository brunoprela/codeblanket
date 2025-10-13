#!/usr/bin/env python3
"""Add missing LeetCode and YouTube links to algorithm problems."""

import re

# Map of problem IDs to their missing links
MISSING_LINKS = {
    # Advanced Graphs
    'network-delay-time': {
        'leetcode': 'https://leetcode.com/problems/network-delay-time/',
        'youtube': 'https://www.youtube.com/watch?v=EaphyqKU4PQ'
    },
    'cheapest-flights-within-k-stops': {
        'leetcode': 'https://leetcode.com/problems/cheapest-flights-within-k-stops/',
        'youtube': 'https://www.youtube.com/watch?v=5eIK3zUdYmE'
    },
    'path-with-minimum-effort': {
        'leetcode': 'https://leetcode.com/problems/path-with-minimum-effort/',
        'youtube': 'https://www.youtube.com/watch?v=XQlxCCx2vI4'
    },
    
    # Arrays & Hashing
    'contains-duplicate': {
        'leetcode': 'https://leetcode.com/problems/contains-duplicate/',
        'youtube': 'https://www.youtube.com/watch?v=3OamzN90kPg'
    },
    'two-sum': {
        'leetcode': 'https://leetcode.com/problems/two-sum/',
        'youtube': 'https://www.youtube.com/watch?v=KLlXCFG5TnA'
    },
    'group-anagrams': {
        'leetcode': 'https://leetcode.com/problems/group-anagrams/',
        'youtube': 'https://www.youtube.com/watch?v=vzdNOK2oB2E'
    },
    
    # Backtracking
    'subsets': {
        'leetcode': 'https://leetcode.com/problems/subsets/',
        'youtube': 'https://www.youtube.com/watch?v=REOH22Xwdkk'
    },
    'permutations': {
        'leetcode': 'https://leetcode.com/problems/permutations/',
        'youtube': 'https://www.youtube.com/watch?v=s7AvT7cGdSo'
    },
    'n-queens': {
        'leetcode': 'https://leetcode.com/problems/n-queens/',
        'youtube': 'https://www.youtube.com/watch?v=Ph95IHmRp5M'
    },
    
    # BFS
    'binary-tree-level-order': {
        'leetcode': 'https://leetcode.com/problems/binary-tree-level-order-traversal/',
        'youtube': 'https://www.youtube.com/watch?v=6ZnyEApgFYg'
    },
    'shortest-path-binary-matrix': {
        'leetcode': 'https://leetcode.com/problems/shortest-path-in-binary-matrix/',
        'youtube': 'https://www.youtube.com/watch?v=caXJJOMLyHk'
    },
    'rotting-oranges': {
        'leetcode': 'https://leetcode.com/problems/rotting-oranges/',
        'youtube': 'https://www.youtube.com/watch?v=y704fEOx0s0'
    },
    
    # Binary Search
    'binary-search': {
        'leetcode': 'https://leetcode.com/problems/binary-search/',
        'youtube': 'https://www.youtube.com/watch?v=s4DPM8ct1pI'
    },
    
    # Time & Space Complexity
    'pivot-index': {
        'leetcode': 'https://leetcode.com/problems/find-pivot-index/',
        'youtube': 'https://www.youtube.com/watch?v=kDfuxY37Zn4'
    },
    'first-unique-char': {
        'leetcode': 'https://leetcode.com/problems/first-unique-character-in-a-string/',
        'youtube': 'https://www.youtube.com/watch?v=5co5Gvp_-S0'
    },
    'duplicate-number': {
        'leetcode': 'https://leetcode.com/problems/find-the-duplicate-number/',
        'youtube': 'https://www.youtube.com/watch?v=wjYnzkAhcNk'
    },
    'valid-number': {
        'leetcode': 'https://leetcode.com/problems/valid-number/',
        'youtube': 'https://www.youtube.com/watch?v=QfRSeibcugw'
    },
    
    # Sorting
    'sort-array-parity': {
        'leetcode': 'https://leetcode.com/problems/sort-array-by-parity/',
        'youtube': 'https://www.youtube.com/watch?v=6YZn-z5jkrg'
    },
    'insertion-sort-list': {
        'leetcode': 'https://leetcode.com/problems/insertion-sort-list/',
        'youtube': 'https://www.youtube.com/watch?v=Kk6mXAzQ3zs'
    },
    'sort-list': {
        'leetcode': 'https://leetcode.com/problems/sort-list/',
        'youtube': 'https://www.youtube.com/watch?v=TGveA1oFhrc'
    },
    'count-smaller': {
        'leetcode': 'https://leetcode.com/problems/count-of-smaller-numbers-after-self/',
        'youtube': 'https://www.youtube.com/watch?v=2SVLYsq5W8M'
    },
    
    # More problems - I'll add more in batches
}

def add_links_to_problem(content, problem_id, links):
    """Add missing links to a problem."""
    # Find the problem block
    pattern = rf"(id:\s*['\"]" + re.escape(problem_id) + r"['\"][^}]*?)((?:leetcodeUrl:|youtubeUrl:|order:\s*\d+,\s*\},))"
    
    def replacement(match):
        problem_block = match.group(1)
        ending = match.group(2)
        
        # Check if links already exist
        has_leetcode = 'leetcodeUrl:' in problem_block
        has_youtube = 'youtubeUrl:' in problem_block
        
        result = problem_block
        
        # Add leetcode if missing
        if not has_leetcode and 'leetcode' in links:
            result += f"\n    leetcodeUrl: '{links['leetcode']}',"
        
        # Add youtube if missing
        if not has_youtube and 'youtube' in links:
            result += f"\n    youtubeUrl: '{links['youtube']}',"
        
        return result + "\n    " + ending
    
    return re.sub(pattern, replacement, content, flags=re.DOTALL)

# Process files
files_to_update = {
    'lib/problems/time-space-complexity.ts': ['pivot-index', 'first-unique-char', 'duplicate-number', 'valid-number'],
    'lib/problems/sorting.ts': ['sort-array-parity', 'insertion-sort-list', 'sort-list', 'count-smaller'],
}

for filepath, problem_ids in files_to_update.items():
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        modified = False
        for problem_id in problem_ids:
            if problem_id in MISSING_LINKS:
                new_content = add_links_to_problem(content, problem_id, MISSING_LINKS[problem_id])
                if new_content != content:
                    content = new_content
                    modified = True
                    print(f"✓ Added links to {problem_id} in {filepath}")
        
        if modified:
            with open(filepath, 'w') as f:
                f.write(content)
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")

print("\n✅ Done!")

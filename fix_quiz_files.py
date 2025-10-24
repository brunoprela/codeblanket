#!/usr/bin/env python3
import re
import sys

files_to_fix = [
    'lib/content/quizzes/ai-safety-guardrails/content-moderation.ts',
    'lib/content/quizzes/ai-safety-guardrails/pii-detection-removal.ts',
    'lib/content/quizzes/ml-advanced-deep-learning/attention-mechanism.ts',
    'lib/content/quizzes/ml-advanced-deep-learning/generative-adversarial-networks.ts',
]

for filepath in files_to_fix:
    print(f"\nProcessing: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_size = len(content)
        
        # Find sampleAnswer fields with multi-line strings
        # Pattern: sampleAnswer:\n      "..." (with actual newlines in the string)
        pattern = r'(sampleAnswer:\s*\n\s*["\'])((?:[^"\'\\]|\\.)*)(["\'],?)'
        
        def escape_string(match):
            prefix = match.group(1)
            string_content = match.group(2)
            suffix = match.group(3)
            
            # Escape the string content
            escaped = string_content.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r')
            
            return f"{prefix}{escaped}{suffix}"
        
        # Try a simpler approach - just escape actual newlines in quoted strings
        lines = content.split('\n')
        result_lines = []
        in_string = False
        string_start_char = None
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this line has sampleAnswer:
            if 'sampleAnswer:' in line and not in_string:
                result_lines.append(line)
                i += 1
                # Next line should start the string
                if i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startsWith('"') or next_line.startsWith("'"):
                        # Start collecting the multi-line string
                        string_start_char = next_line[0]
                        in_string = True
                        collected_string = next_line[1:]  # Remove opening quote
                        indent = len(lines[i]) - len(lines[i].lstrip())
                        
                        # Collect until we find the closing quote
                        i += 1
                        while i < len(lines) and in_string:
                            current = lines[i]
                            # Check if this line ends the string
                            stripped = current.strip()
                            if stripped.endswith(string_start_char + ',') or stripped.endswith(string_start_char):
                                # Found the end
                                collected_string += current.rstrip().rstrip(',').rstrip(string_start_char)
                                in_string = False
                                
                                # Now escape and write as single line
                                escaped = collected_string.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                                result_lines.append(' ' * indent + '"' + escaped + '",')
                                i += 1
                                break
                            else:
                                collected_string += '\n' + current
                                i += 1
                continue
            
            result_lines.append(line)
            i += 1
        
        new_content = '\n'.join(result_lines)
        
        if len(new_content) != original_size:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  ✓ Fixed! ({original_size} → {len(new_content)} bytes)")
        else:
            print(f"  ℹ No changes needed")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\nDone!")


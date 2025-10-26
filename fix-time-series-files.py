#!/usr/bin/env python3
"""
Batch fix all Time Series Analysis module files.
This script:
1. Renames quiz files (removes -discussion suffix)
2. Renames multiple-choice files (removes -quiz suffix)
3. Updates export names in all files
"""

import os
import re

BASE_DIR = '/Users/bruno/Developer/codeblanket/frontend/lib/content'

def fix_quiz_files():
    """Fix all quiz files: rename and update exports."""
    quiz_dir = os.path.join(BASE_DIR, 'quizzes/time-series-analysis')
    
    for filename in os.listdir(quiz_dir):
        if not filename.endswith('.ts'):
            continue
            
        old_path = os.path.join(quiz_dir, filename)
        
        # Read content
        with open(old_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix export name
        content = re.sub(r'export const (\w+)DiscussionQuestions =', r'export const \1Quiz =', content)
        
        # Determine new filename
        if filename.endswith('-discussion.ts'):
            new_filename = filename.replace('-discussion.ts', '.ts')
        else:
            new_filename = filename
        
        new_path = os.path.join(quiz_dir, new_filename)
        
        # Write corrected content
        with open(new_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Delete old file if different
        if old_path != new_path and os.path.exists(old_path):
            os.remove(old_path)
            print(f'✓ Quiz: {filename} → {new_filename}')

def fix_multiple_choice_files():
    """Fix all multiple-choice files: rename and update exports."""
    mc_dir = os.path.join(BASE_DIR, 'multiple-choice/time-series-analysis')
    
    for filename in os.listdir(mc_dir):
        if not filename.endswith('.ts'):
            continue
            
        old_path = os.path.join(mc_dir, filename)
        
        # Read content
        with open(old_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix export name
        content = re.sub(r'export const (\w+)MultipleChoiceQuestions =', r'export const \1MultipleChoice =', content)
        
        # Determine new filename
        if filename.endswith('-quiz.ts'):
            new_filename = filename.replace('-quiz.ts', '.ts')
        else:
            new_filename = filename
        
        new_path = os.path.join(mc_dir, new_filename)
        
        # Write corrected content
        with open(new_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Delete old file if different
        if old_path != new_path and os.path.exists(old_path):
            os.remove(old_path)
            print(f'✓ Multiple Choice: {filename} → {new_filename}')

def verify_files():
    """Verify all files are correctly named and formatted."""
    sections_dir = os.path.join(BASE_DIR, 'sections/time-series-analysis')
    quiz_dir = os.path.join(BASE_DIR, 'quizzes/time-series-analysis')
    mc_dir = os.path.join(BASE_DIR, 'multiple-choice/time-series-analysis')
    
    sections = sorted([f for f in os.listdir(sections_dir) if f.endswith('.ts')])
    quizzes = sorted([f for f in os.listdir(quiz_dir) if f.endswith('.ts')])
    mcs = sorted([f for f in os.listdir(mc_dir) if f.endswith('.ts')])
    
    print(f'\n=== Verification ===')
    print(f'Sections: {len(sections)} files')
    print(f'Quizzes: {len(quizzes)} files')
    print(f'Multiple Choice: {len(mcs)} files')
    
    if len(sections) == len(quizzes) == len(mcs) == 14:
        print('✓ All files present!')
    else:
        print('⚠ Warning: File count mismatch')
    
    # Check for old naming
    problems = []
    for f in quizzes:
        if '-discussion' in f:
            problems.append(f'Quiz file still has -discussion: {f}')
    for f in mcs:
        if '-quiz' in f:
            problems.append(f'MC file still has -quiz: {f}')
    
    if problems:
        print('\n⚠ Issues found:')
        for p in problems:
            print(f'  - {p}')
    else:
        print('\n✓ All files correctly named!')

if __name__ == '__main__':
    print('Fixing Time Series Analysis module files...\n')
    
    fix_quiz_files()
    print()
    fix_multiple_choice_files()
    print()
    verify_files()
    
    print('\n✅ Done!')


#!/usr/bin/env python3
import os
import re

# Rename quiz files (remove -discussion suffix)
quiz_dir = '/Users/bruno/Developer/codeblanket/frontend/lib/content/quizzes/time-series-analysis'
for filename in os.listdir(quiz_dir):
    if filename.endswith('-discussion.ts'):
        old_path = os.path.join(quiz_dir, filename)
        new_filename = filename.replace('-discussion.ts', '.ts')
        new_path = os.path.join(quiz_dir, new_filename)
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')
        
        # Update export name inside the file
        with open(new_path, 'r') as f:
            content = f.read()
        content = content.replace('DiscussionQuestions', 'Quiz')
        with open(new_path, 'w') as f:
            f.write(content)

# Rename multiple-choice files (remove -quiz suffix)
mc_dir = '/Users/bruno/Developer/codeblanket/frontend/lib/content/multiple-choice/time-series-analysis'
for filename in os.listdir(mc_dir):
    if filename.endswith('-quiz.ts'):
        old_path = os.path.join(mc_dir, filename)
        new_filename = filename.replace('-quiz.ts', '.ts')
        new_path = os.path.join(mc_dir, new_filename)
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')
        
        # Update export name inside the file
        with open(new_path, 'r') as f:
            content = f.read()
        content = content.replace('MultipleChoiceQuestions', 'MultipleChoice')
        with open(new_path, 'w') as f:
            f.write(content)

print('Done!')


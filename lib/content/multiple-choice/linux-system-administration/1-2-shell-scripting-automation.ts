/**
 * Multiple choice questions for Shell Scripting for Automation
 */

import { MultipleChoiceQuestion } from '../../../types';

export const shellScriptingAutomationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'shell-script-mc-1',
      question:
        'What is the purpose of "set -euo pipefail" at the beginning of a bash script?',
      options: [
        'It enables debug mode and verbose output',
        'It exits on errors, treats unset variables as errors, and fails on pipe errors',
        'It sets environment variables for production use',
        'It enables strict POSIX compliance mode',
      ],
      correctAnswer: 1,
      explanation:
        'set -euo pipefail is the "strict mode" for bash scripts. -e exits immediately if any command fails, -u treats unset variables as errors (prevents typos), and -o pipefail makes pipelines fail if any command in the pipe fails (not just the last one). This prevents silent failures and makes scripts more reliable in production.',
      difficulty: 'medium',
      topic: 'Error Handling',
    },
    {
      id: 'shell-script-mc-2',
      question:
        'Which command correctly processes 100 files in parallel with a maximum of 10 concurrent operations?',
      options: [
        'for file in *.txt; do process_file "$file" & done',
        'xargs -P 10 -n 1 process_file < file_list.txt',
        'parallel --jobs 10 process_file ::: *.txt',
        'find . -name "*.txt" -exec process_file {} \\;',
      ],
      correctAnswer: 2,
      explanation:
        'GNU Parallel with --jobs 10 correctly limits concurrency to 10 parallel operations. Option 1 would launch all 100 at once (resource exhaustion). Option 2 (xargs) could work but parallel has better features like progress bars, retry logic, and job logging. Option 4 (find -exec) runs serially, one at a time.',
      difficulty: 'advanced',
      topic: 'Parallelization',
    },
    {
      id: 'shell-script-mc-3',
      question:
        'What is the safest way to delete files in a directory stored in a variable?',
      options: [
        'rm -rf $dir/*',
        'rm -rf "$dir"/*',
        'rm -rf "${dir:?}"/*',
        'cd "$dir" && rm -rf ./*',
      ],
      correctAnswer: 2,
      explanation:
        '${dir:?} fails with an error if $dir is unset or empty, preventing catastrophic "rm -rf /*" if the variable is empty. Option 1 is dangerous (unquoted). Option 2 is better but still risky if $dir is empty. Option 4 is safer but cd can fail silently. The :? parameter expansion is the safest approach for destructive operations.',
      difficulty: 'advanced',
      topic: 'Safety',
    },
    {
      id: 'shell-script-mc-4',
      question:
        'Which jq command extracts the "name" field from all items in a JSON array?',
      options: [
        'jq ".items.name" file.json',
        'jq ".items[].name" file.json',
        'jq -r ".items | map(.name)" file.json',
        'Both B and C are correct',
      ],
      correctAnswer: 3,
      explanation:
        'Both options work but produce different formats. jq ".items[].name" outputs each name on a separate line (better for piping to other commands). jq -r ".items | map(.name)" outputs a JSON array of names. The -r flag in C makes output raw (without quotes). Both are valid depending on your use case, making D the most complete answer.',
      difficulty: 'medium',
      topic: 'JSON Processing',
    },
    {
      id: 'shell-script-mc-5',
      question: 'What is the purpose of "trap cleanup EXIT" in a bash script?',
      options: [
        'It catches errors and logs them to a file',
        'It ensures the cleanup function runs when the script exits (success or failure)',
        'It prevents the script from being interrupted by Ctrl+C',
        'It sets up a signal handler for SIGTERM only',
      ],
      correctAnswer: 1,
      explanation:
        'trap cleanup EXIT ensures the cleanup function runs whenever the script exits, regardless of how it exits (normal completion, error, or interrupt). This is essential for cleaning up temp files, releasing locks, or closing connections. EXIT is a pseudo-signal that triggers on any exit. This is different from trapping specific signals like SIGTERM or SIGINT.',
      difficulty: 'medium',
      topic: 'Error Handling',
    },
  ];

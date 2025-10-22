/**
 * Multiple choice questions for File System Operations & Path Handling section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const filesystemoperationspathhandlingMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'fpdu-fs-ops-mc-1',
        question:
            'What is the correct way to create a cross-platform file path in Python?',
        options: [
            'path = "C:\\\\Users\\\\data\\\\file.txt"',
            'path = "/Users/data/file.txt"',
            'path = Path("Users") / "data" / "file.txt"',
            'path = os.path.join("C:", "\\\\Users\\\\", "data", "file.txt")',
        ],
        correctAnswer: 2,
        explanation:
            'Using pathlib\'s Path with the / operator is the most cross-platform approach. It automatically handles path separators for the current OS.',
    },
    {
        id: 'fpdu-fs-ops-mc-2',
        question:
            'Why is atomic file writing important in production applications?\n\nwith open("file.txt", "w") as f:\n    f.write(content)  # vs atomic write',
        options: [
            'It is faster than regular file writing',
            'It prevents partial writes if the process crashes',
            'It automatically creates backup files',
            'It allows multiple processes to write simultaneously',
        ],
        correctAnswer: 1,
        explanation:
            'Atomic writes prevent file corruption by ensuring you never have a partially-written file. You either get the complete new content or the old content remains unchanged.',
    },
    {
        id: 'fpdu-fs-ops-mc-3',
        question:
            'What is the best way to read a 10GB log file for processing with an LLM?',
        options: [
            'content = Path("large.log").read_text()',
            'with open("large.log") as f: content = f.read()',
            'for line in open("large.log"): process(line)',
            'content = open("large.log").readlines()',
        ],
        correctAnswer: 2,
        explanation:
            'Reading line-by-line with iteration uses constant memory regardless of file size. The other options load the entire file into memory, which would fail or cause swapping with a 10GB file.',
    },
    {
        id: 'fpdu-fs-ops-mc-4',
        question:
            'What does path.parent.mkdir(parents=True, exist_ok=True) do?',
        options: [
            'Creates only the immediate parent directory',
            'Creates all parent directories and does not error if they exist',
            'Deletes existing parent directories and creates new ones',
            'Creates the file and all parent directories',
        ],
        correctAnswer: 1,
        explanation:
            'parents=True creates all intermediate directories, and exist_ok=True prevents errors if directories already exist. This is the safe pattern for ensuring parent directories exist before writing a file.',
    },
    {
        id: 'fpdu-fs-ops-mc-5',
        question:
            'Which approach is correct for avoiding race conditions when checking file existence?',
        options: [
            'if Path("file.txt").exists(): content = Path("file.txt").read_text()',
            'try: content = Path("file.txt").read_text()\nexcept FileNotFoundError: handle_error()',
            'assert Path("file.txt").exists()\ncontent = Path("file.txt").read_text()',
            'if os.path.exists("file.txt"): f = open("file.txt")',
        ],
        correctAnswer: 1,
        explanation:
            'EAFP (Easier to Ask Forgiveness than Permission) is the Pythonic approach. Using try/except avoids race conditions where a file could be deleted between the existence check and the read operation.',
    },
];


import { MultipleChoiceQuestion } from '@/lib/types';

export const codeQualityToolsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'cqt-mc-1',
    question: "What is Black\'s main philosophy?",
    options: [
      'Highly configurable formatting with many options',
      'Uncompromising auto-formatting with minimal configuration',
      'Manual code review and formatting',
      'Formatting suggestions without auto-fix',
    ],
    correctAnswer: 1,
    explanation:
      'Black is "uncompromising": One style, minimal config, no debates. Philosophy: Formatters should auto-fix, not suggest. Config: Only line-length (default 88). Benefits: No style debates, consistent across projects, fast (1000 files in seconds), git-friendly diffs. Not configurable (intentional—prevents bikeshedding), not manual (auto-formats), not suggestions (fixes automatically). Usage: black myapp/ → formats everything. Essential for team consistency.',
  },
  {
    id: 'cqt-mc-2',
    question:
      'Why use Ruff instead of Flake8, isort, and pyupgrade separately?',
    options: [
      'Ruff has more rules than the others combined',
      'Ruff is 10-100× faster and replaces all three tools',
      'Ruff is required by Python 3.11+',
      'Ruff auto-fixes all issues while others cannot',
    ],
    correctAnswer: 1,
    explanation:
      'Ruff is extremely fast (written in Rust): 10-100× faster than Flake8+isort+pyupgrade combined. Example: 1000 files: Flake8+isort 30s, Ruff 0.3s. Replaces: Flake8 (linting), isort (import sorting), pyupgrade (syntax upgrades), and 50+ other tools. Auto-fix: Partial (like isort), not all rules. Not more rules (comparable), not required (optional), not unique auto-fix. Benefit: One fast tool instead of 5 slow tools. Essential for large projects.',
  },
  {
    id: 'cqt-mc-3',
    question: 'What does mypy check?',
    options: [
      'Code formatting and style',
      'Type annotations match actual usage',
      'Security vulnerabilities',
      'Test coverage',
    ],
    correctAnswer: 1,
    explanation:
      'mypy checks type hints match actual usage: def add (a: int, b: int) -> int: return a + b; add("5", 10) → mypy error: Expected int, got str. Catches: Type mismatches, missing arguments, wrong return types, None errors. Example: def get_user (id: int) -> User: ... ; user: str = get_user(1) → mypy error: Incompatible types. Not formatting (Black), not security (bandit), not coverage (pytest-cov). Critical for catching bugs before runtime.',
  },
  {
    id: 'cqt-mc-4',
    question: 'What does bandit scan for?',
    options: [
      'Code style violations and formatting issues',
      'Security vulnerabilities like SQL injection and hardcoded passwords',
      'Type annotation errors',
      'Test coverage gaps',
    ],
    correctAnswer: 1,
    explanation:
      'bandit finds security issues: SQL injection (f"SELECT * FROM users WHERE id={user_id}"), shell injection (subprocess.call (shell=True)), insecure deserialization (pickle.loads), hardcoded passwords, debug=True in production, MD5/weak crypto. Severity: HIGH (critical), MEDIUM, LOW. Example: query = f"SELECT * FROM users WHERE id={id}" → bandit: HIGH - SQL injection. Not style (Black/Ruff), not types (mypy), not coverage (pytest-cov). Essential CI check: Fail on HIGH severity.',
  },
  {
    id: 'cqt-mc-5',
    question: 'What does cyclomatic complexity measure?',
    options: [
      'Number of lines of code in a function',
      'Number of independent execution paths through code',
      'Number of function calls',
      'Code duplication percentage',
    ],
    correctAnswer: 1,
    explanation:
      "Cyclomatic complexity = number of paths through code: def simple (x): return x * 2 → complexity 1 (one path). def with_if (x): if x > 0: return x else: return 0 → complexity 2 (two paths). Each if/elif/else/and/or adds path. Thresholds: 1-5 (A, simple), 6-10 (B, OK), 11-20 (C, complex), 21+ (D-F, refactor). Not LOC (that's raw metrics), not calls (separate metric), not duplication (similarity). Use radon cc: radon cc myapp/ -a. High complexity → hard to test, maintain.",
  },
];

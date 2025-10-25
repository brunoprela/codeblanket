import { Quiz } from '@/lib/types';

export const gitTradingStrategiesMultipleChoice: Quiz = {
  title: 'Git for Trading Strategies Quiz',
  description:
    'Test your knowledge of version control best practices for quantitative trading.',
  questions: [
    {
      id: 'git-1',
      question:
        'Why is it critical to use `nbstripout` or similar tools when committing Jupyter notebooks to Git in a quantitative trading project?',
      options: [
        'To reduce file size by compressing notebook contents',
        'To remove cell outputs and metadata, preventing false diffs and merge conflicts from execution state changes',
        'To automatically convert notebooks to Python scripts',
        'To encrypt sensitive code before committing',
      ],
      correctAnswer: 1,
      explanation:
        "`nbstripout` removes cell outputs, execution counts, and metadata from notebooks before Git commits. Without this: (1) Every execution changes the notebook even if code didn't change, creating meaningless diffs, (2) Outputs can make notebooks 100MB+ bloating the repository, (3) Execution metadata differs between users causing merge conflicts, (4) Accidentally committed chart outputs might contain sensitive data. With `nbstripout --install`, outputs are automatically stripped on commit but preserved locally for interactive work.",
    },
    {
      id: 'git-2',
      question:
        'What does semantic versioning (e.g., v2.3.1) communicate about changes to a trading strategy?',
      options: [
        'v2 = year, 3 = month, 1 = day of deployment',
        'MAJOR.MINOR.PATCH where MAJOR = breaking changes, MINOR = new features (compatible), PATCH = bug fixes',
        'The number of developers who worked on the version',
        'The Sharpe ratio multiplied by 100',
      ],
      correctAnswer: 1,
      explanation:
        'Semantic versioning (SemVer) format is MAJOR.MINOR.PATCH. For trading strategies: MAJOR version (2) = breaking changes like complete strategy overhaul or different asset universe; MINOR version (3) = new features that are backward compatible like adding a new indicator or filter; PATCH version (1) = bug fixes and small corrections. This immediately communicates impact: v2.3.1 → v2.3.2 is low-risk (bug fix), v2.3.1 → v2.4.0 is medium-risk (new feature), v2.3.1 → v3.0.0 is high-risk (major changes). Critical for deployment decisions and rollback planning.',
    },
    {
      id: 'git-3',
      question:
        'What is the primary purpose of a pre-commit Git hook in a trading strategy repository?',
      options: [
        'To automatically push changes to the remote repository',
        'To run automated checks (tests, linters, security scans) before allowing the commit, preventing bad code from entering history',
        'To backup the repository to an external drive',
        'To notify team members via email about the commit',
      ],
      correctAnswer: 1,
      explanation:
        'Pre-commit hooks run automatically before `git commit` completes, allowing you to enforce quality standards. Common checks for trading: (1) Run unit tests - prevent commits that break tests, (2) Check code formatting (Black, flake8) - maintain style consistency, (3) Scan for debugger statements (pdb.set_trace()) - prevent debug code in production, (4) Check for exposed secrets (API keys, passwords) - security, (5) Run quick backtest - verify strategy still works. If any check fails, the commit is aborted and the developer must fix issues. This creates a quality gate preventing bad code from polluting Git history.',
    },
    {
      id: 'git-4',
      question:
        "In a quantitative trading team using the 'feature branch workflow,' when should a researcher merge their feature branch into the main branch?",
      options: [
        'Immediately after creating the feature branch',
        'After the feature is complete, tested, peer-reviewed via pull request, and approved by the team lead',
        'Once per day at 5pm to synchronize with the team',
        'Never - feature branches should remain separate indefinitely',
      ],
      correctAnswer: 1,
      explanation:
        'The feature branch workflow ensures quality through a structured process: (1) Create feature branch for new work, (2) Develop and commit frequently to feature branch, (3) When complete, create Pull Request (PR) with detailed description and backtest results, (4) Team lead reviews code checking for correctness, lookahead bias, performance, documentation, (5) Reviewer requests changes if needed, (6) After approval and all tests pass, merge to main. This prevents broken code in main, enables experimentation without risk, facilitates knowledge sharing through code review, and maintains high code quality. The main branch should always be production-ready.',
    },
    {
      id: 'git-5',
      question:
        'What is the correct approach for handling API keys and database passwords in a Git repository for a trading system?',
      options: [
        'Commit them in plain text to config files so everyone has access',
        'Store them in .env or secrets.yaml files that are explicitly excluded in .gitignore and never committed to Git',
        'Encrypt them with ROT13 before committing',
        "Put them in commit messages where they're easy to find",
      ],
      correctAnswer: 1,
      explanation:
        "NEVER commit secrets to Git - they remain in history forever even if deleted later. Correct approach: (1) Store secrets in separate files like `.env` or `config/secrets.yaml`, (2) Add these files to `.gitignore` so Git never tracks them, (3) Load secrets from environment variables or config files at runtime, (4) Commit a template file (e.g., `secrets.yaml.template`) with placeholder values as documentation, (5) For team sharing, use encrypted secrets (git-crypt) or secret management services (AWS Secrets Manager, HashiCorp Vault). If you accidentally commit a secret: rotating the secret immediately is the ONLY solution - removing it from history isn't enough as it may have been pulled by others or scraped by bots.",
    },
  ],
};

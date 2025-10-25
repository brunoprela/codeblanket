import { Discussion } from '@/lib/types';

export const gitTradingStrategiesQuiz: Discussion = {
  title: 'Git for Trading Strategies Discussion Questions',
  description:
    'Deep dive into version control workflows and best practices for quantitative trading teams.',
  questions: [
    {
      id: 'git-disc-1',
      question:
        'Design a comprehensive Git workflow for a quantitative trading team of 5 researchers developing multiple strategies simultaneously. Include branching strategy, code review process, deployment tracking, and how to handle both experimental research and production-critical hotfixes.',
      sampleAnswer: `[Comprehensive answer covering: Gitflow model with main/develop/feature/hotfix branches, pull request workflow with review checklist for trading-specific concerns (lookahead bias, performance metrics, risk management), semantic versioning and tagging for deployments, CI/CD pipeline for automated testing and backtesting, handling notebook-heavy research vs production Python code, managing secrets and credentials, deployment log maintenance, and rollback procedures. Include example branch structure, PR template for strategy changes, and automation workflows.]`,
    },
    {
      id: 'git-disc-2',
      question:
        'Explain the challenges of version controlling Jupyter notebooks in quantitative research and provide a detailed solution strategy. Cover diff viewing, output management, collaboration, and transitioning notebook code to production.',
      sampleAnswer: `[Detailed answer on: Why raw notebooks are problematic for Git (JSON format, outputs change on every execution, large file sizes, merge conflicts), solution approaches including nbstripout for output removal, Jupytext for paired .py files, nbdime for better diffs, versioning strategies (when to commit, what to include), extracting production code from notebooks to modules, testing notebook-derived code, and maintaining reproducibility while enabling collaboration.]`,
    },
    {
      id: 'git-disc-3',
      question:
        'You discover that a researcher accidentally committed an API key to the repository 10 commits ago, and the repository has been pushed to GitHub. Outline the complete incident response procedure, including immediate actions, history cleanup, security considerations, and preventive measures.',
      sampleAnswer: `[Comprehensive incident response: Immediate actions (1. Rotate/revoke the exposed key ASAP - historical removal alone insufficient, 2. Check if key was accessed/abused via provider logs, 3. Notify security team and management), History cleanup (git filter-branch or BFG Repo-Cleaner to remove from all commits, force-push to remote - requires team coordination), Prevention measures (pre-commit hooks to scan for secrets, .gitignore for config files, git-crypt for encrypted secrets, secret management services, team training on security), Documentation (incident report, lessons learned), Ongoing monitoring (secret scanning tools like GitHub Secret Scanning, truffleHog). Include code examples for cleanup and prevention.]`,
    },
  ],
};

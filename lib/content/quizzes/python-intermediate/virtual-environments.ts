/**
 * Quiz questions for Virtual Environments & Package Management section
 */

export const virtualenvironmentsQuiz = [
  {
    id: 'q1',
    question:
      'What problems do virtual environments solve? Why should every Python project use one?',
    hint: 'Think about package versions, system cleanliness, and reproducibility.',
    sampleAnswer:
      'Virtual environments solve several critical problems: 1) **Version conflicts**: Project A needs Django 3.2, Project B needs Django 4.2—without virtual environments, only one can be installed. 2) **Reproducibility**: requirements.txt ensures everyone working on the project has identical dependencies. 3) **Clean system**: Installing packages globally pollutes your system Python, potentially breaking system tools. 4) **Experimentation**: Safely try new packages without affecting other projects. 5) **Multiple Python versions**: Different projects can use Python 3.9, 3.10, 3.11 side-by-side. For example, data science projects often need specific NumPy/Pandas versions that conflict with web projects. Virtual environments make this trivial.',
    keyPoints: [
      'Isolates dependencies per project',
      'Prevents version conflicts',
      'Enables reproducibility with requirements.txt',
      'Keeps global Python clean',
      'Allows multiple Python versions',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the difference between pip freeze and manually maintaining requirements.txt. What are the pros and cons?',
    hint: 'Consider transitive dependencies, version pinning, and maintainability.',
    sampleAnswer:
      "**pip freeze**: Outputs ALL installed packages including transitive dependencies (dependencies of dependencies) with exact versions. **Pros**: Completely reproducible—guarantees exact environment. **Cons**: Includes packages you didn't explicitly install, making it hard to see actual dependencies; updates are all-or-nothing. **Manual requirements.txt**: List only direct dependencies, optionally with version ranges (>=2.0,<3.0). **Pros**: Clear what your project actually needs; allows minor updates for security patches. **Cons**: Different Python versions or platforms might resolve dependencies differently. **Best practice**: Use pip freeze for production (exact reproducibility), manual for development (flexibility). Or use Poetry which tracks both: pyproject.toml for direct deps, poetry.lock for exact versions.",
    keyPoints: [
      'pip freeze: all packages with exact versions',
      'Manual: only direct dependencies',
      'freeze pros: exact reproducibility',
      'Manual pros: clarity and flexibility',
      'Use Poetry for best of both worlds',
    ],
  },
  {
    id: 'q3',
    question:
      'When should you use venv vs virtualenv vs Poetry vs Conda? What are the use cases for each?',
    hint: 'Consider built-in vs third-party, Python-only vs non-Python deps, simplicity vs features.',
    sampleAnswer:
      '**venv**: Built into Python 3.3+, perfect for most projects. Use for: simple projects, when you want standard tooling, teaching. **virtualenv**: Third-party, more features than venv (faster, supports older Python). Use for: need Python 2.7 support (legacy), need certain advanced features. **Poetry**: Modern dependency management with lockfiles. Use for: libraries (publishing to PyPI), complex applications, teams that need reproducible builds. Handles dependency resolution better than pip. **Conda**: Includes non-Python dependencies (C libraries, system tools). Use for: data science (needs NumPy, SciPy with optimized binaries), projects mixing Python with R/Julia, cross-platform scientific computing. **Recommendation**: Start with venv for learning, move to Poetry for serious projects, use Conda only for data science.',
    keyPoints: [
      'venv: built-in, simple, most common',
      'virtualenv: more features, older Python support',
      'Poetry: modern, dependency resolution, lockfiles',
      'Conda: data science, non-Python dependencies',
      'Choose based on project complexity and needs',
    ],
  },
];

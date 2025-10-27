import { Content } from '@/lib/types';

export const gitTradingStrategiesContent: Content = {
  title: 'Git for Trading Strategies',
  subtitle: 'Version control and collaboration for quantitative research',
  description:
    'Master Git and GitHub for managing trading strategies, backtests, and quantitative research. Learn professional workflows used by quant teams at hedge funds and prop trading firms.',
  sections: [
    {
      title: 'Why Version Control for Trading?',
      content: `
# The Critical Need for Version Control in Trading

## The Problem Without Version Control

\`\`\`plaintext
Common Disasters in Trading Development:

├── "It worked yesterday!" Syndrome
│   └── Strategy performed great, now broken
│   └── No record of what changed
│   └── Can't reproduce results
│
├── Multiple Versions Chaos
│   ├── strategy_v1.py
│   ├── strategy_v2.py
│   ├── strategy_v2_final.py
│   ├── strategy_v2_final_REALLY.py
│   └── strategy_v2_final_use_this_one.py
│
├── Collaboration Nightmares
│   ├── Two researchers editing same file
│   ├── Emailing code back and forth
│   ├── "Which version are you using?"
│   └── Lost work from overwriting
│
├── No Audit Trail
│   ├── Can't prove what code was running when
│   ├── Regulatory compliance issues
│   └── Can't investigate why strategy failed
│
└── Cannot Recover
    ├── Accidentally deleted working strategy
    ├── Broke something, can't undo
    └── No backup of research notebooks
\`\`\`

## What Git Provides for Trading

### 1. Complete History
Every change tracked with:
- What changed
- Who changed it
- When it was changed
- Why it was changed (commit message)

### 2. Branching for Experiments
\`\`\`plaintext
main (production strategy)
├── feature/add-volatility-filter
├── feature/optimize-parameters
├── experiment/ml-signals
└── hotfix/fix-calculation-bug

Try ideas without breaking working code
\`\`\`

### 3. Team Collaboration
Multiple researchers work simultaneously without conflicts

### 4. Deployment Tracking
Know exactly which code version is running in production

### 5. Reproducibility
Recreate exact environment and results from any point in history

### 6. Code Review
Team reviews changes before merging to production
      `,
    },
    {
      title: 'Git Fundamentals for Quants',
      content: `
# Essential Git Concepts and Commands

## Repository Structure

\`\`\`plaintext
quantitative-strategy/
├── .git/                      # Git internal database (don't touch)
├── .gitignore                 # Files to NOT track
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── src/                       # Source code (TRACKED)
│   ├── strategies/
│   ├── backtest/
│   └── data/
├── notebooks/                 # Jupyter notebooks (TRACKED, output stripped)
├── data/                      # Data files (NOT tracked - too large)
├── results/                   # Results (NOT tracked)
└── tests/                     # Unit tests (TRACKED)
\`\`\`

## .gitignore for Quant Projects

\`\`\`gitignore
# .gitignore for quantitative trading project

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.pytest_cache/
.coverage

# Jupyter Notebook
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Environment
.env
.venv
env/
venv/
ENV/

# Data files (usually too large for git)
data/raw/*
data/processed/*
!data/*/.gitkeep

# Results and outputs
results/*
!results/.gitkeep
*.log
*.csv
*.h5
*.parquet

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Credentials (CRITICAL - never commit!)
config/secrets.yaml
*.key
*.pem
credentials.json
\`\`\`

## Basic Git Workflow

### Initial Setup
\`\`\`bash
# Configure Git (one-time setup)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Create new repository
cd quantitative-strategy
git init

# Create .gitignore
cat > .gitignore << EOF
__pycache__/
*.pyc
.env
data/
results/
.ipynb_checkpoints/
EOF

# First commit
git add .
git commit -m "Initial commit: Project structure"
\`\`\`

### Daily Workflow

\`\`\`bash
# 1. Check current status
git status

# Output:
# On branch main
# Changes not staged for commit:
#   modified:   src/strategies/momentum.py
#   modified:   notebooks/backtest_analysis.ipynb
# 
# Untracked files:
#   src/strategies/new_strategy.py

# 2. See what changed
git diff src/strategies/momentum.py

# 3. Stage changes
git add src/strategies/momentum.py
git add src/strategies/new_strategy.py

# Or stage all changes
git add .

# 4. Commit with meaningful message
git commit -m "feat: Add volatility filter to momentum strategy

- Added ATR-based volatility calculation
- Only take trades when volatility > 20th percentile
- Backtests show 15% improvement in Sharpe ratio"

# 5. View commit history
git log --oneline --graph

# 6. Push to remote (GitHub/GitLab)
git push origin main
\`\`\`

## Commit Message Best Practices

### Structure
\`\`\`plaintext
<type>: <short summary> (50 chars or less)

<detailed explanation of what and why>
<include performance impact if relevant>

<reference to issue/ticket if applicable>
\`\`\`

### Types
\`\`\`plaintext
feat:     New feature (new indicator, strategy component)
fix:      Bug fix (calculation error, data handling issue)
refactor: Code restructuring (no functionality change)
perf:     Performance improvement
test:     Adding or updating tests
docs:     Documentation changes
data:     Data pipeline changes
backtest: Backtest-related changes
\`\`\`

### Examples

\`\`\`bash
# Good commit messages
git commit -m "feat: Implement RSI-based entry filter

Added 14-period RSI calculation to momentum strategy.
Only enter long positions when RSI < 70 (not overbought).
Reduces drawdowns by 8% while maintaining returns.

Closes #42"

git commit -m "fix: Correct forward-fill logic in data preprocessing

Previously, price data was incorrectly forward-filled across
weekends, creating lookahead bias. Now only fills within
trading days.

Impact: Strategy returns decrease from 25% to 18% (more realistic)"

git commit -m "perf: Vectorize moving average calculations

Replaced pandas apply() with numpy operations.
Backtest time reduced from 45s to 3s (15x speedup).
Results unchanged (verified with assertions)."

# Bad commit messages (don't do this!)
git commit -m "fix"
git commit -m "stuff"
git commit -m "updated code"
git commit -m "asdf"
\`\`\`

## Branching Strategy for Quant Development

### Branch Types

\`\`\`plaintext
Trading Strategy Git Flow:

main
├── develop                    # Integration branch
│   ├── feature/vol-filter     # New features
│   ├── feature/ml-signals
│   ├── experiment/regime-switching
│   └── hotfix/data-bug        # Urgent fixes
├── staging                    # Pre-production testing
└── production                 # Live trading code
\`\`\`

### Branch Workflow

\`\`\`bash
# 1. Create feature branch
git checkout -b feature/add-stop-loss

# 2. Develop and commit
git add src/risk/stop_loss.py
git commit -m "feat: Add ATR-based stop loss"

# 3. Push branch
git push -u origin feature/add-stop-loss

# 4. Create Pull Request on GitHub
# (via web interface)

# 5. After review and merge, clean up
git checkout main
git pull
git branch -d feature/add-stop-loss

# 6. Start next feature
git checkout -b feature/next-thing
\`\`\`

## Handling Notebooks with Git

### Problem: Notebooks Create Messy Diffs

\`\`\`bash
# Every execution changes output, creating false diff
git diff notebook.ipynb
# Shows hundreds of lines of JSON changes
# Even though code didn't change!
\`\`\`

### Solution 1: nbstripout (Recommended)

\`\`\`bash
# Install
pip install nbstripout

# Configure for repository
cd /path/to/repo
nbstripout --install

# Now outputs are auto-stripped on commit
git add notebook.ipynb
git commit -m "Add exploratory analysis notebook"
# Only code changes are committed, not outputs
\`\`\`

### Solution 2: Jupytext (Alternative)

\`\`\`bash
# Install
pip install jupytext

# Convert notebook to .py format
jupytext --to py notebook.ipynb

# Creates notebook.py (plain Python)
# Git tracks .py instead of .ipynb
# Can sync bidirectionally

# Configure Jupyter to auto-sync
jupytext --set-formats ipynb,py notebook.ipynb

# Now editing either file updates both
\`\`\`

## Viewing Notebook Diffs

\`\`\`bash
# Install nbdime for better notebook diffs
pip install nbdime

# Configure git to use nbdime
nbdime config-git --enable --global

# Now notebook diffs are readable
git diff notebook.ipynb
# Shows semantic diff of cells, not JSON

# Web-based diff viewer
nbdiff-web notebook_old.ipynb notebook_new.ipynb
\`\`\`
      `,
    },
    {
      title: 'Collaborative Workflows',
      content: `
# Team Collaboration for Quantitative Research

## Pull Request Workflow

### Step 1: Researcher Proposes Change

\`\`\`bash
# Create feature branch
git checkout -b feature/improve-sharpe-ratio

# Develop strategy improvement
# Edit src/strategies/momentum.py
# Add position sizing based on volatility

# Commit changes
git add src/strategies/momentum.py
git add tests/test_momentum.py
git commit -m "feat: Add vol-weighted position sizing

Implemented Kelly criterion for position sizing.
Positions now scaled by inverse volatility.

Backtest results:
- Sharpe: 1.45 → 1.78 (+23%)
- Max DD: -18% → -14%
- Win rate: unchanged at 58%"

# Push to remote
git push origin feature/improve-sharpe-ratio

# Create Pull Request on GitHub
# Title: "Improve Sharpe ratio with Kelly sizing"
# Description: Link to backtest notebook, explain methodology
\`\`\`

### Step 2: Code Review

Lead researcher reviews code:

\`\`\`markdown
## Pull Request Review Checklist

### Code Quality
- [ ] Code is well-documented with docstrings
- [ ] Variable names are clear and descriptive
- [ ] No hardcoded magic numbers
- [ ] Follows team style guide (PEP 8)

### Testing
- [ ] Unit tests added for new functions
- [ ] All tests pass (pytest tests/)
- [ ] No lookahead bias introduced
- [ ] Edge cases handled (missing data, zero division, etc.)

### Performance Validation
- [ ] Backtest results reproduced independently
- [ ] Walk-forward analysis performed
- [ ] Out-of-sample period tested
- [ ] Transaction costs included
- [ ] Slippage assumptions reasonable

### Risk Assessment
- [ ] Max drawdown acceptable
- [ ] Leverage within limits
- [ ] Correlation with existing strategies checked
- [ ] Stress testing performed

### Documentation
- [ ] README updated if needed
- [ ] Notebook documenting changes included
- [ ] Performance metrics clearly stated
- [ ] Assumptions documented

## Review Comments

**Positive:**
- Kelly criterion implementation looks solid
- Good test coverage
- Backtest improvements are significant

**Concerns:**1. Line 47: Kelly fraction might be too aggressive in high-vol regimes
2. Consider adding maximum position size cap
3. Need to test on 2008-2009 crisis period

**Requested Changes:**
- Add \`max_position_size\` parameter (default 20% of portfolio)
- Include backtest for 2008-2009 period
- Add docstring explaining Kelly fraction calculation

**Action:** Request changes
\`\`\`

### Step 3: Researcher Addresses Feedback

\`\`\`bash
# Make requested changes
git add src/strategies/momentum.py
git commit -m "refactor: Add max position size cap

Added max_position_size parameter (default 20%).
Prevents over-concentration in low-volatility periods.

2008-2009 backtest completed:
- Strategy survived with -25% max DD
- vs Buy-and-hold -55% DD
- Outperformed on risk-adjusted basis"

git push origin feature/improve-sharpe-ratio

# Respond to review comments on GitHub
\`\`\`

### Step 4: Approval and Merge

\`\`\`bash
# After approval, merge to develop branch
git checkout develop
git merge feature/improve-sharpe-ratio

# Run full test suite
pytest tests/

# If tests pass, merge to main
git checkout main
git merge develop

# Tag release
git tag -a v1.2.0 -m "Release 1.2.0: Kelly position sizing"
git push origin v1.2.0
\`\`\`

## Resolving Merge Conflicts

### Scenario: Two Researchers Edit Same File

\`\`\`bash
# Researcher A: Adds RSI filter
# File: src/strategies/momentum.py
def generate_signals (df):
    df['signal'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
    df.loc[df['rsi'] < 30, 'signal'] = 0  # Don't buy if oversold
    return df

# Researcher B: Adds volume filter
# Same file: src/strategies/momentum.py
def generate_signals (df):
    df['signal'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
    df.loc[df['volume'] < df['avg_volume'], 'signal'] = 0  # Need volume confirmation
    return df

# When merging: CONFLICT!
\`\`\`

### Resolving the Conflict

\`\`\`bash
# Try to merge
git merge feature/volume-filter

# Output:
# Auto-merging src/strategies/momentum.py
# CONFLICT (content): Merge conflict in src/strategies/momentum.py
# Automatic merge failed; fix conflicts and then commit the result.

# Open file - Git adds conflict markers
def generate_signals (df):
    df['signal'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
<<<<<<< HEAD
    df.loc[df['rsi'] < 30, 'signal'] = 0  # Don't buy if oversold
=======
    df.loc[df['volume'] < df['avg_volume'], 'signal'] = 0  # Need volume confirmation
>>>>>>> feature/volume-filter
    return df

# Manually resolve - keep both filters!
def generate_signals (df):
    df['signal'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
    df.loc[df['rsi'] < 30, 'signal'] = 0  # Don't buy if oversold
    df.loc[df['volume'] < df['avg_volume'], 'signal'] = 0  # Need volume confirmation
    return df

# Mark as resolved
git add src/strategies/momentum.py
git commit -m "merge: Combine RSI and volume filters"
\`\`\`

## Code Review Best Practices for Quant Teams

### Review Focus Areas

1. **Correctness**
   - Mathematical formulas implemented correctly
   - No off-by-one errors in lookbacks
   - Proper handling of NaN/infinity

2. **No Lookahead Bias**
   - All indicators use only past data
   - Signal generation doesn't peek into future
   - Train/test split doesn't leak

3. **Performance**
   - Vectorized operations where possible
   - No unnecessary loops
   - Database queries optimized

4. **Risk Management**
   - Position sizing reasonable
   - Stop losses implemented
   - Portfolio-level risk limits

5. **Documentation**
   - Clear docstrings
   - Assumptions stated explicitly
   - Example usage provided

### Review Comment Examples

\`\`\`markdown
**Excellent practices:**
👍 Great use of vectorization in returns calculation
👍 Comprehensive unit tests
👍 Clear docstrings with examples

**Suggestions:**
💡 Consider using @lru_cache on indicator calculation for performance
💡 This would benefit from a diagram in the docstring
💡 Might want to add logging for debugging production issues

**Concerns:**
⚠️ Line 34: This creates lookahead bias - signal uses tomorrow's close
⚠️ No test for missing data handling
⚠️ Hard-coded commission (should be parameter)

**Blocking issues:**
🚫 CRITICAL: Position sizing can exceed 100% of portfolio
🚫 Division by zero possible if volatility is zero (line 67)
🚫 This will break in production - needs error handling
\`\`\`

## GitHub Features for Quant Teams

### Issues for Task Tracking

\`\`\`markdown
Title: Backtest shows unexpected drawdown in 2020

**Type:** Bug
**Priority:** High
**Assigned to:** @researcher-alice

**Description:**
Strategy backtest for 2020 shows -35% drawdown, much worse than expected.
Need to investigate if this is due to:
1. COVID-19 volatility (expected)
2. Bug in code (not expected)
3. Data quality issue

**Steps to Reproduce:**1. Run python scripts/backtest.py --start 2020-01-01 --end 2020-12-31
2. Observe max drawdown in results/

**Expected:** ~15-20% max drawdown (based on historical average)
**Actual:** 35% max drawdown

**Environment:**
- Python 3.10
- pandas 2.0.0
- Code version: commit abc123f

**Additional Context:**
March 2020 saw extreme moves. Need to verify if:
- Circuit breakers handled correctly
- Slippage assumptions valid during high volatility
- Position sizing adapted to vol regime
\`\`\`

### Project Boards for Sprint Planning

\`\`\`plaintext
GitHub Project Board: Q1 2024 Strategy Development

├── Backlog
│   ├── Implement ML-based regime detection
│   ├── Add cryptocurrency markets
│   └── Research alternative data sources
├── To Do (This Sprint)
│   ├── Add volatility-based position sizing
│   ├── Optimize MA periods via grid search
│   └── Fix backtest data quality issues
├── In Progress
│   ├── Implement Kelly criterion (Alice)
│   └── Add transaction cost model (Bob)
├── Review
│   ├── Add RSI overbought filter (PR #23)
│   └── Refactor data pipeline (PR #24)
└── Done
    ├── Set up CI/CD pipeline
    ├── Add unit test framework
    └── Document strategy methodology
\`\`\`

### GitHub Actions for Automation

\`\`\`yaml
# .github/workflows/backtest.yml
name: Run Backtests on Push

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run unit tests
      run: |
        pytest tests/ -v
    
    - name: Run quick backtest
      run: |
        python scripts/backtest.py --quick
    
    - name: Check for lookahead bias
      run: |
        python scripts/check_lookahead_bias.py
    
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: backtest-results
        path: results/
\`\`\`
      `,
    },
    {
      title: 'Advanced Git for Production Trading',
      content: `
# Production-Grade Version Control

## Semantic Versioning for Trading Strategies

\`\`\`plaintext
Version Format: MAJOR.MINOR.PATCH

Example: v2.3.1

MAJOR (2): Breaking changes
- Complete strategy overhaul
- Different asset universe
- Incompatible with previous version

MINOR (3): New features (backward compatible)
- Added new indicator
- New position sizing method
- Additional filtering logic

PATCH (1): Bug fixes
- Corrected calculation error
- Fixed data handling issue
- Performance optimization
\`\`\`

### Tagging Releases

\`\`\`bash
# Tag major release
git tag -a v2.0.0 -m "v2.0.0: Major strategy overhaul

Complete rewrite using ML-based signals.
Not backward compatible with v1.x.

Backtest performance:
- Sharpe: 2.1 (was 1.6)
- Max DD: -12% (was -18%)
- Win rate: 63% (was 58%)"

# Push tags
git push origin --tags

# List all tags
git tag -l

# Checkout specific version
git checkout v2.0.0

# Create branch from tag
git checkout -b hotfix/v2.0.1 v2.0.0
\`\`\`

## Git Hooks for Quality Control

### Pre-Commit Hook: Run Tests

\`\`\`bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running pre-commit checks..."

# Run unit tests
echo "Running unit tests..."
pytest tests/ -q
if [ $? -ne 0 ]; then
    echo "Tests failed! Commit aborted."
    exit 1
fi

# Check for debugger statements
echo "Checking for debugger statements..."
if git diff --cached --name-only | xargs grep -n "import pdb\|breakpoint()" ; then
    echo "Found debugger statements! Remove before committing."
    exit 1
fi

# Check for API keys
echo "Checking for exposed API keys..."
if git diff --cached | grep -i "api_key\|secret_key\|password" ; then
    echo "Potential API key detected! Review carefully."
    exit 1
fi

# Run Black formatter
echo "Formatting code with Black..."
black src/ tests/
git add -u

echo "Pre-commit checks passed ✓"
exit 0
\`\`\`

### Pre-Push Hook: Run Backtests

\`\`\`bash
# .git/hooks/pre-push
#!/bin/bash

echo "Running quick backtest before push..."

# Run fast backtest on recent data
python scripts/backtest.py --quick --start 2023-01-01

if [ $? -ne 0 ]; then
    echo "Backtest failed! Push aborted."
    echo "Review strategy changes before pushing."
    exit 1
fi

echo "Backtest passed ✓"
exit 0
\`\`\`

## Managing Secrets and Credentials

### Never Commit Secrets!

\`\`\`python
# BAD: Hardcoded API key (don't do this!)
API_KEY = "sk-1234567890abcdef"  # Will be in git history forever!

# GOOD: Load from environment variable
import os
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# GOOD: Load from config file (not committed)
import yaml
with open('config/secrets.yaml') as f:
    secrets = yaml.safe_load (f)
API_KEY = secrets['alpha_vantage_key']
\`\`\`

### Config File Structure

\`\`\`yaml
# config/config.yaml (committed to git)
data:
  source: alpha_vantage
  start_date: 2020-01-01
  end_date: 2023-12-31

backtest:
  initial_capital: 100000
  commission: 0.001
  
# config/secrets.yaml (NOT committed)
alpha_vantage_key: "sk-1234567890abcdef"
database_password: "supersecret"
\`\`\`

### .gitignore Secrets

\`\`\`gitignore
# .gitignore
config/secrets.yaml
config/production.yaml
*.key
*.pem
credentials.json
.env
\`\`\`

### Encrypted Secrets (for team sharing)

\`\`\`bash
# Install git-crypt
brew install git-crypt  # macOS
apt-get install git-crypt  # Linux

# Initialize
cd /path/to/repo
git-crypt init

# Specify which files to encrypt
cat > .gitattributes << EOF
config/secrets.yaml filter=git-crypt diff=git-crypt
*.key filter=git-crypt diff=git-crypt
EOF

# Add collaborator's GPG key
git-crypt add-gpg-user USER_ID

# Commit - secrets are now encrypted in repo
git add config/secrets.yaml
git commit -m "Add encrypted secrets"

# Collaborators decrypt with:
git-crypt unlock
\`\`\`

## Deployment Tracking

### Track What\'s Running in Production

\`\`\`bash
# Tag deployment
git tag -a prod-20240115 -m "Deployed to production: 2024-01-15

Strategy version: v2.3.0
Server: prod-trading-01
Deployed by: alice@company.com

Changes since last deployment:
- Added volatility filter
- Optimized MA periods
- Improved position sizing

Performance monitoring:
- Dashboard: http://monitoring.company.com/strategy-v2.3.0
"

git push origin prod-20240115

# Know exactly what's running
git log prod-20240115

# Rollback if needed
git checkout prod-20240108  # Previous deployment
git tag -a prod-20240115-rollback -m "Rollback to previous version"
\`\`\`

### Deployment Log

\`\`\`markdown
# deployment_log.md (in repository)

## Deployment History

### 2024-01-15 - v2.3.0
- **Deployed by:** Alice
- **Git commit:** abc123f
- **Git tag:** prod-20240115
- **Changes:** Added volatility filter, optimized parameters
- **Reason:** Improve Sharpe ratio (backtested 1.85)
- **Status:** ✅ Running successfully
- **Performance:** Sharpe 1.92 (first week)

### 2024-01-08 - v2.2.0
- **Deployed by:** Bob
- **Git commit:** def456g
- **Git tag:** prod-20240108
- **Changes:** Fixed data handling bug
- **Reason:** Hotfix for incorrect fills on gaps
- **Status:** ✅ Replaced by v2.3.0
- **Performance:** Sharpe 1.78 (one week)

### 2024-01-01 - v2.1.0
- **Deployed by:** Alice
- **Git commit:** ghi789h
- **Git tag:** prod-20240101
- **Changes:** Initial v2 deployment
- **Reason:** Major strategy update
- **Status:** ⚠️ Had data bug, replaced by v2.2.0
- **Performance:** Sharpe 1.65 (one week, bugged)
\`\`\`

## Git Workflows for Different Team Sizes

### Solo Researcher
\`\`\`plaintext
Simple workflow:
- Work on main branch
- Commit frequently with good messages
- Tag releases
- Use branches for risky experiments
\`\`\`

### Small Team (2-5 people)
\`\`\`plaintext
Feature branch workflow:
main
├── feature/researcher-a-idea
├── feature/researcher-b-optimization
└── hotfix/urgent-fix

- Create branch for each feature
- Pull request for review
- Merge to main after approval
\`\`\`

### Large Team (5+ people)
\`\`\`plaintext
Gitflow workflow:
main (production)
├── develop (integration)
│   ├── feature/team1-ml-signals
│   ├── feature/team2-risk-model
│   ├── feature/team3-data-pipeline
│   └── release/v3.0.0
├── staging (pre-production testing)
└── hotfix/critical-bug

- Features merge to develop
- Periodic releases from develop to staging
- After testing, staging merges to main
- Hotfixes branch from main, merge back to both
\`\`\`

## Continuous Integration for Trading

### GitHub Actions Example

\`\`\`yaml
# .github/workflows/ci.yml
name: Continuous Integration

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python \${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: \${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v2
      with:
        path: ~/.cache/pip
        key: \${{ runner.os }}-pip-\${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov black flake8
    
    - name: Lint with flake8
      run: |
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Check formatting with Black
      run: |
        black --check src/ tests/
    
    - name: Run unit tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Check for lookahead bias
      run: |
        python scripts/validate_strategy.py
    
    - name: Run backtest
      run: |
        python scripts/backtest.py --quick
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
\`\`\`
      `,
    },
    {
      title: 'Git Best Practices for Quants',
      content: `
# Professional Git Practices for Quantitative Trading

## Commit Hygiene

### Commit Often, Push Less Frequently

\`\`\`bash
# Commit after each logical change
git commit -m "Add RSI calculation"
git commit -m "Add entry signal logic"
git commit -m "Add exit signal logic"
git commit -m "Add unit tests for signals"

# Review commits before pushing
git log --oneline

# Squash related commits if needed (advanced)
git rebase -i HEAD~4

# Push clean history
git push
\`\`\`

### Atomic Commits

Each commit should be one logical change:

\`\`\`bash
# ✅ GOOD: Single logical change
git add src/strategies/momentum.py tests/test_momentum.py
git commit -m "Add momentum strategy with tests"

# ❌ BAD: Multiple unrelated changes
git add src/strategies/momentum.py
git add src/data/loader.py
git add notebooks/exploration.ipynb
git commit -m "Various updates"
\`\`\`

## Branch Naming Conventions

\`\`\`plaintext
Format: <type>/<short-description>

Types:
├── feature/    New strategy component
├── experiment/ Research experiment (might not merge)
├── bugfix/     Non-critical bug fix
├── hotfix/     Critical production bug fix
├── refactor/   Code cleanup (no functionality change)
├── docs/       Documentation updates
└── test/       Test additions/improvements

Examples:
feature/add-volatility-filter
experiment/ml-based-signals
bugfix/fix-data-alignment
hotfix/fix-production-crash
refactor/simplify-backtest-engine
docs/update-strategy-readme
test/add-integration-tests
\`\`\`

## Code Review Checklist

### Before Creating Pull Request

\`\`\`bash
# 1. Ensure tests pass
pytest tests/ -v

# 2. Check code formatting
black src/ tests/
flake8 src/

# 3. Update documentation
# Edit README if API changed
# Add docstrings to new functions

# 4. Run backtest
python scripts/backtest.py

# 5. Clean up commits
git log --oneline
# Squash "WIP" and "fix typo" commits if needed

# 6. Write comprehensive PR description
# Include backtest results
# Explain methodology
# Note any breaking changes
\`\`\`

### PR Description Template

\`\`\`markdown
## Description
Brief summary of changes

## Motivation
Why is this change needed?

## Changes Made
- Added X feature
- Modified Y calculation
- Removed Z deprecated code

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Backtests run successfully
- [ ] Manual testing completed

## Performance Impact
### Backtest Results (2015-2023)
- Sharpe Ratio: 1.45 → 1.78 (+23%)
- Max Drawdown: -18% → -14%
- Win Rate: 58% (unchanged)
- Total Return: 145% → 167%

### Computational Performance
- Backtest runtime: 45s → 38s (-15%)
- Memory usage: 2.1GB → 1.8GB (-14%)

## Breaking Changes
None / List any breaking changes

## Additional Context
Any other relevant information, screenshots, links to notebooks, etc.

## Checklist
- [ ] Code follows style guide
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Reviewed own code
\`\`\`

## Handling Data in Git

### Large Files: Use Git LFS

\`\`\`bash
# Install Git LFS
brew install git-lfs  # macOS
apt-get install git-lfs  # Linux

# Initialize in repository
git lfs install

# Track large files
git lfs track "*.parquet"
git lfs track "*.h5"
git lfs track "*.csv"

# .gitattributes updated automatically
cat .gitattributes
# *.parquet filter=lfs diff=lfs merge=lfs -text
# *.h5 filter=lfs diff=lfs merge=lfs -text

# Commit normally
git add data/large_dataset.parquet
git commit -m "Add historical price data"

# LFS uploads to remote storage, not bloating repo
git push
\`\`\`

### Alternative: DVC (Data Version Control)

\`\`\`bash
# Install DVC
pip install dvc

# Initialize
dvc init

# Track data file
dvc add data/sp500_prices.parquet

# Creates data/sp500_prices.parquet.dvc
# Actual data stored in .dvc/cache/

# Commit .dvc file (small)
git add data/sp500_prices.parquet.dvc data/.gitignore
git commit -m "Add SP500 price data"

# Configure remote storage (S3, GCS, Azure, etc.)
dvc remote add -d myremote s3://mybucket/dvcstore

# Push data to remote
dvc push

# Collaborators pull data
dvc pull
\`\`\`

## Recovering from Mistakes

### Undo Last Commit (Not Pushed)

\`\`\`bash
# Keep changes in working directory
git reset --soft HEAD~1

# Discard changes entirely
git reset --hard HEAD~1
\`\`\`

### Undo Changes to File

\`\`\`bash
# Discard uncommitted changes
git checkout -- src/strategies/momentum.py

# Or with newer Git syntax
git restore src/strategies/momentum.py
\`\`\`

### Recover Deleted Branch

\`\`\`bash
# Find commit SHA of branch tip
git reflog

# Recreate branch
git branch recovered-branch abc123f
\`\`\`

### Revert Pushed Commit

\`\`\`bash
# Create new commit that undoes changes
git revert abc123f

# This is safer than git reset for pushed commits
# Preserves history
\`\`\`

## Git Aliases for Efficiency

\`\`\`bash
# ~/.gitconfig
[alias]
    # Short status
    st = status -sb
    
    # Pretty log
    lg = log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit
    
    # Show changes
    df = diff --color
    
    # Commit with message
    cm = commit -m
    
    # Amend last commit
    amend = commit --amend --no-edit
    
    # List branches sorted by last modified
    br = branch --sort=-committerdate
    
    # Checkout
    co = checkout
    
    # Create and checkout branch
    cob = checkout -b
    
    # Push current branch
    push-current = push -u origin HEAD
    
    # Undo last commit (keep changes)
    undo = reset --soft HEAD~1
    
    # See what's changed recently
    recent = log --oneline --no-merges -n 10

# Usage
git st         # instead of git status -sb
git lg         # pretty log
git cm "message"  # commit
git cob feature/new-thing  # create branch
\`\`\`

## Documentation in Repository

### Essential README

\`\`\`markdown
# Quantitative Momentum Strategy

## Overview
Brief description of strategy logic and philosophy.

## Performance Summary
- Sharpe Ratio: 1.85
- Max Drawdown: -14%
- Win Rate: 61%
- CAGR: 18.5%
- Backtest period: 2015-2023

## Setup

### Prerequisites
- Python 3.10+
- 8GB RAM minimum
- API keys: Alpha Vantage, Polygon.io

### Installation
\`\`\`bash
git clone https://github.com/company/momentum-strategy.git
cd momentum-strategy
python -m venv venv
source venv/bin/activate  # or venv\\\\Scripts\\\\activate on Windows
pip install -r requirements.txt
\`\`\`

### Configuration
1. Copy config/secrets.yaml.template to config/secrets.yaml
2. Add API keys to config/secrets.yaml
3. Adjust parameters in config/config.yaml

## Usage

### Run Backtest
\`\`\`bash
python scripts/backtest.py --start 2020-01-01 --end 2023-12-31
\`\`\`

### Run Tests
\`\`\`bash
pytest tests/ -v
\`\`\`

## Project Structure
\`\`\`
├── src/                    # Source code
│   ├── strategies/        # Trading strategies
│   ├── backtest/          # Backtesting engine
│   ├── data/              # Data loading and processing
│   └── risk/              # Risk management
├── tests/                  # Unit and integration tests
├── notebooks/              # Research notebooks
├── scripts/                # Executable scripts
├── config/                 # Configuration files
└── results/                # Backtest results

## Strategy Description
Detailed explanation of strategy logic, indicators used, entry/exit rules, etc.

## Contributing
1. Create feature branch: git checkout -b feature/your-feature
2. Make changes and add tests
3. Run test suite: pytest tests/
4. Commit: git commit -m "feat: add your feature"
5. Push: git push origin feature/your-feature
6. Create Pull Request

## License
Proprietary - Internal use only

## Contact
Questions? Contact: quant-team@company.com
\`\`\`

### CHANGELOG.md

\`\`\`markdown
# Changelog

All notable changes to this project will be documented in this file.

## [2.3.0] - 2024-01-15

### Added
- Volatility-based position sizing using ATR
- Maximum position size cap (20% of portfolio)
- Regime detection using HMM

### Changed
- Optimized MA periods: 20/50 → 15/45 (grid search result)
- Improved data validation in preprocessing pipeline

### Fixed
- Corrected forward-fill logic to prevent lookahead bias
- Fixed division by zero in Sharpe ratio calculation

### Performance
- Sharpe Ratio: 1.78 → 1.85
- Max Drawdown: -14.2% (improved from -16.1%)

## [2.2.0] - 2024-01-08

### Fixed
- Critical: Fixed incorrect fill prices on gap days

### Performance
- No performance change (bug fix)

## [2.1.0] - 2024-01-01

### Added
- Initial v2.0 release with ML-based signals

### Changed
- Complete rewrite from v1.x

### Performance
- Sharpe Ratio: 1.65 (baseline for v2)
\`\`\`

## Summary: Git Excellence in Quant Trading

1. **Commit frequently** with clear messages
2. **Branch for features** - keep main stable
3. **Review code** before merging
4. **Tag releases** for deployment tracking
5. **Protect secrets** - never commit API keys
6. **Document changes** in README and CHANGELOG
7. **Automate testing** with CI/CD
8. **Track data separately** with Git LFS or DVC
9. **Use Git hooks** for quality control
10. **Learn advanced commands** for efficiency

Git is not just version control - it's the foundation for reproducible, collaborative, production-grade quantitative research.
      `,
    },
  ],
  exercises: [
    {
      title: 'Git Workflow Setup',
      description:
        'Initialize a new quantitative trading project with proper Git configuration, including .gitignore, README, and initial structure.',
      difficulty: 'beginner',
      hints: [
        'Create comprehensive .gitignore for Python, Jupyter, data files',
        'Set up nbstripout for notebook output management',
        'Write clear README with setup instructions',
        'Create initial directory structure following best practices',
      ],
    },
    {
      title: 'Feature Branch Workflow',
      description:
        'Implement a new strategy feature using feature branches, pull requests, and code review process.',
      difficulty: 'intermediate',
      hints: [
        'Create feature branch from main',
        'Make multiple atomic commits with good messages',
        'Write tests for new feature',
        'Create pull request with detailed description including backtest results',
      ],
    },
    {
      title: 'Production Deployment Tracking',
      description:
        'Set up a deployment tracking system using Git tags, maintain deployment log, and implement rollback procedure.',
      difficulty: 'intermediate',
      hints: [
        'Create semantic version tags for releases',
        'Maintain deployment_log.md with deployment history',
        'Write rollback script that can revert to previous tag',
        "Document what's currently running in production",
      ],
    },
    {
      title: 'CI/CD Pipeline',
      description:
        'Implement a GitHub Actions workflow that runs tests, performs backtests, and checks for common issues on every push.',
      difficulty: 'advanced',
      hints: [
        'Create .github/workflows/ci.yml',
        'Run pytest, black, flake8',
        'Execute quick backtest to verify strategy still works',
        'Check for lookahead bias programmatically',
        'Upload backtest results as artifacts',
      ],
    },
  ],
};

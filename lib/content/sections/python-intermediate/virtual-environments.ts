/**
 * Virtual Environments & Package Management Section
 */

export const virtualenvironmentsSection = {
  id: 'virtual-environments',
  title: 'Virtual Environments & Package Management',
  content: `# Virtual Environments & Package Management

## Why Virtual Environments?

Virtual environments solve the "dependency hell" problem:
- **Isolation**: Each project has its own dependencies
- **No conflicts**: Different projects can use different package versions
- **Reproducibility**: Easy to replicate the exact environment
- **Clean system**: Don't pollute global Python installation

## Creating Virtual Environments

**Using venv (built-in, Python 3.3+):**
\`\`\`bash
# Create virtual environment
python -m venv myenv

# Or specify Python version
python3.11 -m venv myenv

# Creates directory structure:
# myenv/
#   bin/          # Scripts (Linux/Mac)
#   Scripts/      # Scripts (Windows)
#   lib/          # Installed packages
#   include/      # C headers
#   pyvenv.cfg    # Configuration
\`\`\`

## Activating Virtual Environments

\`\`\`bash
# Linux/Mac
source myenv/bin/activate

# Windows Command Prompt
myenv\\Scripts\\activate.bat

# Windows PowerShell
myenv\\Scripts\\Activate.ps1

# After activation, your prompt changes:
(myenv) $

# Check you're in the virtual environment
which python  # Should point to myenv/bin/python
python --version
\`\`\`

## Deactivating

\`\`\`bash
# From any directory
deactivate

# Your prompt returns to normal
$
\`\`\`

## Installing Packages

\`\`\`bash
# Activate environment first
source myenv/bin/activate

# Install packages
pip install requests
pip install numpy pandas matplotlib

# Install specific version
pip install Django==4.2.0

# Install from requirements.txt
pip install -r requirements.txt

# Upgrade package
pip install --upgrade requests

# Uninstall
pip uninstall requests
\`\`\`

## Managing Dependencies

**requirements.txt:**
\`\`\`bash
# Generate requirements.txt (all installed packages)
pip freeze > requirements.txt

# Example requirements.txt:
'''
requests==2.31.0
numpy==1.24.3
pandas==2.0.2
Django==4.2.0
'''

# Install from requirements.txt
pip install -r requirements.txt
\`\`\`

**Better: Separate dev and prod dependencies:**
\`\`\`bash
# requirements.txt (production)
requests==2.31.0
Django==4.2.0

# requirements-dev.txt (development)
-r requirements.txt  # Include production requirements
pytest==7.4.0
black==23.3.0
mypy==1.3.0
\`\`\`

## Viewing Installed Packages

\`\`\`bash
# List all installed packages
pip list

# Show package details
pip show requests

# Output:
# Name: requests
# Version: 2.31.0
# Summary: HTTP library
# Home-page: https://requests.readthedocs.io
# Location: /path/to/myenv/lib/python3.11/site-packages
# Requires: charset-normalizer, idna, urllib3, certifi
\`\`\`

## Alternative: virtualenv

virtualenv is a more powerful third-party tool:
\`\`\`bash
# Install virtualenv
pip install virtualenv

# Create virtual environment
virtualenv myenv

# Can use different Python versions
virtualenv -p python3.10 myenv

# Or specify full path
virtualenv -p /usr/bin/python3.11 myenv
\`\`\`

## Alternative: Poetry

Poetry is a modern dependency management tool:
\`\`\`bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Create new project
poetry new my-project
cd my-project

# Initialize in existing project
poetry init

# Add dependencies
poetry add requests
poetry add pytest --dev  # Development dependency

# Install all dependencies
poetry install

# pyproject.toml (Poetry's config file):
'''
[tool.poetry]
name = "my-project"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
requests = "^2.31.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
'''

# Poetry automatically creates and manages virtual environment
poetry run python script.py
poetry shell  # Activate virtual environment
\`\`\`

## Alternative: Conda

Conda is popular in data science (includes non-Python dependencies):
\`\`\`bash
# Create environment
conda create -n myenv python=3.11

# Activate
conda activate myenv

# Install packages
conda install numpy pandas matplotlib

# Install from conda-forge
conda install -c conda-forge opencv

# Mix conda and pip
conda install numpy
pip install some-package-not-in-conda

# Export environment
conda env export > environment.yml

# Create from environment.yml
conda env create -f environment.yml

# List environments
conda env list

# Remove environment
conda env remove -n myenv
\`\`\`

## Best Practices

**1. One virtual environment per project:**
\`\`\`bash
my-project/
  venv/          # Virtual environment
  src/           # Source code
  tests/         # Tests
  requirements.txt
  README.md
\`\`\`

**2. Don't commit virtual environment to git:**
\`\`\`bash
# .gitignore
venv/
env/
*.pyc
__pycache__/
\`\`\`

**3. Pin versions in production:**
\`\`\`bash
# Development: allow minor updates
requests>=2.31.0,<3.0.0

# Production: pin exact versions
requests==2.31.0
\`\`\`

**4. Document Python version:**
\`\`\`bash
# .python-version (for pyenv)
3.11.4

# Or in README:
'''
Requires Python 3.11+
'''
\`\`\`

**5. Use requirements.txt for simple projects, Poetry for complex ones:**
- Small scripts, tutorials: requirements.txt
- Libraries, applications: Poetry or pipenv

## Common Workflows

**Starting a new project:**
\`\`\`bash
# Create project directory
mkdir my-project
cd my-project

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate    # Windows

# Install packages
pip install requests pytest

# Save dependencies
pip freeze > requirements.txt

# Create .gitignore
echo "venv/" > .gitignore
\`\`\`

**Cloning an existing project:**
\`\`\`bash
# Clone repository
git clone https://github.com/user/project.git
cd project

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
\`\`\`

**Upgrading dependencies:**
\`\`\`bash
# Check outdated packages
pip list --outdated

# Upgrade specific package
pip install --upgrade requests

# Upgrade all (careful!)
pip list --outdated --format=freeze | grep -v '^\\-e' | cut -d = -f 1 | xargs -n1 pip install -U

# Update requirements.txt
pip freeze > requirements.txt
\`\`\`

## Troubleshooting

**Virtual environment not activating:**
\`\`\`bash
# Linux/Mac: check permissions
ls -la venv/bin/activate
chmod +x venv/bin/activate

# Windows PowerShell: enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
\`\`\`

**Wrong Python version:**
\`\`\`bash
# Specify exact Python version
python3.11 -m venv venv

# Or with virtualenv
virtualenv -p /usr/bin/python3.11 venv
\`\`\`

**Package conflicts:**
\`\`\`bash
# Start fresh
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\``,
};

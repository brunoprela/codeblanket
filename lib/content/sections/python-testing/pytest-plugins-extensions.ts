export const pytestPluginsExtensions = {
  title: 'pytest Plugins & Extensions',
  id: 'pytest-plugins-extensions',
  content: `
# pytest Plugins & Extensions

## Introduction

**pytest's plugin ecosystem extends testing capabilities dramatically**—from parallel execution to benchmark testing, from random test ordering to automatic retries. The right plugins transform pytest from a good framework into an exceptional one. This section covers essential plugins and how to use them professionally.

---

## Understanding pytest Plugins

pytest plugins are Python packages that extend pytest's functionality through **hooks**—special functions that pytest calls at specific points in the test lifecycle. Plugins can:

- **Modify test collection** (which tests run)
- **Add command-line options** (custom flags)
- **Provide fixtures** (reusable test resources)
- **Alter test execution** (parallel, random order)
- **Generate reports** (HTML, JSON, custom formats)
- **Integrate external tools** (coverage, benchmarking)

### Plugin Discovery

pytest automatically discovers plugins:

1. **Installed packages**: Any package named \`pytest-*\` or \`pytest_*\`
2. **conftest.py**: Local plugins in your test directory
3. **Explicit registration**: \`pytest_plugins = ["myapp.plugins"]\`

\`\`\`bash
# List installed plugins
pytest --trace-config

# Disable specific plugin
pytest -p no:warnings
\`\`\`

---

## Essential Plugin: pytest-xdist (Parallel Testing)

**pytest-xdist runs tests in parallel across multiple CPUs**—reducing test suite time from minutes to seconds.

### Installation

\`\`\`bash
pip install pytest-xdist
\`\`\`

### Basic Usage

\`\`\`bash
# Run tests on 4 CPUs
pytest -n 4

# Auto-detect CPU count
pytest -n auto

# Distribute across CPU cores (load balancing)
pytest -n auto --dist loadscope
\`\`\`

### Advanced Distribution Strategies

\`\`\`bash
# loadscope: Distribute by test scope (module/class)
# Tests in same class run on same worker (preserves class-scoped fixtures)
pytest -n auto --dist loadscope

# loadfile: Distribute by file
# All tests in file run on same worker
pytest -n auto --dist loadfile

# loadgroup: Custom groups via @pytest.mark.xdist_group
pytest -n auto --dist loadgroup
\`\`\`

### Custom Test Groups

\`\`\`python
import pytest

@pytest.mark.xdist_group("database")
def test_db_operation_1():
    """Runs on same worker as other database tests"""
    pass

@pytest.mark.xdist_group("database")
def test_db_operation_2():
    """Runs on same worker (prevents connection pool issues)"""
    pass

@pytest.mark.xdist_group("api")
def test_api_call():
    """Runs on different worker"""
    pass
\`\`\`

**Use cases**: Database tests (share connection pool), API tests (share auth tokens), resource-intensive tests (distribute evenly).

### Session-Scoped Fixtures with xdist

**Problem**: Session-scoped fixtures run once per worker, not once globally.

\`\`\`python
@pytest.fixture (scope="session")
def expensive_setup():
    """Runs once per worker (not once globally)"""
    print("Setting up...")
    return ExpensiveResource()
\`\`\`

**Solution**: Use file-based locks or databases for true global state.

\`\`\`python
import pytest
from filelock import FileLock

@pytest.fixture (scope="session")
def global_resource (tmp_path_factory, worker_id):
    """True session-scoped fixture across workers"""
    if worker_id == "master":
        # Not running with xdist
        return setup_resource()
    
    # Get temp directory shared across workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    lock_file = root_tmp_dir / "resource.lock"
    
    with FileLock (str (lock_file) + ".lock"):
        resource_file = root_tmp_dir / "resource.json"
        if resource_file.is_file():
            # Resource already created by another worker
            return load_resource (resource_file)
        else:
            # First worker creates resource
            resource = setup_resource()
            save_resource (resource, resource_file)
            return resource
\`\`\`

### Performance Comparison

\`\`\`bash
# Sequential (baseline)
pytest tests/  # 5 minutes

# Parallel on 4 CPUs
pytest -n 4 tests/  # 1.5 minutes (3.3× faster)

# Parallel with auto-detect (8 CPUs)
pytest -n auto tests/  # 50 seconds (6× faster)
\`\`\`

**Real-world impact**: 100-test suite (30 min) → 5 min with \`pytest -n auto\`.

---

## Essential Plugin: pytest-cov (Coverage Reporting)

**pytest-cov integrates coverage.py with pytest**—measuring which code your tests execute.

### Installation

\`\`\`bash
pip install pytest-cov
\`\`\`

### Basic Usage

\`\`\`bash
# Measure coverage for myapp package
pytest --cov=myapp tests/

# Show missing lines
pytest --cov=myapp --cov-report=term-missing tests/

# Generate HTML report
pytest --cov=myapp --cov-report=html tests/
# Open htmlcov/index.html in browser
\`\`\`

### Coverage Reports

**Terminal Report**:
\`\`\`
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
myapp/__init__.py           2      0   100%
myapp/models.py            45      5    89%   23-27, 89
myapp/services.py          67      2    97%   103, 156
myapp/utils.py             23      0   100%
-----------------------------------------------------
TOTAL                     137      7    95%
\`\`\`

**HTML Report**: Interactive, shows exact lines, branch coverage, per-file analysis.

### Branch Coverage

**Line coverage** measures if code runs. **Branch coverage** measures if all decision paths execute.

\`\`\`python
def process_payment (amount):
    if amount > 0:        # Branch 1: True path
        return "success"
    else:                 # Branch 2: False path
        return "error"
\`\`\`

- **Line coverage**: 100% if both lines execute once
- **Branch coverage**: 100% if both True and False paths tested

\`\`\`bash
# Enable branch coverage
pytest --cov=myapp --cov-branch --cov-report=term-missing tests/
\`\`\`

### Coverage Configuration

\`\`\`.coveragerc
# .coveragerc
[run]
source = myapp
omit =
    */tests/*
    */migrations/*
    */__init__.py
    */config.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstractmethod

precision = 2
show_missing = True

[html]
directory = htmlcov
\`\`\`

### Enforcing Coverage Thresholds

\`\`\`bash
# Fail if coverage < 80%
pytest --cov=myapp --cov-fail-under=80 tests/

# In pytest.ini
[pytest]
addopts = --cov=myapp --cov-fail-under=80 --cov-report=term-missing
\`\`\`

**CI/CD integration**: Prevents merging code that reduces coverage.

### Coverage with xdist

\`\`\`bash
# Parallel execution with coverage (combining workers)
pytest -n auto --cov=myapp --cov-report=term-missing tests/
\`\`\`

pytest-cov automatically combines coverage from all workers.

---

## Essential Plugin: pytest-benchmark

**pytest-benchmark measures function performance**—tracking speed, comparing implementations, detecting regressions.

### Installation

\`\`\`bash
pip install pytest-benchmark
\`\`\`

### Basic Benchmarking

\`\`\`python
def test_sorting_performance (benchmark):
    """Benchmark sorting algorithm"""
    data = list (range(1000, 0, -1))
    result = benchmark (sorted, data)
    assert result == list (range(1, 1001))
\`\`\`

**Output**:
\`\`\`
Name                         Min      Max     Mean  StdDev  Median     IQR  Outliers  Rounds  Iterations
---------------------------------------------------------------------------------------------------------
test_sorting_performance  45.2μs  67.3μs  48.1μs   3.2μs  47.5μs  2.1μs      2;0     100       10
\`\`\`

### Advanced Benchmarking

\`\`\`python
def test_function_with_setup (benchmark):
    """Benchmark with setup phase (not timed)"""
    def setup():
        return [1, 2, 3, 4, 5] * 200
    
    def process_data (data):
        return [x * 2 for x in data]
    
    result = benchmark.pedantic(
        process_data,
        setup=setup,
        rounds=100,        # Number of benchmark rounds
        iterations=10,     # Iterations per round
        warmup_rounds=5    # Warmup (not counted)
    )
\`\`\`

### Comparing Implementations

\`\`\`python
def test_list_comprehension (benchmark):
    data = range(1000)
    benchmark (lambda: [x * 2 for x in data])

def test_map_function (benchmark):
    data = range(1000)
    benchmark (lambda: list (map (lambda x: x * 2, data)))

def test_numpy_array (benchmark):
    import numpy as np
    data = np.arange(1000)
    benchmark (lambda: data * 2)
\`\`\`

**Compare results**:
\`\`\`bash
pytest tests/benchmarks/ --benchmark-compare
\`\`\`

### Regression Testing

\`\`\`bash
# Save baseline
pytest tests/benchmarks/ --benchmark-save=baseline

# Compare against baseline (fails if >10% slower)
pytest tests/benchmarks/ \\
    --benchmark-compare=baseline \\
    --benchmark-compare-fail=max:10%
\`\`\`

**CI/CD usage**: Detect performance regressions before merge.

### Benchmark Groups

\`\`\`python
@pytest.mark.benchmark (group="sorting")
def test_quicksort (benchmark):
    benchmark (quicksort, [5, 2, 8, 1, 9])

@pytest.mark.benchmark (group="sorting")
def test_mergesort (benchmark):
    benchmark (mergesort, [5, 2, 8, 1, 9])

@pytest.mark.benchmark (group="searching")
def test_binary_search (benchmark):
    benchmark (binary_search, list (range(1000)), 500)
\`\`\`

---

## Essential Plugin: pytest-mock

**pytest-mock provides the \`mocker\` fixture**—a convenient wrapper around unittest.mock.

### Installation

\`\`\`bash
pip install pytest-mock
\`\`\`

### Why Use pytest-mock?

**Without pytest-mock** (unittest.mock):
\`\`\`python
from unittest.mock import patch

def test_api_call():
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"status": "ok"}
        result = fetch_data()
        assert result["status"] == "ok"
\`\`\`

**With pytest-mock** (cleaner syntax):
\`\`\`python
def test_api_call (mocker):
    mock_get = mocker.patch("requests.get")
    mock_get.return_value.json.return_value = {"status": "ok"}
    result = fetch_data()
    assert result["status"] == "ok"
\`\`\`

### Automatic Cleanup

pytest-mock automatically undoes mocks after each test—no \`with\` blocks needed.

\`\`\`python
def test_one (mocker):
    mocker.patch("os.path.exists", return_value=True)
    assert check_file_exists("test.txt")
    # Mock automatically removed after test

def test_two():
    # os.path.exists is NOT mocked here
    pass
\`\`\`

### Spy on Functions

**Spies** let real functions run while recording calls.

\`\`\`python
def test_spy_on_function (mocker):
    spy = mocker.spy (math, "sqrt")
    
    result = calculate_distance(3, 4)  # Uses math.sqrt internally
    
    assert result == 5
    spy.assert_called_once_with(25)  # Verify sqrt(25) was called
\`\`\`

### Stub vs Mock

\`\`\`python
def test_stub_example (mocker):
    """Stub: Returns predefined values"""
    stub = mocker.stub (name="database_connection")
    stub.return_value = {"user": "alice"}
    
    result = stub()
    assert result["user"] == "alice"

def test_mock_example (mocker):
    """Mock: Records interactions for verification"""
    mock = mocker.MagicMock()
    
    process_data (mock)
    
    mock.save.assert_called_once()
    mock.validate.assert_called_with (strict=True)
\`\`\`

---

## Essential Plugin: pytest-randomly

**pytest-randomly runs tests in random order**—detecting hidden dependencies between tests.

### Installation

\`\`\`bash
pip install pytest-randomly
\`\`\`

### Why Random Order?

**Problem**: Tests pass when run in specific order but fail in different order.

\`\`\`python
# Test 1 sets up state
def test_create_user():
    global current_user
    current_user = User("alice")

# Test 2 depends on test 1 (BAD)
def test_user_name():
    assert current_user.name == "alice"  # Fails if test_create_user didn't run first
\`\`\`

**Solution**: Random order exposes these dependencies early.

### Usage

\`\`\`bash
# pytest-randomly automatically randomizes order after installation

# Use specific seed (for reproducibility)
pytest --randomly-seed=12345

# Disable randomization for debugging
pytest -p no:randomly
\`\`\`

### Reproducing Failures

When test fails with random order, pytest-randomly prints the seed:

\`\`\`
===== test session starts =====
randomly: seed=12345
\`\`\`

Reproduce with:
\`\`\`bash
pytest --randomly-seed=12345
\`\`\`

---

## Essential Plugin: pytest-timeout

**pytest-timeout kills tests that run too long**—preventing infinite loops from hanging CI/CD.

### Installation

\`\`\`bash
pip install pytest-timeout
\`\`\`

### Usage

\`\`\`python
import pytest

@pytest.mark.timeout(5)  # Timeout after 5 seconds
def test_with_timeout():
    result = slow_operation()
    assert result is not None

# Or set globally in pytest.ini
[pytest]
timeout = 10  # All tests timeout after 10 seconds
\`\`\`

### Debugging Hangs

\`\`\`python
@pytest.mark.timeout(10, method="thread")
def test_threaded_timeout():
    """Uses threading (better stack traces on timeout)"""
    pass
\`\`\`

---

## Essential Plugin: pytest-flask

**pytest-flask provides fixtures for testing Flask applications**.

### Installation

\`\`\`bash
pip install pytest-flask
\`\`\`

### Basic Usage

\`\`\`python
import pytest
from myapp import create_app

@pytest.fixture
def app():
    """Create Flask app for testing"""
    app = create_app (config="testing")
    return app

def test_homepage (client):
    """client fixture from pytest-flask"""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Welcome" in response.data

def test_api_endpoint (client):
    response = client.post("/api/users", json={"name": "Alice"})
    assert response.status_code == 201
    assert response.json["name"] == "Alice"
\`\`\`

---

## Creating Custom Plugins

### Local Plugin (conftest.py)

\`\`\`python
# tests/conftest.py
import pytest

def pytest_configure (config):
    """Called at pytest startup"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )

@pytest.fixture
def custom_resource():
    """Custom fixture available to all tests"""
    resource = setup_expensive_resource()
    yield resource
    teardown_resource (resource)

def pytest_collection_modifyitems (items):
    """Modify collected tests"""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker (pytest.mark.slow)
\`\`\`

### Installable Plugin

\`\`\`python
# pytest_myapp.py
import pytest

@pytest.hookimpl
def pytest_addoption (parser):
    """Add command-line option"""
    parser.addoption(
        "--myapp-env",
        action="store",
        default="test",
        help="Environment to test against"
    )

@pytest.fixture
def myapp_env (request):
    """Fixture providing environment from command line"""
    return request.config.getoption("--myapp-env")
\`\`\`

**Usage**:
\`\`\`bash
pytest --myapp-env=staging tests/
\`\`\`

---

## Plugin Recommendations by Use Case

| Use Case | Plugin | Purpose |
|----------|--------|---------|
| **Speed** | pytest-xdist | Parallel execution |
| **Coverage** | pytest-cov | Code coverage |
| **Performance** | pytest-benchmark | Benchmark functions |
| **Mocking** | pytest-mock | Cleaner mocking syntax |
| **Randomness** | pytest-randomly | Random test order |
| **Timeouts** | pytest-timeout | Kill hanging tests |
| **Flask** | pytest-flask | Flask testing fixtures |
| **Django** | pytest-django | Django testing fixtures |
| **Async** | pytest-asyncio | Test async code |
| **Database** | pytest-postgresql | PostgreSQL fixtures |
| **Retries** | pytest-rerunfailures | Retry flaky tests |

---

## Best Practices

1. **Start with essentials**: pytest-cov, pytest-xdist, pytest-mock
2. **Use -n auto** for parallel tests in CI/CD (6× faster)
3. **Enforce coverage** with --cov-fail-under=80
4. **Benchmark critical paths** (payment processing, data transformations)
5. **Random order** to catch test dependencies
6. **Timeout long tests** to prevent CI hangs
7. **Create custom plugins** for team-specific needs

---

## Summary

**Essential pytest plugins**:
- **pytest-xdist**: Parallel testing (6× faster)
- **pytest-cov**: Coverage reporting (80%+ target)
- **pytest-benchmark**: Performance testing (regression detection)
- **pytest-mock**: Cleaner mocking (auto cleanup)
- **pytest-randomly**: Random order (find dependencies)
- **pytest-timeout**: Kill hanging tests (prevent CI hangs)

Master these plugins to build **professional, fast, comprehensive test suites**.
`,
};

/**
 * Test Generation Section
 * Module 5: Building Code Generation Systems
 */

export const testgenerationSection = {
  id: 'test-generation',
  title: 'Test Generation',
  content: `# Test Generation

Master generating comprehensive unit tests, test cases, and test data automatically with LLMs.

## Overview: Automated Test Generation

Writing tests is time-consuming but critical. LLMs can:
- Generate unit tests from code
- Create test cases covering edge cases
- Generate test data
- Add assertions
- Mock dependencies
- Achieve high coverage

### Why Generate Tests?

**Manual Testing Challenges:**
- Time-consuming
- Incomplete coverage
- Forgotten edge cases
- Inconsistent patterns

**Generated Tests:**
- Fast and comprehensive
- Systematic edge case coverage
- Consistent patterns
- Easy to maintain

## Unit Test Generation

### Basic Test Generator

\`\`\`python
from dataclasses import dataclass
from typing import List, Optional
from openai import OpenAI
import ast

@dataclass
class TestSpec:
    """Specification for test generation."""
    function_name: str
    function_code: str
    test_framework: str = "pytest"  # "pytest", "unittest", "nose"
    coverage_level: str = "comprehensive"  # "basic", "standard", "comprehensive"

class TestGenerator:
    """Generate unit tests for Python functions."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_tests(
        self,
        function_code: str,
        test_framework: str = "pytest",
        coverage_level: str = "comprehensive"
    ) -> str:
        """Generate unit tests for a function."""
        
        # Extract function information
        func_info = self._analyze_function(function_code)
        
        # Build prompt
        prompt = self._build_test_prompt(
            func_info,
            test_framework,
            coverage_level
        )
        
        # Generate tests
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert at writing {test_framework} tests. "
                              "Generate comprehensive, readable tests."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code(response.choices[0].message.content)
    
    def _analyze_function(self, function_code: str) -> dict:
        """Extract function metadata."""
        tree = ast.parse(function_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function details
                args = [arg.arg for arg in node.args.args]
                
                # Get type hints if present
                arg_types = {}
                for arg in node.args.args:
                    if arg.annotation:
                        arg_types[arg.arg] = ast.unparse(arg.annotation)
                
                return_type = None
                if node.returns:
                    return_type = ast.unparse(node.returns)
                
                # Get docstring
                docstring = ast.get_docstring(node)
                
                return {
                    'name': node.name,
                    'args': args,
                    'arg_types': arg_types,
                    'return_type': return_type,
                    'docstring': docstring,
                    'code': function_code
                }
        
        return {}
    
    def _build_test_prompt(
        self,
        func_info: dict,
        test_framework: str,
        coverage_level: str
    ) -> str:
        """Build prompt for test generation."""
        
        coverage_requirements = {
            "basic": "Test happy path only",
            "standard": "Test happy path and common edge cases",
            "comprehensive": """Test:
- Happy path (normal inputs)
- Edge cases (empty, None, boundary values)
- Error cases (invalid inputs, exceptions)
- Type errors (if applicable)
- Integration with any dependencies"""
        }
        
        requirements = coverage_requirements.get(
            coverage_level,
            coverage_requirements["standard"]
        )
        
        args_str = ", ".join(
            f"{arg}: {func_info['arg_types'].get(arg, 'Any')}"
            for arg in func_info['args']
        )
        
        return f"""Generate {test_framework} tests for this function:

\`\`\`python
{func_info['code']}
\`\`\`

Function signature: {func_info['name']}({args_str}) -> {func_info['return_type']}

Requirements:
{requirements}

Generate:
1. Test class or functions
2. Setup/teardown if needed
3. Clear test names (test_<scenario>_<expected>)
4. Arrange-Act-Assert pattern
5. Good assertions with messages
6. Mock any external dependencies

Output complete, runnable test code.
"""
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        if "\`\`\`" in response:
            parts = response.split("\`\`\`")
            if len(parts) >= 3:
                code = parts[1]
                lines = code.split("\\n")
                if lines and lines[0].strip() in {'python', 'py'}:
                    code = "\\n".join(lines[1:])
                return code.strip()
        return response.strip()

# Usage
generator = TestGenerator()

function_code = """
def calculate_average(numbers: List[float]) -> float:
    '''Calculate the average of a list of numbers.'''
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)
"""

tests = generator.generate_tests(
    function_code,
    test_framework="pytest",
    coverage_level="comprehensive"
)

print(tests)

# Output will be something like:
# import pytest
# from typing import List
#
# def test_calculate_average_normal_case():
#     '''Test average with normal positive numbers.'''
#     result = calculate_average([1.0, 2.0, 3.0])
#     assert result == 2.0
#
# def test_calculate_average_single_number():
#     '''Test average with single number.'''
#     result = calculate_average([5.0])
#     assert result == 5.0
#
# def test_calculate_average_negative_numbers():
#     '''Test average with negative numbers.'''
#     result = calculate_average([-1.0, -2.0, -3.0])
#     assert result == -2.0
#
# def test_calculate_average_empty_list_raises_error():
#     '''Test that empty list raises ValueError.'''
#     with pytest.raises(ValueError, match="Cannot calculate average"):
#         calculate_average([])
\`\`\`

## Test Case Generation

### Generate Comprehensive Test Cases

\`\`\`python
class TestCaseGenerator:
    """Generate comprehensive test cases."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_test_cases(
        self,
        function_code: str,
        num_cases: int = 10
    ) -> List[dict]:
        """Generate diverse test cases for a function."""
        
        prompt = f"""Generate {num_cases} diverse test cases for this function:

\`\`\`python
{function_code}
\`\`\`

For each test case, provide:
1. Input values
2. Expected output
3. Description
4. Category (happy_path, edge_case, error_case, boundary)

Output as JSON array:
[
    {{
        "description": "...",
        "category": "happy_path",
        "inputs": {{"param1": value1, "param2": value2}},
        "expected": "expected_value_or_exception",
        "should_raise": false
    }},
    ...
]
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at test case design."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3  # Slightly higher for diversity
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        return result.get("test_cases", [])
    
    def generate_parametrized_test(
        self,
        function_code: str,
        test_cases: List[dict]
    ) -> str:
        """Generate pytest parametrized test from test cases."""
        
        # Build parametrize decorator
        param_values = []
        for case in test_cases:
            inputs_str = ", ".join(str(v) for v in case["inputs"].values())
            expected = case["expected"]
            param_values.append(f"({inputs_str}, {expected})")
        
        # Get function name
        tree = ast.parse(function_code)
        func_name = None
        param_names = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                param_names = [arg.arg for arg in node.args.args]
                break
        
        # Build test
        params = ", ".join(param_names)
        decorator = f"@pytest.mark.parametrize(\\"{params}, expected\\", ["
        decorator += ", ".join(param_values)
        decorator += "])"
        
        test = f"""
{decorator}
def test_{func_name}_parametrized({params}, expected):
    result = {func_name}({params})
    assert result == expected
"""
        
        return test

# Usage
case_gen = TestCaseGenerator()

function_code = """
def is_palindrome(text: str) -> bool:
    '''Check if text is a palindrome.'''
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]
"""

test_cases = case_gen.generate_test_cases(function_code, num_cases=8)

for case in test_cases:
    print(f"{case['category']}: {case['description']}")
    print(f"  Inputs: {case['inputs']}")
    print(f"  Expected: {case['expected']}")
    print()

# Generate parametrized test
parametrized = case_gen.generate_parametrized_test(function_code, test_cases)
print(parametrized)
\`\`\`

## Mock Generation

### Generate Mocks for Dependencies

\`\`\`python
class MockGenerator:
    """Generate mocks for testing."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_mocks(
        self,
        function_code: str,
        dependencies: List[str]
    ) -> str:
        """Generate mock objects for dependencies."""
        
        # Analyze dependencies
        dep_info = self._analyze_dependencies(function_code, dependencies)
        
        prompt = f"""Generate pytest mocks for these dependencies:

Function being tested:
\`\`\`python
{function_code}
\`\`\`

Dependencies to mock:
{self._format_dependencies(dep_info)}

Generate:
1. Fixture(s) for each dependency
2. Mock setup with expected behavior
3. Assertions to verify mock calls

Use pytest and unittest.mock.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at test mocking."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code(response.choices[0].message.content)
    
    def _analyze_dependencies(
        self,
        function_code: str,
        dependencies: List[str]
    ) -> dict:
        """Analyze how dependencies are used."""
        tree = ast.parse(function_code)
        
        dep_usage = {dep: [] for dep in dependencies}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    obj = node.func.value
                    if isinstance(obj, ast.Name) and obj.id in dependencies:
                        method = node.func.attr
                        dep_usage[obj.id].append(method)
        
        return dep_usage
    
    def _format_dependencies(self, dep_info: dict) -> str:
        """Format dependency information."""
        lines = []
        for dep, methods in dep_info.items():
            lines.append(f"- {dep}: methods used = {methods}")
        return "\\n".join(lines)
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        if "\`\`\`" in response:
            parts = response.split("\`\`\`")
            if len(parts) >= 3:
                return parts[1].strip()
        return response.strip()

# Usage
mock_gen = MockGenerator()

function_code = """
def process_payment(payment_gateway, amount: float, card_token: str):
    '''Process a payment through gateway.'''
    if amount <= 0:
        raise ValueError("Amount must be positive")
    
    result = payment_gateway.charge(amount, card_token)
    
    if result.success:
        payment_gateway.send_receipt(result.transaction_id)
        return result.transaction_id
    else:
        raise PaymentError(result.error_message)
"""

mocks = mock_gen.generate_mocks(
    function_code,
    dependencies=["payment_gateway"]
)

print(mocks)

# Output will be something like:
# import pytest
# from unittest.mock import Mock, MagicMock
#
# @pytest.fixture
# def payment_gateway():
#     gateway = Mock()
#     
#     # Setup successful charge
#     result = MagicMock()
#     result.success = True
#     result.transaction_id = "txn_123"
#     gateway.charge.return_value = result
#     
#     return gateway
#
# def test_process_payment_success(payment_gateway):
#     transaction_id = process_payment(payment_gateway, 100.0, "tok_123")
#     
#     # Verify charge was called correctly
#     payment_gateway.charge.assert_called_once_with(100.0, "tok_123")
#     
#     # Verify receipt was sent
#     payment_gateway.send_receipt.assert_called_once_with("txn_123")
#     
#     assert transaction_id == "txn_123"
\`\`\`

## Test Data Generation

### Generate Realistic Test Data

\`\`\`python
class TestDataGenerator:
    """Generate realistic test data."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_test_data(
        self,
        data_spec: str,
        num_examples: int = 5,
        include_edge_cases: bool = True
    ) -> List[dict]:
        """Generate test data matching specification."""
        
        prompt = f"""Generate {num_examples} test data examples:

Specification:
{data_spec}

Include:
- Normal valid examples
"""
        
        if include_edge_cases:
            prompt += """- Edge cases (empty strings, None, boundary values)
- Invalid examples (for negative testing)
"""
        
        prompt += """
Output as JSON array of objects.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at test data generation."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.4  # Higher for diversity
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        return result.get("examples", [])

# Usage
data_gen = TestDataGenerator()

spec = """
User object with:
- username: 3-20 alphanumeric characters
- email: valid email format
- age: integer 18-100
- roles: array of strings (admin, user, moderator)
"""

test_data = data_gen.generate_test_data(spec, num_examples=10)

for i, example in enumerate(test_data):
    print(f"Example {i+1}: {example}")
\`\`\`

## Integration Test Generation

### Generate Integration Tests

\`\`\`python
class IntegrationTestGenerator:
    """Generate integration tests."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_integration_test(
        self,
        components: List[str],
        workflow_description: str
    ) -> str:
        """Generate integration test for workflow."""
        
        prompt = f"""Generate an integration test for this workflow:

Components involved:
{self._format_components(components)}

Workflow:
{workflow_description}

Generate:
1. Setup (database, services, etc.)
2. Test workflow end-to-end
3. Assertions at each step
4. Cleanup

Use pytest and appropriate fixtures.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at integration testing."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return self._extract_code(response.choices[0].message.content)
    
    def _format_components(self, components: List[str]) -> str:
        """Format component list."""
        return "\\n".join(f"- {comp}" for comp in components)
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        if "\`\`\`" in response:
            parts = response.split("\`\`\`")
            if len(parts) >= 3:
                return parts[1].strip()
        return response.strip()

# Usage
integration_gen = IntegrationTestGenerator()

test = integration_gen.generate_integration_test(
    components=[
        "UserService",
        "EmailService",
        "Database",
        "AuthenticationService"
    ],
    workflow_description="""
1. User registers with email and password
2. System sends verification email
3. User clicks verification link
4. User is logged in automatically
5. User profile is created
"""
)

print(test)
\`\`\`

## Coverage-Guided Generation

### Generate Tests for Uncovered Code

\`\`\`python
class CoverageGuidedGenerator:
    """Generate tests to improve coverage."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_for_uncovered_lines(
        self,
        function_code: str,
        uncovered_lines: List[int],
        existing_tests: str
    ) -> str:
        """Generate tests to cover specific lines."""
        
        # Highlight uncovered lines
        lines = function_code.split("\\n")
        highlighted = []
        for i, line in enumerate(lines, 1):
            if i in uncovered_lines:
                highlighted.append(f"{i:4d} | >>> {line}  # UNCOVERED")
            else:
                highlighted.append(f"{i:4d} |     {line}")
        
        prompt = f"""Generate additional tests to cover marked lines:

Function:
{chr(10).join(highlighted)}

Existing tests:
{existing_tests}

Generate new test(s) that will execute the uncovered lines.
Explain what input values trigger those code paths.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at test coverage."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content

# Usage
coverage_gen = CoverageGuidedGenerator()

function_code = """
def process_order(order, inventory):
    if not order:
        raise ValueError("Order cannot be empty")
    
    if order.quantity > inventory.stock:
        return {"status": "insufficient_stock"}  # Line 6 - UNCOVERED
    
    if order.priority == "express":
        fee = order.amount * 0.15  # Line 9 - UNCOVERED
    else:
        fee = order.amount * 0.05
    
    return {"status": "success", "fee": fee}
"""

new_tests = coverage_gen.generate_for_uncovered_lines(
    function_code,
    uncovered_lines=[6, 9],
    existing_tests="""
def test_process_order_success():
    order = Order(quantity=5, amount=100, priority="standard")
    inventory = Inventory(stock=10)
    result = process_order(order, inventory)
    assert result["status"] == "success"
"""
)

print(new_tests)
\`\`\`

## Best Practices Summary

### ✅ DO:
1. **Generate comprehensive tests** - happy path + edge cases + errors
2. **Use descriptive test names** - test_<scenario>_<expected>
3. **Follow AAA pattern** - Arrange, Act, Assert
4. **Mock external dependencies**
5. **Generate parametrized tests** for multiple cases
6. **Include test data generation**
7. **Add clear assertions** with messages
8. **Generate integration tests** for workflows

### ❌ DON'T:
1. **Only test happy path**
2. **Use vague test names**
3. **Skip edge cases**
4. **Test without mocking dependencies**
5. **Generate redundant tests**
6. **Forget error cases**
7. **Skip assertions**
8. **Ignore code coverage**

## Next Steps

You've mastered test generation! Next:
- Documentation generation
- Code review systems
- Building complete code generation platforms

Remember: **Comprehensive Tests = Confident Code**
`,
};

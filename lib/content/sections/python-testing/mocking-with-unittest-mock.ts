export const mockingWithUnittestMock = {
  title: 'Mocking with unittest.mock',
  id: 'mocking-with-unittest-mock',
  content: `
# Mocking with unittest.mock

## Introduction

**Mocking is essential for testing code with external dependencies**—databases, APIs, file systems, external services. Without mocking, tests become slow, flaky, and dependent on external resources. **Mocks replace real objects with controlled substitutes**, allowing you to test your code in isolation.

### The Problem: Testing with Real Dependencies

\`\`\`python
def send_welcome_email(user):
    """Send welcome email to new user"""
    email_service = EmailService()
    email_service.connect("smtp.gmail.com", 587)
    email_service.send(
        to=user.email,
        subject="Welcome!",
        body=f"Hello {user.name}, welcome to our service!"
    )
    return True

def test_send_welcome_email():
    """How do we test this?"""
    user = User(name="Alice", email="alice@example.com")
    result = send_welcome_email(user)
    assert result is True
    # Problems:
    # 1. Actually sends email (slow, costs money)
    # 2. Requires SMTP credentials
    # 3. Fails if network down
    # 4. No control over email_service behavior
\`\`\`

### Solution: Mocking

\`\`\`python
from unittest.mock import Mock, patch

def test_send_welcome_email_mocked():
    """Test with mocked email service"""
    user = User(name="Alice", email="alice@example.com")
    
    with patch('email_service.EmailService') as MockEmailService:
        # Control the mock's behavior
        mock_service = MockEmailService.return_value
        mock_service.send.return_value = True
        
        # Test the function
        result = send_welcome_email(user)
        
        # Verify behavior
        assert result is True
        mock_service.connect.assert_called_once_with("smtp.gmail.com", 587)
        mock_service.send.assert_called_once()
    
    # Benefits:
    # - Fast (no actual email sent)
    # - No external dependencies
    # - Control over behavior
    # - Verify interactions
\`\`\`

---

## Mock Basics

### Creating Mock Objects

\`\`\`python
from unittest.mock import Mock

# Create a mock
mock = Mock()

# Mock can be called
result = mock()
print(result)  # <Mock name='mock()' id='...'>

# Mock has any attribute you access
print(mock.any_attribute)  # <Mock name='mock.any_attribute' id='...'>

# Mock can be called with any arguments
mock(1, 2, 3, key="value")
mock.method(arg1, arg2)

# All calls return Mock objects by default
\`\`\`

### Setting Return Values

\`\`\`python
# Simple return value
mock = Mock(return_value=42)
print(mock())  # 42

# Or set after creation
mock = Mock()
mock.return_value = "hello"
print(mock())  # "hello"

# Method return values
mock = Mock()
mock.method.return_value = "result"
print(mock.method())  # "result"

# Chained calls
mock = Mock()
mock.method().another_method.return_value = 100
print(mock.method().another_method())  # 100
\`\`\`

### Side Effects

Execute custom logic when mock is called:

\`\`\`python
# Raise exception
mock = Mock(side_effect=Exception("Connection failed"))
try:
    mock()
except Exception as e:
    print(e)  # Connection failed

# Return different values on successive calls
mock = Mock(side_effect=[1, 2, 3])
print(mock())  # 1
print(mock())  # 2
print(mock())  # 3

# Custom function
def custom_side_effect(arg):
    return arg * 2

mock = Mock(side_effect=custom_side_effect)
print(mock(5))  # 10
print(mock(10))  # 20
\`\`\`

---

## Patching

**Patching** temporarily replaces an object in the code being tested.

### patch as Context Manager

\`\`\`python
from unittest.mock import patch

# payment_service.py
def process_payment(amount):
    gateway = PaymentGateway()
    return gateway.charge(amount)

# test_payment.py
def test_process_payment():
    """Test with patched PaymentGateway"""
    with patch('payment_service.PaymentGateway') as MockGateway:
        # Configure mock
        mock_gateway = MockGateway.return_value
        mock_gateway.charge.return_value = {"status": "success"}
        
        # Test
        result = process_payment(100.0)
        
        # Verify
        assert result["status"] == "success"
        mock_gateway.charge.assert_called_once_with(100.0)
\`\`\`

### patch as Decorator

\`\`\`python
@patch('payment_service.PaymentGateway')
def test_process_payment(MockGateway):
    """Mock passed as function argument"""
    mock_gateway = MockGateway.return_value
    mock_gateway.charge.return_value = {"status": "success"}
    
    result = process_payment(100.0)
    
    assert result["status"] == "success"
\`\`\`

### Multiple Patches

\`\`\`python
# Patch multiple objects
@patch('module.ThirdDependency')
@patch('module.SecondDependency')
@patch('module.FirstDependency')
def test_multiple_patches(mock_first, mock_second, mock_third):
    """Arguments in reverse order of decorators"""
    # Configure mocks
    mock_first.return_value = "first"
    mock_second.return_value = "second"
    mock_third.return_value = "third"
    
    # Test code that uses all three dependencies
    ...
\`\`\`

### patch.object

Patch a specific method/attribute of an object:

\`\`\`python
class Database:
    def query(self, sql):
        # Actual database query
        ...

def test_with_patch_object():
    """Patch specific method"""
    db = Database()
    
    with patch.object(db, 'query', return_value=[{"id": 1, "name": "Alice"}]):
        result = db.query("SELECT * FROM users")
        assert len(result) == 1
        assert result[0]["name"] == "Alice"
\`\`\`

---

## Assertions on Mocks

Verify how mocks were used:

### Called Assertions

\`\`\`python
mock = Mock()
mock.method(1, 2, key="value")

# Assert called
mock.method.assert_called()

# Assert called once
mock.method.assert_called_once()

# Assert called with specific arguments
mock.method.assert_called_with(1, 2, key="value")

# Assert called once with specific arguments
mock.method.assert_called_once_with(1, 2, key="value")

# Assert any call with arguments (multiple calls allowed)
mock.method.assert_any_call(1, 2, key="value")

# Assert never called
another_mock = Mock()
another_mock.assert_not_called()
\`\`\`

### Call Count and Arguments

\`\`\`python
mock = Mock()
mock(1)
mock(2)
mock(3)

# Check call count
assert mock.call_count == 3

# Get all calls
print(mock.call_args_list)
# [call(1), call(2), call(3)]

# Get most recent call
print(mock.call_args)
# call(3)

# Access arguments
print(mock.call_args.args)  # (3,)
print(mock.call_args.kwargs)  # {}
\`\`\`

---

## MagicMock

\`MagicMock\` supports Python magic methods (\`__str__\`, \`__len__\`, etc.):

\`\`\`python
from unittest.mock import MagicMock

# Regular Mock doesn't support magic methods well
regular_mock = Mock()
print(len(regular_mock))  # Error or unexpected behavior

# MagicMock supports magic methods
magic_mock = MagicMock()
magic_mock.__len__.return_value = 5
print(len(magic_mock))  # 5

magic_mock.__str__.return_value = "custom string"
print(str(magic_mock))  # "custom string"

magic_mock.__iter__.return_value = iter([1, 2, 3])
print(list(magic_mock))  # [1, 2, 3]
\`\`\`

---

## Real-World Examples

### Example 1: Mocking Database

\`\`\`python
# user_service.py
class UserService:
    def __init__(self, db):
        self.db = db
    
    def get_user(self, user_id):
        result = self.db.query(f"SELECT * FROM users WHERE id={user_id}")
        return result[0] if result else None
    
    def create_user(self, name, email):
        self.db.execute(f"INSERT INTO users (name, email) VALUES ('{name}', '{email}')")
        return True

# test_user_service.py
from unittest.mock import Mock

def test_get_user():
    """Test getting user with mocked database"""
    # Create mock database
    mock_db = Mock()
    mock_db.query.return_value = [{"id": 1, "name": "Alice", "email": "alice@example.com"}]
    
    # Test
    service = UserService(mock_db)
    user = service.get_user(1)
    
    # Verify
    assert user["name"] == "Alice"
    mock_db.query.assert_called_once()

def test_get_user_not_found():
    """Test user not found"""
    mock_db = Mock()
    mock_db.query.return_value = []  # No results
    
    service = UserService(mock_db)
    user = service.get_user(999)
    
    assert user is None

def test_create_user():
    """Test creating user"""
    mock_db = Mock()
    
    service = UserService(mock_db)
    result = service.create_user("Bob", "bob@example.com")
    
    assert result is True
    mock_db.execute.assert_called_once()
\`\`\`

### Example 2: Mocking HTTP Requests

\`\`\`python
# weather_service.py
import requests

def get_weather(city):
    """Get weather from external API"""
    response = requests.get(f"https://api.weather.com/v1/weather?city={city}")
    response.raise_for_status()
    return response.json()

# test_weather_service.py
from unittest.mock import patch, Mock

@patch('weather_service.requests.get')
def test_get_weather_success(mock_get):
    """Test successful weather fetch"""
    # Configure mock response
    mock_response = Mock()
    mock_response.json.return_value = {
        "city": "New York",
        "temperature": 72,
        "condition": "sunny"
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    # Test
    weather = get_weather("New York")
    
    # Verify
    assert weather["temperature"] == 72
    assert weather["condition"] == "sunny"
    mock_get.assert_called_once_with("https://api.weather.com/v1/weather?city=New York")

@patch('weather_service.requests.get')
def test_get_weather_api_error(mock_get):
    """Test API error handling"""
    # Configure mock to raise exception
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    mock_get.return_value = mock_response
    
    # Test
    with pytest.raises(requests.HTTPError):
        get_weather("InvalidCity")
\`\`\`

### Example 3: Mocking File Operations

\`\`\`python
# config_loader.py
def load_config(filename):
    """Load configuration from file"""
    with open(filename, 'r') as f:
        content = f.read()
    return json.loads(content)

# test_config_loader.py
from unittest.mock import mock_open, patch

def test_load_config():
    """Test loading config with mocked file"""
    config_data = '{"debug": true, "api_key": "secret"}'
    
    with patch('builtins.open', mock_open(read_data=config_data)):
        config = load_config('config.json')
        
        assert config["debug"] is True
        assert config["api_key"] == "secret"

def test_load_config_file_not_found():
    """Test file not found error"""
    with patch('builtins.open', side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load_config('missing.json')
\`\`\`

### Example 4: Mocking datetime

\`\`\`python
# order_service.py
from datetime import datetime

def create_order(product_id, quantity):
    """Create order with current timestamp"""
    order = {
        "product_id": product_id,
        "quantity": quantity,
        "created_at": datetime.now(),
    }
    return order

# test_order_service.py
from unittest.mock import patch
from datetime import datetime

def test_create_order():
    """Test order creation with fixed timestamp"""
    fixed_time = datetime(2024, 1, 1, 12, 0, 0)
    
    with patch('order_service.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_time
        
        order = create_order(123, 5)
        
        assert order["created_at"] == fixed_time
        assert order["product_id"] == 123
\`\`\`

---

## pytest-mock Plugin

pytest-mock provides a simpler interface for mocking:

\`\`\`python
# Install: pip install pytest-mock

def test_with_mocker(mocker):
    """mocker fixture provided by pytest-mock"""
    # Patch with mocker (cleaner than unittest.mock)
    mock_service = mocker.patch('module.Service')
    mock_service.return_value.method.return_value = "result"
    
    # Test
    result = function_that_uses_service()
    
    # Verify
    assert result == "result"
    mock_service.assert_called_once()
\`\`\`

### mocker vs patch

\`\`\`python
# unittest.mock (verbose)
@patch('module.Service')
def test_function(mock_service):
    ...

# pytest-mock (cleaner)
def test_function(mocker):
    mock_service = mocker.patch('module.Service')
    ...
\`\`\`

---

## Best Practices

### 1. Mock at the Right Level

✅ **Good**: Mock external dependencies

\`\`\`python
@patch('requests.get')  # Mock external HTTP call
def test_api_call(mock_get):
    ...
\`\`\`

❌ **Bad**: Mock internal logic

\`\`\`python
@patch('my_module.calculate_total')  # Don't mock your own code!
def test_order():
    ...
\`\`\`

### 2. Use Meaningful Mock Names

✅ **Good**: Clear mock names

\`\`\`python
@patch('payment_gateway.PaymentGateway')
def test_payment(mock_payment_gateway):
    ...
\`\`\`

❌ **Bad**: Generic names

\`\`\`python
@patch('payment_gateway.PaymentGateway')
def test_payment(mock):
    ...
\`\`\`

### 3. Verify Important Interactions

✅ **Good**: Verify critical calls

\`\`\`python
mock_gateway.charge.assert_called_once_with(amount=100.0, currency="USD")
\`\`\`

❌ **Bad**: Over-verification

\`\`\`python
# Don't verify every single call
mock.method1.assert_called()
mock.method2.assert_called()
mock.method3.assert_called()
# ... (testing implementation details)
\`\`\`

### 4. Keep Mocks Simple

✅ **Good**: Simple mock behavior

\`\`\`python
mock_db.query.return_value = [{"id": 1, "name": "Alice"}]
\`\`\`

❌ **Bad**: Complex mock setup

\`\`\`python
def complex_side_effect(query):
    if "users" in query:
        if "id=1" in query:
            return [{"id": 1, "name": "Alice"}]
        elif "id=2" in query:
            return [{"id": 2, "name": "Bob"}]
    # ... 50 more lines
# If mock logic is complex, your code might need refactoring
\`\`\`

### 5. Use spec for Type Safety

Prevent accessing non-existent attributes:

\`\`\`python
# Without spec
mock = Mock()
mock.non_existent_method()  # Works (returns Mock)

# With spec
mock = Mock(spec=PaymentGateway)
mock.non_existent_method()  # AttributeError (catches typos)
\`\`\`

---

## Common Pitfalls

### Pitfall 1: Patching Wrong Location

\`\`\`python
# module_a.py
from module_b import SomeClass

def function():
    obj = SomeClass()
    ...

# test_module_a.py
# ❌ Wrong: Patches original location
@patch('module_b.SomeClass')
def test_function(mock):
    ...  # Doesn't work!

# ✅ Correct: Patch where it's used
@patch('module_a.SomeClass')
def test_function(mock):
    ...  # Works!
\`\`\`

**Rule**: Patch where the object is **used**, not where it's **defined**.

### Pitfall 2: Forgetting return_value

\`\`\`python
# ❌ Wrong
mock_service = Mock()
mock_service.method = "result"  # Replaces method with string!

result = mock_service.method()  # Error: str not callable

# ✅ Correct
mock_service = Mock()
mock_service.method.return_value = "result"

result = mock_service.method()  # "result"
\`\`\`

### Pitfall 3: Mocking Too Much

\`\`\`python
# ❌ Over-mocking makes tests meaningless
@patch('module.Database')
@patch('module.Cache')
@patch('module.Logger')
@patch('module.Validator')
@patch('module.Formatter')
def test_function(mock_formatter, mock_validator, mock_logger, mock_cache, mock_db):
    # If everything is mocked, what are we actually testing?
    ...

# ✅ Mock only external dependencies
@patch('module.Database')
def test_function(mock_db):
    # Test actual logic with real Validator, Formatter, etc.
    ...
\`\`\`

---

## Summary

**Mocking is essential for isolated, fast, reliable tests**:

**Key concepts**:
- **Mock**: Replace objects with controllable substitutes
- **patch**: Temporarily replace objects during tests
- **Return values**: Control what mocks return
- **Side effects**: Simulate errors or complex behavior
- **Assertions**: Verify how code interacted with mocks

**Benefits**:
- **Fast**: No slow external calls (databases, APIs, files)
- **Reliable**: No dependency on external services
- **Isolated**: Test your code, not dependencies
- **Controllable**: Simulate any scenario (errors, edge cases)

**Best practices**:
- Mock external dependencies (not your own code)
- Use meaningful names
- Verify important interactions only
- Keep mocks simple
- Use \`spec\` for type safety
- Patch where object is used, not defined

**Common patterns**:
- Mock database connections
- Mock HTTP requests
- Mock file operations
- Mock datetime for consistent timestamps
- Mock external services (payment gateways, email, etc.)

Master mocking, and you'll write **fast, reliable, isolated tests** that don't depend on external resources.
`,
};

export const testingBestPractices = {
  title: 'Testing Best Practices & Patterns',
  id: 'testing-best-practices',
  content: `
# Testing Best Practices & Patterns

## Introduction

**Testing is as much art as science**—knowing what to test, how deeply, and when to stop is crucial for productive development. This final section synthesizes everything into actionable best practices, common patterns, and anti-patterns to avoid.

Professional testing isn't about 100% coverage or testing every edge case. It's about strategically applying testing techniques to maximize confidence while minimizing maintenance burden.

---

## The Testing Mindset

### Test Behavior, Not Implementation

❌ **Testing implementation** (brittle):
\`\`\`python
def test_user_creation_implementation():
    user = User()
    user._set_username("alice")  # Testing private method
    user._validate()
    user._save_to_db()
    
    assert user._username == "alice"
    assert user._validated is True
\`\`\`

✅ **Testing behavior** (robust):
\`\`\`python
def test_user_creation_behavior():
    user = User.create(username="alice")
    
    assert user.username == "alice"
    assert User.exists(username="alice")
\`\`\`

**Why**: Implementation tests break when refactoring (even if behavior unchanged). Behavior tests survive refactoring.

### Write Tests That Tell a Story

✅ **Clear test story**:
\`\`\`python
def test_user_cannot_withdraw_more_than_balance():
    """
    Given a user with $100 balance
    When they try to withdraw $150
    Then the withdrawal fails
    And their balance remains $100
    """
    account = Account(balance=100)
    
    with pytest.raises(InsufficientFundsError):
        account.withdraw(150)
    
    assert account.balance == 100
\`\`\`

**Pattern**: Given (setup) → When (action) → Then (assertion)

---

## Test Organization Patterns

### Pattern 1: Arrange-Act-Assert (AAA)

\`\`\`python
def test_shopping_cart_total():
    # Arrange: Setup test data
    cart = ShoppingCart()
    item1 = Product(name="Widget", price=29.99)
    item2 = Product(name="Gadget", price=49.99)
    
    # Act: Perform action
    cart.add(item1)
    cart.add(item2)
    total = cart.calculate_total()
    
    # Assert: Verify result
    assert total == 79.98
\`\`\`

### Pattern 2: Given-When-Then (BDD Style)

\`\`\`python
def test_user_login_with_invalid_password():
    # Given: User exists with password
    user = UserFactory.create(password="correct_password")
    
    # When: User attempts login with wrong password
    result = auth.login(username=user.username, password="wrong_password")
    
    # Then: Login fails
    assert result.success is False
    assert result.error == "Invalid password"
\`\`\`

### Pattern 3: Test Class Organization

\`\`\`python
class TestUserAuthentication:
    """Group related tests"""
    
    @pytest.fixture
    def user(self):
        return UserFactory.create(password="secret123")
    
    def test_login_with_correct_password(self, user):
        result = auth.login(user.username, "secret123")
        assert result.success is True
    
    def test_login_with_wrong_password(self, user):
        result = auth.login(user.username, "wrong")
        assert result.success is False
    
    def test_login_with_nonexistent_user(self):
        result = auth.login("nonexistent", "password")
        assert result.success is False

class TestUserRegistration:
    """Separate class for different feature"""
    
    def test_register_new_user(self):
        result = auth.register(username="alice", password="secret")
        assert result.success is True
    
    def test_register_duplicate_username(self):
        UserFactory.create(username="alice")
        result = auth.register(username="alice", password="secret")
        assert result.success is False
\`\`\`

---

## What to Test

### Test Pyramid (Professional Standard)

\`\`\`
        /\\
       /E2E\\        10 tests  - Full user workflows
      /-----\\
     /       \\
    /Integration\\   100 tests - Component interactions
   /-----------\\
  /             \\
 /   Unit Tests  \\  1000 tests - Individual functions
-------------------
\`\`\`

### Critical Paths (Always Test)

1. **Authentication & Authorization**
   - User registration, login, logout
   - Password reset, email verification
   - Permission checks, role-based access

2. **Payment Processing**
   - Checkout flow, payment gateway integration
   - Refunds, discounts, tax calculation
   - Inventory updates after purchase

3. **Data Integrity**
   - Database constraints (unique, foreign keys)
   - Transaction rollbacks on error
   - Cascade deletes

4. **Business Logic**
   - Pricing calculations
   - Discount/promotion rules
   - Status transitions (order: pending → paid → shipped)

5. **Error Handling**
   - Invalid input validation
   - Network failures, timeouts
   - Database connection loss

### What NOT to Test

❌ **Third-party libraries**:
\`\`\`python
def test_requests_library():
    # Don't test that requests.get() works
    response = requests.get("https://api.example.com")
    assert response.status_code == 200
\`\`\`

❌ **Trivial getters/setters**:
\`\`\`python
class User:
    @property
    def username(self):
        return self._username  # Too simple to test

def test_username_getter():  # Waste of time
    user = User(username="alice")
    assert user.username == "alice"
\`\`\`

❌ **Framework features**:
\`\`\`python
def test_django_orm():
    # Don't test that Django ORM works
    user = User.objects.create(username="alice")
    assert User.objects.count() == 1
\`\`\`

✅ **DO test your usage** of libraries:
\`\`\`python
def test_api_client_handles_timeout():
    """Test OUR error handling, not requests library"""
    with pytest.raises(APIClientError):
        api_client.fetch_data(timeout=0.001)  # Will timeout
\`\`\`

---

## Common Testing Anti-Patterns

### Anti-Pattern 1: Testing Multiple Concerns

❌ **Bad** (tests too much):
\`\`\`python
def test_user_and_posts_and_comments():
    user = User.create(username="alice")
    post = Post.create(author=user, title="Hello")
    comment = Comment.create(post=post, author=user, text="Great!")
    
    assert user.username == "alice"
    assert post.title == "Hello"
    assert comment.text == "Great!"
    assert len(user.posts) == 1
    assert len(post.comments) == 1
\`\`\`

✅ **Good** (focused):
\`\`\`python
def test_user_creation():
    user = User.create(username="alice")
    assert user.username == "alice"

def test_post_creation():
    user = UserFactory.create()
    post = Post.create(author=user, title="Hello")
    assert post.title == "Hello"
    assert post.author == user

def test_comment_creation():
    post = PostFactory.create()
    comment = Comment.create(post=post, text="Great!")
    assert comment.text == "Great!"
    assert comment.post == post
\`\`\`

### Anti-Pattern 2: Test Interdependence

❌ **Bad** (tests depend on each other):
\`\`\`python
class TestUserFlow:
    user = None  # Shared state (BAD)
    
    def test_1_create_user(self):
        self.user = User.create(username="alice")
        assert self.user.id is not None
    
    def test_2_update_user(self):
        self.user.username = "bob"  # Depends on test_1
        self.user.save()
        assert self.user.username == "bob"
\`\`\`

✅ **Good** (independent tests):
\`\`\`python
class TestUser:
    @pytest.fixture
    def user(self):
        return UserFactory.create(username="alice")
    
    def test_create_user(self):
        user = User.create(username="alice")
        assert user.id is not None
    
    def test_update_user(self, user):
        user.username = "bob"
        user.save()
        assert user.username == "bob"
\`\`\`

### Anti-Pattern 3: Excessive Mocking

❌ **Bad** (mocking everything):
\`\`\`python
def test_user_service_create_user(mocker):
    mock_db = mocker.Mock()
    mock_validator = mocker.Mock()
    mock_email = mocker.Mock()
    mock_logger = mocker.Mock()
    mock_cache = mocker.Mock()
    
    service = UserService(mock_db, mock_validator, mock_email, mock_logger, mock_cache)
    service.create_user("alice", "alice@example.com")
    
    mock_db.save.assert_called_once()
    # Testing nothing meaningful—just mocks
\`\`\`

✅ **Good** (mock external dependencies only):
\`\`\`python
def test_user_service_sends_welcome_email(db_session, mocker):
    mock_email = mocker.patch("myapp.email.send_email")
    
    service = UserService(db_session)
    user = service.create_user("alice", "alice@example.com")
    
    assert user.username == "alice"  # Real database
    mock_email.assert_called_once_with(  # Mocked email
        to="alice@example.com",
        subject="Welcome!"
    )
\`\`\`

### Anti-Pattern 4: Slow Tests Without Good Reason

❌ **Bad** (unnecessarily slow):
\`\`\`python
def test_calculation():
    time.sleep(2)  # Why?
    result = calculate(10, 20)
    assert result == 30
\`\`\`

✅ **Good** (fast):
\`\`\`python
def test_calculation():
    result = calculate(10, 20)
    assert result == 30  # <1ms
\`\`\`

---

## Fixture Patterns

### Pattern: Base Fixture + Variants

\`\`\`python
@pytest.fixture
def user():
    """Base user fixture"""
    return UserFactory.create()

@pytest.fixture
def admin_user():
    """Admin variant"""
    return UserFactory.create(role="admin")

@pytest.fixture
def user_with_posts():
    """User with content variant"""
    user = UserFactory.create()
    PostFactory.create_batch(5, author=user)
    return user
\`\`\`

### Pattern: Fixture Composition

\`\`\`python
@pytest.fixture
def authenticated_client(client, user):
    """Compose client + user"""
    token = generate_auth_token(user)
    client.set_header("Authorization", f"Bearer {token}")
    return client

def test_protected_endpoint(authenticated_client):
    response = authenticated_client.get("/api/profile")
    assert response.status_code == 200
\`\`\`

---

## Assertion Best Practices

### Use Specific Assertions

❌ **Vague**:
\`\`\`python
assert user  # What are we testing?
assert result  # Too vague
\`\`\`

✅ **Specific**:
\`\`\`python
assert user is not None
assert user.is_active is True
assert result["status"] == "success"
assert len(items) > 0
\`\`\`

### Multiple Assertions When Appropriate

✅ **Related assertions** (OK together):
\`\`\`python
def test_user_creation():
    user = User.create(username="alice", email="alice@example.com")
    
    # Related assertions about same object
    assert user.id is not None
    assert user.username == "alice"
    assert user.email == "alice@example.com"
    assert user.created_at is not None
\`\`\`

❌ **Unrelated assertions** (split tests):
\`\`\`python
def test_everything():
    # Too many unrelated assertions
    assert add(2, 3) == 5
    assert User.count() == 0
    assert api_client.is_connected
\`\`\`

---

## Error Testing Patterns

### Pattern: Testing Exceptions

\`\`\`python
# Basic exception
def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

# Exception with message
def test_invalid_age():
    with pytest.raises(ValueError, match="Age must be positive"):
        User.create(username="alice", age=-5)

# Exception with inspection
def test_authentication_error():
    with pytest.raises(AuthenticationError) as exc_info:
        auth.login("nonexistent", "password")
    
    assert exc_info.value.code == "USER_NOT_FOUND"
    assert "nonexistent" in str(exc_info.value)
\`\`\`

### Pattern: Testing Error Recovery

\`\`\`python
def test_api_client_retries_on_failure(mocker):
    """Test retry logic"""
    mock_request = mocker.patch("requests.get")
    
    # First 2 calls fail, 3rd succeeds
    mock_request.side_effect = [
        requests.Timeout(),
        requests.ConnectionError(),
        mocker.Mock(status_code=200, json=lambda: {"data": "success"})
    ]
    
    result = api_client.fetch_with_retry(max_attempts=3)
    
    assert result["data"] == "success"
    assert mock_request.call_count == 3
\`\`\`

---

## Test Data Management

### Pattern: Factories Over Raw Objects

❌ **Repetitive**:
\`\`\`python
def test_1():
    user = User(username="user1", email="user1@example.com", age=30, role="user")
    # ...

def test_2():
    user = User(username="user2", email="user2@example.com", age=25, role="user")
    # ...
\`\`\`

✅ **DRY with factories**:
\`\`\`python
def test_1():
    user = UserFactory.create()  # Sensible defaults

def test_2():
    user = UserFactory.create(age=25)  # Override specific fields
\`\`\`

### Pattern: Minimal Test Data

✅ **Only what's needed**:
\`\`\`python
def test_username_validation():
    # Only username matters for this test
    user = User(username="a")
    with pytest.raises(ValidationError):
        user.validate()
\`\`\`

---

## Performance Testing Patterns

### Pattern: Benchmark Critical Paths

\`\`\`python
def test_search_performance(benchmark):
    """Ensure search completes in <100ms"""
    items = [Product(name=f"Item {i}") for i in range(1000)]
    
    result = benchmark(search_products, items, query="Item 500")
    
    assert result is not None
    # benchmark automatically fails if >100ms (configurable)

def test_no_n_plus_1_queries(db_session, django_assert_num_queries):
    """Ensure queries are optimized"""
    users = UserFactory.create_batch(10)
    
    # Should be 1 query (with prefetch_related), not 11 (N+1)
    with django_assert_num_queries(1):
        users = User.objects.prefetch_related("posts").all()
        for user in users:
            _ = user.posts.count()
\`\`\`

---

## Documentation Through Tests

### Pattern: Tests as Examples

\`\`\`python
def test_api_client_usage_example():
    """
    Example of using APIClient:
    
    1. Create client
    2. Authenticate
    3. Make requests
    4. Handle responses
    """
    client = APIClient(base_url="https://api.example.com")
    client.authenticate(api_key="your_key")
    
    response = client.get("/users/123")
    
    assert response.status_code == 200
    assert "username" in response.json()
\`\`\`

---

## Continuous Improvement

### Track Test Metrics

- **Test count**: Unit (1000+), Integration (100+), E2E (10+)
- **Coverage**: 80-90% overall, 95%+ critical paths
- **Speed**: Unit (<5 min), Integration (<15 min)
- **Flakiness**: <1% flaky tests
- **Maintenance**: Test changes per feature (should be low)

### Regular Test Audits

Monthly review:
1. **Remove redundant tests** (duplicate coverage)
2. **Fix flaky tests** (intermittent failures)
3. **Speed up slow tests** (>1s unit tests)
4. **Update stale tests** (testing deprecated features)
5. **Add missing tests** (new critical paths)

---

## Summary of Best Practices

### Always Do:
1. ✅ Test behavior, not implementation
2. ✅ Write independent tests
3. ✅ Use AAA/Given-When-Then pattern
4. ✅ Mock external dependencies only
5. ✅ Use factories for test data
6. ✅ Test critical paths thoroughly
7. ✅ Keep tests fast (<10ms unit tests)
8. ✅ Use descriptive test names
9. ✅ One logical assertion per test
10. ✅ Treat tests as production code

### Never Do:
1. ❌ Test third-party libraries
2. ❌ Share state between tests
3. ❌ Test private methods
4. ❌ Write slow tests without reason
5. ❌ Skip flaky tests (fix them!)
6. ❌ Commit commented-out tests
7. ❌ Use time.sleep() in tests
8. ❌ Test implementation details
9. ❌ Ignore test failures
10. ❌ Write tests without assertions

---

## Final Wisdom

**Testing is an investment**:
- Upfront cost: 20-30% more development time
- Payoff: 80% fewer bugs, 50% faster debugging, confident refactoring

**Quality over quantity**:
- 100 meaningful tests > 1000 weak tests
- 80% coverage with strong assertions > 100% coverage with none

**Balance speed and thoroughness**:
- Unit tests: Fast (1000 tests in 2 min)
- Integration tests: Moderate (100 tests in 10 min)
- E2E tests: Slow (10 tests in 5 min)

**Professional testing mindset**:
- Tests verify behavior, not implementation
- Tests are documentation
- Tests enable refactoring
- Tests prevent regressions
- Tests give confidence

Master these practices for **professional, maintainable, reliable software**.
`,
};

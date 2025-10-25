/**
 * Quiz questions for Test Generation section
 */

export const testgenerationQuiz = [
  {
    id: 'bcgs-testgen-q-1',
    question:
      'Design a comprehensive test generation strategy that creates tests with high coverage but avoids redundant tests. What criteria would you use to determine if a test case adds value vs. being redundant?',
    hint: 'Consider code paths, edge cases, error conditions, and test diversity.',
    sampleAnswer:
      '**Test Generation Strategy for High Coverage Without Redundancy:** **Value Criteria (Generate These):** 1) **Unique Code Paths** - Each test should exercise a different branch/path through code. Track: If statements (test both true and false), Loops (test 0, 1, many iterations), Exception handling (test each catch block). 2) **Edge Cases** - Empty inputs ([], None, ""), Boundary values (0, -1, MAX_INT), Type edges (0.0 vs 0, empty string vs None). 3) **Error Conditions** - Each exception type that can be raised, Invalid input combinations, Resource failures. 4) **Integration Points** - Test each external dependency, Test failure modes of dependencies. **Redundancy Criteria (Skip These):** 1) **Same Code Path** - If test exercises same if/else branches as existing test, only different in data. 2) **Subset Behavior** - Test that checks subset of what another test checks. 3) **Trivial Variations** - Testing `add(2, 3)` and `add(4, 6)` when logic doesn\'t branch on values. **Algorithm:** ```python\\ndef should_generate_test (new_test, existing_tests):\\n    # Check code coverage delta\\n    new_coverage = get_coverage([*existing_tests, new_test])\\n    old_coverage = get_coverage (existing_tests)\\n    \\n    if new_coverage - old_coverage > 0:\\n        return True  # Covers new lines\\n    \\n    # Check if tests unique error condition\\n    if tests_different_exception (new_test, existing_tests):\\n        return True\\n    \\n    # Check if tests different input category\\n    if different_input_category (new_test, existing_tests):\\n        return True\\n    \\n    return False``` **Example:** Function `def divide (a, b)`. Generate: `divide(10, 2)` (happy path), `divide(0, 1)` (zero numerator), `divide(1, 0)` (zero denominator - error), `divide(-5, 2)` (negative). Don\'t generate: `divide(20, 4)` (same path as divide(10, 2)).',
    keyPoints: [
      'Generate tests for unique code paths (branches, loops, exceptions)',
      'Include edge cases (empty, boundary, type edges)',
      'Skip redundant tests that exercise same paths',
      'Measure coverage delta to determine value',
    ],
  },
  {
    id: 'bcgs-testgen-q-2',
    question:
      "You're generating mocks for a function that calls external APIs. Explain how you would determine what the mock should return and what assertions should verify the mock was called correctly.",
    hint: 'Consider analyzing function usage, API contracts, and error scenarios.',
    sampleAnswer:
      '**Mock Generation Strategy:** **1) Analyze Function to Understand API Usage** - Parse function code to find: Which API methods are called, What arguments are passed, How return values are used, Error handling patterns. **2) Determine Mock Return Values** - **Success Case:** Analyze how return value is used: ```python\\nresult = api.fetch_user (id)\\nif result.active:  # Uses .active attribute\\n    return result.name  # Uses .name attribute``` Mock needs: `result.active = True, result.name = "test_user"` - **Error Cases:** Check what exceptions are caught: ```python\\ntry:\\n    api.fetch_user (id)\\nexcept UserNotFoundError:\\n    return None``` Mock should raise `UserNotFoundError` in error test. **3) Generate Assertions** - Verify mock called with correct args: `api.fetch_user.assert_called_once_with(123)`, Verify called in correct order if multiple calls, Verify NOT called if conditional, Verify called N times if in loop. **4) Handle Complex Scenarios** - **Side Effects:** If API call modifies state, mock should too. **Return Value Factory:** For multiple calls, return different values: `mock.side_effect = [result1, result2, exception]`. **5) Complete Example:** ```python\\n# Function using API\\ndef get_active_username (user_id):\\n    user = api.fetch_user (user_id)\\n    if not user.active:\\n        raise UserInactiveError()\\n    return user.name\\n\\n# Generated mock and test\\n@pytest.fixture\\ndef api_mock():\\n    mock = Mock()\\n    user = Mock (active=True, name="test_user")\\n    mock.fetch_user.return_value = user\\n    return mock\\n\\ndef test_get_active_username (api_mock):\\n    result = get_active_username(123)\\n    api_mock.fetch_user.assert_called_once_with(123)\\n    assert result == "test_user"```',
    keyPoints: [
      'Analyze code to determine API usage patterns',
      "Mock return values based on how they're used in code",
      'Generate assertions for call arguments and order',
      'Handle error cases with exception-raising mocks',
    ],
  },
  {
    id: 'bcgs-testgen-q-3',
    question:
      "You've generated tests that achieve 100% code coverage but some bugs still slip through. What additional test generation strategies would catch bugs that coverage alone misses? Provide specific examples.",
    hint: 'Think about property-based testing, mutation testing, and behavior verification.',
    sampleAnswer:
      '**Beyond Coverage - Additional Test Strategies:** **1) Property-Based Testing** - Instead of specific examples, test invariant properties: ```python\\n# Coverage-based (misses bugs)\\ndef test_sort():\\n    assert sort([3,1,2]) == [1,2,3]\\n\\n# Property-based (catches more)\\n@given (lists (integers()))\\ndef test_sort_properties (lst):\\n    result = sort (lst)\\n    assert len (result) == len (lst)  # No elements lost\\n    assert all (result[i] <= result[i+1] for i in range (len (result)-1))  # Actually sorted\\n    assert set (result) == set (lst)  # Same elements``` Coverage test passes even if sort loses elements! **2) Mutation Testing** - Modify code slightly, tests should fail: Original: `if x > 0`, Mutant: `if x >= 0`. If tests still pass, they\'re insufficient. Generate tests for each possible mutation. **3) Edge Case Combinations** - Coverage tests single edges, miss combinations: ```python\\ndef process (items, max_size, allow_duplicates):\\n    # Bug only appears when: empty items + allow_duplicates=False\\n    pass``` Generate tests for all parameter combinations. **4) State-Based Testing** - Test sequences of operations: ```python\\n# Bug: delete then add same item fails\\ndef test_state_sequence():\\n    cache.add("key", "value")\\n    cache.delete("key")\\n    cache.add("key", "value2")  # Should work but doesn\'t``` **5) Behavior Verification** - Check observable behavior, not just return value: ```python\\ndef test_behavior():\\n    result = save_user (user)\\n    assert result.success\\n    assert database.contains (user)  # Verify side effect\\n    assert audit_log.has_entry("user_created")  # Verify logging``` **Real Example:** Function passes coverage but has race condition. Solution: Generate concurrent test with threading. Function passes coverage but leaks memory. Solution: Generate test measuring memory before/after many iterations.',
    keyPoints: [
      'Use property-based testing for invariant properties',
      'Apply mutation testing to ensure tests fail on code changes',
      'Test edge case combinations, not just single edges',
      'Verify behavior and side effects, not just return values',
    ],
  },
];

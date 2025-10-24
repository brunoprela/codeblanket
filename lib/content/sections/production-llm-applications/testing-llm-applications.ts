export const testingLlmApplicationsContent = `
# Testing LLM Applications

## Introduction

Testing LLM applications is challenging because outputs are non-deterministic, expensive to generate, and hard to evaluate objectively. This section covers unit testing, integration testing, mocking strategies, and evaluation frameworks for LLM apps.

## Unit Testing with Mocks

\`\`\`python
import pytest
from unittest.mock import Mock, patch
import openai

class LLMService:
    def generate(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

@pytest.fixture
def mock_openai():
    """Mock OpenAI responses."""
    with patch('openai.ChatCompletion.create') as mock:
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Mocked response"))]
        mock_response.usage = Mock(total_tokens=100)
        mock.return_value = mock_response
        yield mock

def test_llm_service(mock_openai):
    """Test LLM service without calling API."""
    service = LLMService()
    result = service.generate("Test prompt")
    
    assert result == "Mocked response"
    mock_openai.assert_called_once()
\`\`\`

## Snapshot Testing

\`\`\`python
import pytest
import json

@pytest.fixture
def snapshot(tmpdir):
    """Snapshot testing fixture."""
    snapshot_file = tmpdir.join("snapshots.json")
    
    def _snapshot(name: str, data: any):
        if snapshot_file.exists():
            snapshots = json.loads(snapshot_file.read())
        else:
            snapshots = {}
        
        if name in snapshots:
            assert snapshots[name] == data, f"Snapshot {name} changed"
        else:
            snapshots[name] = data
            snapshot_file.write(json.dumps(snapshots, indent=2))
    
    return _snapshot

def test_prompt_output(snapshot):
    """Test that prompt output hasn't changed."""
    result = generate("What is 2+2?", temperature=0)
    snapshot("math_prompt", result)
\`\`\`

## Integration Testing

\`\`\`python
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_generation_endpoint():
    """Test end-to-end generation."""
    response = client.post(
        "/generate",
        json={"prompt": "Say hello"},
        headers={"X-API-Key": "test-key"}
    )
    
    assert response.status_code == 200
    assert "result" in response.json()
    assert len(response.json()["result"]) > 0

@pytest.mark.slow
def test_generation_quality():
    """Test generation quality (expensive, mark as slow)."""
    result = generate("Explain Python in one sentence")
    
    # Basic quality checks
    assert len(result) > 20
    assert "python" in result.lower()
\`\`\`

## Testing for Costs

\`\`\`python
def test_token_estimation():
    """Test that we estimate costs correctly."""
    prompt = "Short prompt"
    estimated_cost = estimate_cost(prompt, "gpt-4")
    
    # Should be roughly $0.01 for short prompt
    assert 0.001 < estimated_cost < 0.05

@pytest.mark.limit
def test_cost_limits():
    """Test that cost limits are enforced."""
    # Try to make expensive request
    with pytest.raises(CostLimitExceeded):
        generate_many([expensive_prompt] * 1000, "gpt-4")
\`\`\`

## Load Testing

\`\`\`python
from locust import HttpUser, task, between

class LLMUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def generate(self):
        self.client.post(
            "/generate",
            json={"prompt": "Test prompt"},
            headers={"X-API-Key": "test-key"}
        )

# Run with: locust -f test_load.py
\`\`\`

## Best Practices

1. **Mock LLM calls** in unit tests to avoid costs
2. **Use deterministic parameters** (temperature=0) for reproducible tests
3. **Snapshot test** important prompts to catch regressions
4. **Test rate limiting** and error handling
5. **Test cost estimation** to avoid surprise bills
6. **Load test** before production
7. **Test with real API** in CI/CD but limit frequency
8. **Evaluate output quality** programmatically when possible
9. **Test graceful degradation** and fallbacks
10. **Monitor test costs** and optimize

Testing LLM applications requires balancing thoroughness with cost and practicality.
`;

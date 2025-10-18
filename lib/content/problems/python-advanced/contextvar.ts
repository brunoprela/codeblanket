/**
 * advanced-contextvar
 * Order: 49
 */

import { Problem } from '../../../types';

export const contextvarProblem: Problem = {
  id: 'advanced-contextvar',
  title: 'Context Variables',
  difficulty: 'Hard',
  description: `Use contextvars for context-local state (better than thread-local).

Context variables features:
- Async-safe (unlike threading.local)
- Context-specific values
- Inherited by child tasks
- Used in async frameworks

**Use Case:** Request IDs, user context, logging

This tests:
- contextvars module
- Context isolation
- Async compatibility`,
  examples: [
    {
      input: 'Request ID per context',
      output: 'Different IDs in different contexts',
    },
  ],
  constraints: ['Use contextvars.ContextVar', 'Values isolated per context'],
  hints: [
    'Create ContextVar instance',
    'Use .set() and .get()',
    'Each context has own value',
  ],
  starterCode: `from contextvars import ContextVar

# Create context variable
request_id: ContextVar[str] = ContextVar('request_id', default='none')


def process_request(req_id: str) -> str:
    """Process request with context-specific ID"""
    # Set context variable
    request_id.set(req_id)
    
    # Get context variable
    current_id = request_id.get()
    
    return current_id


def test_contextvar():
    """Test context variables"""
    result1 = process_request('req-123')
    result2 = process_request('req-456')
    
    # Each call has its own context
    return len(result1) + len(result2)
`,
  testCases: [
    {
      input: [],
      expected: 14,
      functionName: 'test_contextvar',
    },
  ],
  solution: `from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar('request_id', default='none')


def process_request(req_id: str) -> str:
    request_id.set(req_id)
    current_id = request_id.get()
    return current_id


def test_contextvar():
    result1 = process_request('req-123')
    result2 = process_request('req-456')
    
    return len(result1) + len(result2)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1) per context',
  order: 49,
  topic: 'Python Advanced',
};

export const apiDesignForLlmAppsQuiz = [
  {
    id: 'pllm-q-2-1',
    question:
      'Design a comprehensive API for an LLM application that supports both REST and WebSocket patterns, handles streaming responses, implements proper versioning, and provides excellent developer experience. What design decisions would you make and why?',
    sampleAnswer:
      'RESTful design with resource-oriented endpoints (/v1/conversations, /v1/messages, /v1/generate) using proper HTTP methods. WebSocket endpoint (/ws/chat/{session_id}) for bidirectional streaming and real-time updates. Server-Sent Events (/v1/chat/stream) for unidirectional streaming. URL-based versioning (/v1/, /v2/) for clarity and easy routing. Comprehensive error responses with request IDs, error codes, and helpful messages. OpenAPI documentation with examples and code snippets. Rate limiting headers (X-RateLimit-*) on all responses. Support for both synchronous (immediate response) and asynchronous (return task_id, poll for status) patterns. Implement CORS properly, provide SDKs in multiple languages, include sandbox environment. Key design decisions: streaming over polling for better UX, task IDs for long operations, semantic HTTP status codes, detailed error messages with solutions, consistent response format across all endpoints. Developer experience priorities: clear documentation, interactive API explorer, helpful error messages, predictable behavior.',
    keyPoints: [
      'Resource-oriented REST with streaming support via SSE/WebSocket',
      'URL versioning and comprehensive error handling',
      'Developer-friendly documentation and consistent response formats',
    ],
  },
  {
    id: 'pllm-q-2-2',
    question:
      "Compare Server-Sent Events (SSE) and WebSockets for streaming LLM responses. When would you choose each, and how would you implement fallback strategies for clients that don't support either?",
    sampleAnswer:
      'SSE advantages: simpler protocol, automatic reconnection, works over HTTP/1.1, browser EventSource API, one-way is sufficient for LLM streaming. WebSocket advantages: bidirectional, can send control messages (cancel generation), more efficient for high-frequency updates, supports binary data. Choose SSE for: standard LLM streaming, simple implementation, clients behind restrictive proxies. Choose WebSocket for: interactive conversations with cancellation, real-time collaborative features, need for client→server messages during generation. Fallback strategy: 1) Detect client capabilities via feature detection or headers, 2) Offer polling endpoint (/tasks/{id}/poll) as universal fallback, 3) Implement long-polling (hold request until data available) as middle ground, 4) Use chunked transfer encoding for older HTTP clients. Implementation: Provide all three methods, let client choose via Accept header or query parameter, ensure same data format across all methods, document fallback order in API docs. Progressive enhancement: start with polling, upgrade to SSE, then WebSocket if available.',
    keyPoints: [
      'SSE for simple one-way streaming, WebSocket for bidirectional',
      'Multiple transport methods with automatic fallback',
      'Client capability detection and progressive enhancement',
    ],
  },
  {
    id: 'pllm-q-2-3',
    question:
      'Explain how you would implement rate limiting at the API level for an LLM application with multiple user tiers. How would you communicate limits to clients, handle exceeded limits gracefully, and prevent abuse while maintaining good UX?',
    sampleAnswer:
      'Implement token bucket algorithm in middleware: different buckets per tier (Free: 10/min, Pro: 100/min, Enterprise: 1000/min). Use Redis for distributed rate limiting across instances. Rate limit headers on every response: X-RateLimit-Limit (max), X-RateLimit-Remaining (left), X-RateLimit-Reset (timestamp). When limit exceeded: return 429 status with Retry-After header, clear error message explaining limit and upgrade path, suggestion to cache or batch requests. Prevent abuse: require API key authentication, implement anomaly detection for unusual patterns, add CAPTCHA after repeated limit hits, block IPs with sustained abuse. Maintain good UX: provide rate limit dashboard showing usage, send email at 80% of limit, offer burst capacity for occasional spikes, implement request queuing instead of rejection when possible. Multiple limit types: requests per minute (burst), requests per day (sustained), cost per hour (spend), concurrent requests (load). Communicate via: API response headers, usage dashboard, email notifications, webhook alerts. Allow upgrades to higher tiers with immediate limit increase.',
    keyPoints: [
      'Token bucket with Redis for distributed rate limiting',
      'Clear communication via headers and helpful error messages',
      'Multiple limit types and graceful degradation strategies',
    ],
  },
];

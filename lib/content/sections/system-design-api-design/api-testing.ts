/**
 * API Testing Section
 */

export const apitestingSection = {
  id: 'api-testing',
  title: 'API Testing',
  content: `Comprehensive API testing ensures reliability, correctness, and prevents regressions. Different test types serve different purposes.

## Test Pyramid

\`\`\`
        /\\
       /  \\      E2E Tests (Few, Slow, High Confidence)
      /────\\
     /      \\    Integration Tests (Some, Medium Speed)
    /────────\\
   /          \\  Unit Tests (Many, Fast, Low-Level)
  /────────────\\
\`\`\`

## Unit Tests

Test individual functions and modules:

\`\`\`javascript
const { validateEmail, hashPassword } = require('./utils');

describe('Email Validation', () => {
  it('should accept valid emails', () => {
    expect (validateEmail('user@example.com')).toBe (true);
    expect (validateEmail('test+tag@domain.co.uk')).toBe (true);
  });
  
  it('should reject invalid emails', () => {
    expect (validateEmail('notanemail')).toBe (false);
    expect (validateEmail('@example.com')).toBe (false);
    expect (validateEmail('user@')).toBe (false);
  });
});

describe('Password Hashing', () => {
  it('should hash passwords', async () => {
    const hash = await hashPassword('password123');
    expect (hash).not.toBe('password123');
    expect (hash.length).toBeGreaterThan(50);
  });
});
\`\`\`

## Integration Tests

Test API endpoints:

\`\`\`javascript
const request = require('supertest');
const app = require('./app');

describe('User API', () => {
  describe('POST /users', () => {
    it('should create a new user', async () => {
      const response = await request (app)
        .post('/users')
        .send({
          name: 'John Doe',
          email: 'john@example.com',
          password: 'password123'
        })
        .expect(201);
      
      expect (response.body).toMatchObject({
        id: expect.any(String),
        name: 'John Doe',
        email: 'john@example.com'
      });
      expect (response.body.password).toBeUndefined();
    });
    
    it('should reject invalid email', async () => {
      const response = await request (app)
        .post('/users')
        .send({
          name: 'John Doe',
          email: 'invalid-email',
          password: 'password123'
        })
        .expect(400);
      
      expect (response.body.error).toBe('Invalid email');
    });
    
    it('should require authentication', async () => {
      await request (app)
        .get('/users/123')
        .expect(401);
    });
  });
});
\`\`\`

## Contract Testing

Verify API adheres to OpenAPI spec:

\`\`\`javascript
const { matchers } = require('jest-openapi');
const openApiSpec = require('./openapi.json');

expect.extend (matchers);

describe('API Contract', () => {
  it('should match OpenAPI spec', async () => {
    const response = await request (app)
      .get('/users/123')
      .set('Authorization', 'Bearer token');
    
    expect (response).toSatisfyApiSpec (openApiSpec);
  });
});
\`\`\`

## Load Testing

Test performance under load:

\`\`\`javascript
// k6 load test
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '1m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% of requests < 500ms
    http_req_failed: ['rate<0.01'],    // Error rate < 1%
  },
};

export default function () {
  const response = http.get('https://api.example.com/users');
  
  check (response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  sleep(1);
}
\`\`\`

## Best Practices

1. **Test happy path and edge cases**
2. **Mock external dependencies**
3. **Use factories for test data**
4. **Test authentication and authorization**
5. **Validate response schemas**
6. **Test rate limiting**
7. **Test error handling**
8. **Load test before production**
9. **CI/CD integration**
10. **Monitor test coverage (aim for 80%+)**`,
};

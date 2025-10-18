/**
 * Microservices Testing Section
 */

export const microservicestestingSection = {
  id: 'microservices-testing',
  title: 'Microservices Testing',
  content: `Testing microservices is fundamentally different from testing monoliths. With distributed services, you need multiple testing strategies at different levels.

## Testing Pyramid for Microservices

The testing pyramid still applies, but with microservices-specific considerations:

\`\`\`
         ╱╲
        ╱  ╲      End-to-End Tests (Few)
       ╱────╲     - Full system integration
      ╱      ╲    - Expensive, slow, brittle
     ╱────────╲
    ╱ Contract ╲   Contract Tests (Some)
   ╱   Tests    ╲  - Service boundaries
  ╱──────────────╲ - Provider/Consumer
 ╱  Integration   ╲ Integration Tests (More)
╱      Tests       ╲ - Service + dependencies
────────────────────
   Unit Tests        Unit Tests (Most)
   (Many)            - Fast, isolated, cheap
\`\`\`

**Key difference**: Add **Contract Tests** layer to verify service boundaries.

---

## 1. Unit Tests

**What**: Test individual functions/classes in isolation.

**Scope**: Single service, no external dependencies.

**Example**:
\`\`\`javascript
// order-service/calculateTotal.test.js
describe('calculateTotal', () => {
    it('should calculate order total correctly', () => {
        const items = [
            { price: 10, quantity: 2 },
            { price: 5, quantity: 3 }
        ];
        
        const total = calculateTotal(items);
        
        expect(total).toBe(35); // (10*2) + (5*3)
    });
    
    it('should apply discount correctly', () => {
        const items = [{ price: 100, quantity: 1 }];
        const discount = 0.1; // 10%
        
        const total = calculateTotal(items, discount);
        
        expect(total).toBe(90);
    });
});
\`\`\`

**Best Practices**:
- Mock external dependencies (database, other services)
- Fast execution (< 100ms per test)
- High coverage (80%+)
- Run on every code change

---

## 2. Integration Tests

**What**: Test service with its dependencies (database, message queue, cache).

**Scope**: Single service + immediate dependencies.

### Database Integration Tests

**Use test database or Docker containers**:

\`\`\`javascript
// order-service/orderRepository.integration.test.js
describe('OrderRepository Integration', () => {
    let db;
    
    beforeAll(async () => {
        // Start test database (Docker)
        db = await startTestDatabase();
        await db.migrate();
    });
    
    afterAll(async () => {
        await db.close();
    });
    
    afterEach(async () => {
        await db.clearAll(); // Clean between tests
    });
    
    it('should create and retrieve order', async () => {
        const order = {
            userId: 'user-123',
            items: [{ productId: 'prod-1', quantity: 2 }],
            total: 50
        };
        
        const createdOrder = await orderRepository.create(order);
        const retrievedOrder = await orderRepository.findById(createdOrder.id);
        
        expect(retrievedOrder).toMatchObject(order);
    });
    
    it('should update order status', async () => {
        const order = await orderRepository.create({...});
        
        await orderRepository.updateStatus(order.id, 'SHIPPED');
        
        const updated = await orderRepository.findById(order.id);
        expect(updated.status).toBe('SHIPPED');
    });
});
\`\`\`

### Message Queue Integration Tests

\`\`\`javascript
describe('Order Event Publisher Integration', () => {
    let rabbitMQ;
    
    beforeAll(async () => {
        rabbitMQ = await startTestRabbitMQ();
    });
    
    it('should publish OrderCreated event', async () => {
        const order = { id: 'order-123', userId: 'user-1' };
        
        await eventPublisher.publishOrderCreated(order);
        
        // Verify message published
        const messages = await rabbitMQ.getMessages('order-events');
        expect(messages).toHaveLength(1);
        expect(messages[0]).toMatchObject({
            eventType: 'OrderCreated',
            data: order
        });
    });
});
\`\`\`

---

## 3. Contract Tests

**Problem**: Service A depends on Service B. If Service B changes its API, Service A breaks.

**Solution**: Contract tests verify that provider (Service B) fulfills the contract that consumer (Service A) expects.

### Consumer-Driven Contract Testing

**Tool**: Pact

**Flow**:
1. **Consumer** (Order Service) defines expected contract
2. **Consumer** generates Pact file
3. **Provider** (Payment Service) verifies it can fulfill contract

**Example**:

**Consumer side** (Order Service):
\`\`\`javascript
// order-service/payment.contract.test.js
const { Pact } = require('@pact-foundation/pact');
const { chargePayment } = require('./paymentClient');

describe('Payment Service Contract', () => {
    const provider = new Pact({
        consumer: 'OrderService',
        provider: 'PaymentService',
        port: 8989
    });
    
    beforeAll(() => provider.setup());
    afterAll(() => provider.finalize());
    
    describe('charge payment endpoint', () => {
        beforeAll(() => {
            // Define expected interaction
            return provider.addInteraction({
                state: 'user has sufficient balance',
                uponReceiving: 'a charge payment request',
                withRequest: {
                    method: 'POST',
                    path: '/charge',
                    body: {
                        userId: 'user-123',
                        amount: 50,
                        orderId: 'order-456'
                    }
                },
                willRespondWith: {
                    status: 200,
                    body: {
                        transactionId: Matchers.like('txn-789'),
                        status: 'SUCCESS'
                    }
                }
            });
        });
        
        it('should charge payment successfully', async () => {
            const result = await chargePayment({
                userId: 'user-123',
                amount: 50,
                orderId: 'order-456'
            });
            
            expect(result.status).toBe('SUCCESS');
            expect(result.transactionId).toBeTruthy();
        });
    });
});
\`\`\`

This generates a **Pact file** (JSON contract).

**Provider side** (Payment Service):
\`\`\`javascript
// payment-service/pact-verification.test.js
const { Verifier } = require('@pact-foundation/pact');

describe('Pact Verification', () => {
    it('should validate all consumer contracts', async () => {
        await new Verifier({
            provider: 'PaymentService',
            providerBaseUrl: 'http://localhost:8080',
            pactUrls: [
                './pacts/OrderService-PaymentService.json'
            ],
            publishVerificationResult: true
        }).verifyProvider();
    });
});
\`\`\`

**Benefits**:
✅ Catch breaking changes before deployment
✅ Consumer defines contract (knows what it needs)
✅ Automated testing of service boundaries
✅ No need to run both services together

---

## 4. Component Tests

**What**: Test entire service in isolation with mocked dependencies.

**Approach**: Run service, mock external services, test via API.

\`\`\`javascript
// order-service/component.test.js
describe('Order Service Component Tests', () => {
    let app;
    let mockPaymentService;
    let mockInventoryService;
    
    beforeAll(async () => {
        // Start mock services
        mockPaymentService = startMockServer(8081);
        mockInventoryService = startMockServer(8082);
        
        // Start order service (points to mocks)
        app = await startOrderService({
            paymentServiceUrl: 'http://localhost:8081',
            inventoryServiceUrl: 'http://localhost:8082'
        });
    });
    
    afterAll(async () => {
        await app.stop();
        await mockPaymentService.stop();
        await mockInventoryService.stop();
    });
    
    it('should create order end-to-end', async () => {
        // Setup mocks
        mockInventoryService.on('POST', '/reserve', () => ({
            status: 200,
            body: { reserved: true }
        }));
        
        mockPaymentService.on('POST', '/charge', () => ({
            status: 200,
            body: { transactionId: 'txn-123', status: 'SUCCESS' }
        }));
        
        // Test order creation
        const response = await request(app)
            .post('/orders')
            .send({
                userId: 'user-123',
                items: [{ productId: 'prod-1', quantity: 2 }]
            });
        
        expect(response.status).toBe(201);
        expect(response.body.status).toBe('PENDING');
        
        // Verify calls to dependencies
        expect(mockInventoryService.requests).toHaveLength(1);
        expect(mockPaymentService.requests).toHaveLength(1);
    });
    
    it('should handle payment failure', async () => {
        mockInventoryService.on('POST', '/reserve', () => ({
            status: 200,
            body: { reserved: true }
        }));
        
        // Payment fails
        mockPaymentService.on('POST', '/charge', () => ({
            status: 400,
            body: { error: 'Insufficient funds' }
        }));
        
        const response = await request(app)
            .post('/orders')
            .send({...});
        
        expect(response.status).toBe(400);
        expect(response.body.error).toContain('Payment failed');
        
        // Should have called inventory release (compensating transaction)
        expect(mockInventoryService.requests).toContainEqual(
            expect.objectContaining({ path: '/release' })
        );
    });
});
\`\`\`

---

## 5. End-to-End Tests

**What**: Test entire system with all services running.

**Challenges**:
❌ Slow (seconds to minutes per test)
❌ Flaky (network issues, timing problems)
❌ Expensive to maintain
❌ Hard to debug failures

**Keep them minimal!**

### Critical Path Testing

**Only test happy paths and critical failures**:

\`\`\`javascript
// e2e/order-flow.test.js
describe('Order Flow E2E', () => {
    beforeAll(async () => {
        // Start all services (Docker Compose)
        await exec('docker-compose up -d');
        await waitForServices();
    });
    
    afterAll(async () => {
        await exec('docker-compose down');
    });
    
    it('should complete full order flow', async () => {
        // 1. Create user
        const user = await createUser({
            email: 'test@example.com',
            password: 'password123'
        });
        
        // 2. Add to cart
        await addToCart(user.id, { productId: 'prod-1', quantity: 2 });
        
        // 3. Checkout
        const order = await checkout(user.id, {
            paymentMethod: 'credit_card'
        });
        
        expect(order.status).toBe('CONFIRMED');
        
        // 4. Wait for async processing
        await sleep(2000);
        
        // 5. Verify order status
        const orderStatus = await getOrderStatus(order.id);
        expect(orderStatus.status).toBe('SHIPPED');
        
        // 6. Verify email sent
        const emails = await getTestEmails(user.email);
        expect(emails).toContainEqual(
            expect.objectContaining({
                subject: 'Order Confirmed',
                orderId: order.id
            })
        );
    });
});
\`\`\`

**Best Practices**:
- Run in dedicated environment (not production!)
- Use test data that's isolated
- Clean up after tests
- Run less frequently (nightly, not on every commit)

---

## Test Data Management

### Problem: Test Data Conflicts

Multiple tests running in parallel can interfere:

\`\`\`
Test 1: Create user with email "test@example.com"
Test 2: Create user with email "test@example.com" ❌ CONFLICT
\`\`\`

### Solutions:

**1. Generate Unique Data**:
\`\`\`javascript
const userId = \`user-\${uuidv4()}\`;
const email = \`test-\${Date.now()}@example.com\`;
\`\`\`

**2. Use Test Database Per Test**:
\`\`\`javascript
beforeEach(async () => {
    testDb = await createTestDatabase();
});

afterEach(async () => {
    await testDb.drop();
});
\`\`\`

**3. Database Transactions (Rollback)**:
\`\`\`javascript
beforeEach(async () => {
    await db.beginTransaction();
});

afterEach(async () => {
    await db.rollback(); // Undo all changes
});
\`\`\`

---

## Testing Asynchronous Communication

### Testing Event-Driven Services

**Challenge**: Service A publishes event, Service B consumes it asynchronously.

**Solution**: Poll and wait for expected state:

\`\`\`javascript
describe('Order Created Event Processing', () => {
    it('should send email when order created', async () => {
        // Publish event
        await eventBus.publish('OrderCreated', {
            orderId: 'order-123',
            userId: 'user-456'
        });
        
        // Poll for email sent (eventual consistency)
        await waitFor(async () => {
            const emails = await getEmailsSent('user-456');
            return emails.some(e => e.orderId === 'order-123');
        }, { timeout: 5000 });
        
        // Verify email content
        const emails = await getEmailsSent('user-456');
        const orderEmail = emails.find(e => e.orderId === 'order-123');
        expect(orderEmail.subject).toContain('Order Confirmed');
    });
});

// Helper
async function waitFor(condition, options = {}) {
    const timeout = options.timeout || 5000;
    const interval = options.interval || 100;
    const start = Date.now();
    
    while (Date.now() - start < timeout) {
        if (await condition()) {
            return;
        }
        await sleep(interval);
    }
    
    throw new Error('Condition not met within timeout');
}
\`\`\`

---

## Testing Resilience Patterns

### Circuit Breaker Tests

\`\`\`javascript
describe('Circuit Breaker', () => {
    it('should open after threshold failures', async () => {
        // Make service fail
        mockPaymentService.alwaysFail();
        
        // Trigger failures
        for (let i = 0; i < 5; i++) {
            await expect(chargePayment({})).rejects.toThrow();
        }
        
        // Circuit should be open now
        await expect(chargePayment({})).rejects.toThrow('Circuit breaker open');
        
        // Verify no requests sent (circuit is open)
        expect(mockPaymentService.requestCount).toBe(5); // Not 6
    });
    
    it('should half-open after timeout', async () => {
        // Open circuit
        mockPaymentService.alwaysFail();
        for (let i = 0; i < 5; i++) {
            await chargePayment({}).catch(() => {});
        }
        
        // Wait for reset timeout
        await sleep(30000);
        
        // Fix service
        mockPaymentService.alwaysSucceed();
        
        // Should allow test request (half-open)
        const result = await chargePayment({});
        expect(result.status).toBe('SUCCESS');
        
        // Circuit should be closed now
        const result2 = await chargePayment({});
        expect(result2.status).toBe('SUCCESS');
    });
});
\`\`\`

### Retry Tests

\`\`\`javascript
describe('Retry Logic', () => {
    it('should retry on transient failure', async () => {
        // Fail twice, then succeed
        mockPaymentService
            .onCall(1).fail()
            .onCall(2).fail()
            .onCall(3).succeed();
        
        const result = await chargePaymentWithRetry({});
        
        expect(result.status).toBe('SUCCESS');
        expect(mockPaymentService.requestCount).toBe(3);
    });
    
    it('should not retry on permanent failure', async () => {
        // 400 Bad Request (permanent failure)
        mockPaymentService.respondWith(400, { error: 'Invalid card' });
        
        await expect(chargePaymentWithRetry({})).rejects.toThrow('Invalid card');
        
        // Should not retry 400 errors
        expect(mockPaymentService.requestCount).toBe(1);
    });
});
\`\`\`

---

## Performance Testing

### Load Testing

**Tool**: k6, Artillery, Gatling

\`\`\`javascript
// k6 load test
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
    stages: [
        { duration: '1m', target: 50 },   // Ramp up to 50 users
        { duration: '3m', target: 50 },   // Stay at 50 users
        { duration: '1m', target: 100 },  // Ramp up to 100
        { duration: '3m', target: 100 },  // Stay at 100
        { duration: '1m', target: 0 },    // Ramp down
    ],
    thresholds: {
        http_req_duration: ['p(95)<500'], // 95% of requests < 500ms
        http_req_failed: ['rate<0.01'],   // < 1% failure rate
    },
};

export default function () {
    const res = http.post('http://api-gateway/orders', JSON.stringify({
        userId: 'user-123',
        items: [{ productId: 'prod-1', quantity: 2 }]
    }), {
        headers: { 'Content-Type': 'application/json' },
    });
    
    check(res, {
        'status is 201': (r) => r.status === 201,
        'response time < 500ms': (r) => r.timings.duration < 500,
    });
    
    sleep(1);
}
\`\`\`

---

## Chaos Testing

**What**: Intentionally break things to verify resilience.

**Tool**: Chaos Monkey, Gremlin

**Examples**:
- Randomly kill service instances
- Inject network latency
- Fill up disk space
- Corrupt databases

\`\`\`javascript
describe('Chaos Tests', () => {
    it('should handle payment service failure', async () => {
        // Kill payment service
        await killService('payment-service');
        
        // Order creation should still work (degrade gracefully)
        const response = await createOrder({...});
        expect(response.status).toBe(202); // Accepted
        expect(response.body.status).toBe('PENDING_PAYMENT');
    });
    
    it('should recover when service comes back', async () => {
        await killService('payment-service');
        
        // Wait for Kubernetes to restart
        await sleep(10000);
        
        // Should work again
        const response = await createOrder({...});
        expect(response.status).toBe(201);
    });
});
\`\`\`

---

## Testing in Production

### Synthetic Monitoring

Run automated tests against production.

\`\`\`javascript
// Runs every 5 minutes in production
describe('Production Smoke Tests', () => {
    it('should create test order', async () => {
        const order = await createOrder({
            userId: 'SYNTHETIC_USER',
            items: [{ productId: 'TEST_PRODUCT', quantity: 1 }]
        });
        
        expect(order.status).toBe('CONFIRMED');
        
        // Clean up
        await deleteOrder(order.id);
    });
});
\`\`\`

---

## Best Practices

1. **Test Pyramid**: Most unit tests, fewer integration, minimal E2E
2. **Contract Tests**: Verify service boundaries
3. **Independent Tests**: No shared state between tests
4. **Fast Feedback**: Unit tests run on every commit
5. **Isolate Services**: Use mocks, stubs, or Docker
6. **Test Data**: Generate unique data, clean up after tests
7. **Chaos Engineering**: Test failure scenarios
8. **Monitor in Production**: Synthetic tests, real user monitoring

---

## Interview Tips

**Red Flags**:
❌ Only unit tests (ignoring integration)
❌ Too many E2E tests (slow, brittle)
❌ Not testing service boundaries

**Good Responses**:
✅ Explain testing pyramid for microservices
✅ Mention contract testing (Pact)
✅ Discuss trade-offs (speed vs confidence)
✅ Talk about chaos engineering

**Sample Answer**:
*"For microservices testing, I follow the testing pyramid: majority unit tests for fast feedback, integration tests for service + dependencies, contract tests to verify service boundaries, and minimal E2E tests for critical paths. I'd use Pact for consumer-driven contract testing to catch breaking changes early. For async communication, I'd use polling with timeouts. Tests should be independent with unique test data. I'd also implement chaos testing to verify resilience and synthetic monitoring in production."*

---

## Key Takeaways

1. **Testing pyramid** applies but with contract tests added
2. **Contract tests** verify service boundaries without running both services
3. **Minimal E2E tests** (slow, expensive, flaky)
4. **Integration tests** for service + immediate dependencies
5. **Async testing** requires polling and timeouts
6. **Chaos engineering** verifies resilience
7. **Test data** must be unique and isolated
8. **Synthetic monitoring** tests production continuously`,
};

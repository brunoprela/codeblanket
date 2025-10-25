/**
 * Distributed Transactions & Saga Pattern Section
 */

export const distributedtransactionssagaSection = {
  id: 'distributed-transactions-saga',
  title: 'Distributed Transactions & Saga Pattern',
  content: `Maintaining data consistency across multiple microservices is one of the hardest challenges in distributed systems. Traditional ACID transactions don't work across service boundaries.

## The Problem: Distributed Transactions

**Monolith with local transaction**:
\`\`\`sql
BEGIN TRANSACTION;
  INSERT INTO orders (...);
  UPDATE inventory SET quantity = quantity - 1;
  INSERT INTO payments (...);
  UPDATE loyalty_points SET points = points + 100;
COMMIT;
\`\`\`

Either all succeed or all rollback atomically.

**Microservices - each has its own database**:
\`\`\`
Order Service (orders DB)
Inventory Service (inventory DB)  
Payment Service (payments DB)
Loyalty Service (loyalty DB)
\`\`\`

**Cannot use a single database transaction across services!**

**Failure scenario**:
\`\`\`
1. Order Service: Create order ✅
2. Inventory Service: Decrease stock ✅  
3. Payment Service: Charge card ❌ FAILS
4. Loyalty Service: Add points ❌ NOT EXECUTED

Result: Order created, inventory decreased, but no payment!
Customer gets free product 💸
\`\`\`

---

## Why 2PC (Two-Phase Commit) Doesn't Work

**Two-Phase Commit** is a traditional distributed transaction protocol:

**Phase 1 - Prepare**:
\`\`\`
Coordinator: "Can you all commit?"
Service A: "Yes, I'm ready"
Service B: "Yes, I'm ready"  
Service C: "Yes, I'm ready"
\`\`\`

**Phase 2 - Commit**:
\`\`\`
Coordinator: "OK, everyone commit now!"
All services: Commit simultaneously
\`\`\`

**Problems in microservices**:

1. **Blocking**: Services hold locks while waiting for coordinator, reducing throughput
2. **Single point of failure**: If coordinator crashes, all services are stuck
3. **Not supported**: Many modern databases (NoSQL, cloud) don't support 2PC
4. **Latency**: Synchronous protocol adds significant latency
5. **Reduced availability**: System availability = product of all service availabilities

**Example**:
\`\`\`
If each service has 99.9% uptime:
4 services with 2PC = 0.999^4 = 99.6% uptime

Without 2PC (eventual consistency) = 99.9% uptime
\`\`\`

**Microservices prefer availability over consistency** (CAP theorem).

---

## The Saga Pattern

A **saga** is a sequence of local transactions where each service performs its work and publishes events. If one step fails, compensating transactions undo the previous steps.

**Key principles**:
1. Each service performs its local transaction
2. On success, publishes event to trigger next step
3. On failure, executes compensating transactions to rollback

**Two implementations**:
1. **Choreography**: Decentralized, event-driven
2. **Orchestration**: Centralized coordinator

---

## Choreography-Based Saga

Services communicate via events. Each service listens for events, does work, publishes next event.

**Example: E-commerce order**

**Happy path**:
\`\`\`
1. Order Service:     CreateOrder() → OrderCreated event
2. Inventory Service: (listens) → ReserveInventory() → InventoryReserved event
3. Payment Service:   (listens) → ChargeCard() → PaymentCompleted event
4. Shipping Service:  (listens) → ShipOrder() → OrderShipped event
\`\`\`

**Failure path** (payment fails):
\`\`\`
1. Order Service:     CreateOrder() → OrderCreated event
2. Inventory Service: ReserveInventory() → InventoryReserved event
3. Payment Service:   ChargeCard() → ❌ FAILS → PaymentFailed event
4. Inventory Service: (listens) → CancelReservation() → InventoryReleased event
5. Order Service:     (listens) → CancelOrder() → OrderCancelled event
\`\`\`

**Implementation**:
\`\`\`javascript
// Order Service
async function createOrder (orderData) {
    // Local transaction
    const order = await db.orders.insert({
        ...orderData,
        status: 'PENDING'
    });
    
    // Publish event
    await eventBus.publish('OrderCreated', {
        orderId: order.id,
        items: order.items,
        customerId: order.customerId
    });
    
    return order;
}

// Listen for failure events
eventBus.on('PaymentFailed', async (event) => {
    // Compensating transaction
    await db.orders.update (event.orderId, {
        status: 'CANCELLED'
    });
    
    await eventBus.publish('OrderCancelled', {
        orderId: event.orderId
    });
});

// Inventory Service
eventBus.on('OrderCreated', async (event) => {
    try {
        // Reserve inventory
        await db.inventory.update({
            productId: event.items[0].productId,
            reserved: { $inc: event.items[0].quantity }
        });
        
        await eventBus.publish('InventoryReserved', {
            orderId: event.orderId,
            items: event.items
        });
    } catch (error) {
        await eventBus.publish('InventoryReservationFailed', {
            orderId: event.orderId,
            reason: 'Out of stock'
        });
    }
});

// Compensating transaction
eventBus.on('PaymentFailed', async (event) => {
    await db.inventory.update({
        productId: event.items[0].productId,
        reserved: { $dec: event.items[0].quantity }
    });
    
    await eventBus.publish('InventoryReleased', {
        orderId: event.orderId
    });
});
\`\`\`

**Advantages**:
✅ Decentralized (no single point of failure)
✅ Services are loosely coupled
✅ Simple for basic flows

**Disadvantages**:
❌ Hard to understand flow (scattered across services)
❌ Difficult to debug and monitor
❌ Risk of cyclic dependencies
❌ Harder to add new steps

---

## Orchestration-Based Saga

A **saga orchestrator** coordinates the saga, telling each service what to do and handling failures.

**Example: Order orchestrator**

**Happy path**:
\`\`\`
Orchestrator:
  1. Tell Order Service: CreateOrder()
  2. Tell Inventory Service: ReserveInventory()
  3. Tell Payment Service: ChargeCard()
  4. Tell Shipping Service: ShipOrder()
  5. Mark saga COMPLETED
\`\`\`

**Failure path**:
\`\`\`
Orchestrator:
  1. Tell Order Service: CreateOrder() ✅
  2. Tell Inventory Service: ReserveInventory() ✅
  3. Tell Payment Service: ChargeCard() ❌ FAILS
  4. Rollback:
     - Tell Inventory Service: CancelReservation()
     - Tell Order Service: CancelOrder()
  5. Mark saga FAILED
\`\`\`

**Implementation**:
\`\`\`javascript
// Saga Orchestrator
class OrderSagaOrchestrator {
    async execute (orderData) {
        const sagaId = generateId();
        const sagaState = {
            id: sagaId,
            status: 'STARTED',
            steps: []
        };
        
        try {
            // Step 1: Create order
            const order = await orderService.createOrder (orderData);
            sagaState.steps.push({step: 'CreateOrder', status: 'COMPLETED', data: order});
            
            // Step 2: Reserve inventory
            await inventoryService.reserveInventory (order.items);
            sagaState.steps.push({step: 'ReserveInventory', status: 'COMPLETED'});
            
            // Step 3: Charge payment
            const payment = await paymentService.charge({
                amount: order.total,
                cardToken: orderData.paymentToken
            });
            sagaState.steps.push({step: 'ChargePayment', status: 'COMPLETED', data: payment});
            
            // Step 4: Ship order
            await shippingService.ship (order.id);
            sagaState.steps.push({step: 'ShipOrder', status: 'COMPLETED'});
            
            // Success
            sagaState.status = 'COMPLETED';
            await sagaRepository.save (sagaState);
            return order;
            
        } catch (error) {
            // Rollback in reverse order
            sagaState.status = 'ROLLING_BACK';
            await this.rollback (sagaState);
            throw error;
        }
    }
    
    async rollback (sagaState) {
        // Execute compensating transactions in reverse order
        const completedSteps = sagaState.steps.filter (s => s.status === 'COMPLETED').reverse();
        
        for (const step of completedSteps) {
            switch (step.step) {
                case 'CreateOrder':
                    await orderService.cancelOrder (step.data.id);
                    break;
                case 'ReserveInventory':
                    await inventoryService.releaseInventory (step.data.items);
                    break;
                case 'ChargePayment':
                    await paymentService.refund (step.data.paymentId);
                    break;
                // No compensation needed for ShipOrder (we failed before this)
            }
        }
        
        sagaState.status = 'ROLLED_BACK';
        await sagaRepository.save (sagaState);
    }
}
\`\`\`

**Advantages**:
✅ Centralized logic (easy to understand)
✅ Easy to add/remove steps
✅ Better monitoring and debugging
✅ Can implement timeouts, retries centrally

**Disadvantages**:
❌ Orchestrator can become single point of failure (mitigate with HA)
❌ Services coupled to orchestrator
❌ Orchestrator can become complex (god object)

---

## Compensating Transactions

**Compensating transaction**: Undo the effect of a previous transaction.

**Example**:

| Forward Transaction | Compensating Transaction |
|---------------------|-------------------------|
| Create order | Cancel order |
| Reserve inventory | Release inventory |
| Charge credit card | Refund |
| Send email | Send cancellation email |

**Important**: Compensations should be **idempotent** (can be executed multiple times safely).

**Semantic rollback vs Physical rollback**:
- **Physical**: Delete the order record (not recommended - lose history)
- **Semantic**: Mark order as CANCELLED (preferred - audit trail)

**Example**:
\`\`\`sql
-- ❌ Physical rollback
DELETE FROM orders WHERE id = 123;

-- ✅ Semantic rollback
UPDATE orders SET status = 'CANCELLED', cancelled_at = NOW() WHERE id = 123;
\`\`\`

---

## Handling Failures

### Retryable Failures (Transient)

Temporary issues that might succeed on retry:
- Network timeout
- Database temporarily unavailable
- Rate limit exceeded

**Solution**: Retry with exponential backoff

\`\`\`javascript
async function withRetry (fn, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fn();
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            
            // Exponential backoff
            const delay = Math.pow(2, i) * 1000;
            await sleep (delay);
        }
    }
}

// Usage
await withRetry(() => paymentService.charge (payment));
\`\`\`

### Non-Retryable Failures (Permanent)

Business rule violations that won't succeed on retry:
- Insufficient funds
- Invalid card
- Out of stock

**Solution**: Execute compensating transactions immediately

---

## Saga State Machine

Track saga state to handle crashes and recovery.

**States**:
\`\`\`
STARTED → INVENTORY_RESERVED → PAYMENT_COMPLETED → SHIPPED → COMPLETED
                ↓                      ↓                ↓
              FAILED             COMPENSATING     COMPENSATING
                                      ↓                ↓
                                  ROLLED_BACK    ROLLED_BACK
\`\`\`

**Store state in database**:
\`\`\`sql
CREATE TABLE sagas (
    id UUID PRIMARY KEY,
    order_id UUID,
    status VARCHAR(50), -- STARTED, COMPLETED, FAILED, ROLLING_BACK, ROLLED_BACK
    current_step VARCHAR(100),
    steps JSONB, -- Array of completed steps
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
\`\`\`

**Recovery**: If saga orchestrator crashes, another instance can pick up and continue from last saved state.

---

## Saga Execution Coordinator (SEC) Pattern

When using orchestration, implement a **Saga Execution Coordinator**:

**Responsibilities**:
1. Execute saga steps in order
2. Store saga state after each step
3. Handle failures and trigger compensations
4. Implement timeouts
5. Provide monitoring and visibility

**Example (using state machine)**:
\`\`\`javascript
class SagaExecutionCoordinator {
    async run (sagaDefinition, data) {
        const saga = await this.createSaga (sagaDefinition, data);
        
        for (const step of sagaDefinition.steps) {
            try {
                // Execute step
                const result = await this.executeStep (step, saga);
                
                // Save state
                saga.steps.push({name: step.name, status: 'COMPLETED', result});
                saga.currentStep = step.name;
                await this.saveSaga (saga);
                
            } catch (error) {
                // Trigger compensation
                saga.status = 'FAILED';
                await this.compensate (saga);
                throw error;
            }
        }
        
        saga.status = 'COMPLETED';
        await this.saveSaga (saga);
        return saga;
    }
    
    async compensate (saga) {
        const completedSteps = saga.steps.filter (s => s.status === 'COMPLETED').reverse();
        
        for (const step of completedSteps) {
            const compensation = sagaDefinition.compensations[step.name];
            if (compensation) {
                await this.executeCompensation (compensation, step.result);
            }
        }
        
        saga.status = 'ROLLED_BACK';
        await this.saveSaga (saga);
    }
}
\`\`\`

---

## Real-World Example: Amazon Order

**Saga steps**:
1. **Order Service**: Create order (status: PENDING)
2. **Inventory Service**: Reserve items
3. **Payment Service**: Authorize payment (not capture yet)
4. **Shipping Service**: Calculate shipping, reserve slot
5. **Payment Service**: Capture payment (now that shipping confirmed)
6. **Notification Service**: Send confirmation email
7. **Order Service**: Mark order CONFIRMED

**Compensations if step 5 fails**:
- Shipping: Release slot
- Payment: Cancel authorization (it will expire anyway)
- Inventory: Release reserved items
- Order: Mark as CANCELLED
- Notification: Send cancellation email

**Why authorize then capture?**
- Don't charge customer until we're sure we can ship
- Authorization holds the funds but doesn't transfer them

---

## Eventual Consistency

Sagas provide **eventual consistency**, not immediate consistency.

**During saga execution**:
\`\`\`
Time T0: Order created (status: PENDING)
Time T1: Inventory reserved
Time T2: Payment processing...
Time T3: Payment succeeded (status: CONFIRMED)
\`\`\`

**Between T0 and T3, order is in inconsistent state** (created but not paid).

**Handling this**:
1. **Don't expose intermediate states to users**: Show "Processing..." to customer
2. **Status field**: Use status field to track saga progress
3. **Read your writes**: Query same service that wrote the data
4. **Version/timestamp**: Use to detect stale reads

---

## Saga vs 2PC Comparison

| Aspect | Two-Phase Commit (2PC) | Saga |
|--------|----------------------|------|
| **Isolation** | Locks held during transaction | No locks (eventual consistency) |
| **Availability** | Lower (blocking) | Higher (non-blocking) |
| **Consistency** | Strong (ACID) | Eventual |
| **Latency** | Higher (synchronous) | Lower (asynchronous) |
| **Rollback** | Automatic | Manual (compensating transactions) |
| **Complexity** | Database handles it | Application handles it |
| **Use case** | Monoliths, local services | Microservices, distributed systems |

---

## Interview Tips

**Red Flags**:
❌ Suggesting distributed ACID transactions
❌ Not mentioning compensating transactions
❌ Ignoring failure scenarios

**Good Responses**:
✅ Explain Saga pattern (choreography vs orchestration)
✅ Discuss eventual consistency trade-off
✅ Mention specific tools (temporal.io, Netflix Conductor)
✅ Explain compensating transactions

**Sample Answer**:
*"I'd use the Saga pattern with orchestration for complex flows. The order service would coordinate steps: create order, reserve inventory, charge payment, ship order. Each step is a local transaction in its service. If any step fails, we execute compensating transactions in reverse order (refund payment, release inventory, cancel order). We'd store saga state in a database for crash recovery. This provides eventual consistency rather than ACID, but gives us better availability and scalability. For simpler flows, I'd use choreography with event-driven communication."*

---

## Key Takeaways

1. **Distributed transactions** across microservices require special patterns
2. **2PC doesn't work** well in microservices (blocking, reduces availability)
3. **Saga pattern**: Sequence of local transactions with compensating transactions
4. **Choreography**: Decentralized, event-driven (good for simple flows)
5. **Orchestration**: Centralized coordinator (better for complex flows, easier to understand)
6. **Compensating transactions**: Semantic rollback, must be idempotent
7. **Eventual consistency**: Accept that system is temporarily inconsistent during saga
8. **Store saga state**: Enable crash recovery and monitoring`,
};

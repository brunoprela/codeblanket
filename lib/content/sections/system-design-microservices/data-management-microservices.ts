/**
 * Data Management in Microservices Section
 */

export const datamanagementmicroservicesSection = {
  id: 'data-management-microservices',
  title: 'Data Management in Microservices',
  content: `Data management is one of the most challenging aspects of microservices. The "database per service" pattern provides autonomy but introduces complexity in querying and consistency.

## Database Per Service Pattern

**Core Principle**: Each microservice owns its data and database. Other services cannot access it directly.

**Monolith**: Single shared database
\`\`\`
┌─────────────────────────────────┐
│    Application (Monolith)       │
└────────────┬────────────────────┘
             │
    ┌────────▼───────────┐
    │  Shared Database   │
    │ ┌────────────────┐ │
    │ │ orders         │ │
    │ │ users          │ │
    │ │ payments       │ │
    │ │ products       │ │
    │ └────────────────┘ │
    └────────────────────┘
\`\`\`

**Microservices**: Database per service
\`\`\`
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Order   │  │   User   │  │ Payment  │  │ Product  │
│ Service  │  │ Service  │  │ Service  │  │ Service  │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │              │              │
┌────▼────┐   ┌───▼────┐    ┌───▼────┐    ┌───▼────┐
│ Orders  │   │ Users  │    │Payments│    │Products│
│   DB    │   │   DB   │    │   DB   │    │   DB   │
└─────────┘   └────────┘    └────────┘    └────────┘
\`\`\`

**Why?**
✅ **Loose coupling**: Services can evolve databases independently
✅ **Tech diversity**: Use best database for each use case (SQL, NoSQL, graph)
✅ **Scalability**: Scale databases independently
✅ **Failure isolation**: Database failure affects only one service

**Challenges**:
❌ Can't use JOIN across services
❌ Distributed transactions (already covered via Saga)
❌ Data duplication
❌ Eventual consistency

---

## Implementing Database Per Service

### Option 1: Separate Database Instances

Each service has its own database server.

**Pros**: Complete isolation, can scale database independently
**Cons**: Higher cost, more operational overhead

\`\`\`
Order Service  → MySQL instance 1
User Service   → PostgreSQL instance 2
Product Service → MongoDB instance 3
\`\`\`

### Option 2: Separate Schemas

Services share database server but have separate schemas/databases.

**Pros**: Lower cost, easier to manage
**Cons**: Less isolation, database becomes coupling point

\`\`\`
MySQL Server:
├── order_db (Order Service)
├── user_db (User Service)
└── product_db (Product Service)
\`\`\`

**Best Practice**: Start with separate schemas, move to separate instances as you scale.

---

## Handling Cross-Service Queries

Without JOIN, how do you query data across services?

### Problem: Display Order with Product Details

**In Monolith**:
\`\`\`sql
SELECT o.*, p.name, p.price, p.image
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE o.user_id = 123;
\`\`\`

**In Microservices**: Can't JOIN across services!

### Solution 1: API Composition

Application makes multiple API calls and combines results.

\`\`\`javascript
// Get orders from Order Service
const orders = await orderService.getOrdersByUser(userId);

// Extract product IDs
const productIds = orders.map(o => o.productId);

// Get product details from Product Service
const products = await productService.getByIds(productIds);

// Combine in application
const ordersWithProducts = orders.map(order => ({
    ...order,
    product: products.find(p => p.id === order.productId)
}));
\`\`\`

**Pros**: Simple, maintains service boundaries
**Cons**: Multiple round trips (latency), N+1 query problem

### Solution 2: Data Duplication (CQRS)

Store denormalized data for queries.

**Idea**: Order Service stores product name and price (not just ID)

\`\`\`javascript
// When creating order
const order = {
    id: generateId(),
    userId: userId,
    productId: product.id,
    productName: product.name,      // Duplicated!
    productPrice: product.price,    // Duplicated!
    productImage: product.image,    // Duplicated!
    status: 'PENDING'
};
\`\`\`

**Pros**: Fast queries (no extra calls), single service read
**Cons**: Data duplication, staleness (what if product name changes?)

**When to use**: For data that rarely changes or staleness is acceptable

**Updating duplicated data**:
- Product Service publishes ProductUpdated event
- Order Service listens and updates its copies

\`\`\`javascript
// Product Service
await productService.updateProduct(productId, {name: 'New Name'});
await eventBus.publish('ProductUpdated', {
    productId,
    name: 'New Name',
    price: 999
});

// Order Service (listener)
eventBus.on('ProductUpdated', async (event) => {
    // Update all orders with this product
    await db.orders.updateMany(
        {productId: event.productId},
        {
            productName: event.name,
            productPrice: event.price
        }
    );
});
\`\`\`

### Solution 3: CQRS with Read Models

Create specialized read databases optimized for queries.

**Pattern**: Command Query Responsibility Segregation (CQRS)

**Architecture**:
\`\`\`
Write Side:
  Order Service  → Orders DB (write)
  Product Service → Products DB (write)

Read Side:
  Events ↓
  OrderViewService → Order View DB (read-only, denormalized)
    Contains: orders + product details + user details
\`\`\`

**Implementation**:
\`\`\`javascript
// Order View Service
eventBus.on('OrderCreated', async (event) => {
    // Get additional data
    const product = await productService.get(event.productId);
    const user = await userService.get(event.userId);
    
    // Create denormalized read model
    await orderViewDB.insert({
        orderId: event.orderId,
        orderStatus: event.status,
        orderTotal: event.total,
        // Product details
        productId: product.id,
        productName: product.name,
        productPrice: product.price,
        productImage: product.image,
        // User details
        userId: user.id,
        userName: user.name,
        userEmail: user.email
    });
});

// Queries use read model
async function getOrderDetailsForUser(userId) {
    return await orderViewDB.find({userId});
    // Returns everything in one query!
}
\`\`\`

**Pros**: Fast queries, optimized for specific use cases
**Cons**: Eventual consistency, complexity, storage overhead

---

## Choosing the Right Database Per Service

Different services have different data needs.

### Relational (SQL)

**Use for**: Structured data, complex queries, transactions, strong consistency

**Examples**: PostgreSQL, MySQL

**Good for**:
- Order Service (needs ACID for order creation)
- Payment Service (financial transactions)
- User Service (structured user data)

\`\`\`sql
-- Order Service (PostgreSQL)
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    total DECIMAL(10,2),
    status VARCHAR(50),
    created_at TIMESTAMP
);

CREATE TABLE order_items (
    id UUID PRIMARY KEY,
    order_id UUID REFERENCES orders(id),
    product_id UUID,
    quantity INT,
    price DECIMAL(10,2)
);
\`\`\`

### Document (NoSQL)

**Use for**: Flexible schema, nested data, high write throughput

**Examples**: MongoDB, DynamoDB

**Good for**:
- Product Catalog Service (flexible product attributes)
- Content Service (blog posts, comments)
- Session Service (user sessions)

\`\`\`javascript
// Product Service (MongoDB)
{
    _id: "prod_123",
    name: "iPhone 15 Pro",
    price: 999,
    category: "Electronics",
    attributes: {
        // Flexible schema
        color: "Black",
        storage: "256GB",
        processor: "A17 Pro"
    },
    images: ["url1", "url2"],
    reviews: [
        {user: "user_1", rating: 5, comment: "Great phone!"}
    ]
}
\`\`\`

### Key-Value

**Use for**: Simple lookups, caching, sessions

**Examples**: Redis, DynamoDB

**Good for**:
- Cart Service (shopping carts)
- Session Service
- Cache Service

\`\`\`javascript
// Cart Service (Redis)
SET cart:user_123 '{"items": [{"productId": "prod_1", "quantity": 2}]}'
EXPIRE cart:user_123 3600  // Auto-delete after 1 hour
\`\`\`

### Graph

**Use for**: Relationships, recommendations, social graphs

**Examples**: Neo4j, Amazon Neptune

**Good for**:
- Social Network Service
- Recommendation Service

\`\`\`cypher
// Recommendation Service (Neo4j)
CREATE (u:User {id: 'user_123', name: 'John'})
CREATE (p:Product {id: 'prod_456', name: 'iPhone'})
CREATE (u)-[:PURCHASED]->(p)

// Find recommendations
MATCH (u:User {id: 'user_123'})-[:PURCHASED]->(p:Product)
      <-[:PURCHASED]-(other:User)-[:PURCHASED]->(recommendation:Product)
WHERE NOT (u)-[:PURCHASED]->(recommendation)
RETURN recommendation.name
\`\`\`

### Time-Series

**Use for**: Metrics, logs, IoT data

**Examples**: InfluxDB, TimescaleDB

**Good for**:
- Metrics Service
- Logging Service
- IoT Service

---

## Data Ownership and Boundaries

**Rule**: Each entity is owned by exactly ONE service.

**Example**:
- User Service owns users table
- Order Service owns orders table
- Product Service owns products table

**What about foreign keys?**

**In Monolith**:
\`\`\`sql
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),  -- Foreign key
    product_id UUID REFERENCES products(id)  -- Foreign key
);
\`\`\`

**In Microservices**: Can't have foreign keys across services!

**Solution**: Store IDs without foreign key constraints

\`\`\`sql
-- Order Service database
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,  -- No REFERENCES
    product_id UUID NOT NULL,  -- No REFERENCES
    status VARCHAR(50)
);
\`\`\`

**Referential integrity**: Handled by application logic, not database

\`\`\`javascript
// Before creating order, verify user and product exist
const user = await userService.getUser(userId);
if (!user) throw new Error('User not found');

const product = await productService.getProduct(productId);
if (!product) throw new Error('Product not found');

// Now create order
await orderService.createOrder({userId, productId, ...});
\`\`\`

---

## Handling Data Changes Across Services

### Problem: Product price changes

**Scenario**:
1. Order Service stores product price when order created: $99
2. Product Service updates price to $89
3. Old orders still show $99

**Is this a problem?** Usually NO! Historical orders should reflect price at purchase time.

**But what if we need updated data?**

### Solution: Event-Driven Updates

**Pattern**: Services subscribe to relevant events

\`\`\`javascript
// Product Service
async function updateProductPrice(productId, newPrice) {
    await db.products.update(productId, {price: newPrice});
    
    await eventBus.publish('ProductPriceChanged', {
        productId,
        oldPrice: 99,
        newPrice: 89,
        changedAt: new Date()
    });
}

// Order Service (if it needs current prices)
eventBus.on('ProductPriceChanged', async (event) => {
    // Update cached product info for pending orders
    await db.orders.updateMany(
        {productId: event.productId, status: 'PENDING'},
        {currentProductPrice: event.newPrice}
    );
});

// Inventory Service (adjust reorder calculations)
eventBus.on('ProductPriceChanged', async (event) => {
    await recalculateReorderPoint(event.productId);
});
\`\`\`

---

## Shared Data Anti-Patterns

### ❌ Shared Database

**Problem**: Multiple services access the same database

\`\`\`
Order Service  ─┐
User Service   ─┼─→ Shared Database
Product Service─┘
\`\`\`

**Why it's bad**:
- Tight coupling (schema change breaks all services)
- Can't choose different database types
- Scaling nightmare
- Single point of failure

### ❌ Shared Tables

**Problem**: Multiple services read/write same table

\`\`\`
Order Service writes orders table
Reporting Service reads orders table
\`\`\`

**Why it's bad**: Breaks encapsulation, creates hidden dependencies

**Better**: Reporting Service subscribes to OrderCreated events

---

## Transaction Boundaries

**Rule**: Transactions should not span services.

**Example**:

❌ **Bad** (distributed transaction):
\`\`\`
BEGIN TRANSACTION;
  INSERT INTO orders (...);  -- Order Service DB
  UPDATE inventory (...);     -- Inventory Service DB
  INSERT INTO payments (...); -- Payment Service DB
COMMIT;
\`\`\`

✅ **Good** (Saga pattern with local transactions):
\`\`\`
// Order Service
BEGIN TRANSACTION;
  INSERT INTO orders (...);
COMMIT;
Publish OrderCreated event

// Inventory Service
Listen to OrderCreated
BEGIN TRANSACTION;
  UPDATE inventory (...);
COMMIT;
Publish InventoryReserved event

// Payment Service
Listen to InventoryReserved
BEGIN TRANSACTION;
  INSERT INTO payments (...);
COMMIT;
Publish PaymentCompleted event
\`\`\`

---

## Data Migration Strategies

Moving from monolith to microservices?

### Strangler Fig Pattern

**Gradually** extract services while keeping shared database initially.

**Phase 1**: Extract service but keep shared DB
\`\`\`
Monolith     ─┐
Order Service─┼─→ Shared Database
\`\`\`

**Phase 2**: Replicate data to service's own DB
\`\`\`
Monolith     ─┬─→ Shared Database
Order Service─┼─→ Shared Database
              └─→ Orders DB (read-only copy)
\`\`\`

**Phase 3**: Service writes to its own DB, syncs to shared DB
\`\`\`
Monolith     ───→ Shared Database ←─┐
Order Service──→ Orders DB ─────────┘ (sync)
\`\`\`

**Phase 4**: Cut over completely
\`\`\`
Monolith     ───→ Shared Database
Order Service──→ Orders DB (independent)
\`\`\`

---

## Interview Tips

**Red Flags**:
❌ Suggesting shared database for microservices
❌ Not mentioning eventual consistency
❌ Using JOINs across services

**Good Responses**:
✅ Explain database per service pattern
✅ Discuss trade-offs (autonomy vs consistency)
✅ Mention solutions (API composition, CQRS, events)
✅ Talk about data ownership

**Sample Answer**:
*"I'd implement the database per service pattern, where each microservice owns its data and database. This provides loose coupling and allows us to choose the best database for each use case - PostgreSQL for Order Service (ACID), MongoDB for Product Catalog (flexible schema), Redis for Cart Service (fast lookups). For cross-service queries like 'orders with product details', I'd use API composition for simple cases or CQRS with read models for complex queries. Data consistency is eventual, not immediate, which is acceptable for this use case. Services communicate data changes via events."*

---

## Key Takeaways

1. **Database per service**: Each service owns its data and database
2. **No JOINs**: Can't query across services directly
3. **Solutions**: API composition, data duplication, CQRS
4. **Choose right DB**: SQL for structured, NoSQL for flexible, graph for relationships
5. **Event-driven**: Services communicate data changes via events
6. **Eventual consistency**: Accept temporary inconsistency
7. **No shared databases**: Anti-pattern that creates tight coupling
8. **Local transactions**: Transactions don't span services (use Saga)`,
};

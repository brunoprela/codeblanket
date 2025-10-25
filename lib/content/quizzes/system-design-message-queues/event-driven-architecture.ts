/**
 * Discussion Questions for Event-Driven Architecture
 */

import { QuizQuestion } from '../../../types';

export const eventdrivenarchitectureQuiz: QuizQuestion[] = [
  {
    id: 'event-driven-architecture-dq-1',
    question:
      'Compare event notification, event-carried state transfer, and event sourcing patterns. Design an e-commerce system where customers place orders, payment is processed, inventory is updated, and emails are sent. For each step, which event pattern would you use and why?',
    hint: 'Consider coupling, data availability, and audit requirements for each pattern.',
    sampleAnswer: `Event-driven patterns offer different trade-offs between coupling, performance, and data consistency.

**Event Patterns:**

**1. Event Notification:**
- Minimal data (just event happened)
- Receiver queries for details
- Loose coupling, but chatty

**2. Event-Carried State Transfer:**
- Full data in event
- Receiver has all context
- Tight coupling to schema, but efficient

**3. Event Sourcing:**
- Store events as source of truth
- Reconstruct state from events
- Complete audit trail, replay capability

**E-Commerce System Design:**

\`\`\`
Order Flow:
1. Customer places order
2. Payment processed
3. Inventory updated
4. Email sent
5. Analytics tracked

Events:
┌─────────────┬──────────────────────┬──────────────┐
│ Step        │ Pattern              │ Reason       │
├─────────────┼──────────────────────┼──────────────┤
│ Place Order │ Event Sourcing       │ Audit trail  │
│ Payment     │ Event Sourcing       │ Audit trail  │
│ Inventory   │ State Transfer       │ Performance  │
│ Email       │ State Transfer       │ All data     │
│ Analytics   │ State Transfer       │ Aggregation  │
└─────────────┴──────────────────────┴──────────────┘
\`\`\`

**Implementation:**

\`\`\`java
// 1. Place Order (Event Sourcing)
public class OrderAggregate {
    private List<Event> events = new ArrayList<>();
    private String orderId;
    private OrderStatus status;
    
    public void placeOrder(PlaceOrderCommand cmd) {
        // Validate
        if (cmd.getItems().isEmpty()) {
            throw new InvalidOrderException("No items");
        }
        
        // Create event
        OrderPlacedEvent event = new OrderPlacedEvent(
            orderId: generateId(),
            customerId: cmd.getCustomerId(),
            items: cmd.getItems(),
            totalAmount: calculateTotal (cmd.getItems()),
            timestamp: Instant.now()
        );
        
        // Apply event (change state)
        apply (event);
        
        // Store event
        eventStore.save (event);
        
        // Publish event
        eventBus.publish (event);
    }
    
    private void apply(OrderPlacedEvent event) {
        this.orderId = event.getOrderId();
        this.status = OrderStatus.PENDING;
        this.events.add (event);
    }
    
    // Reconstruct from events
    public static OrderAggregate load(String orderId) {
        List<Event> events = eventStore.getEvents (orderId);
        OrderAggregate order = new OrderAggregate();
        events.forEach (order::apply);
        return order;
    }
}

// Event Store
OrderPlacedEvent → orders_events table
PaymentProcessedEvent → payments_events table
OrderShippedEvent → orders_events table

// Benefit: Complete audit trail, replay capability

// 2. Payment (Event Sourcing)
public void processPayment(ProcessPaymentCommand cmd) {
    // Check previous events
    List<Event> events = eventStore.getEvents (cmd.getOrderId());
    
    // Validate (no duplicate payments)
    boolean alreadyPaid = events.stream()
        .anyMatch (e -> e instanceof PaymentProcessedEvent);
    
    if (alreadyPaid) {
        throw new DuplicatePaymentException();
    }
    
    // Process payment
    PaymentResult result = paymentGateway.charge (cmd);
    
    // Create event
    PaymentProcessedEvent event = new PaymentProcessedEvent(
        orderId: cmd.getOrderId(),
        amount: cmd.getAmount(),
        transactionId: result.getTransactionId(),
        timestamp: Instant.now()
    );
    
    // Store and publish
    eventStore.save (event);
    eventBus.publish (event);
}

// Benefit: Audit trail for financial transactions

// 3. Inventory Update (Event-Carried State Transfer)
// Event contains full order details
public class InventoryService {
    @Subscribe
    public void onOrderPlaced(OrderPlacedEvent event) {
        // Event has all data needed (no extra query)
        for (Item item : event.getItems()) {
            inventory.reserve (item.getProductId(), item.getQuantity());
        }
        
        // Publish inventory event (also state transfer)
        InventoryReservedEvent reserved = new InventoryReservedEvent(
            orderId: event.getOrderId(),
            items: event.getItems(),
            reservedAt: Instant.now()
        );
        
        eventBus.publish (reserved);
    }
}

// Benefit: No additional query needed, fast processing

// 4. Email (Event-Carried State Transfer)
public class EmailService {
    @Subscribe
    public void onOrderPlaced(OrderPlacedEvent event) {
        // Event has customer, items, amount
        sendEmail(
            to: getCustomerEmail (event.getCustomerId()),
            subject: "Order Confirmation",
            body: formatOrderEmail (event)
        );
    }
    
    @Subscribe
    public void onPaymentProcessed(PaymentProcessedEvent event) {
        // Event has transaction details
        sendEmail(
            to: getCustomerEmail (event.getCustomerId()),
            subject: "Payment Confirmed",
            body: formatPaymentEmail (event)
        );
    }
}

// Benefit: Self-contained events, no database queries

// 5. Analytics (Event-Carried State Transfer)
public class AnalyticsService {
    @Subscribe
    public void onOrderPlaced(OrderPlacedEvent event) {
        // Aggregate metrics
        metrics.increment("orders_placed");
        metrics.add("revenue", event.getTotalAmount());
        
        // Store in data warehouse
        dataWarehouse.insert(
            table: "orders",
            data: {
                order_id: event.getOrderId(),
                customer_id: event.getCustomerId(),
                total: event.getTotalAmount(),
                items_count: event.getItems().size(),
                placed_at: event.getTimestamp()
            }
        );
    }
}

// Benefit: Denormalized data, fast analytics
\`\`\`

**Why Each Pattern:**

**Event Sourcing (Orders, Payments):**
- **Audit Trail**: Complete history of all changes
- **Compliance**: Financial regulations require audit
- **Debugging**: Replay events to reproduce bugs
- **Temporal Queries**: "What was order status at time X?"
- **Rollback**: Undo operations by replaying without certain events

**Event-Carried State Transfer (Inventory, Email, Analytics):**
- **Performance**: No extra database queries
- **Availability**: Services don't depend on order service uptime
- **Scalability**: Read from event, not central database
- **Decoupling**: Services don't share database

**Trade-offs:**

\`\`\`
Event Notification:
Pros:
✅ Loose coupling (minimal schema dependency)
✅ Small message size

Cons:
❌ Chatty (receivers query for details)
❌ Dependency on sender availability
❌ Higher latency

Example:
{
  "event": "order_placed",
  "order_id": "order_123"
}

Receiver must query: GET /orders/order_123

Event-Carried State Transfer:
Pros:
✅ Self-contained (no queries needed)
✅ Fast (all data in event)
✅ Receiver independent of sender

Cons:
❌ Large messages (full data)
❌ Schema coupling (changes affect receivers)
❌ Data duplication

Example:
{
  "event": "order_placed",
  "order_id": "order_123",
  "customer": {...},  // Full customer data
  "items": [...],     // Full items
  "total": 99.99,
  "shipping_address": {...}
}

Receiver has everything immediately ✅

Event Sourcing:
Pros:
✅ Complete audit trail
✅ Replay capability
✅ Temporal queries
✅ Debugging powerful

Cons:
❌ Complexity (event versioning)
❌ Storage overhead (all events)
❌ Performance (reconstruct from events)
❌ Learning curve

Example:
Events:
1. OrderPlacedEvent
2. PaymentProcessedEvent
3. OrderShippedEvent

Reconstruct current state:
Order order = new Order();
events.forEach (order::apply);

order.status == SHIPPED ✅
\`\`\`

**Key Takeaways:**
- **Event Notification**: Loose coupling, but chatty
- **State Transfer**: Fast, self-contained, but large messages
- **Event Sourcing**: Audit trail, replay, but complex
- **Choose based on requirements**: Audit needs → Event Sourcing, Performance → State Transfer`,
    keyPoints: [
      'Event Notification: Minimal data, receiver queries (loose coupling, chatty)',
      'Event-Carried State Transfer: Full data in event (fast, no queries)',
      'Event Sourcing: Events as source of truth (audit trail, replay)',
      'Use Event Sourcing for orders/payments (audit, compliance)',
      'Use State Transfer for inventory/email (performance, decoupling)',
      'Trade-offs: Coupling vs performance vs complexity',
    ],
  },
  {
    id: 'event-driven-architecture-dq-2',
    question:
      'Design a CQRS (Command Query Responsibility Segregation) system for a social media application with posts, likes, and comments. How would you separate write and read models? Include event sourcing for writes, materialized views for reads, and eventual consistency handling.',
    hint: 'Consider write model (commands), read model (queries), event store, projection services, and consistency guarantees.',
    sampleAnswer: `CQRS separates write operations (commands) from read operations (queries) using different models optimized for each.

**CQRS Architecture:**

\`\`\`
Write Side (Commands):
  Client → Command → Aggregate → Event Store
  
Read Side (Queries):
  Event Store → Projection → Read Model → Client Query

Flow:
1. Command processed (create post)
2. Events stored (PostCreatedEvent)
3. Events published (Kafka)
4. Projections update read models
5. Queries read from optimized read models
\`\`\`

**Implementation:**

\`\`\`java
// WRITE SIDE

// Command (intent to change state)
public class CreatePostCommand {
    private String postId;
    private String userId;
    private String content;
    private List<String> imageUrls;
}

// Command Handler
public class PostCommandHandler {
    private final EventStore eventStore;
    private final EventBus eventBus;
    
    public void handle(CreatePostCommand cmd) {
        // Validate
        if (cmd.getContent().isEmpty()) {
            throw new InvalidCommandException("Content required");
        }
        
        // Create aggregate
        Post post = new Post (cmd.getPostId());
        post.create (cmd.getUserId(), cmd.getContent(), cmd.getImageUrls());
        
        // Get uncommitted events
        List<Event> events = post.getUncommittedEvents();
        
        // Store events
        eventStore.save (cmd.getPostId(), events);
        
        // Publish events
        events.forEach (eventBus::publish);
    }
    
    public void handle(LikePostCommand cmd) {
        // Load aggregate from events
        Post post = loadPost (cmd.getPostId());
        
        // Execute command
        post.like (cmd.getUserId());
        
        // Save new events
        List<Event> events = post.getUncommittedEvents();
        eventStore.save (cmd.getPostId(), events);
        events.forEach (eventBus::publish);
    }
    
    private Post loadPost(String postId) {
        List<Event> events = eventStore.getEvents (postId);
        Post post = new Post (postId);
        events.forEach (post::apply);
        return post;
    }
}

// Aggregate (write model)
public class Post {
    private String postId;
    private String userId;
    private String content;
    private Set<String> likes = new HashSet<>();
    private List<Event> uncommittedEvents = new ArrayList<>();
    
    public void create(String userId, String content, List<String> imageUrls) {
        PostCreatedEvent event = new PostCreatedEvent(
            postId, userId, content, imageUrls, Instant.now()
        );
        apply (event);
        uncommittedEvents.add (event);
    }
    
    public void like(String userId) {
        if (likes.contains (userId)) {
            throw new AlreadyLikedException();
        }
        
        PostLikedEvent event = new PostLikedEvent(
            postId, userId, Instant.now()
        );
        apply (event);
        uncommittedEvents.add (event);
    }
    
    private void apply(PostCreatedEvent event) {
        this.postId = event.getPostId();
        this.userId = event.getUserId();
        this.content = event.getContent();
    }
    
    private void apply(PostLikedEvent event) {
        this.likes.add (event.getUserId());
    }
}

// Event Store (append-only)
CREATE TABLE events (
    aggregate_id VARCHAR,
    version INT,
    event_type VARCHAR,
    event_data JSONB,
    timestamp TIMESTAMP,
    PRIMARY KEY (aggregate_id, version)
);

INSERT INTO events VALUES (
    'post_123',
    1,
    'PostCreatedEvent',
    '{"user_id": "user_456", "content": "Hello world", ...}',
    NOW()
);

// READ SIDE

// Read Models (denormalized, optimized for queries)
CREATE TABLE post_feed (
    post_id VARCHAR PRIMARY KEY,
    user_id VARCHAR,
    username VARCHAR,
    user_avatar_url VARCHAR,
    content TEXT,
    image_urls TEXT[],
    like_count INT,
    comment_count INT,
    created_at TIMESTAMP,
    INDEX (user_id, created_at DESC)
);

CREATE TABLE user_timeline (
    user_id VARCHAR,
    post_id VARCHAR,
    posted_at TIMESTAMP,
    PRIMARY KEY (user_id, posted_at DESC)
);

// Projection (event handlers that update read models)
public class PostFeedProjection {
    @Subscribe
    public void on(PostCreatedEvent event) {
        db.execute(
            "INSERT INTO post_feed (post_id, user_id, username, content, image_urls, like_count, comment_count, created_at) " +
            "VALUES (?, ?, ?, ?, ?, 0, 0, ?)",
            event.getPostId(),
            event.getUserId(),
            getUserName (event.getUserId()),  // Lookup username
            event.getContent(),
            event.getImageUrls(),
            event.getTimestamp()
        );
        
        // Also update timeline
        db.execute(
            "INSERT INTO user_timeline (user_id, post_id, posted_at) VALUES (?, ?, ?)",
            event.getUserId(),
            event.getPostId(),
            event.getTimestamp()
        );
    }
    
    @Subscribe
    public void on(PostLikedEvent event) {
        db.execute(
            "UPDATE post_feed SET like_count = like_count + 1 WHERE post_id = ?",
            event.getPostId()
        );
    }
    
    @Subscribe
    public void on(CommentAddedEvent event) {
        db.execute(
            "UPDATE post_feed SET comment_count = comment_count + 1 WHERE post_id = ?",
            event.getPostId()
        );
    }
}

// Query Service (reads from read models)
public class PostQueryService {
    public PostFeedDto getFeed(String userId, int page, int size) {
        // Following users' posts
        return db.query(
            "SELECT * FROM post_feed " +
            "WHERE user_id IN (SELECT following_id FROM followers WHERE follower_id = ?) " +
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            userId, size, page * size
        );
    }
    
    public PostDto getPost(String postId) {
        return db.queryOne(
            "SELECT * FROM post_feed WHERE post_id = ?",
            postId
        );
    }
    
    public List<PostDto> getUserPosts(String userId) {
        return db.query(
            "SELECT * FROM post_feed WHERE user_id = ? ORDER BY created_at DESC",
            userId
        );
    }
}
\`\`\`

**Eventual Consistency Handling:**

\`\`\`
Timeline:
T0: User creates post (command)
T1: PostCreatedEvent stored in event store
T2: Event published to Kafka
T3: Projection receives event (50ms delay)
T4: Read model updated

Problem: Query between T1-T4 doesn't see new post ❌

Solutions:

1. Return command result immediately (optimistic):
POST /posts
Response: { "post_id": "post_123", "status": "created" }

Client immediately shows post (optimistic UI)
Eventually consistent with read model ✅

2. Version-based consistency:
Response includes version:
{ "post_id": "post_123", "version": 5 }

Query with version:
GET /posts/post_123?min_version=5

Read model checks:
if (read_model_version < min_version) {
    wait_for_projection() or return_stale_data()
}

3. Read-your-writes consistency:
After command, redirect to write model temporarily:

POST /posts → PostCreatedEvent
Cache: user_456 created post_123 at T1

GET /posts/post_123 from user_456:
- Check cache: Created at T1
- Check read model: Updated at T0 (stale!)
- Read from event store (source of truth) ✅
- Return fresh data

After 1 second (projection complete):
- Read model updated
- Read from read model ✅

4. Client-side handling:
// Optimistic update
posts.add (newPost);
displayPost (newPost);

// Eventually consistent
setTimeout(() => {
    refreshPost (newPost.id);  // Fetch from read model
}, 1000);
\`\`\`

**Benefits of CQRS:**

\`\`\`
Write Side:
✅ Optimized for commands (validation, business logic)
✅ Event sourcing (audit trail)
✅ Strong consistency per aggregate

Read Side:
✅ Optimized for queries (denormalized, indexed)
✅ Multiple read models for different views
✅ Scalable independently (read replicas)

Example Read Models:
- post_feed: Chronological feed
- trending_posts: Sorted by engagement
- user_posts: Per-user timeline
- search_index: ElasticSearch for full-text

Same events → Different projections ✅

Performance:
Write: 100 posts/sec (single database)
Read: 10,000 queries/sec (multiple read replicas)

Scale read and write independently ✅
\`\`\`

**Challenges:**

\`\`\`
1. Eventual Consistency
   - Reads may be stale (50-200ms delay)
   - Solution: Optimistic UI, version checks

2. Complexity
   - Two models to maintain
   - Projections can fail (need replay)
   - Solution: Good tooling, monitoring

3. Event Versioning
   - Events evolve over time
   - Old events must be handled
   - Solution: Upcasting, schema versioning

4. Projection Failures
   - Projection crashes mid-update
   - Solution: Idempotent projections, checkpointing

5. Rebuilding Read Models
   - Need to replay all events (slow)
   - Solution: Snapshots, parallel replay
\`\`\`

**Key Takeaways:**
✅ Separate write (commands) and read (queries) models
✅ Write: Event sourcing for audit and consistency
✅ Read: Materialized views optimized for queries
✅ Eventual consistency (50-200ms delay)
✅ Handle consistency: Optimistic UI, versioning, caching`,
    keyPoints: [
      'CQRS: Separate write model (commands) from read model (queries)',
      'Write side: Event sourcing with aggregate + event store',
      'Read side: Projections build materialized views from events',
      'Eventual consistency (50-200ms delay) between write and read',
      'Handle staleness: Optimistic UI, versioning, read-your-writes',
      'Scale reads and writes independently (read replicas, write sharding)',
    ],
  },
  {
    id: 'event-driven-architecture-dq-3',
    question:
      'Design an event versioning and schema evolution strategy for a long-running event-sourced system. Events from 5 years ago must still be processable. Include upcasting, schema registry, backward/forward compatibility, and migration strategies for breaking changes.',
    hint: 'Consider event versioning, upcasting old events, schema registry for validation, and handling breaking changes.',
    sampleAnswer: `Event versioning is critical for event-sourced systems as events are stored forever and must remain processable.

**Schema Evolution Strategies:**

\`\`\`
V1: OrderPlacedEvent (5 years ago)
{
  "order_id": "123",
  "customer_id": "456",
  "total": 99.99
}

V2: OrderPlacedEvent (3 years ago - added items)
{
  "order_id": "123",
  "customer_id": "456",
  "items": [...],
  "total": 99.99
}

V3: OrderPlacedEvent (now - added currency)
{
  "order_id": "123",
  "customer_id": "456",
  "items": [...],
  "total": 99.99,
  "currency": "USD"
}

Challenge: Replay events from 5 years ago (V1)
Current code expects V3 format ❌
\`\`\`

**Solution 1: Upcasting**

\`\`\`java
// Upcast old events to current version
public class EventUpcaster {
    public OrderPlacedEvent upcast(RawEvent rawEvent) {
        int version = rawEvent.getVersion();
        JsonNode data = rawEvent.getData();
        
        // Upcast V1 → V2
        if (version == 1) {
            data = upcastV1ToV2(data);
            version = 2;
        }
        
        // Upcast V2 → V3
        if (version == 2) {
            data = upcastV2ToV3(data);
            version = 3;
        }
        
        // Deserialize to current version
        return objectMapper.readValue (data.toString(), OrderPlacedEvent.class);
    }
    
    private JsonNode upcastV1ToV2(JsonNode v1) {
        ObjectNode v2 = v1.deepCopy();
        
        // V1 didn't have items, infer from total
        ArrayNode items = objectMapper.createArrayNode();
        items.add (createDefaultItem (v1.get("total").asDouble()));
        v2.set("items", items);
        
        return v2;
    }
    
    private JsonNode upcastV2ToV3(JsonNode v2) {
        ObjectNode v3 = v2.deepCopy();
        
        // V2 didn't have currency, default to USD
        v3.put("currency", "USD");
        
        return v3;
    }
}

// Usage:
List<RawEvent> rawEvents = eventStore.getEvents (orderId);
List<OrderPlacedEvent> events = rawEvents.stream()
    .map (upcaster::upcast)
    .collect(Collectors.toList());
\`\`\`

**Solution 2: Schema Registry**

\`\`\`java
// Confluent Schema Registry
// Register schemas with versioning

// V1 Schema
{
  "type": "record",
  "name": "OrderPlacedEvent",
  "namespace": "com.example.events",
  "fields": [
    {"name": "order_id", "type": "string"},
    {"name": "customer_id", "type": "string"},
    {"name": "total", "type": "double"}
  ]
}

// V2 Schema (backward compatible - added optional field)
{
  "type": "record",
  "name": "OrderPlacedEvent",
  "namespace": "com.example.events",
  "fields": [
    {"name": "order_id", "type": "string"},
    {"name": "customer_id", "type": "string"},
    {"name": "total", "type": "double"},
    {"name": "items", "type": ["null", "array"], "default": null}  // Optional
  ]
}

// V3 Schema (backward compatible - added optional field)
{
  "type": "record",
  "name": "OrderPlacedEvent",
  "namespace": "com.example.events",
  "fields": [
    {"name": "order_id", "type": "string"},
    {"name": "customer_id", "type": "string"},
    {"name": "total", "type": "double"},
    {"name": "items", "type": ["null", "array"], "default": null},
    {"name": "currency", "type": "string", "default": "USD"}  // Optional with default
  ]
}

// Producer
SchemaRegistryClient schemaRegistry = new CachedSchemaRegistryClient("http://schema-registry:8081", 100);
KafkaAvroSerializer serializer = new KafkaAvroSerializer (schemaRegistry);

OrderPlacedEvent event = new OrderPlacedEvent("123", "456", items, 99.99, "USD");
byte[] serialized = serializer.serialize("events", event);

// Consumer
KafkaAvroDeserializer deserializer = new KafkaAvroDeserializer (schemaRegistry);
OrderPlacedEvent event = (OrderPlacedEvent) deserializer.deserialize("events", serialized);

// Schema validation automatic ✅
\`\`\`

**Backward vs Forward Compatibility:**

\`\`\`
Backward Compatible:
New code can read old data

Example: Add optional field with default
Old event: {"order_id": "123", "total": 99.99}
New code expects: {"order_id": "123", "total": 99.99, "currency": "USD"}
New code sets default: currency = "USD" ✅

Rules for backward compatibility:
✅ Add optional fields (with defaults)
✅ Remove required fields (make optional first)
❌ Rename fields (breaks old data)
❌ Change field types (breaks old data)

Forward Compatible:
Old code can read new data

Example: Add field that old code ignores
New event: {"order_id": "123", "total": 99.99, "currency": "USD"}
Old code expects: {"order_id": "123", "total": 99.99}
Old code ignores currency ✅

Rules for forward compatibility:
✅ Add fields (old code ignores)
❌ Remove fields (old code expects them)

Full Compatibility:
Both backward and forward compatible

Rules:
✅ Only add optional fields with defaults
❌ Never remove fields
❌ Never rename fields
❌ Never change types
\`\`\`

**Handling Breaking Changes:**

\`\`\`java
// Scenario: Change field type (breaking!)
// Old: "total": 99.99 (double)
// New: "total": { "amount": 99.99, "currency": "USD" } (object)

// Solution 1: New event type
OrderPlacedEvent → OrderPlacedEvent_V2

// Keep both types
public interface OrderPlacedEvent {
    String getOrderId();
    Money getTotal();  // Abstraction
}

public class OrderPlacedEvent_V1 implements OrderPlacedEvent {
    private double total;
    
    public Money getTotal() {
        return new Money (total, "USD");  // Default currency
    }
}

public class OrderPlacedEvent_V2 implements OrderPlacedEvent {
    private Money total;  // Object
    
    public Money getTotal() {
        return total;
    }
}

// Deserialize based on version
public OrderPlacedEvent deserialize (byte[] data, int version) {
    if (version == 1) {
        return objectMapper.readValue (data, OrderPlacedEvent_V1.class);
    } else {
        return objectMapper.readValue (data, OrderPlacedEvent_V2.class);
    }
}

// Solution 2: Weak schema (JSON with manual handling)
public class OrderPlacedEvent {
    @JsonProperty("total")
    private JsonNode totalNode;  // Can be double or object
    
    public Money getTotal() {
        if (totalNode.isNumber()) {
            // V1: double
            return new Money (totalNode.asDouble(), "USD");
        } else {
            // V2: object
            return objectMapper.convertValue (totalNode, Money.class);
        }
    }
}

// Solution 3: Parallel events (migration period)
// Publish both old and new events during migration
public void placeOrder(Order order) {
    // New event
    OrderPlacedEvent_V2 newEvent = new OrderPlacedEvent_V2(...);
    eventBus.publish("orders-v2", newEvent);
    
    // Old event (for consumers not yet migrated)
    OrderPlacedEvent_V1 oldEvent = new OrderPlacedEvent_V1(...);
    eventBus.publish("orders-v1", oldEvent);
}

// After all consumers migrated to V2, stop publishing V1
\`\`\`

**Event Store Schema:**

\`\`\`sql
CREATE TABLE events (
    event_id UUID PRIMARY KEY,
    aggregate_id VARCHAR,
    aggregate_type VARCHAR,
    event_type VARCHAR,
    event_version INT,  -- Schema version
    event_data JSONB,   -- Flexible storage
    metadata JSONB,     -- Additional context
    timestamp TIMESTAMP
);

-- Store V1 event
INSERT INTO events VALUES (
    '...',
    'order_123',
    'Order',
    'OrderPlacedEvent',
    1,  -- Version 1
    '{"order_id": "123", "total": 99.99}',
    '{"user_id": "456", "ip": "1.2.3.4"}',
    NOW()
);

-- Store V3 event (5 years later)
INSERT INTO events VALUES (
    '...',
    'order_456',
    'Order',
    'OrderPlacedEvent',
    3,  -- Version 3
    '{"order_id": "456", "items": [...], "total": 199.99, "currency": "EUR"}',
    '{"user_id": "789", "ip": "5.6.7.8"}',
    NOW()
);

-- Replay events
SELECT * FROM events WHERE aggregate_id = 'order_123' ORDER BY timestamp;

-- Upcast V1 → V3 on read ✅
\`\`\`

**Migration Strategies:**

\`\`\`
Strategy 1: Lazy Migration (Upcast on Read)
- Old events stay in old format
- Upcast when loading aggregate
- Pros: No migration needed, fast
- Cons: Upcasting overhead on every read

Strategy 2: Eager Migration (Transform in Place)
- Batch job transforms all old events
- Update event_version and event_data
- Pros: No upcasting overhead
- Cons: Risky (data loss if bug), slow

Strategy 3: Snapshot Strategy
- Create snapshot of current state
- Discard old events (after backup)
- Pros: Fast replay (no old events)
- Cons: Lose detailed history

Strategy 4: Parallel Streams
- Publish to new topic with new schema
- Keep old topic for compatibility
- Migrate consumers gradually
- Pros: Safe, gradual migration
- Cons: Dual write complexity

Recommended: Strategy 1 (Lazy Migration) ✅
- Safe (no data transformation)
- Flexible (upcasting logic in code)
- Reversible (rollback code, not data)
\`\`\`

**Key Takeaways:**
✅ Upcast old events to current version on read
✅ Schema registry enforces compatibility rules
✅ Backward compatibility: New code reads old events
✅ Add optional fields with defaults (safe)
✅ Breaking changes: New event types or parallel streams
✅ Store event_version for intelligent upcasting`,
    keyPoints: [
      'Upcast old events to current version on read (lazy migration)',
      'Schema registry validates compatibility (Avro, Protobuf)',
      'Backward compatibility: Add optional fields with defaults',
      'Forward compatibility: Old code ignores new fields',
      'Breaking changes: New event types or parallel streams',
      'Store event_version in event store for intelligent upcasting',
    ],
  },
];

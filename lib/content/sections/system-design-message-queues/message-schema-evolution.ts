/**
 * Message Schema Evolution Section
 */

export const messageschemaevolutionSection = {
  id: 'message-schema-evolution',
  title: 'Message Schema Evolution',
  content: `Message schema evolution is the practice of changing message schemas over time while maintaining compatibility with existing producers and consumers. In distributed systems with multiple services deployed independently, managing schema changes is critical for system stability.

## Why Schema Evolution Matters

### **The Problem:**

\`\`\`
Scenario: Microservices architecture, 50 services

Service A (Producer) sends messages:
V1: {orderId: "123", amount: 99.99}

Service B, C, D... (Consumers) expect V1 format

You want to add a field:
V2: {orderId: "123", amount: 99.99, discount: 10.00}

Challenge:
- Can't upgrade all 50 services at once (different teams, deployment schedules)
- Need gradual rollout
- Must maintain compatibility during transition

What happens if:
- Producer upgraded to V2, consumers still V1? ❌ May break
- Consumers upgraded to V2, producer still V1? ❌ May break

Solution: Schema evolution with compatibility rules
\`\`\`

---

## Compatibility Types

### **1. Backward Compatibility**

**Definition:** New schema can read data written by old schema

\`\`\`
Old Schema (Writer):
message UserV1 {
  string userId;
  string email;
}

New Schema (Reader):
message UserV2 {
  string userId;
  string email;
  string phone = "";  // New field with default
}

Old data: {userId: "123", email: "john@example.com"}
Read by new schema: {userId: "123", email: "john@example.com", phone: ""}
                                                                    ↑ default
✅ Works!

Allowed changes (backward compatible):
✅ Add optional fields (with defaults)
✅ Delete required fields → optional

Forbidden changes (breaks backward compatibility):
❌ Add required fields (old data doesn't have them)
❌ Delete optional fields
❌ Change field types

Use case: Upgrade consumers before producers
\`\`\`

### **2. Forward Compatibility**

**Definition:** Old schema can read data written by new schema

\`\`\`
New Schema (Writer):
message UserV2 {
  string userId;
  string email;
  string phone;
}

Old Schema (Reader):
message UserV1 {
  string userId;
  string email;
}

New data: {userId: "123", email: "john@example.com", phone: "555-1234"}
Read by old schema: {userId: "123", email: "john@example.com"}
                                                     ↑ phone field ignored
✅ Works!

Allowed changes (forward compatible):
✅ Add optional fields (old readers ignore them)
✅ Delete optional fields

Forbidden changes (breaks forward compatibility):
❌ Add required fields to old schema
❌ Delete fields that old schema requires

Use case: Upgrade producers before consumers
\`\`\`

### **3. Full Compatibility (Backward + Forward)**

New schema can read old data AND old schema can read new data:

\`\`\`
Allowed changes:
✅ Add optional fields (with defaults)
✅ Delete optional fields

Forbidden changes:
❌ Add required fields
❌ Delete required fields
❌ Change field types
❌ Rename fields

Use case: Safe to upgrade producers/consumers in any order
Best practice: Aim for full compatibility when possible
\`\`\`

---

## Serialization Formats

### **1. JSON (Schema-less)**

\`\`\`json
// No schema enforced at serialization level
{
  "userId": "123",
  "email": "john@example.com",
  "phone": "555-1234"
}

Pros:
✅ Human-readable
✅ Flexible (add fields anytime)
✅ Language-agnostic
✅ No compilation step

Cons:
❌ No built-in schema validation
❌ Larger size (field names repeated)
❌ Slower serialization/deserialization
❌ No type safety

Compatibility:
- Add fields: Old consumers ignore ✅
- Remove fields: New consumers must handle missing ✅
- Rename fields: Breaks compatibility ❌

Use case: APIs, simple messages, small scale
\`\`\`

### **2. Apache Avro (Schema-full)**

\`\`\`
Schema (IDL):
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "userId", "type": "string"},
    {"name": "email", "type": "string"},
    {"name": "phone", "type": ["null", "string"], "default": null}
                                                      ↑ optional
  ]
}

Binary format:
- Schema not included in message (stored separately)
- Compact (field names not repeated)
- Fast serialization

Pros:
✅ Compact binary format
✅ Schema evolution support (built-in)
✅ Schema stored separately (smaller messages)
✅ Dynamic typing (schema included at runtime)

Cons:
❌ Not human-readable
❌ Requires schema registry
❌ More complex setup

Compatibility rules:
- Add field with default: Backward + forward compatible ✅
- Delete field: Backward compatible (forward if had default) ✅
- Rename field with aliases: Backward + forward compatible ✅

Use case: Kafka, Hadoop, high-throughput systems
\`\`\`

### **3. Protocol Buffers (Protobuf)**

\`\`\`protobuf
// user.proto
syntax = "proto3";

message User {
  string user_id = 1;
  string email = 2;
  string phone = 3;  // Field number 3
}

Binary format:
- Field numbers (not names) in binary
- Compact, fast
- Code generation (strongly typed)

Pros:
✅ Compact binary format
✅ Fast serialization (C++ implementation)
✅ Code generation (type safety)
✅ Good schema evolution support

Cons:
❌ Not human-readable
❌ Requires code generation
❌ Less dynamic than Avro

Compatibility rules:
- Add optional fields: Backward + forward compatible ✅
- Delete fields: Backward + forward compatible ✅
- Change field number: Breaks compatibility ❌
- Rename fields: Compatible (uses field numbers) ✅

Use case: gRPC, microservices, performance-critical
\`\`\`

### **4. Apache Thrift**

Similar to Protobuf, used by Facebook:

\`\`\`thrift
struct User {
  1: string userId,
  2: string email,
  3: optional string phone
}

Pros/Cons: Similar to Protobuf
Use case: Legacy Facebook services, cross-language RPC
\`\`\`

---

## Schema Registry

### **What is Schema Registry?**

**Schema Registry** = Centralized repository for message schemas + Version management + Compatibility checking

\`\`\`
Architecture:

Producer → Schema Registry (register/validate) → Kafka
                ↓ Schema ID
           Publish message (includes schema ID)
                ↓
Consumer ← Schema Registry (fetch schema) ← Kafka

Schema Registry:
- Stores schemas with versions
- Enforces compatibility rules
- Returns schema ID for efficient storage
- Validates schema on publish
\`\`\`

### **Confluent Schema Registry (for Kafka):**

**Register Schema:**

\`\`\`bash
# Register Avro schema
curl -X POST http://schema-registry:8081/subjects/users-value/versions \\
  -H "Content-Type: application/vnd.schemaregistry.v1+json" \\
  -d '{
    "schema": "{\\"type\\":\\"record\\",\\"name\\":\\"User\\",\\"fields\\":[{\\"name\\":\\"userId\\",\\"type\\":\\"string\\"},{\\"name\\":\\"email\\",\\"type\\":\\"string\\"}]}"
  }'

Response:
{"id": 1}  # Schema ID

# This schema ID embedded in Kafka messages
\`\`\`

**Produce Message:**

\`\`\`python
from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer

# Schema
value_schema_str = """
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "userId", "type": "string"},
    {"name": "email", "type": "string"}
  ]
}
"""

value_schema = avro.loads(value_schema_str)

# Producer configuration
avro_producer = AvroProducer({
    'bootstrap.servers': 'kafka:9092',
    'schema.registry.url': 'http://schema-registry:8081'
}, default_value_schema=value_schema)

# Produce
avro_producer.produce(
    topic='users',
    value={"userId": "123", "email": "john@example.com"}
)

# Schema Registry automatically:
# 1. Registers schema (if not exists)
# 2. Gets schema ID
# 3. Embeds schema ID in message (magic byte + ID + data)
\`\`\`

**Message Format:**

\`\`\`
Kafka Message:
┌──────────┬────────────┬─────────────────────────┐
│ Magic    │ Schema ID  │ Avro Binary Data        │
│ Byte (0) │ (4 bytes)  │ (variable)              │
└──────────┴────────────┴─────────────────────────┘

Magic byte: Identifies schema registry format
Schema ID: References schema in registry (e.g., ID=1)
Data: Avro-encoded message (compact, no schema included)

Benefits:
✅ Small message size (schema not repeated)
✅ Schema evolution (version tracked)
✅ Compatibility enforcement
\`\`\`

**Consume Message:**

\`\`\`python
from confluent_kafka.avro import AvroConsumer

avro_consumer = AvroConsumer({
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'my-group',
    'schema.registry.url': 'http://schema-registry:8081'
})

avro_consumer.subscribe(['users'])

# Consume
msg = avro_consumer.poll(1.0)
if msg:
    user = msg.value()  # Automatically deserialized
    print(f"User: {user['userId']}, Email: {user['email']}")

# Schema Registry automatically:
# 1. Extracts schema ID from message
# 2. Fetches schema from registry (cached)
# 3. Deserializes using schema
\`\`\`

---

## Schema Evolution Strategies

### **1. Add Optional Fields (Recommended)**

\`\`\`avro
// V1
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "userId", "type": "string"},
    {"name": "email", "type": "string"}
  ]
}

// V2: Add optional phone field
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "userId", "type": "string"},
    {"name": "email", "type": "string"},
    {"name": "phone", "type": ["null", "string"], "default": null}
                                      ↑ union type (null or string)
                                                      ↑ default value
  ]
}

Compatibility: Backward + forward ✅

Old producer (V1) → New consumer (V2):
- Message: {userId, email}
- Consumer reads: {userId, email, phone: null} ✅

New producer (V2) → Old consumer (V1):
- Message: {userId, email, phone}
- Consumer reads: {userId, email} (ignores phone) ✅
\`\`\`

### **2. Remove Fields**

\`\`\`avro
// V1
{
  "fields": [
    {"name": "userId", "type": "string"},
    {"name": "email", "type": "string"},
    {"name": "phone", "type": ["null", "string"], "default": null}
  ]
}

// V2: Remove phone field
{
  "fields": [
    {"name": "userId", "type": "string"},
    {"name": "email", "type": "string"}
    // phone removed
  ]
}

Compatibility: Backward (if phone was optional) ✅

Old producer (V1) → New consumer (V2):
- Message: {userId, email, phone}
- Consumer reads: {userId, email} (ignores phone) ✅

Forward compatibility broken:
New producer (V2) → Old consumer (V1):
- Message: {userId, email}
- Consumer expects phone field
- Reads default (null) if field was optional ✅
\`\`\`

### **3. Rename Fields (with Aliases)**

\`\`\`avro
// V1
{
  "fields": [
    {"name": "userId", "type": "string"},
    {"name": "userName", "type": "string"}
  ]
}

// V2: Rename userName → fullName
{
  "fields": [
    {"name": "userId", "type": "string"},
    {"name": "fullName", "type": "string", "aliases": ["userName"]}
                                              ↑ old name as alias
  ]
}

Compatibility: Backward + forward ✅

Old producer (V1) → New consumer (V2):
- Message uses "userName" field
- Consumer recognizes "userName" as alias for "fullName" ✅

Gradual migration:
1. Add new field with alias (V2)
2. Update all producers to use new field name
3. Update all consumers to use new field name
4. Remove alias (V3)
\`\`\`

### **4. Change Field Types (Carefully)**

\`\`\`avro
// Allowed type changes (Avro):

int → long ✅ (widening)
int → float ✅ (widening)
int → double ✅ (widening)
long → double ✅ (widening)

string → bytes ✅
bytes → string ✅

Forbidden:
long → int ❌ (narrowing, data loss)
float → int ❌ (data loss)
string → int ❌ (incompatible)

Example:
// V1: amount as integer (cents)
{"name": "amount", "type": "int"}

// V2: amount as double (dollars)
{"name": "amount", "type": "double"}

Compatible: int → double ✅

Old data: 9999 (99.99 dollars as cents)
Read by V2: 9999.0 (interpreted as 9999 dollars ❌ Wrong!)

Solution: Don't change semantics!
Instead: Add new field, deprecate old
{"name": "amountCents", "type": "int"}  // Deprecated
{"name": "amountDollars", "type": "double", "default": 0.0}  // New
\`\`\`

---

## Versioning Strategies

### **1. Implicit Versioning (Schema Registry)**

Version tracked by schema registry:

\`\`\`
Subject: "users-value"
Versions:
- V1 (ID: 1): {userId, email}
- V2 (ID: 2): {userId, email, phone}
- V3 (ID: 3): {userId, email, phone, address}

Message includes schema ID (not version number)
Consumer uses schema ID to deserialize

Pros:
✅ Automatic versioning
✅ Centralized management
✅ Compatibility enforcement

Cons:
❌ Requires schema registry
❌ Dependency on external service
\`\`\`

### **2. Explicit Versioning (in Message)**

Include version in message:

\`\`\`json
{
  "version": "2.0",
  "userId": "123",
  "email": "john@example.com",
  "phone": "555-1234"
}

Consumer logic:
if message.version == "1.0":
    process_v1(message)
elif message.version == "2.0":
    process_v2(message)
else:
    raise UnknownVersionError

Pros:
✅ No external dependencies
✅ Clear versioning
✅ Consumer control

Cons:
❌ Manual version management
❌ Larger messages (version field)
❌ Consumer must handle all versions
\`\`\`

### **3. Separate Topics per Version**

Different topic for each major version:

\`\`\`
Topics:
- users-v1 (schema V1)
- users-v2 (schema V2)

Producers write to users-v2
Consumers:
- Old consumers read from users-v1
- New consumers read from users-v2

Migration:
1. Deploy V2 consumers (read users-v2)
2. Dual-write producers (users-v1 AND users-v2)
3. Verify V2 consumers working
4. Stop writing to users-v1
5. Decommission users-v1

Pros:
✅ Clean separation
✅ Easy rollback
✅ No compatibility issues

Cons:
❌ Dual writes (complexity, cost)
❌ Topic proliferation
❌ Longer migration period
\`\`\`

---

## Best Practices

### **1. Use Schema Registry**

\`\`\`
For Kafka: Confluent Schema Registry
For AWS: AWS Glue Schema Registry
For Azure: Azure Schema Registry

Benefits:
✅ Centralized schema management
✅ Automatic compatibility checking
✅ Version tracking
✅ Prevents breaking changes
\`\`\`

### **2. Choose Compatible Serialization Format**

\`\`\`
Recommended: Avro or Protobuf

Why:
✅ Built-in schema evolution
✅ Compact binary format
✅ Strong typing
✅ Compatibility rules enforced

Avoid: Plain JSON (no schema enforcement)
\`\`\`

### **3. Always Add Fields as Optional**

\`\`\`avro
// Good:
{"name": "phone", "type": ["null", "string"], "default": null}

// Bad:
{"name": "phone", "type": "string"}  // Required!

Why:
- Optional with default: Backward + forward compatible
- Required: Breaks backward compatibility
\`\`\`

### **4. Never Remove Required Fields**

\`\`\`
If field must be removed:
1. Make field optional (V2)
2. Deploy V2 producers/consumers
3. Stop using field
4. Remove field later (V3)

Gradual deprecation avoids breakage
\`\`\`

### **5. Use Aliases for Renames**

\`\`\`avro
// Instead of renaming directly, use alias:
{"name": "fullName", "aliases": ["userName"]}

Allows gradual migration
\`\`\`

### **6. Document Schema Changes**

\`\`\`
Schema changelog:
- V1.0 (2023-01-01): Initial schema
- V1.1 (2023-02-01): Added optional phone field
- V1.2 (2023-03-01): Renamed userName → fullName (alias)
- V2.0 (2023-04-01): Removed deprecated field X (BREAKING)

Communicate changes to all teams
\`\`\`

### **7. Test Compatibility**

\`\`\`python
# Test old consumers with new data
def test_backward_compatibility():
    new_schema = load_schema("user_v2.avsc")
    old_consumer = Consumer(schema="user_v1.avsc")
    
    message = produce_with_schema(new_schema, {...})
    result = old_consumer.consume(message)
    
    assert result is not None  # Should deserialize ✅

# Test new consumers with old data
def test_forward_compatibility():
    old_schema = load_schema("user_v1.avsc")
    new_consumer = Consumer(schema="user_v2.avsc")
    
    message = produce_with_schema(old_schema, {...})
    result = new_consumer.consume(message)
    
    assert result is not None  # Should deserialize ✅
\`\`\`

---

## Schema Evolution in System Design Interviews

### **When to Discuss:**

✅ **Microservices architecture** (independent deployments)
✅ **Event-driven systems** (messages across services)
✅ **Long-lived data** (stored messages, replayed events)
✅ **Multiple teams** (coordinated schema changes)

### **Example Discussion:**

\`\`\`
Interviewer: "How would you handle schema changes in an event-driven architecture with 100 microservices?"

You:
"I'd implement schema evolution with Confluent Schema Registry:

Setup:
- Avro serialization for all messages
- Schema Registry for centralized schema management
- Enforce full compatibility mode (backward + forward)

Schema Evolution Process:
1. Propose schema change (add optional field)
2. Register new schema version in Schema Registry
3. Compatibility check (automatic, enforced)
4. If compatible: Approved
5. If incompatible: Rejected, revise

Deployment:
- Gradual rollout (service by service)
- Old services use V1 schema
- New services use V2 schema
- All services compatible during transition

Example Change:
V1: {orderId, amount}
V2: {orderId, amount, discount: null}  // Optional field

Old producer → New consumer:
- Message: {orderId, amount}
- Consumer reads: {orderId, amount, discount: null} ✅

New producer → Old consumer:
- Message: {orderId, amount, discount: 10}
- Consumer reads: {orderId, amount} (ignores discount) ✅

Monitoring:
- Track schema versions per service
- Alert on incompatible schema registrations
- Dashboard showing schema adoption rates

Why Schema Registry:
✅ Prevents breaking changes (enforced compatibility)
✅ Centralized version management
✅ Compact messages (schema ID vs full schema)
✅ Supports Avro, Protobuf, JSON Schema

Alternative (without Schema Registry):
- Separate topics per version (users-v1, users-v2)
- Dual writes during migration
- More operational overhead

Cost-Benefit:
- Schema Registry: Small overhead (REST API call on first use)
- Prevents outages from incompatible changes
- Worth the investment at scale
"
\`\`\`

---

## Key Takeaways

1. **Schema evolution = Changing schemas without breaking systems** → Critical for distributed systems
2. **Compatibility types: Backward, forward, full** → Aim for full compatibility
3. **Use schema registry** → Enforce compatibility, version tracking
4. **Avro/Protobuf recommended** → Built-in schema evolution support
5. **Add fields as optional with defaults** → Backward + forward compatible
6. **Never remove required fields** → Breaks backward compatibility
7. **Use aliases for renames** → Maintains compatibility during migration
8. **Test compatibility** → Automated tests for schema changes
9. **Document changes** → Communicate to all teams
10. **In interviews: Discuss compatibility, registry, migration** → Show production thinking

---

**Congratulations!** You've completed **Module 12: Message Queues & Event Streaming**. You now have a comprehensive understanding of:
- Message queue fundamentals (patterns, guarantees, ordering)
- Apache Kafka architecture (brokers, partitions, replication, consumers)
- Kafka Producers and Consumers (configuration, scaling, best practices)
- Kafka Streams (stateful processing, windowing, joins)
- RabbitMQ (exchanges, queues, routing patterns)
- AWS SQS/SNS (managed messaging, fanout patterns)
- Event-driven architecture (events, saga, CQRS, event sourcing)
- Stream processing (windowing, late data, exactly-once)
- Schema evolution (compatibility, versioning, schema registry)

You're now equipped to design scalable, reliable messaging systems and discuss them confidently in system design interviews!`,
};

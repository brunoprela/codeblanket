/**
 * Discussion Questions for Message Schema Evolution
 */

import { QuizQuestion } from '../../../types';

export const messageschemaevolutionQuiz: QuizQuestion[] = [
  {
    id: 'message-schema-evolution-dq-1',
    question:
      'Explain backward compatibility vs forward compatibility vs full compatibility for message schemas. Design a schema evolution strategy for a payment service where producers and consumers deploy independently. Include schema registry, version validation, and handling of breaking changes.',
    hint: 'Consider producer/consumer deployment order, schema validation rules, and strategies for incompatible changes.',
    sampleAnswer: `Schema compatibility ensures producers and consumers can evolve independently without breaking.

**Compatibility Types:**

**1. Backward Compatibility:**
New consumers can read old messages

**2. Forward Compatibility:**
Old consumers can read new messages

**3. Full Compatibility:**
Both backward and forward compatible

**Payment Service Schema Evolution:**

\`\`\`
V1 Schema (Initial):
{
  "type": "record",
  "name": "PaymentEvent",
  "fields": [
    {"name": "payment_id", "type": "string"},
    {"name": "amount", "type": "double"},
    {"name": "currency", "type": "string"}
  ]
}

V2 Schema (Add optional field - Backward Compatible):
{
  "type": "record",
  "name": "PaymentEvent",
  "fields": [
    {"name": "payment_id", "type": "string"},
    {"name": "amount", "type": "double"},
    {"name": "currency", "type": "string"},
    {"name": "customer_id", "type": ["null", "string"], "default": null}  // Optional
  ]
}

Deployment:
1. Deploy V2 consumers (can read V1 and V2) ✅
2. Deploy V2 producers (write V2 messages) ✅
3. V2 consumers handle missing customer_id ✅

V3 Schema (Add required field - Breaking! ❌):
{
  "type": "record",
  "name": "PaymentEvent",
  "fields": [
    {"name": "payment_id", "type": "string"},
    {"name": "amount", "type": "double"},
    {"name": "currency", "type": "string"},
    {"name": "customer_id", "type": "string"},  // Now required!
    {"name": "merchant_id", "type": "string"}   // New required field
  ]
}

Problem:
- V2 consumers can't read V3 messages (missing merchant_id) ❌
- V3 consumers can't read V2 messages (missing merchant_id) ❌

Solution: Make field optional with default
{"name": "merchant_id", "type": "string", "default": "unknown"}
\`\`\`

**Schema Registry:**

\`\`\`java
import io.confluent.kafka.serializers.KafkaAvroSerializer;
import io.confluent.kafka.serializers.KafkaAvroDeserializer;

// Producer
Properties props = new Properties();
props.put("schema.registry.url", "http://schema-registry:8081");
props.put("value.serializer", KafkaAvroSerializer.class);

KafkaProducer<String, PaymentEvent> producer = new KafkaProducer<>(props);

// Register V2 schema
SchemaRegistryClient schemaRegistry = new CachedSchemaRegistryClient("http://schema-registry:8081", 100);

Schema v2Schema = new Schema.Parser().parse (v2SchemaString);

// Validate compatibility before registering
boolean compatible = schemaRegistry.testCompatibility("payments-value", v2Schema);

if (!compatible) {
    throw new SchemaIncompatibleException("V2 schema incompatible with V1");
}

// Register schema
int schemaId = schemaRegistry.register("payments-value", v2Schema);

// Produce message
PaymentEvent payment = PaymentEvent.newBuilder()
    .setPaymentId("pay_123")
    .setAmount(99.99)
    .setCurrency("USD")
    .setCustomerId("cust_456")  // V2 field
    .build();

producer.send (new ProducerRecord<>("payments", payment));

// Message format in Kafka:
// [Magic Byte (0x00)] [Schema ID (4 bytes)] [Avro serialized data]

// Consumer
props.put("value.deserializer", KafkaAvroDeserializer.class);
props.put("specific.avro.reader", "true");

KafkaConsumer<String, PaymentEvent> consumer = new KafkaConsumer<>(props);

// Automatic schema evolution ✅
consumer.subscribe(Collections.singleton list("payments"));

while (true) {
    ConsumerRecords<String, PaymentEvent> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<String, PaymentEvent> record : records) {
        PaymentEvent payment = record.value();
        
        // V1 consumer reading V2 message:
        // customer_id field is null (default) ✅
        
        // V2 consumer reading V1 message:
        // customer_id field populated with default (null) ✅
    }
}
\`\`\`

**Compatibility Rules:**

\`\`\`
Backward Compatibility (new consumers read old messages):

Allowed:
✅ Add optional field with default
✅ Delete field
✅ Promote field type (int → long)

Forbidden:
❌ Add required field (old messages don't have it)
❌ Delete required field (new consumer expects it)
❌ Change field type (int → string)
❌ Rename field (breaks old messages)

Example (Backward Compatible):
V1: {"amount": 99.99}
V2: {"amount": 99.99, "currency": "USD"}  // Add optional with default ✅

V2 consumer reads V1 message:
- amount: 99.99 ✅
- currency: "USD" (default) ✅

Forward Compatibility (old consumers read new messages):

Allowed:
✅ Add field (old consumer ignores)
✅ Delete optional field
✅ Demote field type (long → int, if values fit)

Forbidden:
❌ Delete required field (old consumer expects it)
❌ Rename field
❌ Change field type

Example (Forward Compatible):
V1: {"amount": 99.99}
V2: {"amount": 99.99, "currency": "USD"}

V1 consumer reads V2 message:
- amount: 99.99 ✅
- currency: ignored ✅

Full Compatibility (both backward and forward):

Allowed:
✅ Add optional field with default (backward + forward)

Forbidden:
❌ Delete any field
❌ Change field type
❌ Rename field
❌ Add required field

Strictest but safest ✅
\`\`\`

**Handling Breaking Changes:**

\`\`\`
Scenario: Change amount from double to object

V1:
{"amount": 99.99}

V3 (Breaking):
{"amount": {"value": 99.99, "precision": 2}}

Solutions:

1. New Topic Strategy:
   - Create new topic: payments-v2
   - Migrate consumers to new topic
   - Deprecated old topic after migration
   - Pro: Clean separation ✅
   - Con: Dual write during migration ❌

// Dual write during migration
producer.send (new ProducerRecord<>("payments", v1Payment));  // Old
producer.send (new ProducerRecord<>("payments-v2", v3Payment));  // New

// After migration:
producer.send (new ProducerRecord<>("payments-v2", v3Payment));  // New only

2. Versioned Message Strategy:
   - Include version field in message
   - Consumers handle multiple versions
   - Pro: Single topic ✅
   - Con: Consumer complexity ❌

{
  "version": 3,
  "payment_id": "pay_123",
  "amount": {"value": 99.99, "precision": 2}
}

// Consumer
PaymentEvent payment = record.value();
if (payment.getVersion() == 1) {
    double amount = payment.getAmountV1();  // double
} else if (payment.getVersion() == 3) {
    Money amount = payment.getAmountV3();  // object
}

3. Parallel Schema Strategy (Recommended):
   - Register V3 schema as new subject
   - Producers write V3 to new subject
   - Consumers support both subjects
   - Gradual migration
   - Pro: Safe, gradual ✅
   - Con: Temporary dual schemas ⚠️

Schema Registry subjects:
- payments-value-v1 (old)
- payments-value-v3 (new)

Producer:
producer.send (new ProducerRecord<>("payments", v3Payment));
// Schema ID references v3 schema

Consumer (migration period):
consumer.subscribe(Arrays.asList("payments"));
PaymentEvent payment = record.value();

// Deserializer automatically uses correct schema based on Schema ID ✅
\`\`\`

**Version Validation Pipeline:**

\`\`\`
CI/CD Pipeline:

1. Schema Change Detected:
   - Developer modifies PaymentEvent.avsc
   - Git commit triggers pipeline

2. Compatibility Check:
   - curl -X POST http://schema-registry:8081/compatibility/subjects/payments-value/versions/latest \\
     -H "Content-Type: application/vnd.schemaregistry.v1+json" \\
     -d '{"schema": "..."}'
   
   - Response: {"is_compatible": true}
   
   - If false: Fail build ❌

3. Integration Tests:
   - Test V2 producer → V1 consumer
   - Test V1 producer → V2 consumer
   - Verify no deserialization errors

4. Deployment:
   - Deploy consumers first (backward compatible)
   - Deploy producers second
   - Monitor for errors

5. Rollback:
   - If errors detected
   - Roll back producers (not consumers)
   - Schema Registry keeps history

// Automated compatibility check
public class SchemaCompatibilityTest {
    @Test
    public void testBackwardCompatibility() {
        Schema v1 = loadSchema("v1/PaymentEvent.avsc");
        Schema v2 = loadSchema("v2/PaymentEvent.avsc");
        
        // V2 consumer reads V1 message
        GenericRecord v1Record = createV1Record();
        GenericRecord v2Record = convertUsingSchema (v1Record, v2);
        
        assertNotNull (v2Record.get("payment_id"));
        assertNotNull (v2Record.get("amount"));
        assertEquals("USD", v2Record.get("currency"));  // Default ✅
    }
    
    @Test
    public void testForwardCompatibility() {
        // V1 consumer reads V2 message
        GenericRecord v2Record = createV2Record();
        GenericRecord v1Record = convertUsingSchema (v2Record, v1);
        
        assertNotNull (v1Record.get("payment_id"));
        assertNotNull (v1Record.get("amount"));
        // customer_id ignored by V1 consumer ✅
    }
}
\`\`\`

**Best Practices:**

\`\`\`
1. Always use schema registry (Confluent, AWS Glue, Apicurio)
   - Centralized schema management
   - Automatic validation
   - Schema evolution tracking

2. Prefer full compatibility
   - Safest option
   - Allows independent deployments
   - Only add optional fields with defaults

3. Version schemas explicitly
   - Include version in schema name or namespace
   - Easier to track changes

4. Test compatibility in CI/CD
   - Automated compatibility checks
   - Integration tests with old/new versions
   - Fail build on incompatible changes

5. Deploy consumers before producers
   - Ensures consumers can handle new messages
   - Backward compatibility critical

6. Document breaking changes
   - Migration guide for consumers
   - Deprecation notices
   - Timeline for migration

7. Monitor deserialization errors
   - Alert on schema mismatches
   - Track version distribution
   - Identify lagging consumers

8. Use Avro/Protobuf over JSON
   - Enforced schema validation
   - Binary format (smaller, faster)
   - Better tooling support

Avro vs Protobuf vs JSON:

Avro:
✅ Schema evolution built-in
✅ Compact binary format
✅ Confluent Schema Registry support
❌ Harder to debug (binary)

Protobuf:
✅ Backward/forward compatible
✅ Compact binary format
✅ Strong typing
❌ Less integration with Kafka ecosystem

JSON Schema:
✅ Human-readable
✅ Easy debugging
❌ Larger message size
❌ No enforced validation (optional)
❌ Slower serialization

Recommendation: Avro for Kafka ✅
\`\`\`

**Key Takeaways:**
✅ Backward compatibility: New consumers read old messages
✅ Forward compatibility: Old consumers read new messages
✅ Full compatibility: Both directions (safest)
✅ Add optional fields with defaults (compatible change)
✅ Never remove required fields or change types (breaking)
✅ Schema Registry validates compatibility automatically
✅ Deploy consumers before producers (backward compatibility)`,
    keyPoints: [
      'Backward compatibility: New consumers read old messages (add optional fields)',
      'Forward compatibility: Old consumers read new messages (ignore new fields)',
      'Full compatibility: Both directions (only add optional fields with defaults)',
      'Schema Registry validates compatibility automatically (fail on incompatible changes)',
      'Breaking changes: New topic, versioned messages, or parallel schemas',
      'Deploy consumers first, then producers (ensures backward compatibility)',
    ],
  },
  {
    id: 'message-schema-evolution-dq-2',
    question:
      'Design a schema migration strategy for migrating 100 million historical messages from JSON to Avro. Include data validation, performance considerations, zero-downtime migration, and rollback plan. How would you handle messages that fail validation?',
    hint: 'Consider dual-write period, reprocessing historical data, validation pipelines, and gradual cutover.',
    sampleAnswer: `Migrating 100M messages from JSON to Avro requires careful planning for data integrity and zero downtime.

**Migration Strategy:**

\`\`\`
Phase 1: Preparation
Phase 2: Dual Write (JSON + Avro)
Phase 3: Consumer Migration
Phase 4: Producer Cutover
Phase 5: Historical Data Migration
Phase 6: Cleanup
\`\`\`

**Phase 1: Preparation**

\`\`\`java
// 1. Define Avro schema from JSON
JSON format:
{
  "payment_id": "pay_123",
  "amount": 99.99,
  "currency": "USD",
  "timestamp": "2023-06-01T10:00:00Z"
}

Avro schema:
{
  "type": "record",
  "name": "PaymentEvent",
  "namespace": "com.example.payments",
  "fields": [
    {"name": "payment_id", "type": "string"},
    {"name": "amount", "type": "double"},
    {"name": "currency", "type": "string", "default": "USD"},
    {"name": "timestamp", "type": "long", "logicalType": "timestamp-millis"},
    {"name": "schema_version", "type": "string", "default": "v2"}  // Track migration
  ]
}

// 2. Validate schema with sample data
public class SchemaValidator {
    public void validateSample() {
        List<String> jsonMessages = loadSample("sample_messages.json", 10000);
        
        int valid = 0;
        int invalid = 0;
        List<String> errors = new ArrayList<>();
        
        for (String json : jsonMessages) {
            try {
                GenericRecord avroRecord = jsonToAvro (json);
                valid++;
            } catch (Exception e) {
                invalid++;
                errors.add (json + ": " + e.getMessage());
            }
        }
        
        System.out.println("Valid: " + valid + ", Invalid: " + invalid);
        
        if (invalid > valid * 0.01) {  // >1% invalid
            throw new ValidationException("Too many invalid messages: " + errors);
        }
    }
}

// 3. Register schema
SchemaRegistryClient schemaRegistry = ...;
int schemaId = schemaRegistry.register("payments-value", avroSchema);
\`\`\`

**Phase 2: Dual Write (JSON + Avro)**

\`\`\`java
// Producer writes both JSON and Avro
public class DualWriteProducer {
    private final KafkaProducer<String, String> jsonProducer;  // JSON
    private final KafkaProducer<String, GenericRecord> avroProducer;  // Avro
    
    public void sendPayment(Payment payment) {
        String key = payment.getPaymentId();
        
        try {
            // Write JSON (original format)
            String jsonPayload = toJson (payment);
            jsonProducer.send (new ProducerRecord<>("payments-json", key, jsonPayload));
            
            // Write Avro (new format)
            GenericRecord avroPayload = toAvro (payment);
            avroProducer.send (new ProducerRecord<>("payments-avro", key, avroPayload));
            
            metrics.increment("dual_write_success");
            
        } catch (Exception e) {
            logger.error("Dual write failed", e);
            
            // Fallback: JSON only (don't break existing system)
            String jsonPayload = toJson (payment);
            jsonProducer.send (new ProducerRecord<>("payments-json", key, jsonPayload));
            
            metrics.increment("dual_write_fallback");
        }
    }
}

// Duration: 1-2 weeks (ensure no data loss)
// Monitor: Avro write success rate (should be >99.9%)
\`\`\`

**Phase 3: Consumer Migration**

\`\`\`java
// Migrate consumers gradually from JSON to Avro

// Old consumer (JSON)
consumer.subscribe(Collections.singletonList("payments-json"));

// New consumer (Avro)
consumer.subscribe(Collections.singletonList("payments-avro"));

// Migration:
// Week 1: Deploy 10% consumers to Avro (canary)
// Week 2: Deploy 50% consumers to Avro
// Week 3: Deploy 100% consumers to Avro

// Monitor:
// - Deserialization errors (should be 0)
// - Processing latency (should be same or better)
// - Consumer lag (should be low)
\`\`\`

**Phase 4: Producer Cutover**

\`\`\`java
// Stop dual write, write Avro only

public class AvroOnlyProducer {
    private final KafkaProducer<String, GenericRecord> avroProducer;
    
    public void sendPayment(Payment payment) {
        String key = payment.getPaymentId();
        GenericRecord avroPayload = toAvro (payment);
        
        avroProducer.send (new ProducerRecord<>("payments-avro", key, avroPayload));
    }
}

// Deploy producers with Avro-only code
// Monitor for 24 hours (ensure no JSON consumers left)
\`\`\`

**Phase 5: Historical Data Migration**

\`\`\`java
// Reprocess 100M historical JSON messages to Avro

public class HistoricalMigrationJob {
    private final KafkaConsumer<String, String> jsonConsumer;
    private final KafkaProducer<String, GenericRecord> avroProducer;
    private final SchemaRegistryClient schemaRegistry;
    
    public void migrate() {
        // Read from payments-json topic (historical data)
        jsonConsumer.subscribe(Collections.singletonList("payments-json"));
        
        long processed = 0;
        long failed = 0;
        List<FailedMessage> failures = new ArrayList<>();
        
        while (true) {
            ConsumerRecords<String, String> records = jsonConsumer.poll(Duration.ofMillis(100));
            
            if (records.isEmpty()) {
                break;  // Reached end
            }
            
            List<ProducerRecord<String, GenericRecord>> avroBatch = new ArrayList<>();
            
            for (ConsumerRecord<String, String> record : records) {
                try {
                    // Convert JSON → Avro
                    GenericRecord avroRecord = jsonToAvro (record.value());
                    
                    // Add to batch
                    avroBatch.add (new ProducerRecord<>(
                        "payments-avro-historical",  // Separate topic for historical
                        record.key(),
                        avroRecord
                    ));
                    
                    processed++;
                    
                } catch (Exception e) {
                    logger.error("Conversion failed for message: " + record.key(), e);
                    failed++;
                    
                    failures.add (new FailedMessage(
                        record.key(),
                        record.value(),
                        e.getMessage()
                    ));
                    
                    // Write to DLQ
                    dlqProducer.send (new ProducerRecord<>(
                        "payments-json-migration-dlq",
                        record.key(),
                        record.value()
                    ));
                }
            }
            
            // Batch write to Avro topic
            for (ProducerRecord<String, GenericRecord> avroRecord : avroBatch) {
                avroProducer.send (avroRecord);
            }
            
            // Commit offset
            jsonConsumer.commitSync();
            
            // Progress logging
            if (processed % 100000 == 0) {
                logger.info("Migrated: {}, Failed: {}, Success rate: {}%",
                           processed, failed, (processed * 100.0 / (processed + failed)));
            }
        }
        
        logger.info("Migration complete. Total: {}, Failed: {}", processed, failed);
        
        // Write failed messages to file for manual review
        writeFailuresToFile (failures);
    }
    
    private GenericRecord jsonToAvro(String json) throws Exception {
        JsonNode jsonNode = objectMapper.readTree (json);
        
        // Validate required fields
        if (!jsonNode.has("payment_id")) {
            throw new ValidationException("Missing payment_id");
        }
        if (!jsonNode.has("amount")) {
            throw new ValidationException("Missing amount");
        }
        
        // Build Avro record
        GenericRecord avroRecord = new GenericData.Record (avroSchema);
        avroRecord.put("payment_id", jsonNode.get("payment_id").asText());
        avroRecord.put("amount", jsonNode.get("amount").asDouble());
        avroRecord.put("currency", jsonNode.has("currency") ? 
                      jsonNode.get("currency").asText() : "USD");  // Default
        avroRecord.put("timestamp", parseTimestamp (jsonNode.get("timestamp").asText()));
        avroRecord.put("schema_version", "v2_migrated");  // Mark as migrated
        
        return avroRecord;
    }
}

// Performance:
// - Throughput: 10K messages/sec per consumer
// - Total: 100M messages / 10K per sec = 10,000 seconds = 2.8 hours
// - Parallelism: 10 consumers = 2.8 hours / 10 = 17 minutes ✅
\`\`\`

**Handling Validation Failures:**

\`\`\`java
// Common validation issues:

1. Missing required field:
   {"payment_id": "pay_123"}  // Missing amount ❌
   
   Solution: Add default or skip
   avroRecord.put("amount", 0.0);  // Default
   logger.warn("Missing amount for payment_id: " + paymentId);

2. Type mismatch:
   {"amount": "99.99"}  // String instead of double ❌
   
   Solution: Type coercion
   double amount = Double.parseDouble (jsonNode.get("amount").asText());
   avroRecord.put("amount", amount);  ✅

3. Invalid timestamp format:
   {"timestamp": "2023-06-01"}  // Missing time ❌
   
   Solution: Default time or skip
   long timestamp = parseTimestampWithDefault (timestampStr, System.currentTimeMillis());

4. Null values:
   {"payment_id": null}  // Null required field ❌
   
   Solution: Skip or use placeholder
   if (paymentId == null) {
       dlqProducer.send(...);  // Send to DLQ
       return null;
   }

// DLQ for failed messages
KafkaTopic: payments-json-migration-dlq

// Manual review process:
1. Investigate DLQ messages
2. Fix data issues (missing fields, wrong types)
3. Reprocess manually:
   python reprocess_dlq.py --dlq-topic payments-json-migration-dlq

// Track failure rate:
Acceptable: <0.1% (100K out of 100M)
Alert if >1% (indicates systemic issue)
\`\`\`

**Zero-Downtime Migration:**

\`\`\`
Timeline:

Week 1: Preparation
- Define Avro schema
- Validate with sample data
- Register schema

Week 2-3: Dual Write
- Producers write JSON + Avro
- New Avro topic: payments-avro
- Monitor Avro write success rate (>99.9%)

Week 4-5: Consumer Migration
- Migrate consumers from payments-json to payments-avro
- 10% → 50% → 100% gradual rollout
- Monitor deserialization errors (0)

Week 6: Producer Cutover
- Stop dual write
- Producers write Avro only
- Monitor for 24 hours

Week 7: Historical Migration
- Migrate 100M historical JSON messages to Avro
- Write to payments-avro-historical topic
- 10 parallel consumers = 17 minutes
- Review DLQ for failed messages

Week 8: Cleanup
- Deprecate payments-json topic (after retention period)
- Delete dual-write code
- Update documentation

Zero downtime achieved ✅
\`\`\`

**Rollback Plan:**

\`\`\`
Rollback Scenario: Avro migration causes issues

Phase 2 (Dual Write) Rollback:
- Stop Avro writes
- Continue JSON writes only
- Impact: None (JSON still works) ✅

Phase 3 (Consumer Migration) Rollback:
- Revert consumers to JSON
- Redeploy old consumer code
- Impact: Temporary consumer lag (reprocessing) ⚠️

Phase 4 (Producer Cutover) Rollback:
- Re-enable dual write
- Producers write JSON again
- Impact: None (consumers handle both) ✅

Phase 5 (Historical Migration) Rollback:
- Stop migration job
- Delete payments-avro-historical topic
- Impact: Historical data not migrated (acceptable) ⚠️

Key: Keep JSON topic active for 30 days after cutover
Allows full rollback if issues discovered ✅
\`\`\`

**Monitoring:**

\`\`\`
Metrics:

1. Dual write success rate (should be >99.9%)
2. Avro deserialization errors (should be 0)
3. Migration progress (messages migrated / total)
4. DLQ depth (failed messages)
5. Consumer lag (both JSON and Avro consumers)

Dashboards:
- Migration progress: 45M / 100M (45% complete)
- Success rate: 99.95%
- DLQ depth: 5,000 messages (0.005%)
- ETA: 12 minutes remaining

Alerts:
- Deserialization errors > 0 (critical)
- DLQ depth > 100K (warning)
- Migration stalled (no progress in 5 min)
\`\`\`

**Key Takeaways:**
✅ Dual write period (JSON + Avro) ensures zero downtime
✅ Migrate consumers before producers (backward compatibility)
✅ Historical migration in parallel (10K msg/sec × 10 consumers = 17 min)
✅ DLQ for failed messages (<0.1% acceptable)
✅ Rollback plan: Keep JSON topic active for 30 days
✅ Monitoring: Success rate, DLQ depth, consumer lag`,
    keyPoints: [
      'Dual write (JSON + Avro) ensures zero downtime during migration',
      'Migrate consumers first, then producers (backward compatibility)',
      'Historical migration: 100M messages in 17 minutes (10 parallel consumers)',
      'DLQ for validation failures (<0.1% acceptable)',
      'Rollback plan: Keep JSON topic active for 30 days post-cutover',
      'Monitor: Success rate (>99.9%), DLQ depth, consumer lag',
    ],
  },
  {
    id: 'message-schema-evolution-dq-3',
    question:
      'Compare Avro, Protobuf, and JSON Schema for Kafka message serialization. For each format, discuss schema evolution, performance, tooling, and when to use it. Which would you choose for: (1) High-throughput analytics, (2) Microservices communication, (3) Public API events?',
    hint: 'Consider serialization speed, message size, schema validation, ecosystem support, and human readability.',
    sampleAnswer: `Different serialization formats offer trade-offs between performance, flexibility, and ease of use.

**Format Comparison:**

| Feature | Avro | Protobuf | JSON Schema |
|---------|------|----------|-------------|
| **Encoding** | Binary | Binary | Text (JSON) |
| **Schema** | Required | Required | Optional |
| **Evolution** | Excellent | Excellent | Good |
| **Size** | Smallest | Small | Largest |
| **Speed** | Fast | Fastest | Slower |
| **Human Readable** | No | No | Yes |
| **Tooling** | Kafka-native | gRPC-native | Universal |

**Use Case 1: High-Throughput Analytics**

**Choice: Avro ✅**

**Why:**
- Smallest message size (5-10× smaller than JSON)
- Fast serialization (2-3× faster than JSON)
- Schema evolution built-in (readers/writers evolve independently)
- Confluent Schema Registry integration
- Column-oriented storage friendly (Parquet uses Avro)

**Example:** Click stream analytics (1M events/sec)

\`\`\`java
// Avro schema
{
  "type": "record",
  "name": "ClickEvent",
  "fields": [
    {"name": "user_id", "type": "long"},
    {"name": "product_id", "type": "long"},
    {"name": "timestamp", "type": "long"},
    {"name": "action", "type": {"type": "enum", "name": "Action", 
                                 "symbols": ["VIEW", "CLICK", "PURCHASE"]}}
  ]
}

// Message size:
JSON: ~120 bytes
Avro: ~20 bytes (6× smaller) ✅

// Throughput:
1M events/sec × 120 bytes = 120 MB/sec (JSON)
1M events/sec × 20 bytes = 20 MB/sec (Avro) ✅

// Storage:
1M events/sec × 86400 sec/day × 120 bytes = 10.4 TB/day (JSON)
1M events/sec × 86400 sec/day × 20 bytes = 1.7 TB/day (Avro) ✅

// Cost savings:
Storage: 6× less
Network: 6× less bandwidth
Kafka brokers: Fewer brokers needed
\`\`\`

**Use Case 2: Microservices Communication**

**Choice: Protobuf ✅**

**Why:**
- Fastest serialization/deserialization
- Strong typing (compile-time validation)
- Code generation (Java, Go, Python, etc.)
- Backward/forward compatible
- gRPC integration (sync + async)

**Example:** Order service → Payment service

\`\`\`protobuf
// Protobuf schema
syntax = "proto3";

message PaymentRequest {
  string order_id = 1;
  double amount = 2;
  string currency = 3;
  string customer_id = 4;
}

message PaymentResponse {
  string payment_id = 1;
  string status = 2;  // SUCCESS, FAILED
  string error_message = 3;
}

service PaymentService {
  rpc ProcessPayment(PaymentRequest) returns (PaymentResponse);
}

// Generated code (Java)
PaymentRequest request = PaymentRequest.newBuilder()
    .setOrderId("order_123")
    .setAmount(99.99)
    .setCurrency("USD")
    .setCustomerId("cust_456")
    .build();

byte[] serialized = request.toByteArray();  // Very fast ✅

PaymentRequest deserialized = PaymentRequest.parseFrom (serialized);

// Performance:
Serialization: ~1 microsecond (vs 5-10 μs for JSON)
Deserialization: ~1 microsecond
Message size: ~30 bytes (vs 100 bytes JSON)

// Type safety:
request.setAmount("invalid");  // Compile error ✅ (vs runtime error in JSON)
\`\`\`

**Use Case 3: Public API Events**

**Choice: JSON Schema ✅**

**Why:**
- Human-readable (debugging, logs)
- Universal support (all languages)
- Easy to inspect (curl, browser)
- Schema validation available
- No code generation needed (flexible)

**Example:** Webhook events for third-party integrations

\`\`\`json
// JSON Schema
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "event_type": {"type": "string", "enum": ["order.created", "order.completed"]},
    "order_id": {"type": "string"},
    "amount": {"type": "number"},
    "currency": {"type": "string", "default": "USD"}
  },
  "required": ["event_type", "order_id", "amount"]
}

// Event payload (JSON)
{
  "event_type": "order.created",
  "order_id": "order_123",
  "amount": 99.99,
  "currency": "USD",
  "timestamp": "2023-06-01T10:00:00Z"
}

// Why JSON for public API:

1. Human-readable ✅
   - Easy to debug
   - View in logs/browser
   - curl https://api.example.com/events | jq

2. Universal support ✅
   - Every language has JSON libraries
   - No code generation needed
   - Flexible (dynamic typing)

3. Webhook-friendly ✅
   - Third parties can parse without SDK
   - Content-Type: application/json
   - Standard HTTP

4. Schema validation ✅
   const Ajv = require('ajv');
   const ajv = new Ajv();
   const validate = ajv.compile (schema);
   const valid = validate (data);  // true/false

// Trade-offs accepted:
❌ Larger size (100 bytes vs 20 bytes Avro)
❌ Slower parsing (5-10 μs vs 1 μs Protobuf)
✅ But: Ease of use for external consumers worth it
\`\`\`

**Performance Benchmarks:**

\`\`\`
Benchmark: Serialize 1M PaymentEvent messages

Message:
{
  "payment_id": "pay_123",
  "amount": 99.99,
  "currency": "USD",
  "timestamp": 1622534400000
}

Results:

JSON:
- Message size: 95 bytes
- Serialization time: 5,000 ms (5 ms/1K messages)
- Deserialization time: 8,000 ms
- Total: 13,000 ms

Avro:
- Message size: 18 bytes (5.3× smaller)
- Serialization time: 2,000 ms (2.5× faster)
- Deserialization time: 3,000 ms (2.7× faster)
- Total: 5,000 ms (2.6× faster)

Protobuf:
- Message size: 25 bytes (3.8× smaller than JSON)
- Serialization time: 1,200 ms (4.2× faster)
- Deserialization time: 1,800 ms (4.4× faster)
- Total: 3,000 ms (4.3× faster) ✅ Fastest

Summary:
- Protobuf: Fastest, small size
- Avro: Smallest size, fast
- JSON: Slowest, largest, but human-readable
\`\`\`

**Schema Evolution Comparison:**

\`\`\`
Scenario: Add optional field "customer_id"

Avro:
{
  "type": "record",
  "name": "PaymentEvent",
  "fields": [
    {"name": "payment_id", "type": "string"},
    {"name": "amount", "type": "double"},
    {"name": "customer_id", "type": ["null", "string"], "default": null}  // Added
  ]
}

// Old writer → New reader:
// customer_id = null (default) ✅

// New writer → Old reader:
// Old reader ignores customer_id ✅

Protobuf:
message PaymentEvent {
  string payment_id = 1;
  double amount = 2;
  string customer_id = 3;  // Added (optional by default)
}

// Old writer → New reader:
// customer_id = "" (default for string) ✅

// New writer → Old reader:
// Old reader ignores field 3 ✅

JSON Schema:
{
  "type": "object",
  "properties": {
    "payment_id": {"type": "string"},
    "amount": {"type": "number"},
    "customer_id": {"type": "string"}  // Added (optional)
  },
  "required": ["payment_id", "amount"]  // customer_id not required
}

// Old writer → New reader:
// customer_id missing, reader handles gracefully ✅

// New writer → Old reader:
// Old reader ignores extra field ✅

All three handle evolution well ✅
\`\`\`

**Ecosystem Integration:**

\`\`\`
Avro:
✅ Confluent Schema Registry (first-class)
✅ Kafka Connect (native support)
✅ Apache Spark (built-in Avro reader)
✅ Parquet (columnar format uses Avro schema)
❌ gRPC (not supported)

Protobuf:
✅ gRPC (native)
✅ Google Cloud (Pub/Sub, Dataflow)
✅ Multi-language code generation
❌ Kafka Schema Registry (third-party support)
❌ Kafka Connect (limited support)

JSON:
✅ Universal (all platforms)
✅ HTTP APIs (webhooks, REST)
✅ Logging (human-readable)
✅ Debugging tools (jq, browsers)
❌ Large size (network/storage costs)

Decision Matrix:

High-throughput Kafka + Analytics → Avro
Microservices + gRPC + Type safety → Protobuf
Public APIs + Webhooks + Debugging → JSON
\`\`\`

**Migration Path:**

\`\`\`
Scenario: Currently using JSON, want to migrate to Avro

Step 1: Define Avro schema from JSON
Step 2: Dual-write (JSON + Avro)
Step 3: Migrate consumers to Avro
Step 4: Producers write Avro only
Step 5: Deprecate JSON

Same strategy as JSON → Avro migration (Phase 1-6) ✅

Cost-benefit:
- Storage: 5× reduction
- Bandwidth: 5× reduction
- Processing: 2× faster
- Migration effort: 2-3 months

Worth it for high-volume systems ✅
\`\`\`

**Key Takeaways:**
✅ Avro: Best for high-throughput analytics (smallest size, Kafka-native)
✅ Protobuf: Best for microservices (fastest, type-safe, gRPC)
✅ JSON: Best for public APIs (human-readable, universal)
✅ All three support schema evolution well
✅ Choose based on: Performance needs, ecosystem, ease of use`,
    keyPoints: [
      'Avro: Smallest size (6× smaller than JSON), Kafka-native, best for analytics',
      'Protobuf: Fastest (4× faster than JSON), type-safe, best for microservices',
      'JSON: Human-readable, universal support, best for public APIs',
      'All three support schema evolution (backward/forward compatible)',
      'Performance: Protobuf > Avro > JSON',
      'Choose based on use case: Analytics (Avro), Microservices (Protobuf), Public API (JSON)',
    ],
  },
];

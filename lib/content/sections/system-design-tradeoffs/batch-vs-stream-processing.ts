/**
 * Batch Processing vs Stream Processing Section
 */

export const batchvsstreamprocessingSection = {
  id: 'batch-vs-stream-processing',
  title: 'Batch Processing vs Stream Processing',
  content: `Batch and stream processing represent fundamentally different approaches to data processing, with distinct trade-offs in latency, complexity, and use cases.

## Definitions

**Batch Processing**:
- **Process large volumes of data at scheduled intervals**
- Data collected over time, processed together
- High throughput, high latency
- Examples: Daily reports, ETL jobs, Apache Hadoop, Spark batch

**Stream Processing**:
- **Process data continuously as it arrives**
- Real-time or near real-time processing
- Low latency, lower throughput per record
- Examples: Real-time analytics, Apache Kafka Streams, Apache Flink, Storm

---

## Batch Processing in Detail

### How It Works

Collect data, process periodically (hourly, daily, weekly).

\`\`\`
Data Collection → Wait → Batch Job → Results
     (24 hours)     (Daily)
\`\`\`

### Use Cases

**1. Daily Reports**

**Example**: E-commerce sales report
- Collect orders all day
- At midnight: Process all orders
- Generate report: Total sales, top products, by region

**2. ETL (Extract, Transform, Load)**

**Example**: Data warehouse
- Extract: Pull data from 10 databases
- Transform: Clean, aggregate, join
- Load: Insert into data warehouse
- Run: Every night at 2 AM

**3. Machine Learning Training**

**Example**: Recommendation model
- Collect user interactions for 1 week
- Train model on entire dataset (billions of records)
- Deploy updated model

### Advantages

✅ **High throughput**: Process billions of records efficiently
✅ **Simple**: Easier to implement and debug
✅ **Cost-effective**: Use same infrastructure during off-peak hours
✅ **Atomic**: Process entire dataset together (easier correctness)
✅ **Replayable**: Easy to reprocess if errors occur

### Disadvantages

❌ **High latency**: Results delayed (hours to days)
❌ **Resource spikes**: Large computational load at scheduled time
❌ **Not real-time**: Cannot respond to events immediately
❌ **Wasted time**: Idle between batches

---

## Stream Processing in Detail

### How It Works

Process each event as it arrives, continuously.

\`\`\`
Event → Process → Output (milliseconds)
Event → Process → Output (milliseconds)
Event → Process → Output (milliseconds)
\`\`\`

### Use Cases

**1. Real-Time Analytics**

**Example**: Website monitoring
- Track page views in real-time
- Alert if traffic drops >50% in 1 minute
- Cannot wait for daily batch job

**2. Fraud Detection**

**Example**: Credit card transactions
- Detect fraudulent charge within seconds
- Block transaction before it completes
- Batch processing too slow (fraud already happened)

**3. Real-Time Recommendations**

**Example**: Netflix
- User watches show
- Immediately update recommendations
- Show relevant suggestions on homepage

### Advantages

✅ **Low latency**: Results in seconds or milliseconds
✅ **Real-time**: Respond to events immediately
✅ **Continuous**: No idle time, constant processing
✅ **Timely alerts**: Detect anomalies as they happen

### Disadvantages

❌ **Complex**: Harder to implement (state management, exactly-once processing)
❌ **Lower throughput per record**: Overhead per event
❌ **Harder to debug**: Distributed, continuous, stateful
❌ **Resource intensive**: Always running (higher cost)

---

## Real-World Examples

### Example 1: Uber Surge Pricing (Stream)

**Use case**: Dynamic pricing based on demand

**Why stream**:
- Demand changes every minute
- Must adjust prices in real-time
- Batch (hourly) too slow (demand spike missed)

**Implementation**:
- Kafka streams: Track ride requests per region
- Sliding window: Count requests in last 5 minutes
- If demand > supply → Increase price
- Update prices every 30 seconds

**Result**: Real-time pricing, optimal driver utilization

---

### Example 2: Netflix Recommendations (Batch + Stream)

**Batch processing**:
- Train recommendation model weekly
- Process billions of view history records
- Compute item similarities, user preferences
- Takes 24 hours to run

**Stream processing**:
- User watches show → Update user profile immediately
- Fetch recommendations from pre-computed model
- Personalize homepage in real-time

**Why hybrid**:
- Model training: Too expensive for real-time (batch)
- Personalization: Must be real-time (stream)

---

### Example 3: Bank Statement (Batch)

**Use case**: Monthly bank statement

**Why batch**:
- Statements generated once per month
- No need for real-time
- Batch more efficient (process millions at once)

**Implementation**:
- Collect transactions for 30 days
- On last day of month: Generate all statements
- Send via email/mail

**Result**: Cost-effective, sufficient for use case

---

## Trade-off Analysis

### Latency

**Batch**: Hours to days
**Stream**: Seconds to milliseconds

**Example**:
- Daily sales report: Batch (24-hour latency okay)
- Fraud detection: Stream (1-second latency required)

---

### Throughput

**Batch**: Billions of records per job
**Stream**: Thousands to millions of records per second

**Example**:
- ML training on 10TB data: Batch (high throughput)
- Processing IoT sensor data: Stream (continuous, lower per-event throughput)

---

### Complexity

**Batch**: Simpler to implement
**Stream**: Complex (state, windowing, fault tolerance)

**Example**:
- Batch: Spark job (100 lines of code)
- Stream: Kafka Streams with state management (500+ lines)

---

### Cost

**Batch**: Lower (run only when needed)
**Stream**: Higher (always running)

**Example**:
- Batch: Run 1 hour per day ($100/month)
- Stream: Run 24/7 ($2,000/month)

---

## Stream Processing Concepts

### Windowing

Group events into time-based windows.

**Tumbling Window**: Fixed, non-overlapping
\`\`\`
Events: [1,2,3,4,5,6,7,8,9,10]
Window size: 3
Windows: [1,2,3], [4,5,6], [7,8,9], [10]
\`\`\`

**Sliding Window**: Overlapping
\`\`\`
Events: [1,2,3,4,5,6,7,8,9,10]
Window size: 3, slide: 1
Windows: [1,2,3], [2,3,4], [3,4,5], ...
\`\`\`

**Session Window**: Based on inactivity
\`\`\`
Events: [1s, 2s, 3s, .... 100s, 101s]
Timeout: 10s
Windows: [1s-3s], [100s-101s] (separate sessions)
\`\`\`

---

### Exactly-Once Semantics

Ensure each event processed exactly once (no duplicates, no loss).

**Challenges**:
- Network failures: Message sent but acknowledgment lost
- Processing failures: Event processed but result not saved

**Solutions**:
- Idempotent processing: Safe to process twice
- Transactional writes: Atomic save + acknowledgment
- Kafka: Built-in exactly-once support

---

## Lambda Architecture (Hybrid)

Combine batch and stream for best of both worlds.

\`\`\`
Batch Layer:
- Process complete historical data (daily)
- High accuracy, full reprocessing
- Output: Comprehensive views

Stream Layer:
- Process recent data (real-time)
- Low latency, approximate
- Output: Real-time updates

Serving Layer:
- Merge batch + stream results
- Serve queries with both historical and real-time data
\`\`\`

**Example**: Twitter Analytics

**Batch**: 
- Daily job: Compute follower count for all users
- Accurate, complete

**Stream**:
- Real-time: Update follower count as follows happen
- Fast, recent

**Serving**:
- User requests follower count → Return batch + stream delta

---

## Best Practices

### ✅ 1. Use Batch for Historical Analysis

Daily reports, ML training, data warehousing → Batch

### ✅ 2. Use Stream for Real-Time Actions

Fraud detection, monitoring, alerts → Stream

### ✅ 3. Consider Hybrid (Lambda Architecture)

Combine batch (accuracy) + stream (latency)

### ✅ 4. Start with Batch, Add Stream When Needed

Batch simpler, stream when latency critical

### ✅ 5. Monitor Processing Lag

For stream: Track how far behind real-time (lag < 1 second)

---

## Interview Tips

### Strong Answer Pattern

"For this system, I'd use:

**Batch processing for**:
- [Historical analysis, ML training, daily reports]
- Reasoning: High throughput, latency not critical
- Trade-off: Delayed results (hours), but simple and cost-effective

**Stream processing for**:
- [Real-time fraud detection, monitoring, alerts]
- Reasoning: Low latency required (seconds)
- Trade-off: Higher complexity and cost, but necessary for real-time

**Hybrid approach**:
- Batch for comprehensive analysis (nightly)
- Stream for immediate actions (real-time)
- Lambda architecture to merge both

**Overall**: This balances latency requirements with implementation complexity and cost."

---

## Summary Table

| Aspect | Batch Processing | Stream Processing |
|--------|------------------|-------------------|
| **Latency** | Hours to days | Seconds to milliseconds |
| **Throughput** | Billions of records | Thousands to millions/sec |
| **Complexity** | Simple | Complex (state, windowing) |
| **Cost** | Lower (scheduled) | Higher (always running) |
| **Use Cases** | Daily reports, ETL, ML training | Fraud detection, monitoring, real-time analytics |
| **Examples** | Hadoop, Spark batch | Kafka Streams, Flink, Storm |
| **When to Use** | Latency not critical, high volume | Real-time actions required |

---

## Key Takeaways

✅ Batch: High throughput, high latency (hours), simple, cost-effective
✅ Stream: Low latency (seconds), complex, higher cost, real-time
✅ Use batch for daily reports, ETL, ML training (latency not critical)
✅ Use stream for fraud detection, monitoring, real-time alerts (latency critical)
✅ Lambda architecture: Combine batch (accuracy) + stream (latency)
✅ Start with batch (simpler), add stream when real-time required
✅ Stream concepts: Windowing (tumbling, sliding), exactly-once semantics`,
};

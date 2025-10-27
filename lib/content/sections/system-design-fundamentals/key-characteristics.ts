/**
 * Key Characteristics of Distributed Systems Section
 */

export const keycharacteristicsSection = {
  id: 'key-characteristics',
  title: 'Key Characteristics of Distributed Systems',
  content: `Modern systems that serve millions of users are distributed across multiple servers, data centers, and even geographical regions. Understanding the key characteristics that make these systems successful is fundamental to system design.

## 1. Scalability

**Definition**: The ability to handle increased load by adding resources.

### **Horizontal Scaling (Scale Out)**
Adding more machines to handle load.

**Example**: Instagram adds 100 servers to handle increased traffic
- **Pros**: Easier to scale indefinitely, better fault tolerance
- **Cons**: More complex architecture, data consistency challenges

### **Vertical Scaling (Scale Up)**
Adding more power (CPU, RAM, disk) to existing machines.

**Example**: Upgrade database server from 64GB to 256GB RAM
- **Pros**: Simpler, no code changes needed
- **Cons**: Hardware limits, single point of failure, expensive

### **Real-World Example: Netflix**
- Horizontally scaled: 10,000+ servers on AWS
- Serves 200M+ subscribers globally
- Scales up/down based on demand (peak evenings, weekends)
- Auto-scaling: Adds servers during high load, removes during low load

---

## 2. Reliability

**Definition**: System continues to work correctly even when components fail.

### **Fault Tolerance**
System tolerates failures without going down.

**Techniques:**
- **Redundancy**: Multiple copies of data/services
- **Replication**: Data replicated across 3+ servers
- **Failover**: Automatic switch to backup when primary fails

### **Example: WhatsApp Message Delivery**
- Message stored in 3 datacenters simultaneously
- If one datacenter fails, message still delivered from others
- Achieves 99.99%+ reliability

### **Metrics:**
- **MTBF** (Mean Time Between Failures): Average time system works before failure
- **MTTR** (Mean Time To Recovery): Average time to fix and recover
- **Reliability = MTBF / (MTBF + MTTR)**

---

## 3. Availability

**Definition**: Percentage of time system is operational and accessible.

### **Availability Tiers:**
| Availability | Downtime/Year | Downtime/Month | Use Case |
|---|---|---|---|
| 90% | 36.5 days | 3 days | Internal tools |
| 99% (two nines) | 3.65 days | 7.2 hours | Basic web apps |
| 99.9% (three nines) | 8.7 hours | 43 minutes | Most SaaS apps |
| 99.99% (four nines) | 52 minutes | 4.3 minutes | Critical services |
| 99.999% (five nines) | 5.26 minutes | 26 seconds | Payment systems |

### **Achieving High Availability:**1. **Redundancy**: No single point of failure
2. **Load Balancers**: Distribute traffic, detect failures
3. **Health Checks**: Monitor service health
4. **Geographic Distribution**: Multi-region deployment
5. **Graceful Degradation**: Serve reduced functionality vs complete failure

### **Real-World: Stripe (99.99%+)**
- Multi-region active-active
- Automatic failover
- Can lose entire datacenter without downtime
- Mission-critical for businesses

---

## 4. Efficiency

**Definition**: How well system uses resources (time, space, bandwidth).

### **Performance Metrics:**

#### **Latency (Response Time)**
Time from request to response.

**Targets by application type:**
- **Real-time chat**: <100ms
- **Search (Google)**: <200ms
- **Social media feed**: <500ms
- **Video streaming start**: <2 seconds
- **Batch processing**: Minutes to hours

#### **Throughput**
Number of operations per unit time.

**Examples:**
- Twitter: 500M tweets/day = 6K tweets/sec
- Instagram: 50B photo views/day = 580K views/sec
- Visa: 65,000 transactions/second peak

### **Improving Efficiency:**1. **Caching**: Reduce redundant work
2. **CDN**: Serve content from edge locations
3. **Compression**: Reduce data transfer
4. **Database Indexing**: Faster queries
5. **Async Processing**: Non-blocking operations

---

## 5. Manageability

**Definition**: How easy is it to operate and maintain the system?

### **Key Aspects:**

#### **Observability**
Can you understand what's happening in the system?

**Three Pillars:**
- **Logs**: Detailed events (errors, transactions)
- **Metrics**: Numerical measurements (CPU, memory, QPS)
- **Traces**: Request flow across services

#### **Operability**
Can you deploy, scale, and fix issues easily?

**Requirements:**
- **Automated deployments**: No manual steps
- **Self-healing**: Auto-restart failed services
- **Rollback capability**: Undo bad deployments
- **Blue-green deployments**: Zero-downtime updates

#### **Debuggability**
Can you diagnose and fix problems quickly?

**Tools:**
- **Centralized logging**: ELK stack, Splunk
- **Distributed tracing**: Jaeger, Zipkin
- **APM** (Application Performance Monitoring): Datadog, New Relic

### **Real-World: Amazon**
- Deploys code every 11.7 seconds
- Automated everything: testing, deployment, rollback
- Self-healing infrastructure
- Comprehensive monitoring and alerting

---

## 6. Fault Tolerance

**Definition**: System continues operating despite component failures.

### **Types of Failures:**1. **Hardware**: Server crash, disk failure, network outage
2. **Software**: Bugs, memory leaks, deadlocks
3. **Human**: Misconfigurations, accidental deletions
4. **Network**: Partitions, latency spikes, packet loss

### **Fault Tolerance Strategies:**

#### **Replication**
Multiple copies of data across servers.

**Example: Cassandra**
- Replication factor = 3
- Data on 3 different nodes
- Can lose 2 nodes and still serve data

#### **Redundancy**
Duplicate components (servers, databases, datacenters).

**Example: Netflix**
- Every service has multiple instances
- Multiple AWS regions
- If one region fails, traffic routes to another

#### **Graceful Degradation**
Reduce functionality vs complete failure.

**Example: Twitter during overload**
- Disable timeline refresh
- Show cached data
- Prioritize core features (viewing tweets)
- Disable non-critical (trending topics, recommendations)

---

## 7. Consistency vs Availability (CAP Theorem)

During network partition, choose between:

### **Consistency (C)**
All nodes see same data at same time.

**Use cases**: Banking, payments, inventory
**Systems**: Traditional SQL, Spanner

### **Availability (A)**
System always accepts requests (may return stale data).

**Use cases**: Social media, caching, DNS
**Systems**: Cassandra, DynamoDB, Couchbase

### **Partition Tolerance (P)**
System works despite network failures.
**(Always required in distributed systems)**

**CAP Theorem**: Can have at most 2 of 3 during partition.
- **CP Systems**: Consistency + Partition Tolerance (sacrifice availability)
- **AP Systems**: Availability + Partition Tolerance (sacrifice consistency)

---

## Putting It All Together

### **Example: Designing Twitter**

**Scalability**: 
- Horizontal scaling: 1000+ servers
- Database sharding by user_id
- CDN for media

**Reliability**:
- Replication factor 3
- Multi-datacenter deployment
- Automatic failover

**Availability**:
- Target: 99.9% (8.7 hours/year downtime acceptable)
- Load balancers with health checks
- Graceful degradation during overload

**Efficiency**:
- Timeline caching in Redis (reduces DB load)
- CDN for photos/videos (reduces bandwidth)
- Async tweet processing (improves write latency)

**Manageability**:
- Centralized logging (Splunk)
- Metrics dashboard (Grafana)
- Automated deployments
- On-call rotation

**Fault Tolerance**:
- Eventual consistency (acceptable for tweets)
- Message queues (Kafka) prevent data loss
- Circuit breakers prevent cascading failures

---

## Interview Tips

### **Discussing Characteristics:**

✅ **Good**: "For Instagram, we need high availability (99.9%) but can tolerate eventual consistency since users don't need real-time like counts. We'll use Cassandra for writes and Redis for caching, which gives us horizontal scalability."

❌ **Bad**: "We need high availability and strong consistency." (Contradicts CAP theorem without acknowledging trade-off)

### **Common Questions:**
- "How do you ensure high availability?"
  → Redundancy, replication, load balancing, health checks, multi-region
  
- "How does your system scale?"
  → Horizontal scaling, sharding, caching, CDN, async processing
  
- "What happens if a server fails?"
  → Automatic failover, load balancer detects failure, traffic routes to healthy servers`,
};

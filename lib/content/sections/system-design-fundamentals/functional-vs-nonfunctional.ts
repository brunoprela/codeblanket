/**
 * Functional vs. Non-functional Requirements Section
 */

export const functionalvsnonfunctionalSection = {
  id: 'functional-vs-nonfunctional',
  title: 'Functional vs. Non-functional Requirements',
  content: `Understanding the difference between functional and non-functional requirements is critical for system design. These requirements define **what** your system does and **how well** it does it.

## Functional Requirements

**Definition**: What the system must DO - the features and capabilities.

### **Examples for Twitter:**
- Users can post tweets (max 280 characters)
- Users can follow other users
- Users can view a timeline of tweets from people they follow
- Users can like and retweet posts
- Users can upload photos and videos
- Users can search for tweets and users
- Users can send direct messages

**Characteristics:**
- **Specific actions** the system must support
- **Testable** - you can verify they work
- **User-facing** - visible to end users
- Define the **scope** of the system

---

## Non-functional Requirements

**Definition**: How the system must PERFORM - quality attributes and constraints.

### **Key Categories:**

#### **1. Scalability**
- **Vertical**: Handle more load by adding resources to existing machines
- **Horizontal**: Handle more load by adding more machines
- Example: "System must handle 10,000 requests per second"

#### **2. Performance**
- **Latency**: Time to process a single request
  - Example: "API responses must be <100ms at p99"
- **Throughput**: Number of requests processed per unit time
  - Example: "System must process 1 million transactions per day"

#### **3. Availability**
- Percentage of time system is operational
- Example: "99.99% uptime (52 minutes downtime per year)"
- Trade-off with consistency (CAP theorem)

#### **4. Reliability**
- System works correctly even with failures
- Example: "No data loss even if 2 servers fail"
- **MTBF** (Mean Time Between Failures)
- **MTTR** (Mean Time To Recovery)

#### **5. Consistency**
- All users see the same data at the same time
- **Strong consistency**: Immediate updates everywhere
- **Eventual consistency**: Updates propagate over time
- Example: "Bank balance must be strongly consistent"

#### **6. Durability**
- Data persists even after system failures
- Example: "Zero data loss for committed transactions"
- Achieved through replication and backups

#### **7. Security**
- Authentication: Who are you?
- Authorization: What can you do?
- Encryption: Data protection
- Example: "All data encrypted at rest and in transit"

#### **8. Maintainability**
- How easy is it to update and fix?
- Code quality, documentation, monitoring
- Example: "New features can be deployed without downtime"

---

## Why This Distinction Matters

### **In Interviews:**

**Interviewer**: "Design Twitter"

**You should clarify BOTH types:**

**Functional**: 
- "Should users be able to retweet? Upload videos? Edit tweets?"
- "Do we need direct messaging? Notifications?"

**Non-functional**:
- "How many daily active users? How many tweets per day?"
- "What\'s acceptable latency for timeline? 100ms? 1 second?"
- "Strong consistency or eventual consistency?"
- "What's the required availability? 99.9%? 99.99%?"

### **Impact on Design:**

**Example: Banking App vs Social Media**

| Requirement | Banking | Social Media |
|-------------|---------|--------------|
| **Consistency** | Strong (must be accurate) | Eventual (delays OK) |
| **Availability** | 99.99%+ (critical) | 99.9% (acceptable) |
| **Latency** | <500ms (acceptable) | <100ms (expected) |
| **Security** | Extremely high | Moderate |
| **Data Loss** | Zero tolerance | Some tolerance |

**Design implications:**
- Banking: SQL database, ACID transactions, synchronous replication
- Social Media: NoSQL, eventual consistency, async replication, heavy caching

---

## Real-World Example: Instagram Stories

### **Functional Requirements:**
- Users can post photos/videos that expire in 24 hours
- Users can view stories from people they follow
- Users can see who viewed their story
- Stories appear in a sequential feed
- Users can add text, stickers, filters

### **Non-functional Requirements:**
- **Scale**: 500 million daily active users
- **Upload latency**: <2 seconds for photo, <10 seconds for video
- **View latency**: <200ms to load story feed
- **Availability**: 99.9% (some downtime acceptable)
- **Consistency**: Eventual (view counts can be delayed)
- **Storage**: Auto-delete after 24 hours
- **Throughput**: Handle 50,000 stories posted per second

**How these affect design:**
- High scale → CDN for media delivery
- Low latency → aggressive caching with Redis
- Eventual consistency → NoSQL database (Cassandra)
- Auto-delete → TTL (Time To Live) in database
- High throughput → message queue for async processing

---

## Common Mistakes

### ❌ **Mistake 1: Jumping to Solutions**
**Bad**: "We'll use microservices and Kubernetes"
**Why bad**: You haven't defined requirements yet!
**Good**: "Let me first clarify: What scale? What latency is acceptable?"

### ❌ **Mistake 2: Ignoring Non-functional Requirements**
**Bad**: Only discussing features without discussing scale
**Why bad**: A Twitter clone for 100 users is VERY different from one for 100 million users
**Good**: "At 300M users and 500M tweets/day, we need sharding and replication..."

### ❌ **Mistake 3: Unrealistic Expectations**
**Bad**: "We need 100% availability, zero latency, strong consistency, and infinite scale"
**Why bad**: These conflict with each other (CAP theorem)
**Good**: "We'll choose eventual consistency for better availability and lower latency"

---

## Template for Clarifying Requirements

### **Functional Questions:**1. What are the core features? (MVP)
2. What features can we skip? (nice-to-have)
3. Who are the users? (internal/external, tech-savvy?)
4. What platforms? (web, mobile, API)

### **Non-functional Questions:**1. **Scale**: How many users? How much data? Growth rate?
2. **Performance**: What's acceptable latency? Required throughput?
3. **Availability**: Tolerate downtime? What SLA?
4. **Consistency**: Strong or eventual? Why?
5. **Durability**: Can we lose data? How much?
6. **Geography**: Single region or global?
7. **Cost**: Any budget constraints?

**Practice using this template in every system design interview!**`,
};

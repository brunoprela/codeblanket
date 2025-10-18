/**
 * Back-of-the-Envelope Estimations Section
 */

export const backofenvelopeSection = {
  id: 'back-of-envelope',
  title: 'Back-of-the-Envelope Estimations',
  content: `Back-of-the-envelope estimations are quick calculations to assess system scale, storage needs, bandwidth, and costs. These estimations help you make informed architectural decisions during interviews.

## Why Estimations Matter

### **Drive Design Decisions**
- Small scale (1K users) → Single server
- Medium scale (1M users) → Load balancer + replicas
- Large scale (100M users) → Distributed system, CDN, caching

### **Show Quantitative Thinking**
Interview

ers want to see you think with numbers, not just concepts.

**Bad**: "We need a database"
**Good**: "At 1M writes/day, that's 12 writes/second. A single MySQL instance can handle this."

---

## Essential Numbers to Memorize

### **Powers of 2:**
- 2^10 = 1 thousand = 1 KB
- 2^20 = 1 million = 1 MB
- 2^30 = 1 billion = 1 GB
- 2^40 = 1 trillion = 1 TB
- 2^50 = 1 quadrillion = 1 PB

### **Time Conversions:**
- 1 day = 86,400 seconds (~100K seconds)
- 1 month = ~2.5M seconds
- 1 year = ~31.5M seconds (~30M for estimates)

### **Latency Numbers Every Programmer Should Know:**
- L1 cache reference: 0.5 ns
- L2 cache reference: 7 ns
- Main memory reference: 100 ns
- Read 1 MB sequentially from memory: 250 μs
- SSD random read: 150 μs
- Read 1 MB sequentially from SSD: 1 ms
- Disk seek: 10 ms
- Read 1 MB sequentially from disk: 30 ms
- Send 1 MB over 1 Gbps network: 10 ms
- Round trip within same datacenter: 0.5 ms
- Round trip CA to Netherlands: 150 ms

### **Throughput Numbers:**
- Modern server CPU: 10K-50K requests/sec
- MySQL: 1K writes/sec, 10K reads/sec (single instance)
- Redis: 100K ops/sec (single instance)
- Cassandra: 10K-100K writes/sec per node
- Kafka: 1M messages/sec

### **Storage Costs (2024):**
- SSD: $0.10/GB/month
- HDD: $0.02/GB/month
- S3: $0.023/GB/month

---

## Estimation Framework

### **Step 1: Clarify the Question**
Ask: Daily Active Users (DAU)? Growth rate? Geographic distribution?

### **Step 2: List Assumptions**
Write them down: "Assuming 100M DAU, 10 tweets/user/day..."

### **Step 3: Calculate**
Break problem into smaller pieces.

### **Step 4: Validate**
Does the answer make sense? Sanity check.

---

## Example 1: Twitter Storage Estimation

**Question**: How much storage does Twitter need for tweets?

### **Step 1: Assumptions**
- 300M Daily Active Users (DAU)
- 2 tweets per user per day on average
- Each tweet: 280 characters = 280 bytes text
- 10% tweets have photo (200 KB avg)
- 5% tweets have video (2 MB avg)
- Data retention: Forever

### **Step 2: Calculate Daily Tweets**
300M users × 2 tweets/day = 600M tweets/day

### **Step 3: Calculate Storage per Tweet**

**Text**: 280 bytes × 600M = 168 GB/day

**Photos**: 600M × 10% = 60M photos
60M × 200 KB = 12 TB/day

**Videos**: 600M × 5% = 30M videos
30M × 2 MB = 60 TB/day

**Total**: 168 GB + 12 TB + 60 TB ≈ **72 TB/day**

### **Step 4: Calculate Yearly Storage**
72 TB/day × 365 = **26 PB/year**

### **Step 5: 5-Year Projection**
26 PB × 5 = **130 PB (5 years)**

**Storage cost**: 130 PB × $0.02/GB/month = $2.6M/month (HDD)

**Conclusion**: Need distributed storage, CDN for media, possibly tiered storage (hot/cold).

---

## Example 2: YouTube Bandwidth Estimation

**Question**: Estimate bandwidth needed for YouTube.

### **Assumptions**
- 2 billion users globally
- 100M daily active users watching videos
- Average watch time: 30 minutes/day
- Average video bitrate: 2 Mbps (720p)

### **Calculate Daily Bandwidth**

**Total viewing time**: 100M users × 30 min = 3B minutes = 50M hours

**Data transferred**: 50M hours × 2 Mbps × 3600 sec/hour
= 50M × 2 Mbps × 3600 s
= 360,000,000,000 Mb
= 360M GB = 360 PB/day

**Peak bandwidth (assume 20% during peak hour)**:
360 PB/day ÷ 24 hours × 1.2 (peak multiplier) = **18 PB/hour** during peak

**Gbps required**: 18 PB/hour = 18,000 TB/hour = 40 million Gbps

**Conclusion**: Need massive CDN infrastructure, edge caching, multi-tier delivery network.

---

## Example 3: QPS (Queries Per Second) Calculation

**Question**: Design Instagram - estimate read/write QPS.

### **Assumptions**
- 500M DAU
- Each user views 100 photos/day (reads)
- Each user uploads 0.5 photos/day (writes)
- Peak traffic: 2× average

### **Calculate Average QPS**

**Read requests/day**: 500M × 100 = 50B reads/day

**Read QPS**: 50B / 86,400 sec ≈ **580K QPS** (average)

**Peak read QPS**: 580K × 2 = **1.2M QPS**

**Write requests/day**: 500M × 0.5 = 250M writes/day

**Write QPS**: 250M / 86,400 ≈ **3K QPS** (average)

**Peak write QPS**: 3K × 2 = **6K QPS**

**Conclusion**: Read-heavy system (200:1 ratio). Need heavy caching (Redis, CDN), read replicas, consider eventual consistency.

---

## Common Estimation Patterns

### **Pattern 1: Storage Estimation**
**Formula:** Total Storage = Users × Data per User × Replication Factor  
**Example:** 1M users × 1GB photos × 3 replicas = 3 PB

### **Pattern 2: Bandwidth Estimation**
**Formula:** Bandwidth = (Data Size × Requests) / Time  
**Example:** (1MB × 10K requests/sec) = 10 GB/sec = 80 Gbps

### **Pattern 3: Server Count Estimation**
**Formula:** Servers Needed = Total QPS / QPS per Server  
**Example:** 100K QPS ÷ 10K QPS/server = 10 servers (add buffer → 15 servers)

### **Pattern 4: Database Sizing**
**Formula:** DB Size = Records × Size per Record × Growth Factor  
**Example:** 1B users × 1KB/user × 1.5 growth = 1.5 TB

---

## Pro Tips

### **1. Round Numbers**
Don't say "86,400 seconds" - say "~100K seconds"
Don't calculate exact percentages - use 10%, 50%, 2×

### **2. Show Your Work**
Write calculations on whiteboard/paper as you go.
Helps you catch errors and interviewer can follow.

### **3. Think in Orders of Magnitude**
Is it KB, MB, GB, TB, PB?
Is it 10s, 100s, 1000s, millions?

### **4. Sanity Check**
Does 1 PB/day make sense? (Probably too high for most apps)
Does 1 server handle 1M QPS? (No way - red flag)

### **5. Consider Peak vs Average**
Peak traffic often 2-3× average.
Design for peak, not average!

### **6. Account for Redundancy**
Replication: Usually 3× storage
Backups: Add 50-100% overhead

---

## Practice Questions

1. **WhatsApp**: Estimate storage needed for messages (2B users, 50 messages/user/day, 100 bytes/message)
2. **Uber**: Estimate GPS updates from drivers (1M drivers, update every 4 sec, 100 bytes/update)
3. **Netflix**: Estimate CDN costs (100M subscribers, 2 hours watch/day, 5 Mbps bitrate)
4. **TikTok**: Estimate video uploads (1B users, 10% upload daily, 30 sec average, 5 MB/video)

**Practice these calculations until they become second nature!**`,
};

/**
 * Things to Avoid During System Design Interviews Section
 */

export const thingstoavoidSection = {
  id: 'things-to-avoid',
  title: 'Things to Avoid During System Design Interviews',
  content: `System design interviews are as much about demonstrating good engineering judgment as they are about technical knowledge. Avoiding common pitfalls can significantly improve your interview performance.

## 1. Jumping to Solutions Without Clarifying Requirements

### **The Mistake:**
Interviewer: "Design Twitter"
Candidate: "We'll use microservices, Kubernetes, Kafka, and React for the frontend..."

**Why it's bad:**
- Haven't defined the scope (which features?)
- Don't know the scale (100 users vs 100M users)
- Don't understand constraints (latency requirements? availability?)
- Designing in a vacuum without context

### **The Right Approach:**
✅ "Before I start, let me clarify a few things:
- Should we focus on core features like posting tweets and viewing timelines, or also direct messaging and notifications?
- What's the expected scale? Daily active users? Tweets per day?
- What are the latency requirements for timeline loading?
- Do we need strong consistency or is eventual consistency acceptable?"

**Key principle:** Spend 10-15 minutes on requirements. It's not wasted time—it's crucial context.

---

## 2. Using Buzzwords Without Understanding

### **The Mistake:**
"We'll use blockchain for immutability, machine learning for recommendations, and microservices for scalability."

**Why it's bad:**
- Buzzwords without justification
- Technologies may not fit the problem
- Shows surface-level knowledge
- Interviewer will probe deeper and expose gaps

### **Red Flag Buzzwords (without context):**
- "We'll use AI/ML" (for what? why?)
- "Blockchain" (rarely needed)
- "Microservices" (not always better than monolith)
- "NoSQL" (without explaining why over SQL)
- "Kubernetes" (adds complexity, justify it)

### **The Right Approach:**
✅ "I'm considering Cassandra over PostgreSQL because:
1. We need to handle 500K writes/second (high write throughput)
2. Eventual consistency is acceptable for social media feeds
3. We need horizontal scaling across multiple datacenters
However, for the user account service, I'd use PostgreSQL because we need strong consistency for authentication."

**Key principle:** Every technology choice should have a **justification** based on requirements.

---

## 3. Ignoring Scale and Numbers

### **The Mistake:**
"We'll just use a database and cache" (without calculating if one database can handle the load)

**Why it's bad:**
- Shows lack of quantitative thinking
- Design may not actually work at scale
- Misses opportunity to demonstrate engineering depth

### **Example of ignoring scale:**
Instagram has 500M DAU viewing 100 photos each = 50 billion reads/day = 580K QPS.
Single MySQL: ~10K QPS → **Need 58 read replicas or different approach (caching, CDN)!**

### **The Right Approach:**
✅ "Let me estimate the load:
- 100M users, 10 posts/user/day = 1B posts/day
- 1B / 86,400 sec = ~12K writes/sec
- MySQL handles ~1K writes/sec, so we need sharding
- With 20 shards, each handles 600 writes/sec comfortably
- For reads (100× more), we'll need caching..."

**Key principle:** Do **back-of-envelope calculations** to validate your design decisions.

---

## 4. Over-Engineering or Under-Engineering

### **Over-Engineering:**
Designing a URL shortener (bit.ly) with:
- 50 microservices
- Event-driven architecture with Kafka
- Machine learning for fraud detection
- Kubernetes clusters across 10 regions

**Why it's bad:** The problem doesn't need this complexity. Shows poor judgment.

**Right approach:** Simple API server, Redis cache, PostgreSQL database. Scales to millions of URLs easily.

### **Under-Engineering:**
Designing Twitter with:
- Single MySQL database
- No caching
- Single server

**Why it's bad:** Won't scale to 300M users and 500M tweets/day. Shows lack of understanding.

**Right approach:** Distributed system with sharding, caching, CDN, message queues.

### **How to avoid:**
✅ Match complexity to scale:
- Small scale (1K-100K users): Monolith, single DB, simple cache
- Medium scale (100K-10M users): Vertical scaling, read replicas, caching layer
- Large scale (10M+ users): Distributed system, sharding, CDN, message queues

---

## 5. Not Discussing Trade-offs

### **The Mistake:**
"We'll use Cassandra for the database."
Interviewer: "Why not PostgreSQL?"
Candidate: "Um... Cassandra is better for scale?"

**Why it's bad:**
- No acknowledgment that both have pros/cons
- Can't articulate trade-offs
- Suggests memorized solution rather than thinking

### **The Right Approach:**
✅ "I'm choosing between PostgreSQL and Cassandra:

**PostgreSQL:**
- ✅ ACID transactions, strong consistency
- ✅ Rich query capabilities (joins, complex queries)
- ❌ Limited write throughput (~1K writes/sec)
- ❌ Vertical scaling limits

**Cassandra:**
- ✅ High write throughput (10K+ writes/sec per node)
- ✅ Horizontal scaling, multi-datacenter replication
- ❌ Eventual consistency (not always acceptable)
- ❌ No joins (need to denormalize)

For Twitter tweets, I choose Cassandra because:
1. Need 50K writes/sec (tweets being posted)
2. Eventual consistency OK (timeline can be slightly delayed)
3. Need global distribution

But for user authentication, I'd use PostgreSQL because we need strong consistency."

**Key principle:** Always discuss **trade-offs**. There's no perfect solution—show you understand pros and cons.

---

## 6. Poor Communication

### **Signs of poor communication:**
- Long silences (30+ seconds without talking)
- Speaking too quietly or mumbling
- Not using the whiteboard
- Going down rabbit holes without explaining
- Ignoring interviewer hints

### **The Right Approach:**
✅ **Think out loud:**
"I'm thinking about the database choice. We have high write load, so I'm considering NoSQL. Let me think about consistency requirements..."

✅ **Use the whiteboard:**
Draw boxes for services, arrows for data flow, add labels

✅ **Check in with interviewer:**
"Does this high-level design make sense before I dive deeper?"
"Should I focus more on the storage layer or the API design?"

✅ **Respond to hints:**
Interviewer: "What about read latency?"
You: "Good point! Let me add a caching layer..."

**Key principle:** Interview is a **conversation**, not a monologue. Engage with the interviewer.

---

## 7. Neglecting Non-Functional Requirements

### **The Mistake:**
Only discussing features (functional requirements) without discussing:
- Scalability (how many users?)
- Latency (how fast?)
- Availability (how much downtime acceptable?)
- Consistency (strong or eventual?)

**Why it's bad:** Non-functional requirements drive architecture decisions.

### **The Right Approach:**
✅ Explicitly clarify:
"What are the availability requirements? 99.9%? 99.99%?"
"Is sub-100ms latency critical, or is 500ms acceptable?"
"How many concurrent users do we expect?"

Then design accordingly:
- 99.99% availability → Multi-region, redundancy, failover
- <100ms latency → Aggressive caching, CDN
- 100M users → Horizontal scaling, sharding

---

## 8. Focusing Only on Happy Path

### **The Mistake:**
Only designing for when everything works perfectly, ignoring:
- What if a server crashes?
- What if the database is down?
- What if there's a network partition?
- What if traffic spikes 10×?

**Why it's bad:** Real systems must handle failures gracefully.

### **The Right Approach:**
✅ Discuss failure scenarios:
"If a web server crashes, the load balancer detects it via health checks and stops routing traffic to it."
"If the primary database fails, we have a hot standby that automatically takes over within 30 seconds."
"If traffic spikes, auto-scaling adds servers within 2 minutes."
"If the cache goes down, requests go directly to database (slower but system stays up)."

**Key principle:** Always discuss **fault tolerance** and **graceful degradation**.

---

## 9. Not Asking Questions

### **The Mistake:**
Interviewer: "Design Instagram"
Candidate: *immediately starts drawing architecture*

**Why it's bad:**
- Problem is intentionally ambiguous
- Not asking shows lack of curiosity
- Miss important context

### **The Right Approach:**
✅ Ask clarifying questions:
- "Should we focus on photo sharing or also stories, reels, messaging?"
- "Is this mobile-first or web-first?"
- "What's more important: upload speed or viewing speed?"
- "Do we need real-time notifications?"
- "Any specific compliance requirements (GDPR, data residency)?"

**Key principle:** Problem is **intentionally vague**. Asking questions is expected and shows thoroughness.

---

## 10. Giving Up or Saying "I Don't Know"

### **The Mistake:**
Interviewer: "How would you handle rate limiting?"
Candidate: "I don't know. I haven't worked with that before."

**Why it's bad:**
- Missed opportunity to show problem-solving
- Interviews test your ability to reason, not just memorize

### **The Right Approach:**
✅ Reason through it:
"I haven't implemented rate limiting specifically, but here's how I'd approach it:
- Need to track requests per user per time window
- Could use a sliding window counter in Redis
- Key: user_id, value: count, TTL: time window
- For each request: increment counter, if exceeds limit, reject
- This handles distributed system by centralizing count in Redis
Does this approach make sense?"

**Key principle:** Show your **thought process**. It's okay not to know everything—it's not okay to give up.

---

## 11. Rigidity - Not Adapting to Feedback

### **The Mistake:**
Interviewer: "What about caching?"
Candidate: "No, I think the database is fine."
Interviewer: "But at this scale..."
Candidate: "I still think database only is fine."

**Why it's bad:**
- Shows inability to adapt
- Ignoring feedback/hints
- Not collaborative

### **The Right Approach:**
✅ Adapt and incorporate feedback:
Interviewer: "What about caching?"
You: "Good point! At 500K QPS, database alone won't scale. Let me add Redis caching. We'll cache user timelines with 5-minute TTL. This reduces database load by 90%+."

**Key principle:** Be **flexible** and **collaborative**. Interviewer hints are there to help you.

---

## 12. Running Out of Time

### **The Mistake:**
- Spending 40 minutes on requirements and high-level design
- No time for deep dive or discussing trade-offs
- Interview ends before you show technical depth

**Why it's bad:** Didn't complete the interview. Interviewer doesn't see your full capability.

### **Time Management:**
\`\`\`
Total: 45 - 60 minutes

✅ Requirements(5 - 10 min)
✅ High - level design(10 - 15 min)
✅ Deep dive(20 - 25 min) ← Most important!
✅ Wrap up(5 min)
\`\`\`

**The Right Approach:**
✅ Be aware of time:
"I've spent 10 minutes on requirements and high-level design. Should I dive deeper into the storage layer or move to discussing scalability?"

---

## Summary: Green Flags vs Red Flags

### **🚩 Red Flags:**
- Jumping to solutions
- Buzzwords without justification
- Ignoring scale/numbers
- Not discussing trade-offs
- Poor communication (silent, not using whiteboard)
- Only happy path, no failure handling
- Not asking questions
- Giving up easily
- Rigid, not adapting to feedback

### **✅ Green Flags:**
- Clarifying requirements thoroughly
- Using numbers and calculations
- Discussing trade-offs explicitly
- Clear communication (thinking out loud, using diagrams)
- Considering failure scenarios
- Asking thoughtful questions
- Showing problem-solving when unsure
- Adapting to interviewer feedback
- Managing time well`,
};

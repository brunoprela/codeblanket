/**
 * Module Review & Next Steps Section
 */

export const modulereviewSection = {
  id: 'module-review',
  title: 'Module Review & Next Steps',
  content: `Congratulations on completing System Design Fundamentals! Let's review what you've learned and prepare for the next modules.

## What You've Learned

### **Section 1: System Design Interview Basics**
- Purpose and format of system design interviews
- What interviewers evaluate: thinking, communication, trade-offs
- Interview structure and time management
- Difference between junior, mid, senior, and principal level interviews

### **Section 2: Requirements Gathering**
- Functional vs non-functional requirements
- How to clarify ambiguous problems
- Importance of stating assumptions explicitly
- How requirements drive architecture decisions

### **Section 3: Back-of-Envelope Estimation**
- Essential numbers to memorize (latency, throughput, storage costs)
- Storage, bandwidth, QPS calculation techniques
- Using numbers to validate design decisions
- Peak vs average traffic considerations

### **Section 4: Distributed Systems Characteristics**
- Scalability: Horizontal vs vertical scaling
- Reliability and fault tolerance
- Availability tiers and their implications (99.9% vs 99.99%)
- Efficiency: Latency vs throughput
- CAP theorem and consistency trade-offs

### **Section 5: Common Pitfalls**
- Jumping to solutions without requirements
- Using buzzwords without justification
- Ignoring scale and calculations
- Not discussing trade-offs
- Poor communication patterns

### **Section 6: Systematic Framework**
- 4-step approach: Requirements → High-Level → Deep Dive → Wrap Up
- Time allocation: 5-10min, 10-15min, 20-25min, 5min
- How to handle each step effectively
- Deep dive strategies and techniques

### **Section 7: Visual Communication**
- Drawing architecture diagrams effectively
- Component organization and grouping
- Data flow visualization
- Using diagrams to facilitate discussion

---

## Self-Assessment

### **Rate Your Understanding (1-5):**

1. **Requirements gathering**: Can you ask clarifying questions effectively?
2. **Back-of-envelope calculations**: Can you estimate storage, QPS, bandwidth?
3. **CAP theorem**: Do you understand consistency/availability trade-offs?
4. **Framework application**: Can you follow the 4-step process?
5. **Diagram drawing**: Can you draw clear architecture diagrams?

**If any rating is <3, review that section again before moving forward.**

---

## Practice Exercises

Before moving to the next module, practice these:

### **Exercise 1: Requirements (15 minutes)**
Problem: "Design WhatsApp"
Task: List 10 clarifying questions you'd ask about functional and non-functional requirements.

### **Exercise 2: Estimation (20 minutes)**
Calculate for a ride-sharing app like Uber:
- Storage needed for GPS updates (1M drivers, update every 4 sec)
- QPS for ride matching service
- Database size for trip history (1B trips over 5 years)

### **Exercise 3: Framework Application (45 minutes)**
Design a URL shortener using the complete 4-step framework:
- Step 1: Requirements (5 min)
- Step 2: High-level design (10 min)
- Step 3: Deep dive on short code generation (20 min)
- Step 4: Wrap up (5 min)

### **Exercise 4: Diagram Practice (10 minutes)**
Draw architecture for Netflix:
- Include: Clients, LB, API servers, metadata DB, video storage, CDN, recommendation service
- Show numbered data flow for video playback

---

## Common Mistakes to Avoid

### **Still seeing these in interviews:**
❌ Proposing microservices for every problem
❌ Not calculating whether single DB can handle load
❌ Forgetting to discuss failure scenarios
❌ Drawing diagram but never referencing it
❌ Spending 30 minutes on requirements, no time for deep dive

### **Green flags to aim for:**
✅ Asking thoughtful clarifying questions
✅ Using numbers to justify decisions
✅ Explicitly stating trade-offs
✅ Adapting to interviewer feedback
✅ Systematic, structured approach

---

## What's Next

### **Module 2: Core Building Blocks**
You'll learn about:
- Load balancing algorithms and strategies
- Caching patterns and eviction policies
- Database sharding and replication
- Message queues and async processing
- CDN and edge computing

### **Module 3: Database Design & Theory**
Topics include:
- SQL vs NoSQL decision framework
- CAP and PACELC theorems deep dive
- Consistency models
- Database scaling patterns
- Indexing strategies

### **Preparation for Next Modules:**
1. Review this module's key takeaways
2. Practice the 4-step framework on 2-3 problems
3. Get comfortable with back-of-envelope calculations
4. Practice drawing diagrams

---

## Interview Readiness Checklist

### **After this module, you should be able to:**
- [ ] Explain what system design interviews evaluate
- [ ] Ask 5-10 clarifying questions for any design problem
- [ ] Calculate storage, QPS, bandwidth for common scenarios
- [ ] Explain CAP theorem and give real-world examples
- [ ] Follow the 4-step framework consistently
- [ ] Draw clean architecture diagrams
- [ ] Discuss trade-offs for major decisions
- [ ] Handle interviewer feedback gracefully
- [ ] Manage time effectively (45-60 minutes)
- [ ] Identify common red flags and avoid them

**If you can check all boxes confidently, you're ready for more advanced topics!**

---

## Resources for Practice

### **Mock Interview Practice:**
- Practice with peers (30-45 min sessions)
- Record yourself and review
- Use timer to practice time management
- Get feedback on communication

### **System Design Practice Sites:**
- System Design Primer (GitHub)
- Grokking the System Design Interview
- ByteByteGo newsletter
- Real-world tech blogs (Netflix, Uber, Instagram)

### **Study Real Systems:**
- Read engineering blogs from FAANG companies
- Study architecture diagrams from tech talks
- Understand how real companies solve scale problems

---

## Final Thoughts

**System design is a skill, not memorization.**

- You don't need to know every technology
- You need to demonstrate: structured thinking, trade-off analysis, communication, adaptability
- Practice the framework until it's natural
- Learn from mistakes (yours and others')
- Stay curious about how real systems work

**Key principle:** There's no single "correct" answer. Interviews are about the journey (your thought process) more than the destination (the final architecture).

**Keep practicing, stay systematic, and remember: even senior engineers are continuously learning system design!**

Ready for Module 2? Let's dive into the core building blocks that power modern distributed systems! 🚀`,
};

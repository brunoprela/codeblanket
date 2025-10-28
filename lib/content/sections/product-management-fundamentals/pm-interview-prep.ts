/**
 * Section 13: PM Interview Preparation
 * Product Management Fundamentals Module
 */

import { ModuleSection } from '../../../types';

const content = `# PM Interview Preparation

## Introduction

PM interviews are notoriously difficult and varied. Companies test strategy, execution, technical knowledge, and cultural fit through multiple interview rounds. This section provides a comprehensive guide to preparing for and succeeding in PM interviews.

## PM Interview Types

### **Common Interview Formats**

**1. Product Design** ("How would you design X?")  
**2. Strategy** ("How would you grow Y?")  
**3. Analytical/Metrics** ("How would you measure success?")  
**4. Behavioral** ("Tell me about a time when...")  
**5. Technical** ("Explain how X works")  
**6. Execution** ("How would you prioritize?")  
**7. Case Study** (Analyzing data, making decisions)

Most PM interviews include 4-6 rounds covering these types.

## Product Design Questions

### **What They're Testing**

- User empathy
- Problem-solving process
- Creativity
- Structured thinking
- Trade-off navigation

### **Common Questions**

- "Design a product for X"
- "Improve Y product"
- "Design X for Y user segment"

**Examples**:
- Design a better alarm clock
- Improve Instagram Stories
- Design a product for blind users

### **The Framework (CIRCLES)**

**C - Comprehend the situation**  
**I - Identify the customer**  
**R - Report customer needs**  
**C - Cut through prioritization**  
**L - List solutions**  
**E - Evaluate trade-offs**  
**S - Summarize recommendation**

### **Step-by-Step Approach**

**1. Clarify (2-3 min)**:
- "When you say alarm clock, are we designing mobile app, physical device, or feature in existing product?"
- "Who's the target user?"
- "What's the goal: acquire users, increase engagement, monetize?"

**2. Define User (2 min)**:
- Identify user segments
- Pick one to focus on
- "Let's focus on professionals aged 25-40 who struggle to wake up on time"

**3. User Needs (3 min)**:
- List 5-7 user problems/needs
- "They need to: wake up on time, wake up gradually (not startled), snooze without oversleeping, track sleep quality"
- Pick top 2-3 to solve

**4. Solution (5 min)**:
- Brainstorm 3-5 solutions
- Pick best one
- "I'll design a smart alarm with: gradual light+sound, sleep cycle optimization, intelligent snooze"

**5. Trade-offs (2 min)**:
- Discuss pros/cons
- "Pros: Better wake experience, backed by sleep science. Cons: Complex to build, requires hardware integration"

**6. Success Metrics (2 min)**:
- "Measure success with: % of users who wake on first alarm, user satisfaction rating, daily active usage"

**Total: 15-20 minutes**

### **Example Answer**

**Q: "Design a better alarm clock"**

**Clarify**: "Are we designing an app, physical device, or feature? For this answer, I'll assume it's a mobile app. Target user?"

**User**: "Let's focus on professionals (25-40) who struggle to wake up consistently."

**Needs**:
1. Wake up on time (miss work is bad)
2. Wake up gradually (sudden alarm is jarring)
3. Not snooze too many times
4. Sleep better overall

**Solution**: Smart alarm app with:
- **Sleep cycle tracking**: Wakes you in light sleep phase
- **Gradual wake**: Light+sound intensity increases slowly
- **Intelligent snooze**: Limits snoozes to 2, adjusts alarm time
- **Sleep insights**: Track sleep quality, give tips

**Why this works**: Addresses core needs (wake on time, pleasant experience, better sleep)

**Trade-offs**: Complex to build (requires accurate sleep tracking), battery drain concern

**Metrics**: 
- % users wake on first alarm (currently 40%, target 70%)
- Morning satisfaction rating (target 4+/5)
- Daily active users (retention)

## Strategy Questions

### **What They're Testing**

- Business thinking
- Market understanding
- Strategic frameworks
- Prioritization

### **Common Questions**

- "How would you grow X?"
- "Should we enter Y market?"
- "How would you monetize Z?"

**Examples**:
- How would you grow Spotify?
- Should Google build a social network?
- How would you monetize YouTube?

### **Framework**

**1. Clarify goal** (1 min):
- "When you say grow, do you mean: users, revenue, engagement?"
- "Any constraints: geography, budget, timeline?"

**2. Understand current state** (2 min):
- "What's the current user base, revenue, key metrics?"
- (If you don't know, make reasonable assumptions)

**3. Identify growth levers** (5 min):
- Acquisition: How do we get more users?
- Activation: How do we get users to core value faster?
- Retention: How do we keep users coming back?
- Monetization: How do we increase revenue per user?
- **Pick one to focus on**

**4. Brainstorm tactics** (5 min):
- List 5-7 specific tactics
- Estimate impact and effort for each
- Prioritize top 2-3

**5. Success metrics** (2 min):
- How we measure success
- Set targets

**6. Risks & mitigations** (1 min):
- What could go wrong?
- How we'll address it

### **Example Answer**

**Q: "How would you grow Spotify?"**

**Clarify**: "Are we growing users or revenue? I'll focus on growing users."

**Current state assumptions**: 200M users, competing with Apple Music/YouTube Music

**Growth levers**: I'll focus on **Acquisition** (get more users)

**Tactics**:
1. **Viral loops**: "Share playlist" feature with network effects
2. **Free tier**: Improve free experience to attract users, then convert
3. **Partnerships**: Bundle with phone carriers, student discounts
4. **Content**: Exclusive podcasts (Joe Rogan-style deals)
5. **Geographic expansion**: Focus on India, Brazil (large untapped markets)

**Prioritization**: 
- **High impact, low effort**: Partnerships (carrier bundles reach millions fast)
- **High impact, high effort**: Geographic expansion
- **Medium impact, low effort**: Improve viral loops

**I'd prioritize**: Partnerships (carrier bundles) + improve viral loops

**Metrics**:
- New user acquisition: +20% MoM
- Viral coefficient: From 0.3 to 0.5 (each user brings 0.5 more)
- Conversion (free → paid): 10% → 15%

**Risks**: Cannibalization (free tier hurts paid). Mitigation: Feature differentiation.

## Analytical/Metrics Questions

### **What They're Testing**

- Data literacy
- Metrics understanding
- Problem diagnosis
- SQL knowledge (sometimes)

### **Common Questions**

- "How would you measure success for X?"
- "Metric Y dropped 10%, investigate"
- "Define metrics for new feature"

### **Framework**

**1. Clarify metric** (1 min):
- "When you say measure success, are we talking about user engagement, revenue, or something else?"

**2. Define success metrics** (3-5 min):
- **North Star Metric**: One key metric (e.g., DAU, revenue)
- **Supporting metrics**: 3-5 metrics that drive north star
- **Counter metrics**: What we DON'T want to hurt (e.g., user satisfaction)

**3. How to measure** (2-3 min):
- What data do we track?
- How do we calculate it?
- What tools do we use?

**4. Targets** (1-2 min):
- What's baseline?
- What's target?
- What's timeline?

### **Example Answer**

**Q: "How would you measure success for a new recommendation engine?"**

**North Star Metric**: User engagement (time spent on platform)

**Supporting Metrics**:
1. **Click-through rate**: % of recommendations clicked (target: 15% → 25%)
2. **Content discovery**: # of new content pieces consumed per user (target: +50%)
3. **Session length**: Average time per session (target: 20 min → 30 min)
4. **Return rate**: % of users who return next day (target: 40% → 50%)

**Counter Metrics** (things we DON'T want to hurt):
1. User satisfaction: NPS (keep >40)
2. Content diversity: Ensure we're not creating echo chambers

**How to measure**:
- Track clicks, views, session time in analytics (Amplitude/Mixpanel)
- A/B test: 50% get new recommendations, 50% get old
- Measure difference in metrics

**Timeline**: Ship V1 in 8 weeks, evaluate after 4 weeks of data

**Success criteria**: 
- If engagement +20% and NPS stable → Success, roll out to 100%
- If engagement +5% or NPS drops → Iterate
- If no change → Kill

## Behavioral Questions

### **What They're Testing**

- Past experience
- Leadership
- Conflict resolution
- Learning from failures
- Cultural fit

### **Common Questions**

- "Tell me about a time you failed"
- "Tell me about a conflict with engineering"
- "Tell me about your biggest product launch"
- "Why PM? Why this company?"

### **The STAR Framework**

**S - Situation**: Set the context  
**T - Task**: What were you trying to achieve?  
**A - Action**: What did you do? (Be specific)  
**R - Result**: What happened? (Quantify impact)

### **Preparing STAR Stories**

**Have 8-10 prepared stories covering**:
1. Success story (launched impactful product)
2. Failure story (what you learned)
3. Conflict story (resolved disagreement)
4. Leadership story (influenced without authority)
5. Data-driven decision (used analytics)
6. User research story (discovered insight)
7. Technical challenge (worked with engineering)
8. Prioritization story (made tough trade-off)

### **Example Answer**

**Q: "Tell me about a time you failed"**

**S (Situation)**: "At Company X, I was PM for onboarding feature. We had 60% drop-off after signup."

**T (Task)**: "I needed to increase activation rate to 40% → 60%."

**A (Action)**: "I designed a 5-step guided onboarding based on competitor research, without doing user testing. We spent 6 weeks building it."

**R (Result)**: "After launch, activation stayed at 40%—no improvement. Users found the guided flow annoying and skipped it."

**Learning**: 
1. "I should have done user testing before building (not after)"
2. "I assumed competitors were right, but our users were different"
3. "Now I always prototype and test with 5-10 users before engineering starts"

**Follow-up**: "Next iteration, I tested 3 concepts with users, found simpler approach worked better, activation increased to 55%."

## Technical Questions

### **What They're Testing**

- Technical fluency
- Can you work with engineers?
- Understanding of systems
- API/architecture knowledge

### **Common Questions**

- "How does Google Search work?"
- "Explain how Uber matches riders and drivers"
- "Design an API for X"
- "What happens when you type google.com?"

### **How to Approach**

**1. High-level architecture** (2-3 min):
- Draw boxes and arrows
- Explain major components
- Show data flow

**2. Deep dive into one part** (3-5 min):
- Pick one component
- Explain how it works technically

**3. Scale considerations** (2 min):
- How does it handle millions of users?
- What are bottlenecks?

**4. Trade-offs** (1-2 min):
- Different approaches and why we chose this one

### **Example Answer**

**Q: "How does Uber match riders and drivers?"**

**High-level**:
"When rider requests ride:
1. **Rider app** → sends location to **backend**
2. **Backend** queries **database** for nearby available drivers
3. **Matching algorithm** picks best driver (proximity, rating, etc.)
4. **Push notification** sent to **driver app**
5. Driver accepts → **real-time connection** established (WebSocket)
6. **GPS tracking** updates location every few seconds"

**Deep dive (matching algorithm)**:
"Matching considers:
- Distance (closest driver)
- Driver rating (prioritize 4.8+ rated)
- Estimated time to pickup
- Driver acceptance history

Algorithm uses **geospatial indexing** (like quadtrees) to quickly find drivers within radius."

**Scale**:
"To handle millions of requests:
- **Sharding database** by geography (SF database, NYC database)
- **Load balancing** across servers
- **Caching** driver locations (update every 5 sec, not real-time query)
- **Queueing system** for high demand (riders wait if no drivers)"

**Trade-offs**:
"Could match on price (riders pay more for faster pickup), but prioritized simplicity for V1."

## Preparation Strategy

### **3-Month Prep Plan**

**Month 1: Learn Frameworks**
- Week 1: Study product design framework (CIRCLES)
- Week 2: Study strategy frameworks
- Week 3: Study metrics frameworks
- Week 4: Prepare STAR stories

**Month 2: Practice**
- Practice 20 product design questions
- Practice 15 strategy questions
- Practice 10 metrics questions
- Write out answers, time yourself

**Month 3: Mock Interviews**
- Do 5-10 mock interviews with peers
- Record yourself, watch back
- Get feedback, iterate

**Resources**:
- Book: "Cracking the PM Interview" (Gayle McDowell)
- Website: exponent.com, IGotAnOffer
- Practice partner: Find PM buddy

### **Interview Day Tips**

**Before interview**:
- Research company (product, strategy, news)
- Review your stories (STAR)
- Arrive 10 min early
- Bring notebook + pen

**During interview**:
- Think out loud (show your process)
- Ask clarifying questions
- Be structured (use frameworks)
- Watch time (don't go over)
- Be enthusiastic (energy matters)

**After interview**:
- Send thank you email within 24 hours
- Personalize based on conversation
- Reiterate enthusiasm

### **Common Mistakes**

**Mistake #1: Jumping to solution too fast**
- Don't: Immediately answer "I'd build X"
- Do: Ask clarifying questions first

**Mistake #2: Vague answers**
- Don't: "I'd use data to make decisions"
- Do: "I'd run A/B test with 10K users for 2 weeks, measuring CTR and conversion"

**Mistake #3: Not structuring answers**
- Don't: Ramble unstructured
- Do: Use frameworks (CIRCLES, STAR)

**Mistake #4: Not tracking time**
- Don't: Spend 15 min on clarifying questions
- Do: Budget time (2 min clarify, 5 min solution, etc.)

**Mistake #5: Being too quiet**
- Don't: Think silently for 5 minutes
- Do: Think out loud, show your process

## Key Takeaways

1. **7 interview types**: Product design, strategy, analytical, behavioral, technical, execution, case study
2. **Use frameworks**: CIRCLES (design), STAR (behavioral)
3. **Practice 50+ questions**: Can't wing PM interviews
4. **Think out loud**: Show your process
5. **Ask clarifying questions**: Shows thoughtfulness
6. **Be structured**: Framework over rambling
7. **Prepare STAR stories**: 8-10 stories ready
8. **Do mock interviews**: Practice with peers
9. **Research company**: Know product, strategy, news
10. **Be enthusiastic**: Energy and passion matter

## Practical Exercise

**This week**:
1. Practice 3 product design questions (record yourself)
2. Write 3 STAR stories
3. Do 1 mock interview with peer

**This month**:
- Practice 20 questions across types
- Review recordings, identify gaps
- Schedule 5 mock interviews

**Remember**: PM interviews are skill-based. With practice, anyone can master them. Most candidates under-prepare—investing time here gives you huge advantage.

Good luck! 🚀
`;

export const pmInterviewPrepSection: ModuleSection = {
  id: 'pm-interview-prep',
  title: 'PM Interview Preparation',
  content,
};

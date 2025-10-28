/**
 * Discussion questions for Product Discovery vs Product Delivery
 * Product Management Fundamentals Module
 */

export const discoveryVsDeliveryQuiz = [
  {
    id: 1,
    question:
      "Your engineering manager is frustrated because your team has spent 3 weeks doing user research and prototyping for a new feature, and they say: 'We should be writing code by now, not doing endless research. Our competitors are shipping faster than us.' Meanwhile, your designer wants to do another round of user testing before finalizing the design. How do you handle this tension between discovery and delivery? When is it time to stop discovery and start delivery? Create a framework for making this decision.",
    answer: `## Comprehensive Answer:

This is a classic discovery-delivery tension that every PM faces. Let's systematically address it.

### **Understanding the Underlying Tension**

**Engineering Manager's perspective**:
- Values: Speed, shipping, visible progress
- Fear: Endless research, analysis paralysis, competitors winning
- Metric: Features shipped, velocity
- Valid concern: Discovery can become excuse for not shipping

**Designer's perspective**:
- Values: User validation, quality, getting it right
- Fear: Shipping something users won't use/like
- Metric: User satisfaction, usability
- Valid concern: Rushing leads to rework

**Your job as PM**: Balance both, make the call on when to transition.

---

### **Step 1: Diagnose the Situation**

**Ask yourself these questions**:

**Question 1: Have we validated the problem?**
- Do 10+ users confirm this problem exists and matters?
- Is it painful enough that they'd use a solution?
- Have we quantified the problem (how often, how severe)?

**If NO**: Continue discovery (problem validation)
**If YES**: Move to solution exploration

**Question 2: Have we tested our solution hypothesis?**
- Have users seen and reacted to our prototype?
- Did they understand how to use it?
- Did it solve their problem?
- Did we test with the right user segment?

**If NO**: One more round of prototype testing
**If YES**: Ready for delivery

**Question 3: What's the risk of being wrong?**
- **High risk** (core product change, affects all users, hard to reverse): More discovery
- **Low risk** (small feature, feature flag, easy to iterate): Less discovery, ship and learn

**Question 4: What's the opportunity cost of delay?**
- Is competitor launching similar feature? (Time-sensitive)
- Are users churning because they lack this? (Urgent)
- Is this strategic but not urgent? (Can do more discovery)

---

### **Step 2: The Discovery Exit Criteria Framework**

**Create explicit criteria BEFORE starting discovery**:

\`\`\`
Discovery Exit Criteria (Define Week 1):

Problem Validation:
✓ 10+ user interviews confirming problem
✓ Problem occurs weekly+ (frequency threshold)
✓ Users rate problem 7+/10 in severity

Solution Validation:
✓ 10 users tested interactive prototype
✓ 80%+ understood how to use it without help
✓ 70%+ said they'd use this if available
✓ Identified and addressed major usability issues

Feasibility:
✓ Engineering estimates < 8 weeks
✓ No major technical blockers
✓ Can ship MVP version in 4 weeks

Business Viability:
✓ Aligns with product strategy
✓ Expected to move key metric (predicted impact)
✓ Stakeholders aligned
\`\`\`

**If all criteria met → Start delivery**
**If criteria not met → Continue discovery OR kill feature**

---

### **Step 3: The Conversation with Engineering Manager**

**What I'd say**:

*"I hear your concern about speed, and I share it. Let me explain where we are and propose a path forward.*

**Where we are (Week 3 of discovery)**:
- ✓ Problem validated: 15 users confirmed pain point
- ✓ Solution hypothesis: Built interactive prototype
- ⚠️ Testing incomplete: Only 5 users tested prototype (need 10)
- ⚠️ One major usability issue surfaced in testing

**What we've learned**:
- The problem is real and painful (validated)
- Our first solution approach had a major flaw (good we caught it now!)
- We've pivoted prototype based on feedback
- Need one more round to validate the fix

**Trade-off analysis**:
- **Option A: Start coding now**
  - Pro: Ship in 6 weeks
  - Con: 50% chance we built wrong solution (based on early test signals)
  - Con: If wrong, waste 6 weeks of eng time + need to rebuild

- **Option B: One more week of discovery**
  - Pro: Validate solution with 10 users
  - Pro: 80% confidence we're building right thing
  - Con: Delay delivery by 1 week
  - Net: 1 week delay vs. potential 6 week waste

**My recommendation**: One more week of discovery (test with 10 users), then commit to delivery. This is the last round—we have clear exit criteria.

**What changes**:
- Week 4: Final prototype testing (10 users, specific questions)
- Week 5: Start delivery (no more discovery)
- Commit: If we hit exit criteria, we build. If we don't, we kill the feature.

**Why this is responsible**:
- We're not doing "endless research"—we have specific exit criteria
- One week of extra discovery dramatically reduces risk of 6-week waste
- We're being evidence-based, not opinion-based

*Does this address your concern about speed while ensuring we build the right thing?"*

---

### **Step 4: The Conversation with Designer**

**What I'd say**:

*"I appreciate your commitment to quality and user validation. Let's talk about the testing plan:*

**Current state**:
- 5 users tested, identified major issue
- We've revised prototype based on feedback
- Designer wants another round with 10 more users

**My proposal**:
- **This week**: Test with 10 users (your request)
- **Specific goals**: 
  1. Validate users can complete task without help (80%+ success rate)
  2. Confirm our fix addressed the usability issue
  3. Surface any other major problems

**Exit criteria for testing**:
- If 80%+ users succeed → We're done, start building
- If 50-80% succeed → One focused iteration on the issue, then build
- If <50% succeed → Rethink solution or kill feature

**What we're NOT doing**:
- Endless rounds of testing for perfection
- Testing with 50+ users
- Waiting for 100% user love

**Why this is the last round**:
- Diminishing returns after 10 user tests (Nielsen Norman Group: 5-8 users find 80% of issues)
- Real user feedback from production > more prototype testing
- We can iterate after launch (feature flag + progressive rollout)

**Trust in the process**:
- Your design is strong (based on early testing)
- We've incorporated feedback (you're doing great work)
- Real users will teach us more than extended prototype testing

*Can we commit to this timeline and then move to delivery?"*

---

### **Step 5: Decision Framework for "When to Stop Discovery"**

**Use this decision tree**:

\`\`\`
START: Are we in discovery mode?
  ↓
Question 1: Is the problem validated? (10+ users, painful, frequent)
  NO → Continue discovery (problem validation)
  YES ↓
Question 2: Have we tested a solution? (interactive prototype, 5-10 users)
  NO → Build prototype, test it
  YES ↓
Question 3: Did 70%+ users understand and want to use it?
  NO → If major issues: Iterate prototype, test again (max 1 more round)
       If fundamental issues: Kill feature
  YES ↓
Question 4: Is engineering feasibility confirmed? (< 8 weeks)
  NO → If too expensive: Scope down or kill feature
  YES ↓
Question 5: Have we been in discovery > 4 weeks?
  YES → STOP. Make decision: Ship MVP or kill feature
  NO ↓

DECISION: START DELIVERY
  - Write PRD
  - Begin engineering
  - No more user testing until after ship
\`\`\`

**The 4-week rule**: If you've been in discovery for 4 weeks and haven't hit exit criteria, either:
1. Ship an MVP and learn in production, OR
2. Kill the feature (it's not working)

**Never**: Spend 6+ weeks in discovery. That's analysis paralysis.

---

### **Step 6: Common Discovery Traps to Avoid**

**Trap #1: Perfectionism**
- Symptom: "Just one more round of testing"
- Reality: Diminishing returns after 10 tests
- Fix: Set exit criteria upfront, stick to them

**Trap #2: Lack of Exit Criteria**
- Symptom: "We'll know we're done when we're done"
- Reality: Research can go on forever without clear goals
- Fix: Define success metrics for discovery (% who can use it, % who want it)

**Trap #3: Avoiding the Decision**
- Symptom: Continuing research to delay difficult decision
- Reality: More research won't make tough calls easier
- Fix: Timebox discovery, make decision even with incomplete data

**Trap #4: Ignoring Opportunity Cost**
- Symptom: "We need to get this perfect"
- Reality: While perfecting Feature A, Feature B (more impactful) isn't getting built
- Fix: Ruthlessly prioritize—is this the highest-impact thing we could work on?

---

### **Step 7: The "Ship and Learn" Approach**

**After reasonable discovery, ship an MVP and learn from real usage**:

**Discovery (3-4 weeks)**:
- Validate problem
- Test prototype with 10 users
- Confirm basic usability

**Delivery (4-6 weeks)**:
- Build MVP (minimal viable product)
- Feature flag to 10% of users
- Measure actual usage, not just stated intent

**Iterate (ongoing)**:
- Weekly data review
- Monthly user interviews
- Continuous improvement

**Why this works**:
- Real usage data > prototype testing
- Users behave differently in production
- Faster learning cycles

**Example: Instagram Stories**

**Discovery** (2-3 weeks):
- Validated: Users want ephemeral sharing (Snapchat proof)
- Prototype tested with users
- Confirmed demand

**Delivery** (12 weeks):
- Built MVP Stories feature
- Launched to small % of users
- Iterated based on real usage

**Result**: Massive success, but learned and iterated after launch (not perfected before launch)

---

### **Step 8: Managing Stakeholder Expectations**

**Set expectations at the beginning of discovery**:

*"We're entering a 3-week discovery phase. Here's what that means:*

**What we'll do**:
- Validate the problem (user interviews)
- Explore solutions (prototypes)
- Test with users (usability)

**Exit criteria**:
- Problem validated by 10+ users
- Solution tested with 10+ users
- 70%+ users can use it and want it
- Engineering feasibility confirmed

**Timeline**:
- Week 1: Problem validation
- Week 2: Prototype and test
- Week 3: Iterate and validate
- Week 4: Decision to build or kill
- Week 5+: Delivery (if we proceed)

**Commitment**: We will not spend more than 4 weeks in discovery. At that point, we ship an MVP or kill the feature.

*Does everyone understand and agree to this process?"*

---

### **Step 9: Red Flags that Discovery Has Gone Too Long**

**You've been in discovery too long if**:

1. **Time**: > 4 weeks without decision
2. **User tests**: > 15 users tested (diminishing returns)
3. **Iterations**: > 3 prototype iterations (overthinking)
4. **Team morale**: Engineers disengaged, designers frustrated
5. **Business impact**: Competitors shipping similar feature
6. **Confidence**: Still not confident after 4 weeks (kill the feature)

**Action**: Make a decision NOW. Ship MVP or kill feature.

---

### **Step 10: The Final Decision**

**Based on the scenario (3 weeks in, 5 users tested, one major issue found)**:

**My decision**:

*"Here's what we're doing:*

**Week 4 (This week)**:
- Test revised prototype with 10 users
- Specific questions: Can they complete task? Do they want this?
- Designer leads this, PM observes 5 sessions

**Week 4 (End)**:
- Friday: Decision meeting (PM, EM, Designer)
- Review results against exit criteria
- Decision: Build, kill, or one focused iteration

**Week 5+ (If we build)**:
- Start engineering
- No more prototype testing
- Ship MVP in 4-6 weeks with feature flag
- Learn from real usage

**Commitment**:
- This is the last round of discovery
- We make a decision by end of Week 4
- No extending the timeline

**To Engineering**: I hear your concern about speed. One more week of validation dramatically reduces risk of wasting 6 weeks of your time on the wrong thing.

**To Design**: This is the last round of testing. Focus on critical usability issues, not perfection. We'll iterate after launch based on real user behavior.

*Everyone aligned?"*

---

### **Key Takeaways**

**1. Discovery needs exit criteria** (define upfront, stick to them)

**2. The right amount of discovery** varies by:
- Risk (high risk = more discovery)
- Complexity (complex = more discovery)
- Novelty (new problem space = more discovery)

**3. Common range**: 2-4 weeks for most features
- < 2 weeks: Probably not enough validation
- > 4 weeks: Diminishing returns, ship and learn

**4. Balance perspectives**:
- Engineering: Values speed, shipping
- Design: Values quality, validation
- PM: Balances both, makes the call

**5. "Ship and learn" beats "perfect then ship"**:
- Real usage teaches more than prototypes
- MVPs with feature flags enable safe learning
- Iteration after launch is expected

**6. Never**: Spend 6+ weeks in discovery without shipping

**The art of PM**: Knowing when to stop researching and start building. Too little discovery = wrong product. Too much discovery = analysis paralysis. Getting it right = validated learning + bias for action.
`,
  },
  {
    id: 2,
    question:
      "You're at a startup that just raised Series A funding. The CEO wants to 'move fast and break things' and is pressuring the team to ship features weekly. However, you notice that 70% of features shipped in the past quarter had < 10% adoption and several were later removed. The CEO says 'We learn by shipping, not by researching.' How would you introduce discovery practices to this team without being seen as slowing down? Design a discovery process that feels fast but ensures you're building the right things.",
    answer: `## Comprehensive Answer:

This is a common Series A challenge: the "ship first, ask questions later" culture meets the reality that most features fail. Let's systematically introduce discovery without appearing to slow down.

### **Understanding the CEO's Mindset**

**What the CEO values**:
- Speed (weekly shipping cadence)
- Action over analysis
- Learning by doing
- Not being "corporate" or bureaucratic

**What the CEO fears**:
- Slowing down = losing to competitors
- "Analysis paralysis"
- Becoming like big, slow companies
- Losing startup scrappiness

**Your challenge**: Introduce discipline without feeling like bureaucracy.

---

### **Step 1: Reframe Discovery as "Shipping Faster"**

**Don't say**: "We need to slow down and do more research"  
**Do say**: "We can ship winners more often if we validate earlier"

**The pitch to CEO**:

*"I've been analyzing our last quarter's launches:*
- *We shipped 12 features*
- *8 had < 10% adoption (70% failure rate)*
- *3 were removed (wasted engineering time)*
- *Only 1 meaningfully moved metrics*

**Cost analysis**:
- *Average feature: 2 weeks eng time*
- *8 failed features = 16 weeks wasted*
- *1 successful feature = 2 weeks*

**The insight**: We're spending 8x more time on failures than successes.

**What if we could**:
- *Validate features in 3 days (before 2 weeks of eng time)*
- *Kill bad ideas before coding*
- *Spend 16 weeks on 8 winners instead of 8 failures*
- *Ship MORE successful features with the same eng capacity*

**This isn't about slowing down—it's about shipping winners faster.**

*Can I show you a 4-week experiment?"*

---

### **Step 2: Design "Fast Discovery" Process**

**The 3-Day Discovery Sprint** (not 3 weeks!):

#### **Day 1: Problem Validation (4 hours)**

**Morning** (2 hours):
- Hypothesis: Write one-page doc
  - What problem are we solving?
  - Who has this problem?
  - How do they solve it today?
  - What's our solution hypothesis?

**Afternoon** (2 hours):
- Talk to 5 users (30 min each, can be phone/Zoom)
- Ask: "Tell me about the last time you tried to [do task]"
- Validate: Does the problem exist and matter?

**Output by end of Day 1**:
- ✓ Problem validated, OR
- ✗ Problem doesn't exist (kill feature, save 2 weeks)

---

#### **Day 2: Solution Validation (6 hours)**

**Morning** (3 hours):
- Build clickable prototype (Figma/Framer)
- Not full design—just clickable flows
- "Good enough to test" (not pixel-perfect)

**Afternoon** (3 hours):
- Test with 5 users
- Show prototype, ask them to complete task
- Watch where they struggle
- Listen for "I would use this" vs. "Interesting..."

**Output by end of Day 2**:
- ✓ Users understood and wanted it, OR
- ✗ Major usability issues (iterate prototype)
- ✗ Users don't want it (kill feature)

---

#### **Day 3: Decision Day (4 hours)**

**Morning** (2 hours):
- Synthesize learnings
- Engineering feasibility check (how hard to build?)
- Calculate expected impact

**Afternoon** (2 hours):
- Go/No-Go decision meeting
- Present: Problem + Solution + User validation + Feasibility
- Decide: Build, Kill, or Iterate

**Output by end of Day 3**:
- Clear decision: Build this or don't
- If build: PRD and kick off engineering Monday
- If don't: Saved 2 weeks, move to next idea

---

### **Key Insight: 3 Days Discovery < 2 Weeks Delivery**

**If feature is bad**:
- Old way: 2 weeks building + realize it's bad = 2 weeks wasted
- New way: 3 days discovering it's bad = 1.5 weeks saved

**If feature is good**:
- Old way: 2 weeks building
- New way: 3 days validation + 2 weeks building = 2.3 weeks total
- Trade-off: 3 extra days for much higher hit rate

**Net result**: Ship more winners with same eng capacity.

---

### **Step 3: Make Discovery Visible**

**Create "Discovery Dashboard"** (CEO loves dashboards):

\`\`\`
Feature Pipeline Dashboard

Discovery (This Week):
- Feature A: Day 2 - Prototype testing
- Feature B: Day 3 - Go/No-Go decision Friday
- Feature C: Day 1 - User interviews

Delivery (In Progress):
- Feature D: 50% complete (validated in discovery)
- Feature E: 80% complete (shipped next week)

Killed This Month:
- Feature F: Users didn't have problem (saved 2 weeks)
- Feature G: Too complex to build (saved 8 weeks)
- Feature H: Users couldn't figure out prototype (saved 2 weeks)
- Total Time Saved: 12 weeks

Features Shipped This Month:
- Feature I: 35% adoption (predicted 30%) ✓
- Feature J: 18% adoption (predicted 20%) ✓
- Hit rate: 100% (vs. 30% last quarter)
\`\`\`

**Metrics that matter to CEO**:
- **Time saved** (killed features before building)
- **Hit rate** (% of shipped features that work)
- **Velocity** (still shipping regularly)

---

### **Step 4: Run a 4-Week Pilot**

**Pitch to CEO**:

*"Give me 4 weeks to experiment with this 3-day discovery process. If it slows us down or doesn't improve hit rate, we'll stop. If it helps us ship more winners, we'll keep it.*

**Weeks 1-4: Run the experiment**

- Week 1: Discovery sprint for 2 features → Kill 1, Build 1
- Week 2: Deliver validated feature from Week 1
- Week 3: Discovery sprint for 2 more features → Kill 1, Build 1
- Week 4: Results review

**Expected outcomes**:
- Same shipping cadence (2 features/month)
- Higher hit rate (60-80% vs. 30%)
- Less engineering waste

*Can we try this?"*

---

### **Step 5: Address "We Learn by Shipping" Objection**

**CEO's point has merit**: Real usage teaches more than research.

**Your counter**: "Agree! Let's ship smarter"

**The "Fast Feedback" Model**:

\`\`\`
3 Days Discovery
    ↓
2 Weeks Building MVP
    ↓
Ship to 10% users (Day 1)
    ↓
Measure and learn (Week 1)
    ↓
Iterate or expand (Week 2-3)
\`\`\`

**Key insight**: We're still shipping fast AND learning from real users. We just validate direction first.

**Example**:

**Old way**:
- Week 1-2: Build Feature X (full version)
- Week 3: Ship to 100%
- Week 4: Realize no one uses it
- Result: 3 weeks wasted

**New way**:
- Day 1-3: Discovery (validate with users)
- Week 1-2: Build MVP
- Week 3: Ship to 10% (learn from real usage)
- Week 4: Iterate based on data
- Result: Real usage data after 3 weeks, smarter iteration

**Still learning by shipping—just with validated direction first.**

---

### **Step 6: Cultural Shift (Gentle)**

**Don't**: Create "mandatory discovery process" that feels bureaucratic  
**Do**: Make discovery a competitive advantage

**Reframe discovery as**:
- "Customer obsession" (not research)
- "de-risking" (not slowing down)
- "shipping winners" (not perfectionism)

**Language changes**:

| Old (sounds slow) | New (sounds fast) |
|-------------------|-------------------|
| "We need to do research" | "Let's talk to 5 users this afternoon" |
| "We need more data" | "3-day validation sprint" |
| "Let's not rush this" | "Let's ship the right thing fast" |
| "Discovery phase" | "Quick validation before building" |

**Make heroes of PMs who kill bad features early**:
- "Sarah saved us 4 weeks by validating before building"
- "This feature would've failed—good catch in discovery"

---

### **Step 7: The Lean Startup Approach**

**Leverage CEO's love of "Lean Startup" methodology**:

*"Eric Ries (Lean Startup) says: 'The goal is not to ship. The goal is to learn.' We can learn faster by validating cheaply first."*

**Build-Measure-Learn Loop**:

**Traditional loop** (slow learning):
1. Build full feature (2 weeks)
2. Ship and measure (1 week)
3. Learn it doesn't work (0 weeks, it's done)
4. Total: 3 weeks to learn

**Fast loop** (fast learning):
1. Prototype and test (3 days)
2. Learn and iterate (1 day)
3. Build validated feature (2 weeks)
4. Total: 2.5 weeks to ship winner

**Faster learning = faster winning.**

---

### **Step 8: Success Stories to Share**

**Find internal examples of discovery wins**:

**Example 1: The feature you almost built**
- "We were about to build [Feature X]"
- "Talked to 5 users, learned they already had a workaround"
- "Killed feature, saved 2 weeks"
- "Built [Feature Y] instead, 40% adoption"

**Example 2: The pivot that worked**
- "Original idea was [X]"
- "User testing revealed different need"
- "Pivoted to [Y] in Day 2 of discovery"
- "Shipped [Y], huge success"
- "Without discovery, would've built [X] and wasted time"

**Share these stories in all-hands, Slack, standups.**

---

### **Step 9: Metrics to Track**

**Show CEO data proving discovery works**:

**Track these metrics**:

1. **Feature Hit Rate**:
   - % of shipped features with >20% adoption
   - Before discovery: 30%
   - After discovery: 70%

2. **Engineering Efficiency**:
   - Weeks spent on successful features
   - Before: 30% of time on winners
   - After: 70% of time on winners

3. **Time Saved**:
   - Weeks saved by killing bad features early
   - Track monthly

4. **Shipping Velocity**:
   - Features shipped per month (should stay same or increase)
   - Prove: Discovery doesn't slow us down

**Present monthly to CEO**: "Discovery ROI Report"

---

### **Step 10: The Gradual Rollout**

**Don't**: Force all teams to adopt discovery immediately  
**Do**: Pilot with one team, showcase results, spread organically

**Month 1**: Pilot with your team
**Month 2**: Share results, invite another team to try
**Month 3**: Three teams using discovery
**Month 4**: CEO asks why other teams aren't doing it

**Let success speak for itself.**

---

### **The Final Pitch to CEO**

*"I love our 'move fast' culture. I don't want to slow us down—I want to ship more winners.*

**Here's what I'm proposing**:
- 3-day discovery sprint before building
- Still ship features regularly
- Higher hit rate (more winners, fewer failures)
- Same or better velocity

**Give me 4 weeks to prove it**:
- Run discovery on next 4 features
- Track hit rate vs. last quarter
- Measure time saved from killed features

**If it slows us down or doesn't improve results, we stop.**

**If it helps us ship more winners with the same eng capacity, we keep it.**

*Can I run this experiment?"*

---

### **Key Takeaways**

**1. Reframe discovery as "shipping winners faster"** (not slowing down)

**2. Make discovery fast** (3 days, not 3 weeks)

**3. Speak CEO's language**:
- "Velocity" not "quality"
- "De-risking" not "research"
- "Customer obsession" not "user interviews"

**4. Show data**:
- Features killed (time saved)
- Hit rate improvement
- Engineering efficiency

**5. Run experiments, not mandates**:
- 4-week pilot
- Prove value
- Spread organically

**6. "Ship and learn" is right—just validate direction first**

**7. Make discovery feel like startup behavior** (scrappy, fast, user-focused)

**The ultimate goal**: Build a culture where discovery feels fast and shipping without validation feels reckless.

**Success = CEO saying**: "How did we ever ship without validating first? This is how we move fast without breaking things that matter."
`,
  },
  {
    id: 3,
    question:
      "Explain dual-track Agile to a team that's never used it. Your engineering team is used to waterfall (3-month planning cycles) and your designers want more time for exploration before features are 'locked in.' Show specifically how dual-track Agile would work for your team: what each week looks like, who's involved, what artifacts are created, and how you prevent discovery from becoming a bottleneck. Include a 6-week example timeline with 3 features.",
    answer: `## Comprehensive Answer:

Let me introduce dual-track Agile step-by-step with a concrete example.

### **What is Dual-Track Agile?**

**Simple definition**: Run discovery and delivery in parallel, but for different features.

**Visual**:
\`\`\`
Track 1 (Discovery):  Feature A → Feature B → Feature C → Feature D
Track 2 (Delivery):             Feature A → Feature B → Feature C
\`\`\`

**Key insight**: Discovery happens 2-4 weeks ahead of delivery.

**Benefits**:
1. **For engineering**: Always have validated features ready to build
2. **For design**: Time to explore before features are "locked in"
3. **For product**: Continuous validation without blocking delivery

---

### **The Two Tracks Explained**

#### **Track 1: Discovery Track**

**Team**: PM + Designer + 1-2 Engineers (part-time)

**Goal**: Validate what to build

**Activities**:
- User research (interviews, observations)
- Problem validation
- Prototyping (low-fi → high-fi)
- User testing (prototype validation)
- Feasibility assessment (technical spikes)

**Output**: Validated feature ready for delivery track

**Duration**: 2-4 weeks per feature

---

#### **Track 2: Delivery Track**

**Team**: PM + Full Engineering Team + Designer (implementation support)

**Goal**: Build and ship

**Activities**:
- Requirements definition (PRD)
- Engineering implementation
- Design handoff and QA
- Testing and bug fixes
- Release and monitoring

**Output**: Shipped feature

**Duration**: 2-4 weeks per feature (2-week sprints typical)

---

### **How the Tracks Work Together**

**The Flow**:
1. Discovery validates Feature A (Weeks 1-2)
2. Feature A enters delivery queue
3. Engineering builds Feature A while discovery works on Feature B
4. Feature A ships, Feature B enters delivery
5. Continuous cycle

**Key principle**: Features only enter delivery after discovery validates them.

---

### **6-Week Example Timeline (3 Features)**

Let's walk through a real example:

**Features**:
- **Feature A**: In-app notifications (already in discovery)
- **Feature B**: User profile customization (starting discovery)
- **Feature C**: Social sharing (not started yet)

---

#### **Week 1-2: Getting Started**

**Discovery Track (Feature A - weeks 3-4 of discovery)**:
- Feature A is finishing discovery validation
- **Week 1**:
  - PM + Designer: Final prototype testing (5 users)
  - 1 Engineer: Technical feasibility spike (can we do push notifications?)
  - Wednesday: Review findings
- **Week 2**:
  - Synthesize research
  - Engineering estimates effort (3 weeks)
  - Write PRD draft
  - **Friday**: Discovery review meeting → Feature A APPROVED for delivery

**Delivery Track (Starting sprint)**:
- No features ready yet (this is first sprint with dual-track)
- Engineering: Tech debt, refactoring, infrastructure
- **Or**: Engineering helps with Feature A discovery (tech spike)

**Artifacts created**:
- Feature A: Research summary (10 user tests, 80% usability success)
- Feature A: PRD draft
- Feature A: Engineering estimate (3 weeks)
- Feature A: Go decision (validated)

---

#### **Week 3-4: Full Dual-Track Mode**

**Discovery Track (Feature B)**:
- **Week 3**:
  - Monday: Kick off Feature B discovery
  - PM: User interviews (5 users about profile customization)
  - Designer: Explore UI concepts (3 different approaches)
  - Engineer: Research technical constraints (user data model)
  
- **Week 4**:
  - Designer: Build interactive prototype (one chosen approach)
  - PM: Test prototype with 5 users
  - Engineer: Feasibility assessment
  - **Friday**: Mid-discovery review (findings so far)

**Delivery Track (Feature A - Sprint 1)**:
- **Week 3**:
  - Monday: Sprint planning (Feature A PRD review)
  - Engineering: Start implementation (backend notification system)
  - Designer: Create final UI designs (implementation-ready)
  - Daily standups (15 min)
  
- **Week 4**:
  - Engineering: Continue implementation
  - Mid-week: Design review (UI adjustments)
  - QA: Test notification system
  - **Friday**: Sprint review (demo progress)

**Key Activities by Day (Week 3 example)**:

**Monday**:
- 9 AM: Sprint planning (delivery track - Feature A)
- 11 AM: Discovery kickoff (discovery track - Feature B)

**Tuesday**:
- PM: User interviews for Feature B (discovery)
- Engineering: Building Feature A (delivery)

**Wednesday**:
- Designer: UI explorations for Feature B (discovery)
- Designer: Design review for Feature A implementation (delivery)

**Thursday**:
- PM: More user interviews for Feature B (discovery)
- Engineering: Feature A implementation (delivery)
- Engineer helping discovery: Technical spike for Feature B

**Friday**:
- 10 AM: Sprint review for Feature A (delivery track)
- 2 PM: Discovery review for Feature B (discovery track)
- 4 PM: Sprint retro (delivery track)

**People's Time Allocation (Week 3)**:

**PM**:
- 40% - Feature B discovery (interviews, synthesis)
- 40% - Feature A delivery support (PRD questions, priorities)
- 20% - Planning Feature C, stakeholder updates

**Designer**:
- 60% - Feature B discovery (UI exploration)
- 30% - Feature A delivery support (design reviews, adjustments)
- 10% - Design system maintenance

**Lead Engineer** (helping discovery):
- 20% - Feature B discovery (technical spike)
- 80% - Feature A delivery (implementation)

**Rest of Engineering**:
- 100% - Feature A delivery

---

#### **Week 5-6: Continuous Flow**

**Discovery Track (Feature C)**:
- **Week 5**:
  - Feature B discovery complete, approved for delivery
  - Start Feature C discovery
  - PM: User interviews for social sharing
  
- **Week 6**:
  - Continue Feature C discovery
  - Build prototype, test with users

**Delivery Track (Feature A → Feature B transition)**:
- **Week 5**:
  - Feature A: Final testing, bug fixes
  - **Tuesday**: Feature A ships (feature flag to 10% users)
  - **Wednesday**: Sprint planning for Feature B
  - **Thursday-Friday**: Start Feature B implementation
  
- **Week 6**:
  - Feature B: Sprint 1 implementation
  - Feature A: Monitor adoption, gather feedback
  - Feature A: Quick iteration if needed

---

### **Weekly Cadence Details**

**Monday**:
- **9-10 AM**: Sprint planning (delivery track)
  - Review PRD for feature entering delivery
  - Break down into tasks
  - Commit to sprint goals
  
- **11-12 AM**: Discovery planning (discovery track)
  - If starting new feature: Kickoff meeting
  - If mid-discovery: Weekly sync (findings so far)
  
**Tuesday-Thursday**:
- **9:15 AM**: Daily standup (delivery track)
  - What did I ship yesterday?
  - What am I shipping today?
  - Any blockers?
  
- **Throughout day**:
  - Discovery team: User research, prototyping
  - Delivery team: Implementation
  - PM/Designer: Split time between both tracks

**Friday**:
- **10-11 AM**: Sprint review (delivery track)
  - Demo what we shipped
  - Stakeholder feedback
  
- **2-3 PM**: Discovery review (discovery track)
  - PM presents: What we learned this week
  - Show: Prototype or research findings
  - Discuss: Is this ready for delivery?
  
- **4-5 PM**: Retro (delivery track)
  - What went well?
  - What can improve?
  - Action items for next sprint

---

### **Artifacts and Documentation**

**Discovery Track Artifacts**:

1. **Discovery Brief** (Week 1, Day 1)
   - Problem hypothesis
   - User segment
   - Research plan
   - Success criteria

2. **Research Summary** (Week 2)
   - User interview insights (10+ users)
   - Key problems identified
   - Quotes and patterns

3. **Prototype** (Week 2-3)
   - Lo-fi → Hi-fi progression
   - Interactive (Figma, Framer)
   - Test-ready

4. **Test Results** (Week 3)
   - Usability metrics (% who completed task)
   - User feedback
   - Issues identified

5. **PRD Draft** (Week 3-4)
   - Problem statement
   - Solution overview
   - User stories
   - Success metrics
   - Engineering estimate

6. **Go/No-Go Decision** (Week 4)
   - Recommendation: Build / Don't Build
   - Supporting evidence
   - Next steps

**Delivery Track Artifacts**:

1. **Final PRD** (Sprint Day 1)
   - Refined from discovery
   - Detailed requirements
   - Acceptance criteria

2. **Design Files** (Sprint Week 1)
   - Implementation-ready designs
   - Component specifications
   - Asset handoff

3. **Sprint Board** (Ongoing)
   - User stories in Jira/Linear
   - Progress tracking
   - Burndown chart

4. **Sprint Demo** (End of sprint)
   - Working software demo
   - Stakeholder feedback captured

5. **Release Notes** (Ship day)
   - What changed
   - How to use
   - Known issues

---

### **Preventing Discovery Bottlenecks**

**Problem**: Discovery could slow down delivery if not managed well.

**Solutions**:

**1. Maintain a "Validated Backlog"**
- Goal: Always have 2-3 features validated and ready for delivery
- If backlog < 2 features → Speed up discovery
- If backlog > 4 features → Slow down discovery, focus on delivery

**2. Timebox Discovery**
- Maximum 4 weeks per feature
- Week 4: Make decision (build/kill) even with incomplete data
- Prevents endless research

**3. Parallel Discovery**
- Run discovery on 2 features simultaneously if needed
- Different PM or PM + Researcher

**4. "Fast Fail" in Discovery**
- If Week 1 research shows problem doesn't exist → Kill feature immediately
- Don't spend 4 weeks on dead ends

**5. Engineering Involvement in Discovery**
- 1-2 engineers help with technical spikes (20% time)
- Prevents "we can't build this" surprises later
- Engineers understand context before building

---

### **How This Solves Your Team's Problems**

**For Engineering (Used to Waterfall)**:

**Old pain**:
- 3-month planning cycle
- Build features that don't work
- Requirements change mid-sprint
- Unclear priorities

**Dual-track solution**:
- 2-week sprints (faster feedback)
- Only build validated features
- Requirements stable (discovery validated them)
- Clear priorities (from validated backlog)

**For Designers (Want More Exploration Time)**:

**Old pain**:
- Not enough time to explore before "locked in"
- Designs rushed
- Engineering starts before design validated

**Dual-track solution**:
- 2-4 weeks of exploration in discovery
- Test prototypes before finalizing
- Designs validated before engineering starts
- Time to iterate based on user feedback

**For Product (You)**:

**Old pain**:
- Waterfall too slow
- Features fail often
- Hard to balance research and shipping

**Dual-track solution**:
- Continuous discovery AND delivery
- Higher success rate (validated features)
- Never blocked (always have next feature ready)

---

### **Transition Plan: Moving from Waterfall to Dual-Track**

**Month 1: Pilot**
- Start with one team
- Run dual-track for 6 weeks
- Document results

**Month 2: Refine**
- Adjust cadence based on learnings
- Improve ceremonies
- Get team feedback

**Month 3: Scale**
- Roll out to other teams
- Share success stories
- Train new teams

---

### **Common Questions**

**Q: "What if discovery takes longer than 4 weeks?"**
A: Make a decision anyway. Ship MVP and learn in production, or kill the feature. Don't let discovery drag on.

**Q: "What if engineering finishes a feature before next feature is validated?"**
A: Have a validated backlog of 2-3 features. If we burn through it, engineers work on tech debt, infrastructure, or help with discovery.

**Q: "What if we discover the feature won't work after engineering starts?"**
A: Rare if discovery is thorough. If it happens: stop, pivot, or kill. Sunk cost fallacy is worse than admitting mistake.

**Q: "How do we handle urgent features?"**
A: Define "urgent" criteria. True emergencies skip discovery (but are rare). Most "urgent" features can wait 3 days for quick validation.

---

### **Success Metrics**

**Track these to prove dual-track works**:

1. **Feature Success Rate**: % of shipped features with >20% adoption
   - Target: 70%+ (vs. <30% with waterfall)

2. **Engineering Efficiency**: % of eng time on successful features
   - Target: 70%+ (vs. 30% with waterfall)

3. **Shipping Velocity**: Features shipped per quarter
   - Target: Same or better (dual-track doesn't slow down)

4. **Team Satisfaction**: Survey scores
   - Engineers: Feel less wasted effort
   - Designers: Feel designs are validated
   - PM: Feel more confident in decisions

---

### **Key Takeaways**

1. **Dual-track = Discovery and Delivery in parallel** (for different features)
2. **Discovery runs 2-4 weeks ahead of delivery**
3. **Only validated features enter delivery**
4. **Prevents**: Building wrong things, designer rushing, unclear requirements
5. **Enables**: Continuous discovery, stable sprints, higher success rate
6. **Not slower**: Same velocity, fewer failures, more winners

**The result**: Engineering builds validated features, designers have time to explore and validate, PM balances both tracks. Everyone wins.
`,
  },
];

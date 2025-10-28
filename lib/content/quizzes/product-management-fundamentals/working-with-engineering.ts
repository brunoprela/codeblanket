/**
 * Discussion questions for Working with Engineering
 * Product Management Fundamentals Module
 */

export const workingWithEngineeringQuiz = [
  {
    id: 1,
    question:
      "Your engineering team built exactly what you specified in the PRD, but users hate the feature. The engineering manager says 'We built what you asked for—this is a PM problem, not an engineering problem.' However, you feel frustrated because engineers didn't raise concerns about usability issues they must have noticed while building. How do you address this situation? What changes would you make to prevent this in the future? How do you balance PM ownership with engineering partnership?",
    answer: `## Comprehensive Answer:

This scenario reveals a common problem: engineers built to spec but didn't feel ownership or empowerment to raise concerns. Let's address it systematically.

### **Step 1: Take Ownership (Immediately)**

**What NOT to say**:
- "Why didn't you tell me this wouldn't work?"
- "Engineers should have raised concerns"
- "This is on both of us"

**What TO say**:
- "You're right—I specified the solution, and it didn't work. That's on me as the PM. I should have validated better before we built it."

**Why**: Taking full ownership builds trust. Deflecting blame erodes the relationship.

### **Step 2: Root Cause Analysis (Private Reflection)**

**Ask yourself honestly**:

**1. Did I write a prescriptive PRD?**
- Did I specify the exact solution instead of the problem?
- Did I leave room for engineering input?
- Did I ask for their ideas?

**2. Did I involve engineering in discovery?**
- Were engineers in user interviews?
- Did they see the problem firsthand?
- Did they test the prototype?

**3. Did I create psychological safety?**
- Do engineers feel comfortable challenging my ideas?
- Have I reacted poorly to pushback in the past?
- Do they feel like "order takers" or "partners"?

**4. Was the relationship already broken?**
- Is there trust between PM and engineering?
- Are there previous incidents of blame?
- Is this symptomatic of larger issues?

**Likely root causes**:
- **PM prescribed solution** (didn't give space for input)
- **Engineers weren't involved in discovery** (no user context)
- **Lack of psychological safety** (engineers afraid to speak up)
- **Order-taker dynamic** (engineers build to spec, don't question)

### **Step 3: The Conversation with Engineering**

**Setting**: 1-on-1 with EM, then team discussion

**What to say**:

*"I want to talk about the [feature name] launch and what we can learn.*

**What I got wrong**:
- I wrote a prescriptive PRD without involving you in the solution
- I didn't invite anyone to user interviews to see the problem firsthand
- I should have done more prototype testing before we built
- I may have created an environment where you felt like you couldn't push back

**What I'm curious about**:
- Did anyone on the team have concerns while building this?
- If so, what stopped you from raising them?
- How can I make it easier for you to flag concerns early?

**What I want to change going forward**:
- Involve 1-2 engineers in discovery (user interviews, prototype testing)
- Write problem-focused PRDs, not solution prescriptions
- Hold solution design sessions where you propose technical approaches
- Create space for you to challenge my ideas

**I need your help**: 
- If you see something that doesn't make sense, please tell me
- If you have a better idea, please share it
- I want us to be partners who ship great products together

*What would make it easier for you to raise concerns?"*

### **Step 4: Changes to Prevent This**

**Change #1: Include Engineering in Discovery**

**Before** (wrong way):
- PM does all user research alone
- PM designs solution alone
- PM hands finished PRD to engineering
- Engineering builds without context

**After** (right way):
- Invite 1-2 engineers to user interviews
- Show engineers user research insights
- Brainstorm solutions together
- Engineers understand WHY before building

**Example**:
*"Sarah (engineer), I'm doing 5 user interviews about our checkout flow this week. Can you join 2 of them? I think seeing users struggle will give you great context for what we'll build next sprint."*

**Change #2: Write Problem-Focused PRDs**

**Before** (prescriptive PRD):
\`\`\`
Technical Requirements:
            - Add "Save for Later" button(blue, 16px font)
                - Store in localStorage
                - Display in sidebar
                - Show notification on save
\`\`\`

**After** (problem-focused PRD):
\`\`\`
Problem: Users abandon cart when they're not ready to buy
    - 40 % of users leave without completing purchase
    - User feedback: "I want to think about it but don't want to lose items"

User Need: "Save my cart so I can come back later"

Success Metrics:
    - 20 % of users who save cart return within 7 days
        - Saved cart conversion rate > 50 %

            Open Questions for Engineering:
                - Best way to persist cart(localStorage, database, both) ?
                    - How to handle logged out vs logged in users ?
                        - How to handle item availability changes ?
                            \`\`\`

**Change #3: Solution Design Sessions**

**New ritual**: Before sprint starts, hold 60-min collaborative design session

**Attendees**: PM + EM + 2-3 engineers + designer

**Agenda**:
1. PM shares problem and user research (15 min)
2. Designer shares design concepts (15 min)  
3. Engineers propose technical approaches (20 min)
4. Discuss trade-offs together (10 min)

**Outcome**: Collaborative decision, engineers have ownership

**Change #4: Create Psychological Safety**

**Actions**:

**1. Explicitly invite pushback**:
- "I want to hear concerns early, not after we ship"
- "Challenge my ideas—I might be wrong"
- "If something doesn't make sense, speak up"

**2. React well to criticism**:
- Engineer: "This approach won't work because X"
- PM: "Thank you for catching that. Help me understand X better."
- Not: "Why didn't you say this earlier?" (punishes speaking up)

**3. Celebrate good pushback**:
- "Alex saved us 2 weeks by catching this in design review"
- "Sarah's technical insight led to a better approach"
- Make heroes of people who speak up

**4. Make it easy to give feedback**:
- Office hours: "Come talk to me about any concerns"
- Slack channel: #product-feedback
- Anonymous feedback: Survey or suggestion box

**Change #5: Prototype Testing Together**

**New process**: Before writing code, test interactive prototype

**Step 1**: Designer creates Figma prototype
**Step 2**: PM + 2 engineers test with 5 users
**Step 3**: Everyone watches users struggle (or succeed)
**Step 4**: Discuss learnings, adjust design
**Step 5**: Only then write PRD and build

**Why this works**: Engineers see problems firsthand, feel ownership

### **Step 5: Balancing PM Ownership with Engineering Partnership**

**The Paradox**: PMs are accountable for outcomes but need engineering partnership.

**Resolution**: PM owns decisions, but makes them collaboratively.

**PM Owns** (accountability):
- Final call on WHAT to build
- Final call on WHEN to ship
- Responsibility for outcomes (success or failure)

**Engineering Owns** (accountability):
- Final call on HOW to build
- Final call on technical architecture
- Responsibility for technical quality

**Shared** (collaboration):
- Scope negotiation
- Trade-offs (time vs quality vs features)
- Problem-solving

**Example of balanced partnership**:

**Scenario**: Engineers say feature will take 6 weeks, PM needs it in 4 weeks

**Bad PM response** (too controlling):
- "We need it in 4 weeks. Figure it out."

**Bad PM response** (too deferential):
- "Okay, we'll wait 6 weeks" (without exploring options)

**Good PM response** (partnership):
- "Help me understand the 6-week estimate"
- "What drives the complexity?"
- "If we cut features X and Y, how long?"
- "What's an MVP we could ship in 4 weeks?"
- "Can we ship basic version in 4 weeks, full version in 6 weeks?"

**Outcome**: 
- Engineers explain: Core feature is 3 weeks, advanced features are 3 weeks
- PM decides: Ship core in 4 weeks (basic feature in 3 weeks + 1 week buffer)
- Engineers agree: That's achievable
- PM owns the trade-off, but engineers shaped the solution

### **Step 6: Long-Term Relationship Building**

**This incident is a turning point**:

**Option A**: Blame engineers → Trust erodes further
**Option B**: Fix the process → Trust rebuilds

**Actions over next 3 months**:

**Month 1**: Include engineers in discovery (2-3 interviews)
**Month 2**: Hold solution design sessions before sprints
**Month 3**: Ship feature collaboratively designed → Success!

**Result**: Engineers say "PM finally gets it" and relationship improves

### **Key Takeaways**

**1. PM owns outcomes** (including failures), even if engineers built to spec

**2. "We built what you asked" signals broken partnership**
- Engineers feel like order-takers
- No shared ownership
- Communication problems

**3. Fix root cause** (process), not symptom (this feature)
- Involve engineering in discovery
- Write problem-focused PRDs
- Create psychological safety
- Collaborative solution design

**4. Take full responsibility publicly**
- "I specified the wrong solution"
- Builds trust through humility

**5. Create partnership through process**
- Engineers aren't contractors executing specs
- Engineers are problem-solvers with user context
- Best solutions emerge from collaboration

**6. Balance accountability with collaboration**
- PM accountable for WHAT and WHEN
- Engineering accountable for HOW
- Both collaborate on solutions

**The ultimate goal**: Engineers say "We shipped this together" not "We built what PM asked for."

**Healthy partnership**: When feature succeeds, engineering gets credit. When it fails, PM takes responsibility. This asymmetry builds trust.
`,
  },
  {
    id: 2,
    question:
      "Your engineering team wants to spend 6 weeks refactoring the payment system before adding new payment methods. Sales is pressuring you to add Apple Pay and Google Pay immediately because we're losing deals. The CTO supports engineering. The VP Sales supports rushing. You're caught in the middle. How do you make this decision? Create a framework for deciding between technical debt paydown and new features.",
    answer: `## Comprehensive Answer:

This is a classic PM dilemma: technical debt vs. new features. Let's approach it systematically.

### **Step 1: Understand Both Perspectives**

**Engineering's perspective** (refactor first):
- Current payment code is "fragile"
- Adding features on broken foundation creates more debt
- Future velocity will suffer
- Technical incidents likely (payment failures = revenue loss)

**Sales's perspective** (ship new features now):
- Losing deals today
- Competitors have Apple Pay/Google Pay
- Every week of delay = lost revenue
- Customers asking for this

**Both are right**: Engineering protects long-term velocity, Sales protects revenue.

### **Step 2: Quantify the Trade-offs**

**Ask Engineering**:

**Option A: Add Apple Pay/Google Pay now (no refactor)**
- How long? (Estimate: 3 weeks)
- What risks? (Payment failures? Future slowdown?)
- What technical debt created? (Quantify: X weeks of future work)
- What's worst case scenario? (System breaks? Revenue loss?)

**Option B: Refactor first, then add new payment methods**
- Refactor time? (6 weeks)
- Add features after refactor? (2 weeks, because clean code)
- Total: 8 weeks
- Benefits? (Faster future development, fewer incidents)

**Option C: Minimal refactor + new features**
- Critical refactor only? (2 weeks)
- Add new payment methods? (3 weeks on somewhat cleaner code)
- Total: 5 weeks
- Trade-offs? (Some debt remains, but manageable)

**Ask Sales**:

**Revenue impact**:
- How many deals are we losing per week?
- What's the $ value?
- Are we actually losing deals or just hearing requests?
- How many customers actively use Apple Pay/Google Pay?

**Example math**:
- 2 deals lost per week = $50K/month in lost revenue
- 8 week delay (Option B) = $100K lost revenue
- But: If payment system breaks, we lose ALL revenue

### **Step 3: Decision Framework**

**The key questions**:

**1. How critical is the technical risk?**
- Low risk (code ugly but stable) → Ship features first
- Medium risk (occasional issues) → Middle ground (Option C)
- High risk (system at breaking point) → Refactor first (Option B)

**2. How urgent is the business need?**
- Not urgent (nice-to-have) → Refactor first
- Urgent (losing deals daily) → Find middle ground
- Critical (existential threat) → Ship features now

**3. What's the long-term cost?**
- If we ship now, does it cost 2x the time later?
- If we refactor first, do we lose the market window?

**4. Can we de-risk both options?**
- Ship features with feature flags (kill switch if breaks)
- Refactor incrementally (2 weeks now, 4 weeks later)
- Add monitoring (catch issues before revenue impact)

### **Step 4: My Recommended Approach**

**Option C+: Incremental refactor + new features + de-risk**

**Week 1-2: Critical refactor**
- Fix the most fragile parts of payment system
- Add comprehensive monitoring and alerting
- Create rollback plan
- Engineers focus on high-risk areas only

**Week 3-5: Add new payment methods (on improved foundation)**
- Build Apple Pay and Google Pay
- Feature flags for progressive rollout
- Comprehensive testing
- Start with 1% of users

**Week 6-8: Monitor + iterate**
- Watch for issues
- If stable, roll out to 100%
- If issues, rollback and fix

**Weeks 9-14: Complete refactor (scheduled)**
- Now that urgent feature shipped, finish refactoring
- Allocate 50% of sprint to debt paydown
- Improve long-term velocity

**Total time to new features**: 5 weeks (vs. 8 weeks full refactor)
**Technical debt addressed**: Yes, incrementally
**Risk mitigated**: Monitoring, feature flags, progressive rollout

### **Step 5: The Communication Plan**

**To Engineering**:

*"I hear your concern about technical debt. I agree we need to address it. Here's what I'm proposing:*

*Short-term (Weeks 1-5)*:
- 2 weeks: Refactor the most critical parts (you tell me which)
- 3 weeks: Add new payment methods on improved foundation
- We add monitoring, feature flags, and progressive rollout to de-risk

*Long-term (Weeks 6-14)*:
- After urgent feature ships, we allocate 50% of time to complete refactor
- This becomes a priority, not "someday"
- We build a sustainable balance: 30% of every sprint for technical work

*Why this approach*:
- Addresses your technical concerns (incremental refactor)
- Allows us to ship urgent features (business need)
- De-risks through monitoring and progressive rollout
- Commits to long-term debt paydown

*What I need from you*: 
- Tell me which parts of payment system are most critical to refactor
- Help me understand risks and how to mitigate them
- Trust that I'm committed to technical health, not just features

*Does this address your concerns?"*

**To Sales**:

*"I hear you're losing deals because we lack Apple Pay and Google Pay. Here's the plan:*

*Timeline*:
- 5 weeks to ship new payment methods (vs. 3 weeks if we rush)
- Extra 2 weeks because we're doing critical technical work to ensure stability

*Why the delay*:
- Current payment system has technical debt
- If we ship on broken foundation, risk payment failures (lost revenue)
- 2 weeks of refactoring protects our entire payment system
- Then we ship new features on stable foundation

*De-risking*:
- Progressive rollout (1% → 10% → 100%)
- Can roll back if issues
- Comprehensive monitoring

*Result*: 
- New payment methods in 5 weeks
- Stable, reliable payment system
- No risk of payment failures that lose ALL revenue

*What I need from you*:
- Communicate 5-week timeline to customers who asked
- If we're truly losing deals, can we offer beta access to early customers?

*Does this work?"*

**To CTO**:

*"I've talked to both engineering and sales. Here's how I'm balancing:*

*Engineering's concern*: Technical debt in payment system
*Sales's concern*: Losing deals without Apple Pay/Google Pay

*My recommendation*: Incremental approach
- 2 weeks critical refactor (high-risk areas only)
- 3 weeks add new payment methods (on improved foundation)
- 6+ weeks complete refactor (scheduled after urgent feature)
- 30% of every sprint for technical work going forward

*This balances*:
- Technical health (addresses most critical debt, commits to full refactor)
- Business needs (ships features in 5 weeks, not 8 weeks)
- Risk mitigation (monitoring, feature flags, progressive rollout)

*I'm accountable for*:
- Ensuring technical work happens (not just promising it)
- Managing stakeholder expectations
- Balancing short-term needs with long-term velocity

*What do you think?"*

### **Step 6: The Sustainable Solution**

**The real problem**: We're making a binary choice because we never allocated time for technical work.

**Long-term fix**: Allocate 20-30% of every sprint to technical debt and infrastructure.

**How to communicate this to stakeholders**:

*"We're implementing a sustainable engineering practice:*

*70-80% of sprint*: Feature development
*20-30% of sprint*: Technical debt, refactoring, infrastructure

*Why this matters*:
- Prevents crises like this (debt paid down continuously)
- Maintains velocity long-term (without maintenance, velocity declines 50% per year)
- Reduces incidents and technical failures
- Enables faster feature development (clean code = faster changes)

*This might feel like we're "slowing down," but we're actually speeding up long-term.*

*Analogy*: Regular car maintenance vs. waiting for breakdown. 20% of time on maintenance prevents 100% downtime when the engine fails.

*Commitment*: We'll track velocity over 12 months and show that this actually increases our output."*

### **Key Takeaways**

**1. Quantify the trade-offs** (not just feelings)
- Engineering: How long for each option? What risks?
- Sales: How much revenue at stake? How urgent?

**2. Find the middle ground** (usually exists)
- Incremental refactor (not all-or-nothing)
- Progressive rollout (de-risk new features)
- Commit to long-term debt paydown

**3. Communicate transparently** to all stakeholders
- Engineering: We hear your technical concerns
- Sales: We hear your business needs
- Here's how we balance both

**4. PM makes the call** (but collaboratively)
- PM owns the decision
- But makes it with input from all sides
- Takes accountability for outcome

**5. Prevent future crises** with sustainable practices
- 20-30% of every sprint for technical work
- Continuous debt paydown, not crisis-driven

**6. Both sides are right** (not a zero-sum game)
- Engineering's concerns are valid
- Sales's concerns are valid
- Good PMs find solutions that address both

**The PM's role**: Navigate trade-offs, make balanced decisions, communicate clearly, take accountability.

**Success**: Engineering feels heard, Sales gets features reasonably quickly, system stays healthy long-term.
`,
  },
  {
    id: 3,
    question:
      "Create a 'PM-Engineering Partnership Charter' for your team. Include: communication norms, decision-making boundaries, collaboration rituals, how to handle disagreements, and success metrics for the relationship. Make it practical and specific enough that a new PM or engineer could follow it.",
    answer: `## Comprehensive Answer:

Here's a comprehensive PM-Engineering Partnership Charter:

---

# PM-Engineering Partnership Charter

## Purpose

This charter defines how Product Managers and Engineers collaborate to ship great products. It clarifies roles, establishes norms, and creates a foundation for trust.

---

## Section 1: Role Boundaries

### **PM Owns (WHAT & WHY)**

**Product Strategy**:
- ✓ What problems to solve
- ✓ Why they matter (user needs, business impact)
- ✓ Success metrics (how we measure impact)
- ✓ Feature prioritization
- ✓ When to ship (based on priorities, market timing)

**PM makes final call on**: Product roadmap, feature scope, shipping dates, user-facing trade-offs

### **Engineering Owns (HOW)**

**Technical Execution**:
- ✓ How to build it (technical approach)
- ✓ Technical architecture decisions
- ✓ Technology choices (languages, frameworks, tools)
- ✓ Code quality standards
- ✓ Technical feasibility estimates

**Engineering makes final call on**: Technical architecture, implementation approach, code quality, technical debt decisions

### **Shared Ownership (COLLABORATION)**

**Together we decide**:
- ✓ Scope negotiation (what fits in timeline?)
- ✓ Trade-offs (speed vs. quality vs. features)
- ✓ MVP definition (what's minimum viable?)
- ✓ Launch readiness (is it good enough?)
- ✓ Technical debt vs. features (when to refactor?)

---

## Section 2: Communication Norms

### **Daily Communication**

**Standups** (9:00 AM, 15 minutes max):
- **PM attends**: 3x per week (Mon, Wed, Fri)
- **PM provides**: Priorities, blockers to unblock, stakeholder updates
- **PM does NOT**: Micromanage tickets, ask for detailed status

**Slack Communication**:
- **Response time**: 4 hours for non-urgent, 1 hour for urgent
- **Urgent defined as**: Production issue, launch blocker, customer escalation
- **Use threads**: Keep conversations organized
- **Status updates**: Post in #product-updates channel (not DMs)

**Focus Time Protection**:
- **No meetings 2-5 PM**: Engineers have focused coding time
- **No Slack for non-urgent**: Use async communication (Notion comments)
- **Batch questions**: PM batches questions for specific sync times

### **Weekly Communication**

**Sprint Planning** (Monday 9-10 AM):
- PM presents: Prioritized backlog, context for upcoming work
- Engineering breaks down: Stories into tasks, estimates effort
- Together decide: Sprint commitments

**Design Review** (Wednesday 2-3 PM):
- Designer presents: UI mockups for upcoming features
- PM provides: Product context, user needs
- Engineers provide: Technical feasibility, implementation questions

**Sprint Review** (Friday 3-4 PM):
- Engineers demo: Work completed this sprint
- PM provides: Feedback, user perspective
- Together discuss: What's ready to ship

**Sprint Retro** (Friday 4-4:30 PM):
- Discuss: What went well, what didn't, how to improve
- Action items: Specific changes for next sprint

### **As-Needed Communication**

**Technical Design Reviews**:
- When: Before major technical decisions
- Who: PM, EM, engineers working on feature
- Purpose: PM provides product context, engineers propose architecture
- Outcome: Documented decision with rationale

**Solution Design Sessions**:
- When: After discovery, before sprint planning
- Who: PM, EM, 2-3 engineers, designer
- Purpose: Collaboratively design solution to validated problem
- Duration: 60-90 minutes

**1-on-1s**:
- **PM-EM**: Weekly, 30 minutes (sync on priorities, team health)
- **PM-Individual Engineers**: Monthly, 30 minutes (build relationships, get feedback)

---

## Section 3: Collaboration Rituals

### **Discovery Phase**

**Engineer Involvement**:
- ✓ PM invites 1-2 engineers to 2-3 user interviews per project
- ✓ Engineers observe user research sessions
- ✓ Engineers participate in problem definition
- ✓ Engineers do technical spikes for feasibility

**Outcome**: Engineers understand user problems before building

### **Planning Phase**

**Solution Design Session** (before sprint planning):

**Agenda**:
1. PM presents problem + user research (15 min)
2. Designer shares design concepts (15 min)
3. Engineers propose technical approaches (30 min)
   - Option A: [Approach + trade-offs]
   - Option B: [Approach + trade-offs]
   - Recommendation: [Which and why]
4. Discussion: Trade-offs, scope, risks (20 min)
5. Decision: Approach, scope, timeline (10 min)

**Outcome**: Collaborative decision, shared ownership

### **Execution Phase**

**Sprint Execution**:
- ✓ PM attends 3 standups per week (not all 5)
- ✓ PM available for questions (but doesn't micromanage)
- ✓ PM unblocks issues (stakeholders, design, product questions)
- ✓ Engineers self-organize on implementation

**Mid-Sprint Check-in** (Wed afternoon):
- Quick sync: Are we on track? Any blockers?
- Scope adjustment if needed (but minimal)

### **Shipping Phase**

**Launch Checklist** (together):
- ✓ Feature works as intended (QA passed)
- ✓ Edge cases handled
- ✓ Monitoring and alerting set up
- ✓ Rollback plan exists
- ✓ Documentation written
- ✓ Support team trained

**Progressive Rollout**:
- Day 1: 1% of users (watch for issues)
- Day 3: 10% of users
- Day 7: 50% of users
- Day 14: 100% of users

---

## Section 4: Decision-Making Framework

### **When PM Decides Alone**

- Feature prioritization (based on strategy)
- Shipping dates (based on priorities)
- Product positioning and messaging
- User-facing trade-offs (UX vs. functionality)

**PM must**: Explain reasoning, provide context, listen to input (but makes final call)

### **When Engineering Decides Alone**

- Technical architecture
- Technology choices (languages, frameworks)
- Code structure and organization
- Technical quality standards

**Engineering must**: Explain reasoning to PM, educate PM on trade-offs (but makes final call)

### **When We Decide Together**

- Scope negotiations
- MVP definition
- Quality vs. speed trade-offs
- Technical debt vs. new features
- Launch readiness

**Process**: PM and EM discuss → PM makes recommendation → EM provides input → PM makes final call (with engineering buy-in)

### **Escalation Process**

If PM and EM disagree:
1. **First**: Discuss one-on-one (most issues resolved here)
2. **Second**: Get more data (user research, technical spike, customer feedback)
3. **Third**: Escalate to VP Product + CTO (rare, only for major decisions)

**Principle**: Disagree and commit. After decision is made, everyone executes fully.

---

## Section 5: Handling Disagreements

### **Healthy Disagreement Process**

**When PM and Engineer disagree**:

1. **Understand the disagreement**:
   - PM: "Help me understand your concern"
   - Engineer: Explains technical perspective
   - PM: Paraphrases to confirm understanding

2. **Share data**:
   - PM: User research, business impact, strategic context
   - Engineer: Technical constraints, architecture implications, risks
   
3. **Explore options**:
   - What are alternative approaches?
   - Can we run an experiment?
   - Is there a middle ground?

4. **Make decision with clear reasoning**:
   - PM (or EM, depending on decision type) makes call
   - Document: What we decided, why, what we considered

5. **Disagree and commit**:
   - Even if you disagree, commit to decision
   - Revisit if new data emerges

### **When NOT to Escalate**

- **Product scope**: PM decides (with engineering input)
- **Technical implementation**: Engineering decides (with PM input)
- **Normal trade-offs**: Work through together

### **When TO Escalate**

- **Fundamental disagreement** on product direction (rare)
- **Major technical decisions** with broad impact
- **Resource allocation** (need more people, time, budget)
- **Deadlock** that can't be resolved (rare)

### **Conflict Resolution Principles**

✓ **Assume good intent**: Everyone wants to ship great products
✓ **Focus on data**: Not opinions or politics
✓ **Seek to understand**: Before seeking to be understood
✓ **Disagree and commit**: After decision, everyone executes fully
✓ **No blame**: Focus on learning, not finger-pointing

---

## Section 6: Success Metrics

### **Healthy Relationship Indicators**

**Green flags** (relationship is healthy):
- ✅ Engineers proactively suggest product improvements
- ✅ Engineers defend product decisions in technical discussions
- ✅ Engineers raise concerns early (not after shipping)
- ✅ PM and EM have transparent, trust-based relationship
- ✅ Engineers feel PM "gets it" technically
- ✅ PM feels engineers understand user needs
- ✅ Low escalations to leadership
- ✅ Team enjoys working together

**Red flags** (relationship needs work):
- ⚠️ Frequent escalations to VP Product / CTO
- ⚠️ Engineers complain PM doesn't understand technical complexity
- ⚠️ PM feels engineers aren't prioritizing user needs
- ⚠️ Requirements frequently misunderstood
- ⚠️ Blame culture (pointing fingers when things fail)
- ⚠️ Engineers surprised by priorities
- ⚠️ PM surprised by timelines

### **Quarterly Health Check**

**Every quarter, we assess**:

**1. Trust** (1-5 scale):
- Do engineers trust PM's product decisions?
- Does PM trust engineering's technical decisions?
- Target: 4+/5

**2. Communication** (1-5 scale):
- Are requirements clear?
- Do we communicate proactively?
- Target: 4+/5

**3. Collaboration** (1-5 scale):
- Do we feel like partners?
- Do we solve problems together?
- Target: 4+/5

**4. Outcomes**:
- Did we ship on time?
- Did shipped features work?
- Are users happy?

**If scores <4**: Retro to identify specific improvements

---

## Section 7: Onboarding (For New Team Members)

### **New PM Onboarding**

**Week 1**:
- ✓ Read this charter
- ✓ 1-on-1 with every engineer (understand their expertise)
- ✓ System architecture walkthrough (understand technical landscape)
- ✓ Attend all rituals (standups, reviews, retros)

**Week 2-4**:
- ✓ Shadow experienced PM
- ✓ Join user interviews with engineers
- ✓ Review past PRDs (understand documentation standards)
- ✓ Get feedback from EM on collaboration style

### **New Engineer Onboarding**

**Week 1**:
- ✓ Read this charter
- ✓ 1-on-1 with PM (understand product strategy)
- ✓ Review product roadmap (understand priorities)
- ✓ Attend user interview with PM (understand users)

**Week 2-4**:
- ✓ Pair program with experienced engineer
- ✓ Attend solution design session
- ✓ Ship first feature (with PM guidance)
- ✓ Get feedback from PM on collaboration

---

## Commitment

**We commit to**:
- ✅ Respect each other's expertise
- ✅ Communicate transparently
- ✅ Collaborate on solutions
- ✅ Make decisions with data
- ✅ Give credit, take blame
- ✅ Build products together

**PM signs**: _____________  
**EM signs**: _____________  
**Date**: _____________

**Review cadence**: Quarterly (adjust charter based on learnings)

---

**This charter is a living document. We'll update it based on what we learn working together.**
`,
  },
];

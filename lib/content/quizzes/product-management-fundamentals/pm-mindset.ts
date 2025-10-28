/**
 * Discussion questions for The Product Manager Mindset
 * Product Management Fundamentals Module
 */

export const pmMindsetQuiz = [
  {
    id: 1,
    question:
      "You're a PM at a fast-growing B2B SaaS company. Your CEO pressures you to ship a major enterprise feature that a $500K customer is demanding, even though your user research with 20 small customers shows they want something completely different. The CEO says: 'We need to close this deal this quarter. Data from small customers doesn't matter—enterprise is our future.' How do you handle this situation? Walk through your decision-making process, considering the tension between data-driven decisions, customer obsession, and business reality.",
    answer: `## Comprehensive Answer:

This scenario tests multiple PM mindset elements simultaneously: data-driven decision making vs. business pragmatism, customer obsession vs. revenue pressure, and strategic thinking about product direction. Let's work through this systematically.

### **Step 1: Understand the Full Context**

Before making any decision, I need to gather information (avoiding snap judgments):

**Questions for CEO**:
1. "Help me understand the strategic importance of this customer. Are they representative of our target enterprise segment?"
2. "If we build this feature for them, will it attract similar enterprise customers, or is this unique to them?"
3. "What's the expected deal size and LTV of this customer vs. our typical customers?"
4. "What happens if we don't close this deal? Are there other enterprise opportunities?"
5. "What's our long-term strategy: Move upmarket to enterprise or stay focused on SMB?"

**Questions for Sales/Customer Success**:
1. "How many other enterprise prospects are asking for this exact feature?"
2. "What's the competitive dynamic? Will we lose deals without it?"
3. "Is this customer's use case representative or an edge case?"

**My own analysis**:
1. Review product strategy: What segment are we targeting?
2. Analyze data: What % of revenue comes from enterprise vs. SMB?
3. Review roadmap: Where does this fit vs. current priorities?
4. Estimate effort: How expensive is this to build? (2 weeks vs. 6 months makes a difference)

### **Step 2: Frame the Trade-Offs Explicitly**

**Option A: Build the enterprise feature**

**Pros**:
- Close $500K deal (immediate revenue)
- Learn about enterprise needs (might open new market)
- Build relationships with large customer
- Show sales team we can deliver

**Cons**:
- Ignore data from 20 customers
- Potentially derail roadmap
- Build feature that won't generalize
- Opportunity cost: Can't build what small customers need
- Risk of becoming custom dev shop for one customer

**Option B: Don't build it (stick with small customer needs)**

**Pros**:
- Stay true to product strategy
- Honor data from user research
- Build features that scale across all customers
- Maintain product coherence

**Cons**:
- Lose $500K deal
- Disappoint CEO and sales team
- Miss potential enterprise opportunity
- Slow down enterprise motion

**Option C: Hybrid approach (what I'd likely recommend)**

Find creative middle ground that addresses both needs.

### **Step 3: My Recommended Approach**

Here's what I would actually do:

#### **Immediate Action: Deep Dive on the Request**

**Week 1**:
- Talk to the enterprise customer directly (not just sales' interpretation)
- Understand the *problem* they're solving (not just the feature they're requesting)
- Validate if this problem exists across enterprise segment
- Estimate engineering effort

**Example conversation with enterprise customer**:

*PM: "I understand you need [Feature X]. Can you walk me through the specific problem you're trying to solve?"*

*Customer: "We need to integrate with our legacy ERP system for compliance."*

*PM: "Got it. What if we provided an API that lets you build that integration yourself? Would that solve your problem, even if we don't build the full UI?"*

*Customer: "Actually, yes! We have eng resources to build the integration if you give us the API."*

**Key insight**: Often customers request features, but what they really need is a solution to a problem. There might be a lightweight way to solve it.

#### **Strategic Framework: The Three Questions**

**Question 1: Is this enterprise customer representative of our future?**

**If YES** (we're moving upmarket):
- Their needs are a leading indicator
- Building for them is strategic
- → Consider prioritizing their request

**If NO** (they're an outlier):
- Their needs might lead us astray
- Building for them is tactical revenue, not strategic
- → Deprioritize or find lightweight solution

**Question 2: Can we solve their problem AND honor small customer research?**

**Look for creative solutions**:
- Can we build a lightweight version? (MVP of their request that also helps small customers)
- Can we solve the underlying problem differently? (API instead of full feature)
- Can we use feature flags? (enterprise-specific functionality)
- Can we partner? (integrate with 3rd party instead of building)

**Example**: Instead of building custom "Legacy ERP Integration," build a general "Webhooks API" that:
- Solves enterprise customer's problem (they can integrate themselves)
- Solves small customer problems (they can integrate with other tools)
- Scalable and future-proof

**Question 3: What's the long-term product strategy?**

This is where "Think Like a CEO" mindset comes in:

**If our strategy is enterprise**:
- This is the first of many enterprise features
- We need to start somewhere
- Build it, learn, iterate
- → Align with CEO, prioritize enterprise feature

**If our strategy is SMB**:
- Chasing this deal distracts from core market
- Better to lose this deal and stay focused
- → Push back respectfully, offer alternative

**If our strategy is unclear** (red flag!):
- Use this as opportunity to force strategic clarity
- Can't make good product decisions without strategy
- → Escalate: "Before deciding, we need clarity on our target market"

### **Step 4: The Conversation with CEO**

**How I'd frame it** (respectful pushback with data and options):

*"I understand the importance of closing this enterprise deal. I've spent time with the customer and our small customers to understand the trade-offs. Let me share what I learned and propose some options.*

**Context**:
- Enterprise customer needs [Feature X] primarily for [Problem Y]
- Our 20 small customers need [Feature Z] for [Problem A]
- Effort estimates: Enterprise feature = 3 months, SMB feature = 6 weeks

**Strategic Question** (forcing clarity):
- Are we doubling down on enterprise or staying focused on SMB?
- This decision will set our direction for the next year

**Option 1: Enterprise-first strategy**
- Build full enterprise feature (3 months)
- Pros: Close deal, learn enterprise needs, open new market
- Cons: Delay SMB roadmap, may not generalize

**Option 2: SMB-first strategy**
- Build SMB feature (6 weeks), offer lightweight solution to enterprise
- Pros: Honor user research, scalable across customers
- Cons: Likely lose enterprise deal

**Option 3: Hybrid approach** (my recommendation):
- Build general API/hooks (8 weeks) that solves both
- Enterprise can integrate themselves
- SMB customers can connect to tools they need
- Scalable, future-proof architecture

**My recommendation**: Option 3, because:
- Solves immediate problem (enterprise deal)
- Honors user research (benefits all customers)
- Strategic investment (pays off long-term)
- Reduces future custom dev requests

**Decision needed**: Which strategic direction should we go? I'll execute on whatever you decide, but I need clarity to make good product decisions going forward."*

### **Step 5: What If CEO Insists on Building Custom Feature?**

**If CEO overrules and demands we build exactly what enterprise wants**:

**What I do**:
1. **Document the decision**: Write down the strategic trade-off
2. **Define success metrics**: "We're building this to close enterprise deals. Success = signing 3 more similar deals in 6 months"
3. **Set clear scope**: "We'll build V1 for this customer, but won't expand it unless we see demand from others"
4. **Communicate to team**: Explain *why* we're doing this (revenue, strategic test)
5. **Execute well**: Since we're building it, make it excellent
6. **Review after 6 months**: Did it work? Did we get more enterprise customers?

**What I don't do**:
- Passive-aggressively build it poorly
- Complain to team that "CEO made us do it"
- Ignore the decision and build SMB features anyway
- Quit on the spot

**Why**: PM's job is to influence, advocate, provide options—but ultimately CEO makes strategic calls. If you disagree with every decision, you're at the wrong company.

### **Step 6: Balancing Multiple PM Mindsets**

This scenario reveals tension between different mindsets:

**Data-Driven Decision Making**:
- ✓ Research with 20 customers shows different need
- ⚠️ But sample might not include enterprise customers
- ⚠️ Revenue data might contradict user research

**Customer Obsession**:
- ⚠️ Which customers? All customers or target customers?
- ✓ Solve real problems (not just take feature requests)

**Strategic Thinking (CEO Mindset)**:
- ✓ What's our long-term strategy?
- ✓ Sometimes short-term revenue enables long-term vision

**Bias for Action**:
- ✓ Don't overthink—make a decision and move forward
- ✓ Can iterate based on results

**The resolution**: All mindsets matter, but prioritize based on context:
1. **Clarify strategy first** (can't make good decisions without it)
2. **Use data to inform** (but recognize limitations)
3. **Think long-term** (what builds sustainable product?)
4. **Move fast** (don't let decision paralysis slow you down)

### **Conclusion: What I'd Actually Do**

**My recommendation**:

1. **Week 1**: Deeply understand both enterprise and SMB needs (talk to customers)
2. **Week 1**: Estimate engineering effort for different options
3. **Week 2**: Present CEO with options (enterprise vs. SMB vs. hybrid)
4. **Force strategic clarity**: "Are we moving upmarket?"
5. **Recommend hybrid approach**: Build general solution that addresses both
6. **If overruled**: Document decision, execute well, measure results

**The key lesson**: Great PMs don't just follow data or just follow CEO demands. They:
- Bring data AND strategic context
- Present options with trade-offs
- Recommend based on product strategy
- Execute excellently on whatever is decided
- Measure and learn

**Red flags I'd watch for**:
- If CEO consistently overrules product strategy → strategy problem
- If we build many custom features → becoming services company
- If we chase every big deal → no product coherence

**When I'd consider leaving**:
- If there's no clear strategy (just reactive to deals)
- If CEO dismisses user research consistently
- If we're building features for revenue without considering product impact

But one disagreement? That's normal. Great PM-CEO relationships involve healthy tension, respectful debate, and collaborative decisions.
`,
  },
  {
    id: 2,
    question:
      "Your team ships a major feature you've been working on for 3 months. In the first week, you discover that only 5% of users are trying it, and of those who try it, 80% never use it again. Your manager asks what went wrong and what you're learning. Your engineering manager is frustrated that the team worked hard on something users don't want. How do you demonstrate the right mindset in handling this failure? What would you say in the team post-mortem? How do you rebuild team morale and extract valuable lessons?",
    answer: `## Comprehensive Answer:

This scenario tests the "comfort with being wrong" and "learning from failures systematically" mindsets. Let's walk through exactly how I'd handle this situation.

### **Initial Reaction (First 24 Hours): Own It**

**What NOT to do**:
- Blame engineers ("If you'd built it better...")
- Blame users ("They just don't understand it")
- Make excuses ("We needed more time")
- Get defensive ("My research was solid!")
- Hide from it ("Let's just move on")

**What TO do**:
- Acknowledge the failure immediately
- Take ownership as PM
- Show curiosity about what happened
- Commit to learning from it

**What I'd say to my manager** (within 24 hours):

*"The feature launched, but early data shows low adoption (5% tried it, 80% churned). This is clearly not the outcome we expected. I take responsibility for this—I was the PM, and it's my job to ensure we build the right thing.*

*Here's what I'm doing immediately:*
1. Analyzing usage data to understand what happened
2. Scheduling user interviews to understand why people aren't using it
3. Meeting with engineering to discuss what we learned
4. Planning a team post-mortem for Friday

*I'll have a full analysis and recommended next steps by [specific date]. The good news is we're learning fast, and we can iterate or pivot based on what we discover."*

**Why this works**:
- Takes ownership (doesn't blame others)
- Shows action plan (not paralyzed)
- Frames as learning (not disaster)
- Commits to analysis (data-driven)
- Sets timeline (bias for action)

### **Step 1: Deep Dive Analysis (Week 1)**

Before the post-mortem, gather all the data:

#### **Quantitative Analysis**

**Questions to answer with data**:
1. **Adoption**: Why did only 5% try it?
   - Do users know it exists? (awareness problem)
   - Can they find it? (discoverability problem)
   - Do they understand what it does? (positioning problem)

2. **Engagement**: Why did 80% never use it again?
   - Did it work technically? (check error logs)
   - Was it too slow? (performance metrics)
   - Was the experience confusing? (funnel analysis)
   - Did it solve their problem? (need qualitative data)

3. **Segmentation**: Are there users who DID find it valuable?
   - Which user segments tried it?
   - Any power users? (small group who love it = potential)
   - Differences between users who adopted vs. churned?

**SQL queries to run**:
\`\`\`sql
-- Adoption funnel
SELECT 
    COUNT(DISTINCT user_id) as total_users,
    COUNT(DISTINCT CASE WHEN viewed_feature THEN user_id END) as viewed,
    COUNT(DISTINCT CASE WHEN clicked_feature THEN user_id END) as clicked,
    COUNT(DISTINCT CASE WHEN completed_action THEN user_id END) as completed
FROM feature_usage
WHERE date >= 'launch_date';

-- Retention by cohort
SELECT 
    user_segment,
    COUNT(DISTINCT user_id) as users,
    AVG(CASE WHEN day_7_return THEN 1 ELSE 0 END) as day_7_retention
FROM feature_usage
GROUP BY user_segment;

-- Error rates
SELECT error_type, COUNT(*) as count
FROM error_logs
WHERE feature = 'new_feature'
GROUP BY error_type
ORDER BY count DESC;
\`\`\`

#### **Qualitative Research**

**User interviews** (schedule 10-15 in Week 1):

**Users who tried it once and left** (the 80%):
- "Can you walk me through what you were trying to do?"
- "What did you expect the feature to do?"
- "What happened when you tried it?"
- "Why didn't you come back?"
- "What would make this useful for you?"

**Users who didn't try it at all** (the 95%):
- "Did you notice we launched [feature]?"
- "What do you think it does?"
- "Does this solve a problem you have?"
- "What would make you try it?"

**Power users** (if any exist):
- "Why do you use this?"
- "What value are you getting?"
- "What makes you different from others?"
- "What could be better?"

### **Step 2: Synthesis (Mid-Week 1)**

**Pattern-matching from research + data**:

Typical failure patterns:

**Pattern 1: Awareness Problem**
- Data: Low views, low clicks
- Insight: Users don't know it exists
- Fix: Better launch marketing, UI prominence

**Pattern 2: Positioning Problem**
- Data: High views, low clicks
- Insight: Users don't understand value proposition
- Fix: Better messaging, clearer benefits

**Pattern 3: Usability Problem**
- Data: High clicks, low completion, high errors
- Insight: Feature works but UX is confusing
- Fix: Redesign onboarding, simplify flow

**Pattern 4: Wrong Problem**
- Data: Low engagement across all segments
- Insight: We built something users don't need
- Fix: Pivot or sunset feature

**Pattern 5: Wrong Solution**
- Data: Users tried it but it didn't solve their problem
- Insight: We understood problem but built wrong solution
- Fix: Rebuild with different approach

**Pattern 6: Right Product, Wrong Audience**
- Data: One segment loves it, others don't
- Insight: Valuable but for different personas
- Fix: Reposition and relaunch to right segment

**My hypothesis**:
Based on interviews and data, which pattern matches? (Be honest!)

### **Step 3: The Team Post-Mortem (End of Week 1)**

**Setup**:
- Schedule 90 minutes
- Invite: PM (me), EM, engineers, designer, data analyst
- Bring: Data, user quotes, analysis
- Set tone: Learning, not blame

**Agenda**:

#### **1. State the Facts (10 min)**

*"Here's what happened:*
- Launched [feature] on [date]
- Expected: 20% adoption, 50% retention
- Actual: 5% adoption, 20% retention
- Gap: Significant miss on both metrics

*This is a failure by any measure. I take responsibility—I was the PM, and it's my job to ensure we build the right things.*

*But failures are only valuable if we learn from them. Today's goal: Understand what happened and extract lessons so we don't repeat these mistakes."*

#### **2. What We Expected vs. What Happened (20 min)**

**What I expected**:
- Researched with 15 users who said they needed this
- Competitive products had this feature
- Data showed gap in user journey
- Expected strong adoption

**What actually happened**:
- Low awareness (only 5% even tried)
- High bounce (80% tried once and left)
- Segment differences (power users in [segment] loved it, others didn't)

**Why the gap?**:
- [Specific insights from user research]
- [Data showing the real problem]

**Example**:
*"I interviewed 15 users who said they wanted [feature]. But what I missed: They were describing a problem, not a solution. When I asked in interviews 'Would you use X?', they said yes. But I should have asked 'Show me how you solve this problem today.' If I had, I would have seen that the current workaround was actually fine for most users. Only a small segment ([power users]) had the problem severe enough to need a new solution."*

#### **3. What We Learned (30 min - Most Important)**

**About Users**:
- [Insight 1]: Users don't have the problem we thought
- [Insight 2]: The problem exists but only for [segment]
- [Insight 3]: Our solution didn't match their mental model

**About Our Process**:
- [Process Insight 1]: User interviews had leading questions
- [Process Insight 2]: Didn't validate with usage data before building
- [Process Insight 3]: Didn't do prototype testing before full build

**About Execution**:
- [Execution Insight]: Built too much before validating
- [Execution Insight]: Launch marketing was weak

**Key lesson**: [The ONE thing we'll do differently next time]

**Example**:
*"The biggest lesson: We asked users 'Would you use this?' instead of 'Show me how you currently solve this problem.' This led to false positives. Next time, we'll do prototype testing with interactive mocks before writing code."*

#### **4. What We Do Now (20 min)**

**Options** (present to team):

**Option A: Iterate**
- Fix [specific issues] (2 weeks)
- Relaunch with better marketing
- Target [specific segment]
- Metrics: Expect 15% adoption, 40% retention

**Option B: Pivot**
- Keep core functionality, but reposition
- Focus on [different use case]
- Adjust UX for [new persona]
- Metrics: [different success criteria]

**Option C: Sunset**
- Feature didn't solve a real problem
- Cut losses, move to next priority
- Remove feature in 30 days

**My recommendation**: [Option + reasoning]

**Example**:
*"I recommend **Option A: Iterate**, because:*
1. Data shows [power user segment] gets real value
2. Other segments didn't understand the value prop
3. We can fix with better onboarding + clearer messaging
4. Low risk (2 weeks), potentially high reward

*But I'm open to feedback. What do you all think?"*

#### **5. Discussion (20 min)**

**Open floor for team input**:
- Engineers: Technical insights we missed?
- Designer: UX problems we didn't foresee?
- EM: Process improvements?

**Listen for**:
- Are people blaming others? (redirect to learning)
- Are there systemic issues? (document for improvement)
- Is team demoralized? (address morale)

#### **6. Action Items (10 min)**

**Concrete next steps**:
1. [Action item] - Owner: [Name] - Due: [Date]
2. [Action item] - Owner: [Name] - Due: [Date]
3. Process change: [What we'll do differently] - Owner: PM

**Example**:
*Action items:*
1. Rewrite onboarding flow based on user feedback - Owner: Designer - Due: Next week
2. Add feature discovery to home screen - Owner: Engineer - Due: Next week
3. Process change: All future features require prototype testing before development - Owner: PM

**Close on positive note**:
*"Thanks, everyone, for the hard work on this. Yes, it didn't land the way we hoped, but we learned a ton. The best teams aren't the ones that never fail—they're the ones that learn from failures and get better. Let's apply these lessons and make the next feature a win."*

### **Step 4: Rebuilding Team Morale**

**Address engineering frustration directly**:

**1:1 with Engineering Manager**:

*"I know the team worked really hard on this, and it's frustrating to build something users don't immediately love. I want to acknowledge that and also share how I'm thinking about this.*

*First, I take responsibility. It's my job to ensure we're building the right thing, and I missed some signals in my research.*

*Second, this is a learning moment, not a waste. We learned [specific insights] that will make our next features better. That knowledge is valuable.*

*Third, I've identified specific changes to our process [prototype testing, better validation] so we don't repeat this.*

*What can I do to make sure the team feels like their time was valued, even though the feature didn't land as we hoped?"*

**Team-level morale building**:

1. **Acknowledge hard work**: "Thank you for the excellent execution. The code quality was great, the design was polished. This wasn't an execution problem—it was a product decision problem (on me)."

2. **Share learnings**: "Because of what we learned, our next features will be stronger."

3. **Quick win**: "Let's get a win fast. I'm proposing we iterate quickly and relaunch in 2 weeks."

4. **Celebrate learning**: "The best teams learn fast. We just learned something important."

### **Step 5: Communicating Up (To Leadership)**

**Update to leadership** (end of Week 1):

*"Here's the analysis on [feature launch]:*

**What happened**:
- 5% adoption, 20% retention (vs. expected 20% / 50%)
- Clear miss

**Why it happened**:
- [Root cause from analysis]
- [Process gaps we identified]

**What we learned**:
- [Key insight about users]
- [Key insight about our process]

**What we're doing**:
- [Action plan: iterate / pivot / sunset]
- [Process improvements]
- [Timeline for next steps]

**Lessons applied to future work**:
- [Specific process changes]

*I take responsibility for this miss and am committed to learning from it. The team executed well; this was a product strategy error (on me). We're applying these lessons to upcoming features."*

### **Step 6: Applying Lessons to Next Feature**

**New process** (documented and shared):

**Before building anything**:
1. User research (but better questions!)
2. Build interactive prototype
3. Test prototype with 10 users
4. Validate with data (does usage data support the hypothesis?)
5. Define success metrics BEFORE building
6. Build MVP (smallest version that tests hypothesis)
7. Launch, measure, iterate

**Changed behaviors**:
- No more "Would you use this?" questions → "Show me how you solve this today"
- No building before prototype testing
- Smaller MVPs (test faster)
- Clear success metrics upfront

### **Conclusion: The Mindset on Display**

This response demonstrates key PM mindsets:

**1. Comfort with being wrong**:
- Acknowledge failure immediately
- Don't get defensive
- Take ownership

**2. Learning from failures systematically**:
- Analyze what happened (data + user research)
- Extract learnings
- Apply to future work
- Document process improvements

**3. Growth mindset**:
- View failure as learning opportunity
- Improve process based on feedback
- Get better over time

**4. Leadership without authority**:
- Take responsibility (even though EM and engineers built it)
- Rebuild team morale
- Communicate clearly to all stakeholders

**5. Bias for action**:
- Don't wallow in failure
- Make decision quickly (iterate/pivot/sunset)
- Move forward

**The result**: Team learns, improves process, moves forward stronger. That's what great PMs do when things go wrong.

**Final thought**: The worst outcome isn't shipping a feature that fails. It's shipping a feature that fails and learning nothing from it. Great PMs extract maximum learning from every failure.
`,
  },
  {
    id: 3,
    question:
      'Compare and contrast the PM mindset needed at three different stages: (1) 0-to-1 product (finding product-market fit), (2) 1-to-10 product (scaling an existing product), and (3) 10-to-100 product (maintaining and optimizing a mature product). For each stage, explain which of the 10 mindset elements from this section are most critical and why. Give specific examples of how the same PM might need to adjust their mindset when moving between these stages.',
    answer: `## Comprehensive Answer:

Excellent question! The PM mindset needs to adapt as products evolve. Let's break down each stage, prioritize the critical mindsets, and understand why.

### **Overview: The Three Stages**

| Stage | Description | Primary Goal | Key Challenge |
|-------|-------------|-------------|---------------|
| **0-to-1** | Building first version | Find product-market fit (PMF) | Uncertainty about what to build |
| **1-to-10** | Scaling traction | Grow user base, optimize product | Scaling while maintaining quality |
| **10-to-100** | Mature product | Defend position, maximize value | Innovation while maintaining stability |

### **Stage 1: 0-to-1 Product (Finding PMF)**

**Context**: You're building something that doesn't exist yet. No users, no product, high uncertainty.

**Examples**: Early Airbnb (2008), Uber (2010), Clubhouse (2020 pre-explosion)

#### **Critical Mindsets (Ranked)**

**#1: Customer Obsession (Most Critical)**

**Why**: When you don't have product-market fit, the ONLY thing that matters is deeply understanding user problems.

**How it shows up**:
- Talking to users 20+ hours per week
- Personally doing user interviews (not delegating)
- Living with users (Airbnb founders stayed in Airbnbs)
- Obsessively analyzing why users churn

**Example - Airbnb's founding story**:
- Brian Chesky personally stayed in Airbnb listings
- Asked hosts detailed questions about their problems
- Photographed listings himself to understand quality issues
- Discovered: Professional photos increased bookings 2-3x
- Learning: Host experience was as important as guest experience

**Anti-pattern**: Building in a vacuum, not talking to users until after launch

---

**#2: Bias for Action (Critical)**

**Why**: At 0-to-1, speed of learning > perfection. You need to test hypotheses fast.

**How it shows up**:
- Shipping MVPs in weeks, not months
- Using no-code tools (Zapier, Airtable, Webflow) before building
- Doing things manually before automating
- Running experiments constantly

**Example - Dropbox MVP**:
- Drew Houston didn't build full product first
- Created **3-minute demo video** showing concept
- Posted on Hacker News
- Waitlist went from 5K → 75K overnight
- Validated demand before writing significant code
- Only then built the actual product

**Example - DoorDash's "Concierge" MVP**:
- Founders didn't build restaurant platform first
- Created simple website, took orders
- Personally picked up food and delivered it
- Validated demand before building logistics platform

**Anti-pattern**: Spending 12 months building "perfect" product without user validation

---

**#3: Comfort with Ambiguity (Critical)**

**Why**: Everything is uncertain at 0-to-1. No playbook exists.

**How it shows up**:
- Comfortable making decisions with 20% of information (vs. 80%)
- Pivoting quickly when hypotheses are wrong
- Not knowing what metrics to track yet (still figuring it out)
- Experimenting with positioning, pricing, messaging

**Example - Slack's origin story**:
- Started as internal tool for gaming company (Tiny Speck)
- Gaming company failed
- Pivoted to selling the internal tool
- Didn't know if anyone would want it
- Tried it with tech startups (worked!)
- Tried it with enterprises (worked even better!)
- Comfort with ambiguity allowed massive pivot

**Anti-pattern**: Paralysis waiting for perfect data before deciding

---

**#4: Comfort with Being Wrong (Critical)**

**Why**: Most 0-to-1 hypotheses are wrong. You must iterate quickly.

**How it shows up**:
- Expecting most ideas to fail
- Running many experiments simultaneously
- Pivoting without ego
- Celebrating learning, not just wins

**Example - Instagram's pivot**:
- **Original product**: Burbn (location check-in app like Foursquare)
- Users didn't care about check-ins
- Noticed: Everyone loved photo filters feature
- Stripped everything except photos + filters
- Relaunched as Instagram
- Massive success

**If founders weren't comfortable being wrong**: They'd still be building a failing check-in app

**Anti-pattern**: Defending original idea despite user evidence

---

**#5: Strategic Thinking (Important)**

**Why**: Even at 0-to-1, you need product strategy (not just random features).

**How it shows up**:
- Choosing target user segment deliberately
- Deciding what NOT to build
- Thinking about competitive moats early
- Planning 2-3 feature iterations ahead

**Example - Facebook's initial strategy**:
- Zuckerberg: "Move fast and break things"
- But also: Highly strategic about rollout
- Started with Harvard only (controlled launch)
- Then Ivy League (validated scaling)
- Then all colleges (network effects)
- Then everyone (after PMF confirmed)

**Strategic discipline**: Resisting pressure to open to everyone immediately

---

**Less critical at 0-to-1**:
- **Data-driven decisions**: Not enough data yet; rely more on qualitative insights
- **Long-term thinking**: Can't think 5 years ahead when you might not exist in 6 months
- **Building intuition**: Don't have patterns yet; learning as you go

---

### **Stage 2: 1-to-10 Product (Scaling)**

**Context**: You've found product-market fit. Now you need to scale user base, improve product, and build for growth.

**Examples**: Uber (2012-2015), Airbnb (2010-2014), Notion (2020-2023)

#### **Critical Mindsets (Ranked)**

**#1: Data-Driven Decision Making (Most Critical)**

**Why**: You now have usage data. Every decision should be informed by metrics.

**How it shows up**:
- Defining North Star metric (what drives growth?)
- Running A/B tests on everything
- SQL queries daily
- Funnel analysis to find conversion bottlenecks
- Cohort analysis for retention

**Example - Facebook's growth team (2010-2014)**:
- Chamath Palihapitiya led growth
- Data-driven mindset: Identify metric that predicts retention
- Discovery: "7 friends in 10 days" = retained user
- Laser focus: Optimize onboarding for this metric
- Result: Scaled from 100M → 1B users

**Without data-driven mindset**: Random feature launches, no systematic improvement

**Anti-pattern**: Still relying primarily on user interviews vs. quantitative data

---

**#2: Strategic Thinking (Critical)**

**Why**: At scale, every decision has bigger impact. Must think long-term.

**How it shows up**:
- Building competitive moats (network effects, data advantages)
- Platform thinking (should we open an API?)
- International expansion (what markets next?)
- Balancing growth with quality
- Deciding when to add new products vs. improving core product

**Example - Stripe's platform strategy (2012-2016)**:
- **Core product**: Payments API (already PMF)
- **Strategic question**: Build adjacent products or stay focused?
- **Decision**: Build platform (Stripe Connect, Billing, Radar)
- **Rationale**: Increase stickiness, expand TAM, build moat
- **Result**: Went from payments to financial infrastructure platform

**Anti-pattern**: Growing without strategy (just adding features randomly)

---

**#3: Bias for Action (Critical)**

**Why**: You're still in growth mode. Speed matters. Competitors are catching up.

**How it shows up**:
- Weekly or bi-weekly release cycles
- Launching features to subset of users (feature flags)
- Fast follow on competitor features (when strategic)
- Breaking down big projects into smaller milestones

**Example - Spotify (2012-2016)**:
- Launched Discover Weekly in 2015 (major personalization feature)
- Didn't build perfect recommendation system first
- Shipped basic version, measured, iterated weekly
- Became their most loved feature through rapid iteration

**Balance needed**: Bias for action + strategic thinking (don't just copy competitors)

---

**#4: Long-Term Thinking (Important)**

**Why**: At 1-to-10, decisions now affect you for years. Start building for the long-term.

**How it shows up**:
- Investing in infrastructure (scaling tech)
- Building platform capabilities
- Thinking about ecosystem (partners, integrations)
- Balancing short-term growth with long-term moat

**Example - Amazon Prime (2005, during 1-to-10 phase)**:
- **Short-term**: Lose money on shipping ($79/year, cost more than that)
- **Long-term bet**: Increased purchase frequency = higher LTV = sustainable advantage
- **Result**: Created massive competitive moat

**Anti-pattern**: Only optimizing for this quarter's growth (burns out long-term potential)

---

**#5: Building Product Intuition (Important)**

**Why**: You've shipped enough features to develop pattern recognition.

**How it shows up**:
- Knowing which features will work before testing (but still testing!)
- Spotting problems before data confirms
- Understanding your users' mental models deeply
- Making better bets based on past learnings

**Example - Brian Chesky (Airbnb, ~2014)**:
After 5 years, developed intuition:
- "This feature feels right for our brand"
- "Users will misunderstand this UI"
- "This will create support burden"

**Note**: Intuition informs hypotheses, but data still validates

---

**Less critical at 1-to-10**:
- **Comfort with ambiguity**: Less ambiguous now (you know the market, product works)
- **Comfort with being wrong**: Still important, but you're right more often now

---

### **Stage 3: 10-to-100 Product (Mature, Optimizing)**

**Context**: Large user base, established market position, competitive pressure, need to innovate while maintaining stability.

**Examples**: Gmail (2010-present), Salesforce (2015-present), Netflix (2020-present)

#### **Critical Mindsets (Ranked)**

**#1: Strategic Thinking (Most Critical)**

**Why**: Every decision affects millions of users. Long-term moat is everything.

**How it shows up**:
- Thinking 3-5 years ahead
- Platform strategy (ecosystem, partnerships)
- Competitive positioning
- Portfolio management (which products to invest in?)
- Innovation vs. optimization balance

**Example - Microsoft under Satya Nadella (2014-present)**:
- **Strategic shift**: "Mobile-first, cloud-first"
- **Big bet**: Azure cloud vs. Windows dominance
- **Rationale**: Cloud is future, Windows is mature
- **Execution**: Gradual portfolio shift over 5+ years
- **Result**: MSFT market cap went from $300B → $2.5T

**Anti-pattern**: Focusing on incremental optimization, missing big strategic shifts

---

**#2: Long-Term Thinking (Critical)**

**Why**: Mature products need to think in decades, not quarters.

**How it shows up**:
- Investing in R&D with 5-10 year payoff
- Building infrastructure that scales to 100M+ users
- Creating competitive moats that compound (network effects, switching costs)
- Balancing today's business with tomorrow's opportunities

**Example - Amazon's long-term thinking (2000s-present)**:
- **AWS**: Started as internal infrastructure (2002), became product (2006)
- **Payoff timeline**: Took 10+ years to become profitable
- **Long-term value**: Now $80B+ annual revenue, higher margin than retail
- **Bezos mindset**: "We're willing to be misunderstood for long periods"

**Anti-pattern**: Optimizing for quarterly earnings at expense of long-term position

---

**#3: Data-Driven Decision Making (Critical)**

**Why**: At scale, small % improvements = huge impact. Must be rigorous.

**How it shows up**:
- Sophisticated experimentation (multi-variate tests, long-term tests)
- Causal inference (beyond simple A/B tests)
- Machine learning for personalization
- Data infrastructure for self-serve analytics
- Statistical rigor (not just "this looks better")

**Example - Netflix recommendation system**:
- Invests heavily in ML/AI for personalization
- **Impact**: 80% of watch time comes from recommendations
- **Approach**: Rigorous A/B testing (thousands of tests per year)
- **Sophistication**: Test not just UI, but algorithm parameters

**At 10-to-100, you can afford to**:
- Hire specialized data scientists
- Build internal experimentation platforms
- Run long-term tests (months)

---

**#4: Building Product Intuition (Important)**

**Why**: After 10+ years, you deeply understand your users and market.

**How it shows up**:
- Knowing which features align with brand
- Understanding second-order effects
- Pattern matching from similar features
- Quickly identifying doomed ideas

**Example - Susan Wojcicki (YouTube CEO, 2014-2023)**:
After years at YouTube:
- Intuition about what content policies would work
- Understanding creator needs without asking
- Knowing when to copy competitors vs. differentiate
- Sensing cultural shifts before data confirmed

**But still validates with data**: Intuition generates hypotheses; experiments confirm

---

**#5: Customer Obsession (Important but Different)**

**Why**: Easy to lose touch with users at scale. Must stay connected.

**How it shows up differently at 10-to-100**:
- More indirect (can't talk to every user)
- Systematized (research teams, NPS surveys, focus groups)
- Segment-specific (B2B vs. B2C, power users vs. casual)

**Example - Jeff Bezos's "empty chair" practice**:
- Put empty chair in meetings
- Represents the customer
- Forces team to think about user impact
- **Scaled customer obsession**: Institutionalized the mindset

**Challenge at scale**: Staying close to users through layers of abstraction

---

**Less critical at 10-to-100**:
- **Bias for action**: Must balance speed with stability (can't break things for 100M users)
- **Comfort with ambiguity**: Less ambiguous (market proven, product established)
- **Comfort with being wrong**: Your hit rate is higher now

---

### **Comparison Table: Mindset Priorities by Stage**

| Mindset | 0-to-1 | 1-to-10 | 10-to-100 |
|---------|--------|---------|-----------|
| **Customer Obsession** | 🔥🔥🔥 Critical | 🔥🔥 Important | 🔥🔥 Important (different) |
| **Data-Driven** | 🔥 Less critical | 🔥🔥🔥 Critical | 🔥🔥🔥 Critical |
| **Bias for Action** | 🔥🔥🔥 Critical | 🔥🔥🔥 Critical | 🔥🔥 Important (balanced) |
| **CEO Thinking** | 🔥🔥 Important | 🔥🔥🔥 Critical | 🔥🔥🔥 Critical |
| **Long-Term Thinking** | 🔥 Less critical | 🔥🔥 Important | 🔥🔥🔥 Critical |
| **Comfort with Ambiguity** | 🔥🔥🔥 Critical | 🔥🔥 Important | 🔥 Less critical |
| **Comfort with Being Wrong** | 🔥🔥🔥 Critical | 🔥🔥 Important | 🔥 Less critical |
| **Strategic Thinking** | 🔥🔥 Important | 🔥🔥🔥 Critical | 🔥🔥🔥 Critical |
| **Learning from Failure** | 🔥🔥🔥 Critical | 🔥🔥 Important | 🔥🔥 Important |
| **Building Intuition** | 🔥 Can't yet | 🔥🔥 Developing | 🔥🔥🔥 Critical asset |

### **Example: Same PM Across Stages**

Let's follow a PM through all three stages:

**Sarah, PM at Acme (hypothetical SaaS product)**

#### **2018: 0-to-1 Phase (Finding PMF)**

**Sarah's mindset**:
- Talks to users 20 hours/week
- Ships MVPs weekly (feature flags, manual processes)
- Pivots product 3 times based on feedback
- Runs on gut + user insights (not much data yet)
- Comfortable with 80% of ideas failing

**Example**:
- Builds feature in 2 weeks
- 30% of users try it (good!)
- But they churn (bad!)
- Sarah talks to users, learns why
- Pivots feature completely
- Takes 1 more week
- Better adoption

**Key mindset**: Bias for action + comfort with being wrong + customer obsession

---

#### **2020: 1-to-10 Phase (Scaling)**

**Sarah's mindset shifts**:
- Still talks to users, but now 8 hours/week (has research team)
- Relies heavily on data (SQL queries daily)
- Runs A/B tests on all features
- Thinks about competitive positioning
- Building for scale (can't pivot easily with 10K users)

**Example**:
- Wants to add feature
- First: Checks data (do users need this?)
- Second: Runs prototype test with 20 users
- Third: Builds MVP with feature flag (10% of users)
- Fourth: Measures impact (retention up 5%)
- Fifth: Rolls out to everyone

**Key mindset**: Data-driven + strategic thinking + bias for action (but measured)

---

#### **2023: 10-to-100 Phase (Mature product)**

**Sarah's mindset shifts again**:
- Focuses on long-term strategy (2-3 year roadmap)
- Thinks about platform (should we open an API?)
- Analyzes competitors deeply
- Makes fewer, bigger bets
- Rigorously measures everything

**Example**:
- Considering major redesign
- Timeline: 6 months (can't break things for 100K users)
- Builds business case (impact on retention, revenue)
- Runs extensive user research (100+ users)
- A/B tests with 10% of users for 2 months
- Measures long-term impact (not just short-term metrics)
- Gradually rolls out over 3 months

**Key mindset**: Strategic thinking + long-term thinking + rigorous data analysis

---

### **Key Takeaways**

**1. No single "PM mindset" works for all stages**
- Adapt based on product stage
- What works at 0-to-1 breaks at 10-to-100 (and vice versa)

**2. Common mistake: Using wrong mindset for the stage**
- 0-to-1 PM at mature product → moves too fast, breaks things
- 10-to-100 PM at startup → too slow, over-analyzes

**3. Great PMs adapt their mindset as product evolves**
- Start with customer obsession + bias for action
- Add data-driven + strategic thinking as you scale
- Layer on long-term thinking + intuition at maturity

**4. Core principle across all stages: User-centricity**
- 0-to-1: Talk to every user
- 1-to-10: Systematize user research
- 10-to-100: Institutionalize customer obsession

**5. The best PMs can switch between stages**
- Work on 0-to-1 project within mature company
- Apply mature product discipline to early-stage product

**The ultimate PM skill**: Recognize what stage you're at and adapt your mindset accordingly.
`,
  },
];

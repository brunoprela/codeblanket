/**
 * Section 2: The Product Manager Mindset
 * Product Management Fundamentals Module
 */

import { Section } from '../../../types';

const content = `# The Product Manager Mindset

## Introduction

Technical skills and frameworks are important, but what truly separates great product managers from mediocre ones is **mindset**—the mental models, principles, and ways of thinking that guide decision-making.

The PM mindset isn't something you're born with; it's developed through conscious practice, experience, and reflection. This section explores the core elements of the product management mindset and how to cultivate them.

## The 10 Core Elements of PM Mindset

### 1. **Customer Obsession and Empathy**

**What it means**: Genuinely caring about understanding and solving user problems, not just shipping features.

**Why it matters**: Products exist to solve real problems for real people. Without deep user empathy, PMs build features nobody wants.

**How great PMs think**:
- "What problem is the user actually trying to solve?"
- "Why do users behave this way?"
- "What job is this product being hired to do?"
- "How does this feel from the user's perspective?"

**Example - Airbnb's "11-star experience"**:

Brian Chesky (Airbnb CEO) asks: "What would a **11-star experience** look like?"

- **1-star**: Dirty apartment, host doesn't show up
- **3-star**: Clean apartment, smooth check-in
- **5-star**: Bottle of wine, fresh flowers, personal greeting
- **7-star**: Limo pickup from airport, chef-prepared meal
- **11-star**: Elon Musk picks you up in a rocket, takes you to the International Space Station

**The point**: Push your thinking beyond obvious improvements. What would blow users away?

**Practical application**:
- Spend 10+ hours per month talking to users (not just UX researchers doing it)
- Shadow users in their environment (contextual inquiry)
- Use your own product daily (dogfooding)
- Read every piece of customer feedback
- Feel user pain personally

**Red flags of missing customer obsession**:
- "Users don't know what they want"
- "That's how we've always done it"
- "The data doesn't support that" (without talking to users)
- "We can't make everyone happy"
- Building features without user validation

### 2. **Data-Driven Decision Making**

**What it means**: Using quantitative and qualitative data to inform decisions, not gut instinct or HiPPO (Highest Paid Person's Opinion).

**Why it matters**: Opinions are cheap; data reveals truth. Great PMs combine data with judgment.

**The Data-Driven Framework**:

\`\`\`
1. Define hypothesis
2. Identify metrics
3. Collect data
4. Analyze results
5. Make decision
6. Iterate
\`\`\`

**Example - Netflix removing star ratings**:

**Old system**: 5-star rating system (like IMDb)  
**Problem**: Users rated movies they *thought* they should rate highly (Oscar winners) vs. movies they actually enjoyed watching (guilty pleasures)

**Data insight**: Thumbs up/down engagement was **200% higher** than star ratings  
**Decision**: Replace stars with thumbs  
**Result**: Better predictions, higher engagement

**How to develop this mindset**:
- Learn SQL (query data yourself, don't wait for analysts)
- Define success metrics before building anything
- Run experiments (A/B tests) on everything
- Track leading indicators (not just lagging)
- Question your assumptions ("How do I know this is true?")

**Balance**: Data informs, but doesn't decide

**When data isn't enough**:
- 0 to 1 products (no data yet)
- Long-term strategic bets
- User experience decisions (sometimes you need design taste)
- Ethical questions

**Quote - Jeff Bezos**: "When the data and the anecdotes disagree, the anecdotes are usually right."

### 3. **Bias for Action and Speed**

**What it means**: Preferring to ship and iterate rather than perfect planning. Move fast, learn fast.

**Why it matters**: The biggest risk is shipping nothing. Perfect is the enemy of done.

**Mental model - The Speed/Quality/Scope Trade-off**:

You can pick two:
- Fast + High Quality = Small Scope
- Fast + Large Scope = Lower Quality
- High Quality + Large Scope = Slow

**Great PMs default to: Fast + Small Scope + High Quality**

**Example - Facebook's "Move Fast and Break Things"**:

Early Facebook shipped code to production multiple times per day. Yes, things broke. But they learned faster than anyone else.

(Later changed to "Move Fast with Stable Infrastructure" as the company matured)

**How great PMs ship faster**:
- Define MVP ruthlessly (what's the smallest thing that tests our hypothesis?)
- Ship incomplete features behind feature flags (get early signal)
- Don't wait for perfect data (make reversible decisions quickly)
- Timebox decisions (30 minutes for minor decisions, 1 week for major ones)
- Prefer small batches (many small releases vs. one big launch)

**When to slow down**:
- Irreversible decisions (database migrations, public APIs)
- Security and privacy issues
- Regulatory compliance
- Brand-defining moments

**Real example**:

**Slow approach**: 
- 6 months building "perfect" notification system
- Launch, users hate it
- 6 months wasted

**Fast approach**:
- Week 1: Ship email notifications only (basic)
- Week 2: Measure open rates, gather feedback
- Week 3: Add in-app notifications
- Week 4: Iterate based on data
- Result: Better product in 1 month instead of 6

### 4. **Thinking Like a CEO of the Product**

**What it means**: Taking ownership of product success as if it's your company.

**CEO responsibilities applied to PM**:
- **Strategy**: Where are we going and why?
- **Resource allocation**: What should we work on?
- **Team building**: Who do we need to succeed?
- **Stakeholder management**: Keep everyone aligned
- **P&L responsibility**: How does this make money?
- **Long-term thinking**: What about 3 years from now?

**Key differences from actual CEO**:
- No formal authority
- Smaller scope (product, not company)
- Support from functional leaders

**Example - Satya Nadella on PMs at Microsoft**:

"PMs are the CEOs of their products. They need to think about the full P&L, from engineering costs to sales to customer success. If the product fails, it's on you."

**How to think like a CEO**:
- Ask "What would I do if this were my company?"
- Understand unit economics (cost to build, cost to acquire customer, lifetime value)
- Think 2-3 years ahead, not just next quarter
- Consider all stakeholders (users, engineers, sales, support, executives)
- Make trade-offs (saying yes to X means saying no to Y)

**Red flags**:
- "That's not my problem"
- "I just do what my manager tells me"
- "I don't know our business model"
- Optimizing for your product at the expense of company goals

### 5. **Long-Term Thinking vs. Short-Term Wins**

**What it means**: Balancing immediate results with building sustainable competitive advantages.

**The time horizon framework**:

**Now** (0-3 months): Ship quick wins, fix critical issues  
**Next** (3-12 months): Build core product improvements  
**Later** (1-3 years): Invest in strategic bets and infrastructure

**Amazon's framework**:
- **Inputs**: Activities you can control (features shipped, experiments run)
- **Outputs**: Short-term metrics (engagement, revenue)
- **Outcomes**: Long-term value (customer trust, market leadership)

Focus on inputs and outcomes, not just outputs.

**Example - Google Search's long-term thinking**:

**Short-term**: Maximize ad clicks (more revenue now)  
**Long-term**: Deliver best results even if fewer ad clicks (user trust = sustainable revenue)

**Decision**: Prioritize search quality over short-term ad revenue  
**Result**: Dominant market share for 20+ years

**How to balance**:
- Allocate time: 70% now, 20% next, 10% later (Google's approach)
- Always have 1-2 "long-term bets" on the roadmap
- Resist pressure to only ship short-term wins
- Define "success" for both short and long term

**When short-term wins are necessary**:
- Product-market fit not found yet (need quick feedback loops)
- Company in crisis (survival mode)
- New PM building credibility (show value fast)

**When long-term bets are crucial**:
- Established product (competitive moats)
- Platform products (ecosystem effects take time)
- Technical infrastructure (pay now, benefit for years)

### 6. **Comfort with Ambiguity and Uncertainty**

**What it means**: Thriving when there are no clear answers, undefined problems, and incomplete information.

**Why PM is ambiguous**:
- Unclear if users will want the feature
- Uncertain how long it will take to build
- Unknown if competitors will copy
- Ambiguous success metrics
- Undefined scope

**Great PMs mindset**:
- "I don't have all the answers, but I can figure it out"
- "Ambiguity is an opportunity to define clarity for others"
- "Uncertainty means we need experiments, not perfect plans"

**Framework for handling ambiguity**:

**Step 1: Define what you DO know**
- What problem are we solving?
- Who has this problem?
- What does success look like?

**Step 2: Identify what you DON'T know**
- List assumptions
- Rank by importance and uncertainty
- Focus on "critical unknowns"

**Step 3: Reduce uncertainty through experiments**
- Talk to users (qualitative)
- Run prototypes (validation)
- Build MVPs (real-world test)
- A/B test (quantitative proof)

**Example - Airbnb during COVID**:

**Situation** (March 2020): Travel collapsed overnight. What should Airbnb do?

**Ambiguity**:
- How long will this last?
- Will travel ever recover?
- What do users need now?

**PM approach**:
- Talk to hosts and guests daily
- Notice: Remote workers want to "live anywhere"
- Hypothesis: Long-term stays (1+ month) might be the opportunity
- Quick prototype: "Monthly Stays" feature
- Test and learn

**Result**: Airbnb pivoted to long-term stays, survived, and thrived.

**Red flags**:
- Paralysis ("We need more data before deciding")
- Overconfidence ("I know exactly what will work")
- Avoiding ambiguous projects ("Too risky")

### 7. **Comfort with Being Wrong**

**What it means**: Viewing failures as learning opportunities, not personal deficiencies.

**The reality**: Most product decisions are wrong initially
- 90% of startups fail
- 70% of features don't move metrics
- 50% of A/B tests show no significant difference

**Great PMs mindset**:
- "My job is to be wrong as fast as possible, then iterate"
- "Strong opinions, weakly held"
- "What did we learn from this failure?"

**Example - Jeff Bezos on failure**:

"If you're not failing, you're not innovating enough. Failure and invention are inseparable twins."

Amazon's failures: Fire Phone, Amazon Destinations, Amazon Auctions
Amazon's wins: AWS, Prime, Alexa

The wins wouldn't exist without the failures.

**How to develop this mindset**:
- Run premortem ("What could go wrong?")
- Write down predictions before launching
- Celebrate learning, not just wins
- Share failures openly (build psychological safety)
- Iterate based on feedback

**Real example - Slack's "Searchable Log of All Conversation and Knowledge"**:

**Original product**: Gaming company communication tool  
**Hypothesis**: Gamers will love this for coordinating  
**Result**: Failed (wrong market)

**Pivot**: Tried with tech companies instead  
**Result**: Product-market fit! → Became Slack

**Key insight**: They were wrong about the market, but right about the product. Being wrong quickly allowed them to find the right direction.

### 8. **Strategic Thinking (Vision → Execution)**

**What it means**: Connecting daily decisions to long-term vision and company strategy.

**The strategy hierarchy**:

\`\`\`
Company Mission (Why we exist)
    ↓
Company Strategy (How we'll win)
    ↓
Product Vision (What we're building)
    ↓
Product Strategy (Our approach)
    ↓
Product Roadmap (What we'll ship)
    ↓
Features (What we're building this quarter)
\`\`\`

**Great PMs can articulate**:
- "This feature ladders up to [product goal]"
- "This product goal supports [company strategy]"
- "This aligns with our mission because..."

**Example - Stripe's strategy pyramid**:

**Mission**: Increase the GDP of the internet  
**Company Strategy**: Make online payments easy for developers  
**Product Vision**: Best payment API, comprehensive financial tools  
**Product Strategy**: Developer-first, global, compliant  
**Roadmap**: Stripe Checkout, Billing, Terminal, etc.  
**Feature**: "Payment Links" (zero-code payment acceptance)

**How to develop strategic thinking**:
- Always ask "why" 3 times (get to root strategy)
- Read company strategy docs (most PMs don't!)
- Connect your roadmap to company OKRs
- Think about competitive moats ("Why can't competitors copy this?")
- Consider second-order effects ("If we do X, what happens next?")

**Red flags**:
- Building features disconnected from strategy
- Can't explain how your work supports company goals
- "My manager told me to build this" (without knowing why)

### 9. **Learning from Failures Systematically**

**What it means**: Treating failures as data, not disasters.

**The failure learning framework**:

**Step 1: Acknowledge the failure**
- Feature didn't move metrics
- Launch went poorly
- Lost to competitor

**Step 2: Analyze what happened**
- What did we expect?
- What actually happened?
- What caused the gap?

**Step 3: Extract learnings**
- What did we learn about users?
- What did we learn about our process?
- What assumptions were wrong?

**Step 4: Apply learnings**
- How will we do it differently next time?
- What guardrails should we add?
- What should we double down on?

**Step 5: Share learnings**
- Write postmortem
- Share with team and company
- Build institutional knowledge

**Example - Superhuman's "Not Quite PMF" Moment**:

**Launch**: Superhuman launched to much hype  
**Problem**: Growth stalled, users churning  
**Analysis**: Ran PMF survey → Only 22% said "very disappointed" without it (need 40%)  
**Learning**: Product wasn't essential enough for most users  
**Action**: Segmented users, doubled down on "very disappointed" segment  
**Result**: Reached PMF by focusing on core users who truly loved it

**Key insight**: They treated "not quite PMF" as data, not failure. Systematically analyzed and iterated.

### 10. **Building Product Intuition Over Time**

**What it means**: Developing a "sixth sense" for what will work through experience and pattern recognition.

**Product intuition is NOT**:
- Guessing randomly
- Ignoring data
- Trusting gut over evidence

**Product intuition IS**:
- Pattern matching from past experience
- Quickly identifying problems before data confirms
- Sensing when something "feels off"
- Having hypotheses before testing

**How product intuition develops**:

\`\`\`
Experience → Patterns → Mental Models → Intuition
\`\`\`

**Example - Steve Jobs' intuition**:

Jobs famously said: "People don't know what they want until you show it to them."

His intuition came from:
- Decades of experience
- Deep understanding of design principles
- Seeing how users actually behave (vs. what they say)
- Pattern matching from previous products

**How to build product intuition**:
1. **Use lots of products**: Try every new app, analyze why it works/doesn't work
2. **Study product history**: Why did X succeed? Why did Y fail?
3. **Reflect on decisions**: What did you expect? What happened? Why the gap?
4. **Learn frameworks**: Pattern match from PM principles
5. **Talk to users constantly**: Develop sense for what resonates

**Intuition + Data is powerful**:
- Intuition generates hypotheses
- Data validates or refutes
- Intuition helps interpret ambiguous data

**Example - Spotify Discover Weekly**:

**Intuition**: Users might like algorithmically curated playlists (based on seeing personalization work elsewhere)  
**Data**: Ran experiment, validated with usage metrics  
**Result**: Massive hit (40M+ users)

**Key**: Intuition suggested the idea; data proved it worked.

## The Growth Mindset for PMs

### Fixed vs. Growth Mindset

**Fixed mindset**: "I'm either good at PM or I'm not"  
**Growth mindset**: "I can develop PM skills through practice"

**Carol Dweck's research**:
- People with growth mindset achieve more
- They view challenges as opportunities
- They learn from criticism
- They persist despite setbacks

**How to cultivate growth mindset as a PM**:

1. **View skills as learnable**: "I'm not good at X yet, but I can learn"
2. **Embrace challenges**: Take on hard problems
3. **Learn from criticism**: Ask for feedback, don't get defensive
4. **Celebrate effort**: Not just outcomes
5. **Study great PMs**: Learn from their approaches

**Example areas to develop**:
- Not technical enough? → Take coding course, learn SQL
- Weak at communication? → Practice writing, get feedback
- Poor at stakeholder management? → Read *Influence* by Cialdini, practice
- Missing design sense? → Study UX principles, do design critiques

## Common Mindset Pitfalls

### Pitfall #1: Feature Factory Mindset

**Problem**: Measuring success by features shipped, not impact

**Better mindset**: Measure success by outcomes (user value, business results)

### Pitfall #2: Perfectionism

**Problem**: Waiting for perfect product before launching

**Better mindset**: Ship MVPs, iterate based on feedback

### Pitfall #3: Analysis Paralysis

**Problem**: Over-analyzing, never deciding

**Better mindset**: Make reversible decisions quickly, irreversible decisions carefully

### Pitfall #4: Not Invented Here (NIH)

**Problem**: Rejecting ideas from others

**Better mindset**: "Strong opinions, weakly held." Listen to all input.

### Pitfall #5: Hero Complex

**Problem**: Trying to do everything yourself

**Better mindset**: PM is orchestrator, not superhero. Enable others.

## Developing Your PM Mindset: Practical Exercises

### Exercise 1: Customer Obsession

**This week**:
- Talk to 5 users about their problems
- Use a competitor's product for 1 hour
- Read 20 customer support tickets
- Ask yourself: "What am I missing about user needs?"

### Exercise 2: Data-Driven Thinking

**This week**:
- Define success metrics for your current project
- Write SQL query to measure it
- Look at data daily
- Question one assumption with data

### Exercise 3: Bias for Action

**This week**:
- Identify one thing you've been overthinking
- Make a decision in 30 minutes
- Ship something small this week
- Reflect: What did you learn?

### Exercise 4: Strategic Thinking

**This week**:
- Write down your company's mission and strategy
- Connect your current work to it
- Ask yourself: "Does this feature ladder up to our strategy?"
- If not, why are you building it?

### Exercise 5: Learning from Failure

**This week**:
- Write down one thing that didn't work
- Analyze why (be honest)
- Extract learnings
- Share with team

## Conclusion

The PM mindset isn't something you either have or don't have—it's a set of mental habits you cultivate over time through deliberate practice.

**The 10 core elements**:
1. Customer obsession
2. Data-driven decisions
3. Bias for action
4. Think like a CEO
5. Balance short and long term
6. Comfort with ambiguity
7. Comfort with being wrong
8. Strategic thinking
9. Learn from failures systematically
10. Build product intuition

**Start with one**: Pick the mindset element you're weakest in. Focus on it for 30 days. Practice daily. Reflect weekly. Iterate.

Over time, these mindsets become automatic. You'll instinctively think about users, question assumptions with data, ship faster, and connect decisions to strategy.

**That's when you become a truly great product manager.**

## Key Takeaways

1. **Mindset matters more than frameworks**: You can learn tools; mindset determines how you apply them
2. **Customer obsession is non-negotiable**: Without it, everything else fails
3. **Balance intuition and data**: Use both, not either-or
4. **Ship fast, learn fast**: Bias for action beats perfect planning
5. **Think CEO, act PM**: Ownership mindset without formal authority
6. **Embrace failure**: View it as learning, not personal deficiency
7. **Develop pattern recognition**: Product intuition comes from experience + reflection
8. **Growth mindset**: All PM skills are learnable
9. **Strategic thinking**: Connect daily work to long-term vision
10. **Build habits**: Practice these mindsets daily until they're automatic

## Practical Self-Assessment

Rate yourself (1-5 scale) on each mindset element:

1. **Customer obsession**: Do I talk to users weekly?
2. **Data-driven**: Do I make decisions based on data, not opinion?
3. **Bias for action**: Do I ship quickly and iterate?
4. **CEO thinking**: Do I own outcomes, not just outputs?
5. **Long-term thinking**: Do I balance quick wins with strategic bets?
6. **Comfort with ambiguity**: Do I thrive in unclear situations?
7. **Comfort with being wrong**: Do I learn from failures?
8. **Strategic thinking**: Can I connect features to strategy?
9. **Systematic learning**: Do I extract lessons from failures?
10. **Building intuition**: Am I developing pattern recognition?

**Pick your lowest score. Make it your focus for the next month.**

Remember: Great PMs aren't born—they're made through deliberate practice of these mindsets.
`;

export const pmMindsetSection: Section = {
  id: 'pm-mindset',
  title: 'The Product Manager Mindset',
  content,
};

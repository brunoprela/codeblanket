/**
 * Section 4: Product Discovery vs Product Delivery
 * Product Management Fundamentals Module
 */

import { ModuleSection } from '../../../types';

const content = `# Product Discovery vs Product Delivery

## Introduction

One of the most critical PM concepts is the distinction between **product discovery** (figuring out what to build) and **product delivery** (building it). Many product failures come from great delivery of the wrong product. This section explores both phases, how to balance them, and frameworks for each.

**The fundamental insight**: Discovery and delivery require different mindsets, processes, and success metrics.

## What is Product Discovery?

**Definition**: The process of figuring out **what** to build and **why** it will create value for users and the business.

**Core question**: "Are we building the right thing?"

**Marty Cagan's definition** (*Inspired*):
*"The purpose of product discovery is to address these critical risks:*
- *Will the customer buy this (or choose to use it)?*
- *Can the user figure out how to use this?*
- *Can our engineers build this?*
- *Can our stakeholders support this?"*

### **Discovery Activities**

1. **User research** (interviews, observations, surveys)
2. **Problem validation** (does this problem exist and matter?)
3. **Solution exploration** (what are different ways to solve it?)
4. **Prototyping** (test ideas cheaply before building)
5. **Testing assumptions** (what must be true for this to work?)

### **Discovery Outputs**

- User insights and validated problems
- Solution hypotheses
- Prototypes and mockups
- Test results and learning
- Decision to build (or not build)

### **Example: Slack's Discovery Process**

**Discovery phase** (before building):
- Problem: Team communication is fragmented (email, meetings, chat)
- Validation: Talked to 100+ teams, confirmed pain point
- Solution exploration: Tried different UIs, features, integrations
- Prototyping: Built clickable prototype, tested with 10 teams
- Result: Validated that teams wanted "searchable log of all conversation"

**Key insight**: Spent 3 months in discovery before writing production code.

## What is Product Delivery?

**Definition**: The process of **building and shipping** the product.

**Core question**: "Are we building the thing right?"

### **Delivery Activities**

1. **Requirements definition** (PRDs, user stories)
2. **Design and engineering** (building the product)
3. **QA and testing** (ensuring quality)
4. **Release and deployment** (shipping to users)
5. **Monitoring and iteration** (measuring impact)

### **Delivery Outputs**

- Shipped product or feature
- Working code in production
- User-facing functionality
- Measured impact on metrics

### **Example: Slack's Delivery Process**

**Delivery phase** (after discovery):
- Wrote comprehensive PRD
- Eng team built backend infrastructure
- Design team created polished UI
- Beta launch to select customers
- Full launch after iteration
- Result: Product that users loved (because discovery was done well)

## Discovery vs Delivery: Key Differences

| Dimension | Discovery | Delivery |
|-----------|-----------|----------|
| **Goal** | Validate what to build | Build and ship it |
| **Question** | "Is this the right product?" | "Did we build it well?" |
| **Risk** | Building wrong thing | Building slowly or poorly |
| **Speed** | Fast, cheap experiments | Methodical, quality-focused |
| **Mindset** | Learning, exploring, failing | Executing, shipping, delivering |
| **Success** | Validated learning | Shipped product |
| **Cost** | Low (prototypes, tests) | High (engineering time) |
| **Reversibility** | Highly reversible | Less reversible once shipped |
| **Team involvement** | PM, design, few engineers | Full engineering team |
| **Output** | Insights, prototypes | Working code |
| **Failure mode** | Build wrong product | Ship slowly or with bugs |

## The Discovery-Delivery Framework

\`\`\`
Discovery Phase                  Delivery Phase
--------------                  --------------
Problem Definition       →      Requirements (PRD)
User Research           →      Design
Solution Exploration    →      Engineering
Prototyping             →      Development
Testing                 →      QA
Learning                →      Launch
Iteration               →      Monitoring
\`\`\`

**The flow**:
1. Discovery identifies the right problem and solution
2. Delivery builds and ships that solution
3. Post-launch measurement informs next discovery cycle

## Dual-Track Agile

**The framework**: Run discovery and delivery in parallel, but for different features.

\`\`\`
Week 1   Week 2   Week 3   Week 4   Week 5   Week 6
--------------------------------------------------
Discovery:   Feature A → Feature B → Feature C
Delivery:             Feature A → Feature B
\`\`\`

**How it works**:
- **Discovery track**: PM + Designer + 1-2 Engineers exploring next features
- **Delivery track**: Full engineering team building validated features
- Features enter delivery only after discovery validates them

**Benefits**:
- Continuous discovery (not just waterfall "requirements phase")
- Engineering always has validated features to build
- Reduces risk of building wrong thing

**Example: Spotify**

- **Discovery team**: Testing recommendation algorithm concepts
- **Delivery team**: Building validated Discover Weekly feature
- Result: Discovery happens continuously while delivery ships steadily

## Common Discovery Mistakes

### **Mistake #1: Skipping Discovery**

**What it looks like**:
- "We know what to build" (without user validation)
- Jump straight from idea to PRD
- No prototyping or testing

**Why it fails**:
- 70% of features don't move metrics (Booking.com data)
- Building wrong thing wastes months of engineering time

**Fix**: Mandate discovery before delivery

### **Mistake #2: Discovery Paralysis**

**What it looks like**:
- Endless research, never shipping
- "We need more user interviews"
- Over-analyzing, under-delivering

**Why it fails**:
- Perfect is enemy of done
- Learning happens from real usage, not just research

**Fix**: Timebox discovery (2-4 weeks max), then ship MVP

### **Mistake #3: Fake Discovery**

**What it looks like**:
- "Discovery" = confirming pre-decided solution
- Leading questions in user interviews
- Ignoring contradictory evidence

**Why it fails**:
- Confirmation bias
- Build what you wanted, not what users need

**Fix**: Genuinely open exploration, test to fail

### **Mistake #4: No Prototype Testing**

**What it looks like**:
- User research → Straight to code
- No design mockups tested with users
- First user exposure is production

**Why it fails**:
- Usability problems discovered after shipping
- Expensive to fix after engineering time invested

**Fix**: Always test interactive prototypes before coding

## Discovery Methods and Tools

### **Method 1: User Interviews (Qualitative)**

**When to use**: Understanding problems, motivations, context

**How**:
- 10-20 interviews per project
- Open-ended questions
- Jobs-to-be-Done framework
- Look for patterns

**Example questions**:
- "Tell me about the last time you tried to [do task]"
- "What was frustrating about that?"
- "How do you solve this today?"

### **Method 2: Surveys (Quantitative)**

**When to use**: Validating hypotheses at scale

**How**:
- 100+ responses for statistical significance
- Mix of multiple choice and open-ended
- NPS, feature prioritization, problem validation

**Example**: Sean Ellis PMF Survey
- "How would you feel if you could no longer use [product]?"
- 40%+ saying "very disappointed" = PMF signal

### **Method 3: Prototype Testing**

**When to use**: Before writing code

**How**:
- Figma interactive prototypes
- 5-10 users testing
- Think-aloud protocol
- Observe where they struggle

**Tools**: Figma, Framer, Maze, Lookback

### **Method 4: Fake Door Tests**

**When to use**: Validating demand for feature

**How**:
- Add button/link for non-existent feature
- Measure click rate
- Show "coming soon" message
- Learn from interest level

**Example**: Dropbox tested "Dropbox for Business" with fake door
- High click rate validated enterprise demand
- Built actual feature after validation

### **Method 5: Wizard of Oz MVP**

**When to use**: Testing complex features manually

**How**:
- Build simple frontend
- Do backend manually
- Users think it's automated
- Validate value before automation

**Example**: Zapier early version
- UI looked automated
- Founders manually created integrations
- Validated demand before building automation

## Delivery Best Practices

### **Practice #1: Clear Requirements**

**Good PRD structure**:
1. Problem statement (why)
2. User stories (who, what)
3. Success metrics (how we measure)
4. Design mockups (what it looks like)
5. Technical considerations (how to build)
6. Out of scope (what we're NOT building)

### **Practice #2: Agile/Scrum Process**

**Sprint structure** (2-week sprints typical):
- **Monday**: Sprint planning
- **Daily**: 15-min standups
- **Mid-sprint**: Design reviews, demos
- **Friday**: Sprint review, retrospective

### **Practice #3: Feature Flags**

**What**: Code deployed but hidden behind flag

**Benefits**:
- Deploy anytime (not tied to release)
- Test with subset of users (10% rollout)
- Kill switch if problems
- A/B test easily

**Tools**: LaunchDarkly, Split, Optimizely

### **Practice #4: Beta Programs**

**What**: Launch to select users first

**Benefits**:
- Real user feedback before full launch
- Find bugs in lower-stakes environment
- Build champions who help with launch

**Example**: Gmail beta (by invitation only)

### **Practice #5: Progressive Rollout**

**What**: Gradual release from 1% → 100%

**Schedule**:
- Day 1: 1% of users (catch major bugs)
- Day 3: 10% (measure impact)
- Day 7: 50% (validate at scale)
- Day 14: 100% (full rollout)

**Benefit**: Reduces blast radius of issues

## Balancing Discovery and Delivery

### **The Time Allocation Question**

**How much time on discovery vs. delivery?**

**It depends on product stage**:

**0-to-1 (Finding PMF)**:
- Discovery: 60-70%
- Delivery: 30-40%
- Why: Still figuring out what to build

**1-to-10 (Scaling)**:
- Discovery: 30-40%
- Delivery: 60-70%
- Why: Product validated, focus on shipping

**10-to-100 (Mature)**:
- Discovery: 20-30%
- Delivery: 70-80%
- Why: Optimizing existing product

### **The Discovery-Delivery Balance**

**Red flag: Too much discovery**
- Endless research, no shipping
- "We need more data"
- Analysis paralysis

**Red flag: Too little discovery**
- Ship features that don't work
- High failure rate
- Team frustration

**Right balance**:
- Discovery ahead of delivery (2-4 weeks)
- Every feature validated before build
- Shipping regularly (weekly or bi-weekly)

## Continuous Discovery Practices

**Teresa Torres's framework** (*Continuous Discovery Habits*):

### **Practice #1: Weekly Touchpoints with Customers**

- Minimum: 2 hours/week talking to users
- PMs, designers, engineers involved
- Rotating responsibility
- Build continuous understanding

### **Practice #2: Opportunity Solution Trees**

- Map opportunities (user problems)
- Generate multiple solutions per opportunity
- Test cheapest/fastest solution first
- Iterate based on learning

### **Practice #3: Assumption Testing**

For each feature, identify:
1. **Desirability**: Do users want this?
2. **Usability**: Can they figure it out?
3. **Feasibility**: Can we build it?
4. **Viability**: Should we build it (business)?

Test riskiest assumption first.

### **Practice #4: Lightweight Prototyping**

- Sketch → Lo-fi wireframe → Hi-fi mockup → Interactive prototype
- Test at each stage
- Fail cheap, iterate fast

## Real-World Examples

### **Example 1: Instagram Stories**

**Discovery**:
- Snapchat's ephemeral content was working
- Instagram interviewed users about sharing behavior
- Insight: Users wanted to share without permanent record
- Prototyped Stories concept, tested with users
- Validated high interest

**Delivery**:
- Built Stories feature in 3 months
- Beta with select creators
- Full rollout after polish
- Result: Massive success (killed Snapchat's growth)

**Key**: Good discovery identified right opportunity

### **Example 2: Google Wave (Failure)**

**Discovery mistake**:
- Google engineers designed collaboration tool
- Minimal user input in discovery
- Assumed users wanted real-time everything
- No prototype testing with target users

**Delivery**:
- Built complex, technically impressive product
- Launched to big fanfare
- Users confused, didn't adopt

**Result**: Shut down after 2 years

**Key**: Great delivery of wrong product

### **Example 3: Superhuman (Success)**

**Discovery**:
- Surveyed users monthly: "How disappointed without Superhuman?"
- Initially only 22% "very disappointed" (below 40% PMF threshold)
- Analyzed who loved it vs. who didn't
- Doubled down on "very disappointed" segment

**Delivery**:
- Built features that segment wanted
- Polished to extremely high quality
- High-touch onboarding

**Result**: Reached PMF systematically

**Key**: Rigorous discovery guided delivery

## Key Takeaways

1. **Discovery validates WHAT to build; delivery builds it well**
2. **Most failures come from building wrong thing, not building poorly**
3. **Always do discovery before committing engineering resources**
4. **Use prototypes to test before code**
5. **Dual-track Agile balances discovery and delivery**
6. **Time allocation depends on product stage (0-to-1 vs. mature)**
7. **Continuous discovery = weekly customer touchpoints**
8. **Test riskiest assumptions first**
9. **Discovery should be cheap and fast (days/weeks, not months)**
10. **Delivery requires clear requirements and quality execution**

## Practical Exercise

For your current feature idea:

**Discovery checklist**:
- [ ] Have you validated the problem with 10+ users?
- [ ] Have you explored 3+ different solutions?
- [ ] Have you built and tested an interactive prototype?
- [ ] Have you identified your riskiest assumption and tested it?
- [ ] Do you have evidence users will use this?

**Only proceed to delivery if you answered yes to all.**

**Delivery checklist**:
- [ ] Clear PRD with success metrics
- [ ] Design mockups validated with users
- [ ] Engineering estimates and feasibility confirmed
- [ ] Plan for beta/progressive rollout
- [ ] Monitoring and rollback plan

Remember: **Great discovery enables great delivery. Skip discovery at your own risk.**
`;

export const discoveryVsDeliverySection: ModuleSection = {
  id: 'discovery-vs-delivery',
  title: 'Product Discovery vs Product Delivery',
  content,
};

/**
 * Section 9: Writing for Product Managers
 * Product Management Fundamentals Module
 */

import { ModuleSection } from '../../../types';

const content = `# Writing for Product Managers

## Introduction

Writing is a core PM skill. PMs write PRDs, strategy docs, roadmaps, emails, Slack messages, and more. Clear writing drives alignment, prevents confusion, and accelerates execution. Poor writing wastes time, creates misalignment, and frustrates teams.

This section covers how to write effectively as a PM.

## Why Writing Matters for PMs

### **PMs Write Constantly**

**Documents PMs write**:
- **PRDs** (Product Requirements Documents)
- **Strategy documents** (product vision, roadmaps)
- **Specifications** (feature specs, technical requirements)
- **Emails** (to stakeholders, customers, teams)
- **Slack messages** (quick updates, questions)
- **Presentations** (decks for leadership, customers)
- **Documentation** (help articles, release notes)

**Estimate**: 30-50% of PM time is writing.

### **Good Writing Accelerates Teams**

**Clear writing**:
- Reduces meetings (answers questions upfront)
- Prevents confusion (everyone understands requirements)
- Enables async work (teams in different timezones align)
- Builds trust (stakeholders feel informed)

**Poor writing**:
- Creates endless meetings ("What did you mean by X?")
- Causes misalignment (teams build wrong thing)
- Frustrates stakeholders (unclear communication)
- Wastes time (rework, clarifications)

### **Writing Is Thinking**

**Writing forces clarity**:
- Vague thoughts become precise ideas
- Assumptions surface
- Gaps in logic reveal themselves
- Better decisions emerge

**Jeff Bezos banned PowerPoint at Amazon** in favor of 6-page memos. Why? Writing forces rigorous thinking.

## Principles of Effective PM Writing

### **Principle #1: Know Your Audience**

**Different audiences need different information.**

**For Engineers**:
- Technical details matter
- "Why" and "what" are most important
- Specs should be detailed but not prescriptive
- Include edge cases and constraints

**For Designers**:
- User problems and context
- Success metrics
- Examples and references
- Constraints and guidelines

**For Executives**:
- Bottom line upfront (BLUF)
- Business impact (revenue, retention, cost)
- Strategic rationale
- Ask (what decision do you need?)

**For Sales/Marketing**:
- Customer benefits (not features)
- Competitive positioning
- Messaging and talking points
- Use cases and examples

**Example: Announcing same feature to different audiences**:

**To Engineering**:
"We're building real-time Salesforce sync. Requirements: bidirectional sync, <5 sec latency, handle 10K objects/day. Technical constraints: use Salesforce REST API, implement webhook listeners, queue for async processing. Edge cases: handle rate limits, conflicts, deletes."

**To Sales**:
"We're launching Salesforce integration next month. Customer benefit: No more manual CRM updates—changes sync instantly. Use case: When sales rep updates deal in Salesforce, it appears in our product automatically (and vice versa). This unblocks 8 deals worth $750K."

**To Executives**:
"Salesforce integration ships Q1, unlocking $750K pipeline (8 deals currently blocked). Investment: 3 engineering weeks. ROI: $250K per engineering week. Competitive: All competitors have this, table stakes. Recommendation: Approve for Q1."

**Same feature, different framing for each audience.**

### **Principle #2: Start with the Bottom Line**

**Don't bury the lead. Lead with the conclusion.**

**Bad structure** (suspense novel):
- Background and context
- Analysis and options
- Pros and cons
- Recommendation (finally!)
- Reader gave up by page 3

**Good structure** (newspaper article):
- **Bottom line** (recommendation, decision, key point)
- **Why it matters** (impact, rationale)
- **Supporting details** (data, analysis, options)
- **Appendix** (optional deep dives)

**Example**:

**Bad email** (buried lead):
"Over the past 3 months, we've been analyzing user feedback from 47 customers. We conducted 12 user interviews and reviewed 200 support tickets. Users mentioned various pain points including X, Y, and Z. We created a scoring framework to evaluate these. After analysis, we believe..."

**Good email** (bottom line first):
"**Recommendation**: Build Salesforce integration in Q1.

**Why**: Unblocks $750K pipeline (8 deals), all competitors have it, table stakes for enterprise.

**Investment**: 3 engineering weeks

**Supporting data**: [details below]"

**Busy executives read first paragraph. Make it count.**

### **Principle #3: Be Concise**

**Shorter is better. Respect readers' time.**

**Bad** (wordy):
"At this point in time, we are currently experiencing a significant amount of customer feedback that indicates there is a substantial need for the implementation of a Salesforce integration capability within our product offering."

**Good** (concise):
"Customers are requesting Salesforce integration."

**Tips for conciseness**:
- Remove filler words ("at this point in time" → "now")
- Use active voice ("is being built by engineering" → "engineering is building")
- Cut redundancy ("past history" → "history")
- One idea per sentence

**Example revision**:

**Before** (120 words):
"In our recent conversations with the sales team, it has come to our attention that there have been multiple instances where potential customers have expressed concerns regarding the absence of a Salesforce integration in our current product offering. This feedback has been consistent across multiple sales conversations, and it appears to be emerging as a pattern that is worth paying attention to. Given the frequency with which this topic has been raised, and considering the potential impact on our ability to close deals with enterprise customers, we believe it would be prudent to consider prioritizing the development of this integration capability in the near future."

**After** (30 words):
"Sales reports Salesforce integration is blocking multiple enterprise deals. This is a consistent pattern (8 customers, $750K pipeline). Recommendation: Prioritize for Q1 to unlock these deals."

**75% shorter, equally clear.**

### **Principle #4: Use Structure**

**Structure helps readers navigate.**

**Good structure elements**:
- **Headers** (break up text, make scannable)
- **Bullets** (list items clearly)
- **Numbered lists** (for sequences)
- **Tables** (compare options)
- **Bold** (highlight key points)
- **Short paragraphs** (3-4 sentences max)

**Bad structure**:
"We need to build X because of Y and also Z. The problem is A and B and C. We should do D or E or F. The timeline is G and the cost is H."

**Good structure**:
\`\`\`markdown
## Problem
Users struggle with X, causing Y impact.

## Proposed Solution
Build [feature] to solve X.

## Why This Matters
- Business impact: $500K revenue
- User impact: 10K users affected
- Strategic importance: Competitive gap

## Options Considered
1. Option A: [Pros/Cons]
2. Option B: [Pros/Cons]
3. Option C: [Pros/Cons]

## Recommendation
Option B (rationale).

## Timeline & Resources
- 4 weeks, 2 engineers
- Ships Q1
\`\`\`

**Scannable, logical, clear.**

### **Principle #5: Show, Don't Tell**

**Use examples, data, and specifics.**

**Weak** (abstract):
"Users are frustrated with the current experience."

**Strong** (specific):
"Users abandon checkout 40% of the time. User quote: 'I couldn't figure out how to apply my discount code and gave up.'"

**Weak**:
"This feature will improve engagement."

**Strong**:
"Spotify saw 25% increase in daily active users after adding personalized playlists. We expect similar impact."

**Tips**:
- Replace "many" with numbers ("50% of users")
- Include quotes (from users, customers, team)
- Show data (graphs, charts, tables)
- Give examples (user scenarios, use cases)

### **Principle #6: Make It Actionable**

**Every document should have clear next steps.**

**Bad ending**:
"Let me know your thoughts."

**Good ending**:
"**Next Steps**:
1. Leadership: Approve/reject by Friday
2. If approved, Engineering begins Q1
3. I'll update stakeholders by Monday"

**Or**:

"**Decisions Needed**:
- [ ] Approve Q1 prioritization (Owner: VP Product, By: Dec 15)
- [ ] Allocate 2 engineers (Owner: EM, By: Dec 20)
- [ ] Confirm go-to-market plan (Owner: Marketing, By: Jan 5)"

**Make it clear who does what by when.**

## Key Document Types for PMs

### **Type #1: PRD (Product Requirements Document)**

**Purpose**: Define what to build and why.

**Structure**:

\`\`\`markdown
# [Feature Name] PRD

## Problem Statement
[What user problem are we solving?]

## User Needs
[What do users need to accomplish?]

## Success Metrics
[How do we measure success?]

## Requirements
[What must the feature do?]
- Must have: [Non-negotiable]
- Should have: [Important but flexible]
- Nice to have: [Defer to V2]

## User Flows
[How do users interact with feature?]

## Edge Cases
[What happens when...?]

## Out of Scope
[What are we NOT building?]

## Open Questions
[What needs to be decided?]

## Timeline
[When does this ship?]
\`\`\`

**Tips**:
- Focus on "what" and "why," not "how" (let engineering decide "how")
- Include user quotes and research
- Define success metrics upfront
- List what's out of scope (prevents scope creep)

### **Type #2: Strategy Document**

**Purpose**: Define direction and priorities.

**Structure**:

\`\`\`markdown
# [Product Name] Strategy - [Year]

## Vision
[Where are we going?]

## Market Context
[What's happening in the market?]

## Strategic Goals
[What are we trying to achieve?]
1. Goal 1 (Metric: X)
2. Goal 2 (Metric: Y)
3. Goal 3 (Metric: Z)

## Key Initiatives
[What are we building to achieve goals?]
- Initiative 1: [Description] (Impact: X)
- Initiative 2: [Description] (Impact: Y)

## What We're NOT Doing
[Conscious trade-offs]

## Success Metrics
[How do we measure success?]

## Risks & Mitigations
[What could go wrong?]
\`\`\`

### **Type #3: Decision Document**

**Purpose**: Get stakeholder approval for decisions.

**Structure**:

\`\`\`markdown
# Decision: [Topic]

## Decision Needed
[What are we deciding?]

## Recommendation
[What should we do?]

## Why This Matters
[Business/user impact]

## Options Considered
1. Option A: [Pros/Cons]
2. Option B: [Pros/Cons]
3. Option C: [Pros/Cons]

## Recommendation Rationale
[Why Option B?]

## Decision Owners
[Who decides?]

## Timeline
[When do we need decision?]
\`\`\`

**Amazon's "6-pager" format** is a great decision document template.

### **Type #4: Status Update**

**Purpose**: Keep stakeholders informed.

**Structure**:

\`\`\`markdown
# [Project Name] - Weekly Update

## Summary
[One sentence: Are we on track?]

## Completed This Week
- [Accomplishment 1]
- [Accomplishment 2]

## Planned Next Week
- [Plan 1]
- [Plan 2]

## Blockers
- [Blocker 1] (Owner: X, Resolving by: Y)

## Risks
- [Risk 1] (Mitigation: X)

## Metrics
- [Key metric]: [Current value] (Target: X)
\`\`\`

**Keep it brief** (5 min read max).

## Writing for Different Mediums

### **Email Writing**

**Best practices**:

**1. Subject line is critical**:
- Bad: "Update"
- Good: "Q1 Roadmap Approval Needed by Friday"

**2. First sentence = bottom line**:
"Recommending we prioritize Salesforce integration in Q1."

**3. Keep it short**:
- <150 words for updates
- <500 words for decisions

**4. Make action clear**:
"Please approve by Friday so we can start Q1."

**5. Use formatting**:
- **Bold** key points
- Bullets for lists
- Line breaks for readability

### **Slack Writing**

**Best practices**:

**1. One message per topic** (not wall of text)

**2. Use threads** (keep conversations organized)

**3. Tag people** (@name) when you need response

**4. Lead with context**:
- Bad: "Can you review this?"
- Good: "[Salesforce Integration PRD] Can you review by EOD? Need your feedback on technical approach."

**5. Use formatting**:
\`\`\`
*bold*
_italic_
> quote
- bullets
\`\`\`

### **Presentation Writing**

**Best practices**:

**1. One idea per slide**

**2. Minimal text** (slides are visual aids, not documents)

**3. Headlines tell the story**:
- Bad headline: "Q1 Roadmap"
- Good headline: "Q1 Focus: Enterprise Features to Unlock $2M Pipeline"

**4. Use visuals** (charts, diagrams, screenshots)

**5. Include "So what?"** on every slide

## Common Writing Mistakes

### **Mistake #1: Jargon Overload**

**Bad**:
"We're leveraging our core competencies to synergize cross-functional stakeholder alignment on strategic initiatives that will drive measurable KPIs."

**Good**:
"We're building features that increase revenue."

**Use plain English. Avoid buzzwords.**

### **Mistake #2: Passive Voice**

**Bad** (passive):
"The feature is being built by engineering."

**Good** (active):
"Engineering is building the feature."

**Passive voice is wordy and unclear.**

### **Mistake #3: Vague Language**

**Bad**:
"Many users want this feature soon."

**Good**:
"50% of users (5,000 people) requested this feature. Shipping Q1."

**Be specific.**

### **Mistake #4: No Clear Ask**

**Bad**:
"Let me know your thoughts on this proposal."

**Good**:
"Please approve this proposal by Friday so we can start development Monday."

**Every document should have clear next steps.**

### **Mistake #5: Too Long**

**Bad**: 10-page PRD that nobody reads

**Good**: 2-page PRD with appendix for details

**Respect readers' time. Be concise.**

## Editing Your Own Writing

### **The Editing Process**

**1. Write first draft** (don't edit while writing)
**2. Walk away** (take a break, clear your head)
**3. Read aloud** (catches awkward phrasing)
**4. Cut 30%** (remove unnecessary words)
**5. Check structure** (is it scannable?)
**6. Get feedback** (ask colleague to review)
**7. Ship it** (done is better than perfect)

### **Self-Editing Checklist**

- [ ] Bottom line upfront?
- [ ] Audience-appropriate?
- [ ] Concise (no filler words)?
- [ ] Structured (headers, bullets)?
- [ ] Specific (data, examples)?
- [ ] Actionable (clear next steps)?
- [ ] Proofread (no typos)?

## Key Takeaways

1. **Writing is thinking** - Clear writing reflects clear thinking
2. **Know your audience** - Engineers, designers, executives need different information
3. **Bottom line first** - Don't bury the lead
4. **Be concise** - Shorter is better
5. **Use structure** - Headers, bullets, formatting
6. **Show, don't tell** - Use data, examples, specifics
7. **Make it actionable** - Clear next steps always
8. **Edit ruthlessly** - Cut 30% of first draft
9. **Practice daily** - Writing is a skill that improves with practice

## Practical Exercise

**Take a recent email or document you wrote**:

1. Identify the audience (who's reading?)
2. Find the bottom line (what's the key point?)
3. Cut 30% of words (where's the filler?)
4. Add structure (headers, bullets)
5. Make it actionable (what are next steps?)

**Before/after comparison**: Share with a colleague. Which is clearer?

**Remember**: Great PMs are great writers. Invest in this skill—it compounds over your career.
`;

export const writingForPMsSection: ModuleSection = {
  id: 'writing-for-pms',
  title: 'Writing for Product Managers',
  content,
};

/**
 * Section 6: Working with Engineering
 * Product Management Fundamentals Module
 */

import { ModuleSection } from '../../../types';

const content = `# Working with Engineering

## Introduction

The PM-Engineering relationship is the most critical partnership in product development. Great PMs build trust with engineering teams through technical credibility, clear communication, and mutual respect. Poor PM-Engineering relationships lead to failed products, frustrated teams, and PM turnover.

This section covers how to build exceptional partnerships with engineering teams.

## Understanding the Engineering Mindset

### **What Engineers Value**

**1. Technical Excellence**
- Clean, maintainable code
- Elegant solutions to complex problems
- Learning new technologies
- Solving interesting technical challenges

**2. Autonomy**
- Ownership over technical decisions
- Trust in their expertise
- Freedom to choose implementation approach
- Not being micromanaged

**3. Clear Requirements**
- Understanding the "why" behind features
- Well-defined problem statements
- Stable requirements (not changing mid-sprint)
- Context about users and business goals

**4. Respect for Complexity**
- Acknowledgment that building software is hard
- Understanding that estimates are uncertain
- Appreciation for technical debt and infrastructure work
- Recognition of invisible work (refactoring, testing, debugging)

**5. Predictability**
- Consistent processes
- Reliable priorities
- Advance notice of changes
- Honest communication

### **What Frustrates Engineers About PMs**

**Common complaints**:
- "PMs don't understand technical complexity"
- "Requirements change constantly"
- "PMs want everything done yesterday"
- "PMs don't value technical debt"
- "PMs are too prescriptive about implementation"
- "PMs don't listen to technical concerns"
- "PMs take credit for our work"

**Each of these is avoidable through good PM practices.**

## The PM-Engineering Partnership Model

### **Clear Role Boundaries**

**PM Owns** (the WHAT and WHY):
- **What** problem to solve
- **Why** it matters (user needs, business impact)
- **When** to ship (priorities, deadlines)
- **Success metrics** (how we measure impact)

**Engineering Owns** (the HOW):
- **How** to build it (technical approach)
- **Technical architecture** decisions
- **Technology choices** (languages, frameworks, tools)
- **Code quality** standards
- **Technical feasibility** and constraints

**Shared Ownership** (collaboration):
- **Scope negotiation** (what fits in timeline?)
- **Trade-offs** (speed vs. quality vs. features)
- **Launch readiness** (is it good enough to ship?)
- **Technical debt decisions** (when to refactor?)

### **The Respect Boundary**

**Never**:
- Tell engineers how to code
- Question their technical decisions without understanding them
- Promise timelines without consulting them
- Change requirements mid-sprint without discussion
- Take sole credit for successes

**Always**:
- Ask questions to understand technical constraints
- Trust their judgment on technical matters
- Involve them early in problem definition
- Protect their time from constant interruptions
- Share credit generously

## Best Practices for PM-Engineering Collaboration

### **Practice #1: Involve Engineering in Discovery**

**Bad approach**:
- PM does user research alone
- PM designs solution alone
- PM presents finished PRD to engineering
- Engineers push back: "This is technically infeasible"

**Good approach**:
- Invite 1-2 engineers to user interviews
- Include engineering in solution brainstorming
- Do technical feasibility checks during discovery
- Engineers understand context before building

**Example**:

*"Hey Alex (engineer), I'm doing 5 user interviews this week about our onboarding flow. Would you like to join 1-2 of them? I think hearing directly from users would give you great context for the work we'll be doing next month."*

**Benefits**:
- Engineers develop user empathy
- Technical constraints surface early
- Engineers spot solutions PMs might miss
- Stronger buy-in to the solution

### **Practice #2: Write Problem-Focused PRDs**

**Bad PRD** (prescriptive, solution-focused):

\`\`\`markdown
## Technical Requirements
- Use PostgreSQL table "notifications"
- Implement Redis caching with 24hr TTL
- Send emails via SendGrid API
- Add notification bell icon (red, #FF0000)
- Use React hooks for state management
\`\`\`

**Good PRD** (problem-focused, gives engineers space):

\`\`\`markdown
## Problem Statement
Users miss important updates because they don't check the app daily.
- 40% don't return within 7 days of important events
- Support tickets: "I didn't know X happened"

## User Need
"I want to stay informed without constantly checking the app"

## Success Metrics
- 7-day retention increases 10%
- 50% of users enable notifications
- 30% open rate on notifications

## Requirements
- Notify users of important events (messages, comments, mentions)
- Support email and in-app notifications
- User control over notification preferences
- Reliable delivery at scale (1M+ users)

## Constraints
- Must respect user preferences (not spam)
- GDPR/CAN-SPAM compliance
- Mobile-friendly

## Open Questions for Engineering
- What's the best architecture for reliable delivery?
- How do we handle scale?
- What failure modes should we plan for?
- How do we make this extensible for future notification types?
\`\`\`

**Key differences**:
- Focus on problem and "why"
- Share success metrics
- Ask questions instead of prescribing
- Give engineers space to design solution
- Include constraints, not implementation

### **Practice #3: Collaborative Solution Design**

**Hold a "Solution Design Session" before writing code**:

**Attendees**: PM, EM, 2-3 engineers, designer  
**Duration**: 60-90 minutes  
**Timing**: After discovery, before sprint planning

**Agenda**:

**1. PM presents problem** (15 min)
- User research insights
- Problem definition
- Success metrics
- Constraints

**2. Designer shares design exploration** (15 min)
- User flow concepts
- UI mockups
- Open questions

**3. Engineering proposes technical approaches** (30 min)
- Architecture options (2-3 alternatives)
- Trade-offs (time, quality, scope, complexity)
- Technical risks
- Questions for PM/design

**4. Collaborative decision-making** (30 min)
- Discuss trade-offs together
- PM prioritizes scope and timeline
- Engineering chooses technical approach
- Document decisions and rationale

**Example dialogue**:

\`\`\`
PM: "Users are missing important events. We need notifications."

Engineer: "I see three approaches:
1. Email only (fast: 2 weeks, simple)
2. Email + in-app (medium: 4 weeks, better UX)
3. Email + in-app + push (slow: 8 weeks, complete)

What's most important: speed, UX, or completeness?"

PM: "Competitors just launched push. I think we need at least 
email + in-app. But 8 weeks feels slow. Can we do email + in-app 
in 4 weeks, then add push later?"

Engineer: "Yes. If we architect it right, push will be easy to add. 
Let me propose an event-driven architecture that makes extensions simple..."

PM: "Perfect. And I'm okay with basic functionality for V1. 
We can add batching, smart scheduling, etc. in V2 based on usage."

Designer: "For V1, should we support all notification types or start 
with just messages and mentions?"

PM: "Let's start with messages and mentions (80% of use cases). 
We can add others based on user feedback."

Decision: Email + in-app notifications, messages and mentions only, 
event-driven architecture for extensibility, 4-week timeline.
\`\`\`

### **Practice #4: Protect Engineering Time**

**Engineers need focused time** for complex problem-solving.

**Bad PM behavior**:
- Slack interruptions throughout the day
- Pulling engineers into every meeting
- Asking for "quick" estimates constantly
- Changing priorities daily

**Good PM behavior**:

**1. Batch communication**:
- Morning standup: Quick sync
- Afternoon: Engineers have focus time
- EOD: Async updates via Slack/email

**2. Respect "No Meeting" blocks**:
- Engineers set focus time (e.g., 2-5 PM)
- No meetings during these blocks
- Async communication only

**3. Provide advance notice**:
- Week ahead: "Next sprint we'll work on X"
- Day before: "Tomorrow we'll discuss technical approach to Y"
- Not: "Can you give me an estimate in 5 minutes?"

**4. Shield from stakeholder chaos**:
- PMs handle stakeholder management
- Engineers focus on building
- Don't involve engineers in every political discussion

### **Practice #5: Trust Technical Decisions**

**When engineers make technical recommendations, trust them.**

**Example scenario**:

**Engineer**: "We should refactor the authentication system before adding SSO. Current code is fragile and adding SSO on top will create technical debt that will slow us down for months."

**Bad PM response**: "That sounds like engineering wanting to rewrite things for perfection. We need SSO now for the enterprise deal. Just add it to the existing system."

**Good PM response**: "Help me understand the trade-offs:
- If we add SSO now without refactoring: How long? What risks?
- If we refactor first then add SSO: How long total? What benefits?
- Is there a middle ground?"

**Engineer**: "Here's the math:
- Add SSO now: 2 weeks, but creates 4 weeks of future debt
- Refactor first: 2 weeks, then add SSO in 1 week = 3 weeks total
- Middle ground: Minimal refactor (1 week), add SSO (1.5 weeks) = 2.5 weeks

Refactor saves us time long-term."

**Good PM decision**: "Let's do the 3-week option (refactor + SSO). I'll explain to stakeholders why it's faster long-term. Thank you for thinking ahead."

**Key insight**: Engineers aren't being perfectionists—they're protecting future velocity.

### **Practice #6: Understand Technical Debt**

**Technical debt** = shortcuts taken for speed that create future costs.

**PM's role**:
- Understand debt exists (not all code is perfect)
- Balance new features with debt paydown
- Advocate for infrastructure work
- Don't treat debt as "engineering's problem"

**Good practice**: 20-30% of each sprint for technical debt and infrastructure.

**How to communicate this to stakeholders**:

*"We're allocating 25% of engineering time to technical foundations. This might seem like it slows down feature development, but it actually accelerates long-term. Without this investment, our velocity would decline by 50% over the next year. Think of it like maintaining a car—regular maintenance prevents breakdowns."*

### **Practice #7: Give Credit, Take Blame**

**When things go well**:
- "The engineering team did amazing work building this feature"
- "Alex came up with the elegant solution for real-time sync"
- "Engineering shipped this on time despite complexity"

**When things go poorly**:
- "I didn't specify requirements clearly enough"
- "I should have caught this in discovery"
- "I underestimated the complexity"

**Never**:
- "Engineering took longer than expected"
- "Engineering didn't deliver what I asked for"
- "Engineering missed the deadline"

**Why this matters**: Blame erodes trust. Taking responsibility builds respect.

## Common PM-Engineering Conflicts

### **Conflict #1: "This Feature is Too Complex"**

**Engineer**: "This will take 8 weeks"  
**PM**: "Competitors shipped it in 4 weeks"

**Resolution approach**:

**1. Understand the complexity**:
- "Walk me through what makes this complex"
- "What are the technical challenges?"
- "What are we building that competitors aren't?"

**2. Explore scope options**:
- "What if we cut features X and Y?"
- "What's an MVP version we could ship in 4 weeks?"
- "Can we ship basic version, iterate later?"

**3. Find the path forward**:
- "Let's build the 4-week MVP that solves 80% of the problem"
- "We'll add advanced features in V2 based on usage"

### **Conflict #2: "Stop Changing Requirements"**

**Engineer**: "You keep changing what we're building mid-sprint"  
**PM**: "User feedback showed we need adjustments"

**Resolution approach**:

**1. Acknowledge the problem**:
- "You're right, I've been changing things too much"
- "That's frustrating and I understand why"

**2. Set boundaries**:
- "No requirement changes after sprint starts"
- "New learnings go into next sprint"
- "Emergencies only (and define 'emergency')"

**3. Improve discovery**:
- "I'll do more validation before sprint starts"
- "Let's use prototypes to catch issues earlier"

### **Conflict #3: "You Don't Understand Technical Feasibility"**

**PM**: "Why can't we just add this button?"  
**Engineer**: "It requires rewriting the entire data model"

**Resolution approach**:

**1. Ask for education**:
- "Help me understand why this is complex"
- "What would need to change in the system?"
- "Can you draw the architecture for me?"

**2. Learn the system**:
- "Can you give me a system architecture walkthrough?"
- "I want to understand our technical constraints better"

**3. Build technical fluency over time**:
- Read architecture docs
- Ask questions (not just when you need something)
- Attend technical design reviews

## Building Long-Term Trust

### **Consistent Behaviors That Build Trust**

**1. Be technically curious** (not just when you need something)
- Ask: "How does our caching system work?"
- Attend architecture reviews
- Read technical documentation

**2. Respect their time**
- Batch questions
- Don't interrupt focus time
- Provide context upfront

**3. Be honest about priorities**
- Don't say everything is urgent
- Explain why something is actually urgent
- Take responsibility for priority changes

**4. Ship together**
- Celebrate launches together
- PM doesn't take solo credit
- Acknowledge engineering excellence

**5. Protect them from chaos**
- Shield engineers from stakeholder politics
- Make decisions so they can focus
- Handle communication overhead

## Key Takeaways

1. **PM defines WHAT and WHY; Engineering defines HOW**
2. **Involve engineering early** in discovery and problem definition
3. **Write problem-focused PRDs**, not implementation specifications
4. **Trust technical decisions** and respect engineering expertise
5. **Protect engineering time** from interruptions and meetings
6. **Balance new features with technical debt** (20-30% for infrastructure)
7. **Give credit, take blame** to build trust
8. **Communicate clearly and consistently** with stable requirements
9. **Learn technical concepts** to have credible conversations
10. **Build partnership, not hierarchy** - PM and engineering are equals

## Practical Exercise

**Assess your PM-Engineering relationship**:

**Green flags** (healthy relationship):
- [ ] Engineers proactively suggest product improvements
- [ ] Engineers defend your product decisions
- [ ] Engineers come to you early with technical concerns
- [ ] Engineers appreciate your user insights
- [ ] Low turnover on engineering team
- [ ] Engineers say you "get it" technically

**Red flags** (unhealthy relationship):
- [ ] Engineers complain about you to EM
- [ ] Frequent escalations to leadership
- [ ] Engineers work around you
- [ ] High tension in planning meetings
- [ ] Requirements frequently misunderstood
- [ ] Engineers avoid talking to you

**If you have red flags, ask yourself**:
- Am I being too prescriptive?
- Do I respect their technical judgment?
- Are my requirements clear and stable?
- Do I give them context and autonomy?
- Am I protecting their time?

**Action plan**: Pick one behavior to improve this month.

**Remember**: Great PM-Engineering relationships are built through consistent respect, clear communication, and shared ownership of outcomes.
`;

export const workingWithEngineeringSection: ModuleSection = {
  id: 'working-with-engineering',
  title: 'Working with Engineering',
  content,
};

/**
 * Section 7: Working with Design
 * Product Management Fundamentals Module
 */

import { Section } from '../../../types';

const content = `# Working with Design

## Introduction

The PM-Design partnership is crucial for creating products users love. Great PMs and designers collaborate deeply on user problems, co-create solutions, and push each other to higher standards. Poor PM-Design relationships lead to beautiful products nobody wants or useful products nobody uses.

This section covers how to build exceptional partnerships with designers and integrate design thinking into your PM practice.

## Understanding the Design Mindset

### **What Designers Value**

**1. User-Centered Thinking**
- Deep understanding of user needs and behaviors
- Empathy for user struggles
- Solutions that delight, not just function
- Obsession with UX quality

**2. Craft and Quality**
- Pixel-perfect execution
- Consistent design systems
- Attention to visual and interaction details
- Beautiful, elegant solutions

**3. Creative Freedom**
- Space to explore multiple concepts
- Time to iterate on designs
- Ownership over visual and interaction design
- Not being handed solutions to execute

**4. Design Process**
- Research → Exploration → Iteration → Refinement
- Not jumping straight to execution
- Time for critique and feedback
- Evidence-based decision making

**5. Impact and Influence**
- Designers want strategic influence, not just execution
- Want to shape product direction
- Want designs shipped (not shelved)
- Recognition for design contribution

### **What Frustrates Designers About PMs**

**Common complaints**:
- "PMs tell me exactly what to design"
- "PMs don't understand good UX"
- "PMs rush design process"
- "PMs override design decisions for business reasons"
- "PMs don't involve me early enough"
- "PMs focus on features, ignore user experience"
- "PMs cut design time to ship faster"

**Each of these signals broken PM-Design partnership.**

## The PM-Design Partnership Model

### **Clear Role Boundaries**

**PM Owns**:
- **Problem definition** (what user problem to solve)
- **Business requirements** (constraints, success metrics)
- **Prioritization** (what to work on when)
- **Scope and timeline** (when to ship, what's in/out)

**Designer Owns**:
- **User experience** (how users interact with product)
- **Visual design** (aesthetics, brand consistency)
- **Interaction design** (flows, patterns, micro-interactions)
- **Design system** (components, patterns, standards)

**Shared Ownership**:
- **User research** (understanding user needs together)
- **Solution design** (exploring approaches collaboratively)
- **Trade-offs** (UX vs. feasibility vs. timeline)
- **Launch decisions** (is UX quality good enough?)

### **The Respect Boundary**

**Never**:
- Tell designers what the design should look like
- Mock up designs yourself and hand them over
- Override design decisions without discussion
- Rush design process to "just make it work"
- Treat designers as execution resources

**Always**:
- Present problems, not solutions
- Collaborate on solution exploration
- Respect design expertise and craft
- Give designers time to explore and iterate
- Involve designers early in problem definition

## Integrating Design into Product Development

### **Phase 1: Problem Definition (Week 1-2)**

**Bad approach** (PM alone):
- PM does user research alone
- PM defines problem alone
- PM hands problem to designer
- Designer doesn't understand context

**Good approach** (PM + Designer together):

**Activities**:
1. **Joint user interviews** (PM + Designer)
   - Both observe user struggles
   - Designer sees interaction problems
   - PM sees business implications
   
2. **Research synthesis** (together)
   - What user problems did we observe?
   - What jobs are users trying to accomplish?
   - What pain points exist today?
   
3. **Problem framing** (collaborative)
   - PM: Business context and constraints
   - Designer: User experience implications
   - Together: Align on problem to solve

**Output**: Shared problem definition document

**Example**:

\`\`\`markdown
## Problem Statement
Users struggle to track expenses across multiple bank accounts.

## User Research Insights
- 5/8 users use spreadsheets (tedious, error-prone)
- 6/8 want automatic categorization (manual entry too slow)
- All users want to see spending patterns over time

## Jobs to Be Done
- Track where money is going
- Stay within budget
- Identify wasteful spending

## Success Metrics
- Users log expenses weekly (engagement)
- Users feel in control of finances (satisfaction)
- Users reduce unnecessary spending (outcome)

## Constraints
- Must work across multiple banks
- Must protect financial data (security)
- Must be mobile-friendly (80% usage mobile)
\`\`\`

### **Phase 2: Solution Exploration (Week 3-4)**

**This is where design magic happens.**

**PM's role**: Provide context, constraints, and feedback (not solutions)

**Designer's role**: Explore multiple approaches, iterate based on feedback

**Process**:

**1. Kickoff Session** (PM + Designer, 60 min):
- Review problem definition together
- Share constraints (technical, business, timeline)
- Discuss inspiration and references
- Align on exploration timeline (1-2 weeks)

**2. Designer Explores** (solo, 1 week):
- Research existing solutions (competitors, patterns)
- Sketch 3-5 different approaches
- Create low-fidelity concepts
- Identify key UX questions to test

**3. Concept Review** (PM + Designer, 60 min):
- Designer presents 3-5 concepts
- PM provides feedback:
  - Which approaches solve user problems?
  - Which fit business constraints?
  - Which are technically feasible (rough sense)?
- Together narrow to 1-2 directions

**4. Designer Refines** (solo, 1 week):
- High-fidelity mockups of 1-2 directions
- Interactive prototypes for testing
- Document open UX questions

**5. Prototype Testing** (PM + Designer, with users):
- Test prototypes with 5-8 users
- Observe what works and what doesn't
- Validate assumptions
- Identify improvements

**Output**: Validated design direction

### **Phase 3: Detailed Design (Week 5-6)**

**Designer creates high-fidelity designs**:
- Pixel-perfect mockups
- Interaction specifications
- Edge cases and error states
- Design system components
- Documentation for engineers

**PM's role during detailed design**:

**1. Provide continuous feedback** (not at the end):
- Weekly design reviews
- Quick Slack check-ins
- Clarify requirements as questions arise

**2. Manage scope** (protect design quality):
- If timeline slips, cut scope (don't cut design time)
- "Let's ship V1 with fewer features but great UX"
- Not "Let's ship everything with okay UX"

**3. Prepare for handoff**:
- Review designs with engineering early
- Discuss technical feasibility and edge cases
- Ensure designs are buildable within timeline

**4. Write complementary PRD**:
- Designer owns UI specs (in Figma)
- PM owns business logic and requirements (in PRD)
- Together they tell complete story

### **Phase 4: Build Phase (Week 7-10)**

**Designer's role during build**:
- Answer engineer questions about design
- Review implementations for design quality
- Make adjustments based on technical constraints
- QA visual quality before launch

**PM's role**:
- Facilitate PM-Designer-Engineering collaboration
- Make trade-off decisions (scope vs. quality vs. timeline)
- Protect design quality (don't cut corners)
- Shield designer from constant stakeholder questions

**Weekly Design QA**:
- Designer reviews implemented features
- Flags quality issues
- PM decides: Fix now or post-launch?
- Goal: Ship high-quality UX, not perfect UX

## Best Practices for PM-Design Collaboration

### **Practice #1: Involve Designers Early**

**Bad timeline**:
- Week 1-3: PM does research, writes PRD
- Week 4: PM hands PRD to designer
- Week 5-6: Designer creates mockups
- Week 7: Designer hands mockups to engineering
- Result: Designer feels like "order taker"

**Good timeline**:
- Week 1-2: PM + Designer do research together
- Week 3-4: PM + Designer explore solutions together
- Week 5-6: Designer creates detailed designs (PM provides feedback)
- Week 7+: PM + Designer + Engineering build together
- Result: Designer feels like "partner"

**Key insight**: Involve designers at problem definition, not solution specification.

### **Practice #2: Present Problems, Not Solutions**

**Bad PM brief** (prescriptive):

\`\`\`
Design Request:
- Add expense tracking dashboard
- Use bar chart for spending by category
- List view of recent transactions
- Filter dropdown (date range, category, account)
- Green for income, red for expenses
\`\`\`

**Good PM brief** (problem-focused):

\`\`\`
Problem:
Users don't know where their money goes. They want to identify 
wasteful spending and stay within budget.

User Needs:
- See spending patterns over time
- Identify highest spending categories
- Find specific transactions
- Compare to previous months

Success Metrics:
- Users check dashboard weekly
- Users can identify wasteful spending within 30 seconds
- User satisfaction >4/5

Constraints:
- Works on mobile (80% of usage)
- Loads in <2 seconds
- Accessible (WCAG AA)

Questions for Design:
- How might we visualize spending patterns intuitively?
- How might we help users discover insights?
- What's the best way to handle large transaction lists?
\`\`\`

**Difference**: PM defines problem and constraints, designer explores solutions.

### **Practice #3: Give Designers Time and Space**

**Design process needs time**:

**Exploration**: 1-2 weeks
- Research existing patterns
- Sketch multiple approaches
- Create low-fidelity concepts

**Refinement**: 1-2 weeks
- High-fidelity mockups
- Interactive prototypes
- User testing and iteration

**Detailed Design**: 1-2 weeks
- Pixel-perfect specs
- Edge cases and states
- Component documentation

**Total**: 3-6 weeks (depending on complexity)

**What PMs get wrong**:
- "Can you mock this up by EOD?" (No.)
- "Engineering starts Monday, need designs Friday" (No.)
- "Just use competitor's design" (No.)

**What good PMs do**:
- Plan design time into project timeline
- Give advance notice (not last-minute)
- Protect design time from scope creep

### **Practice #4: Understand Design Principles**

**PM doesn't need to be a designer, but should understand design basics.**

**Core UX Principles**:

**1. User-Centered Design**
- Design for user needs, not company wants
- Validate designs with real users
- Prioritize usability over aesthetics

**2. Consistency**
- Use established patterns (don't reinvent)
- Follow design system
- Create predictable experiences

**3. Hierarchy**
- Guide user attention (most important first)
- Clear visual structure
- Reduce cognitive load

**4. Feedback**
- Tell users what's happening (loading states, errors, confirmations)
- Make system state visible
- Provide clear affordances

**5. Accessibility**
- Design for everyone (including disabilities)
- Keyboard navigation, screen readers
- Color contrast, text sizing

**Learn these** so you can have informed discussions with designers.

### **Practice #5: Balance Business and UX**

**PMs bridge business needs and user experience.**

**Common tension**: Business wants feature, designer says it hurts UX.

**Example scenario**:

**Business**: "Add promotional banner for premium upgrade"  
**Designer**: "Banner clutters interface and distracts from core task"

**Bad PM response**: "Business needs this, just add it" (dismisses design)

**Good PM response**: "Let's find a solution that achieves business goal without hurting UX"

**Collaborative problem-solving**:
- Business goal: Increase premium conversions
- UX concern: Banner is disruptive
- Alternative approaches:
  - In-context upgrade prompts (when user hits free limit)
  - Subtle inline upsells (less disruptive)
  - Post-task upgrade suggestions (after user completes action)
- Solution: In-context prompts (achieves business goal, better UX)

**PM's role**: Find solutions that satisfy both business and UX, not compromise both.

### **Practice #6: Critique, Don't Dictate**

**When reviewing designs, critique don't dictate.**

**Bad feedback** (dictating):
- "Make the button blue"
- "Move this to the top"
- "Use a dropdown instead of radio buttons"

**Good feedback** (questioning):
- "What's the reasoning behind the button color?"
- "I'm concerned users won't see this below the fold. How did you think about prominence?"
- "How does dropdown compare to other input methods for this use case?"

**Great feedback** (user-centered):
- "I worry users won't notice this button. How can we make the primary action clearer?"
- "In user testing, did users find this feature easily?"
- "How does this compare to patterns users expect?"

**Framework for feedback**:
1. Ask questions (understand designer's reasoning)
2. Share concerns (user or business perspective)
3. Collaborate on solutions (don't prescribe)

### **Practice #7: Defend Design Quality**

**PMs protect design quality from business and engineering pressure.**

**Scenario 1: Business pressure**

**Sales**: "Competitor has 50 features, we have 10. We need feature parity now!"

**Bad PM**: "Design team, we need to ship 40 features this quarter"

**Good PM**: "We'll prioritize the 10 features that matter most. I'd rather ship 10 great features than 50 mediocre ones. Quality over quantity."

**Scenario 2: Engineering pressure**

**Engineer**: "This animation is complex, let's skip it"

**Bad PM**: "Okay, let's ship without animation"

**Good PM**: "Let's discuss. How long would animation take? Designer, how important is it for UX? Let's make an informed trade-off."

**Sometimes you'll compromise, but don't compromise by default.**

## Common PM-Design Conflicts

### **Conflict #1: "Design Takes Too Long"**

**PM**: "We need to ship faster, design is slowing us down"  
**Designer**: "Good design takes time, you're rushing the process"

**Resolution**:

**1. Plan design time upfront**:
- Don't treat design as overhead
- Build design time into project plans
- 3-6 weeks for design (depending on complexity)

**2. Parallelize when possible**:
- Designer works on feature B while engineering builds feature A
- Don't wait for one phase to complete

**3. Adjust scope, not quality**:
- Ship V1 with fewer features but great UX
- Not all features with okay UX

**4. Use design systems**:
- Established patterns → faster design
- Don't redesign everything from scratch

### **Conflict #2: "PM Dictates Design"**

**Designer**: "PM tells me exactly what to design, I'm just executing"  
**PM**: "I'm just providing clear requirements"

**Resolution**:

**1. Shift from solutions to problems**:
- PM defines problem, not solution
- Present user needs, not design specs

**2. Invite designer into exploration**:
- Collaborative ideation
- Designer proposes multiple approaches
- PM provides feedback, designer refines

**3. Respect design expertise**:
- Designer makes visual and interaction decisions
- PM makes business and prioritization decisions

**4. Build trust through small projects**:
- Start with collaborative project
- Show designer you value their input
- Build track record of partnership

### **Conflict #3: "Business Trumps UX"**

**PM**: "I know UX isn't ideal, but business needs this"  
**Designer**: "We're shipping bad UX for short-term gain"

**Resolution**:

**1. Acknowledge the tension**:
- "You're right, this isn't ideal UX"
- "I hear your concern"

**2. Explain the trade-off**:
- "Here's why this matters for business: [specific impact]"
- "Here's the cost of not doing it: [specific consequence]"

**3. Explore alternatives**:
- "How can we achieve business goal with better UX?"
- "What's a middle ground?"

**4. Set expectations**:
- "We're shipping V1 quickly, we'll improve UX in V2"
- "This is a conscious trade-off, not permanent"

**5. Follow through**:
- Actually improve UX in V2 (don't just say it)
- Build trust through action

## Key Takeaways

1. **Involve designers early** in problem definition, not just solution execution
2. **Present problems, not solutions** to give designers creative space
3. **Give designers time** for exploration, refinement, and quality
4. **Learn design principles** to have informed conversations
5. **Balance business and UX** through collaborative problem-solving
6. **Critique, don't dictate** when reviewing designs
7. **Defend design quality** from pressure to cut corners
8. **Ship great experiences**, not just functional features
9. **Build partnership** through respect and collaboration
10. **Design is strategy**, not just execution

## Practical Exercise

**Assess your PM-Design relationship**:

**Green flags** (healthy partnership):
- [ ] Designer involved in early discovery
- [ ] Designer proposes multiple solutions (not just executing)
- [ ] You discuss design trade-offs together
- [ ] Designer feels ownership over UX
- [ ] You learn from designer's expertise
- [ ] Designs ship consistently (not shelved)

**Red flags** (broken partnership):
- [ ] Designer complaints about PM dictating
- [ ] PM creates mockups, hands to designer
- [ ] Design review happens at the end
- [ ] Frequent compromises on UX quality
- [ ] Designer says "PM doesn't get good UX"
- [ ] PM says "Designer doesn't understand business"

**If you have red flags**:
- Start involving designer earlier
- Present problems, not solutions
- Protect design time and quality
- Build trust through collaboration

**Remember**: Great products require great PM-Design partnerships. Invest in the relationship.
`;

export const workingWithDesignSection: Section = {
  id: 'working-with-design',
  title: 'Working with Design',
  content,
};

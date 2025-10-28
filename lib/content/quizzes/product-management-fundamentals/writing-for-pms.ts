/**
 * Discussion questions for Writing for Product Managers
 * Product Management Fundamentals Module
 */

export const writingForPMsQuiz = [
  {
    id: 1,
    question:
      "You need to write a PRD for a complex feature (multi-step user onboarding flow). Your engineering team complains that your previous PRDs were either 'too vague' (leaving them guessing) or 'too prescriptive' (dictating implementation). Write a sample 2-page PRD outline that balances clarity with flexibility. Show how you'd structure it to give engineers what they need without dictating how to build it.",
    answer: `## Comprehensive Answer:

The challenge is balancing specificity (clear requirements) with flexibility (engineering autonomy). Here's how:

---

# User Onboarding Flow - PRD

**Author**: [PM Name] | **Date**: Jan 15, 2024 | **Status**: Draft for Review

---

## Executive Summary (30 seconds)

**What**: Redesigned onboarding flow to guide new users to first value  
**Why**: 60% of users don't complete setup, never become active  
**Impact**: Increase activation rate from 40% to 60% (2,000 more activated users/month)  
**Timeline**: 6 weeks (Q1 2024)

---

## Problem Statement (What problem are we solving?)

**Current state**:
- New users sign up but don't complete setup (60% drop off)
- Users don't understand how to get started
- No guided path from signup to first value

**User pain** (from 10 user interviews):
- "I signed up but didn't know what to do next"
- "Too many options, felt overwhelming"
- "I couldn't figure out how to [accomplish core task]"

**Business impact**:
- 6,000 signups/month, only 2,400 activate (40%)
- Lost revenue: 3,600 users × $10/month = $36K/month

---

## User Needs (Jobs to be done)

**Primary**: "Get started quickly without reading documentation"  
**Secondary**: "Understand core value within 5 minutes"  
**Tertiary**: "Complete setup without getting stuck"

---

## Success Metrics (How we measure success)

**Primary metric**:
- Activation rate increases from 40% to 60% within 30 days of launch
- Activation = User completes first [core action] within 7 days

**Secondary metrics**:
- Time to first [core action] decreases from 45 min to 15 min
- Setup completion rate increases from 50% to 75%
- User satisfaction (post-onboarding survey) >4/5

**How we'll measure**: Track in Amplitude, cohort analysis (before vs. after)

---

## Requirements (What must the feature do?)

### **Must Have** (Non-negotiable for V1)

**R1: Multi-step guided flow**
- User sees step-by-step guidance from signup to first value
- Clear progress indicator (e.g., "Step 2 of 5")
- Cannot skip steps (enforced sequence)

**R2: Contextual help**
- Each step explains WHY it matters ("This helps you [benefit]")
- Tooltips for unfamiliar terms
- Links to help docs (optional, not required)

**R3: Input validation**
- Real-time validation (not just on submit)
- Clear error messages ("Email format invalid" not "Error 400")
- Cannot proceed with invalid inputs

**R4: Progress persistence**
- User can exit and return (progress saved)
- Email reminder if user abandons mid-flow ("Finish your setup")
- Resume from last completed step

**R5: Success state**
- Clear completion message ("You're all set!")
- Next steps guidance ("Here's what to do next")
- Optional: Celebrate with animation/confetti

### **Should Have** (Important but flexible)

**R6: Personalization**
- Ask user's role/use case upfront
- Customize flow based on answer
- Example: "Developer" sees technical setup, "Marketer" sees campaign setup

**R7: Skip option** (for advanced users)
- "Skip to dashboard" link (but hidden, not prominent)
- Analytics: Track skip rate

### **Nice to Have** (Defer to V2)

**R8: Interactive demo**
- Sample data pre-populated for immediate experimentation
- "Try it yourself" step

**R9: Video tutorials**
- Embedded videos explaining each step

---

## User Flow (How users interact)

### **Happy Path** (90% of users)

\`\`\`
1. User signs up → Welcome screen
2. "Let's get started" → Step 1: Profile setup
3. Complete profile → Step 2: Team invitation
4. Invite team (or skip) → Step 3: Connect integration
5. Connect integration → Step 4: First [core action]
6. Complete action → Success! ("You're all set!")
7. Dashboard with next steps
\`\`\`

### **Alternative Paths**

**Path A: User exits mid-flow**
- Progress saved automatically
- Email sent after 24 hours: "Finish your setup in 3 minutes"
- User clicks link → Resumes from last step

**Path B: User skips setup** (for advanced users)
- "Skip to dashboard" link (small, bottom of page)
- User sees dashboard with incomplete setup banner
- Can restart onboarding anytime

**Path C: User encounters error**
- Clear error message with action ("Email already in use. Try logging in instead.")
- Option to get help (chat, email support)

---

## Edge Cases & Open Questions (What needs clarification?)

### **Edge Cases**

**E1: User already completed some steps manually**
- **Question**: Do we detect this and skip those steps?
- **Suggestion**: Yes, check if user has [integration connected, team invited, etc.] and skip completed steps

**E2: Multiple team members sign up**
- **Question**: Does each see onboarding, or only admin?
- **Suggestion**: Only admin sees full onboarding; team members see simplified "welcome tour"

**E3: User closes browser mid-step**
- **Question**: Save partial step progress or only completed steps?
- **Suggestion**: Save completed steps only (partial step discarded)

**E4: User wants to change answer from earlier step**
- **Question**: Can user go back, or only forward?
- **Suggestion**: "Back" button allowed, but warn if changes affect later steps

### **Open Questions for Engineering**

**Q1: Technical approach**
- How to structure: single-page with progressive disclosure, or multi-page?
- Your recommendation based on performance and UX?

**Q2: Progress persistence**
- Where to store progress: localStorage, database, or both?
- Trade-offs?

**Q3: Email reminders**
- Trigger: 24 hours after abandonment?
- How to handle user who completed manually without finishing flow?

**Q4: Analytics instrumentation**
- What events should we track? (Suggest)
- Amplitude, Mixpanel, or custom?

---

## Out of Scope for V1 (What we're NOT building)

❌ Video tutorials (defer to V2)  
❌ Interactive product tours (defer to V2)  
❌ A/B testing different flows (V1 = single flow, optimize in V2)  
❌ Mobile app onboarding (V1 = web only)  
❌ Localization (V1 = English only)

---

## Design & Technical Constraints

### **Design Constraints**
- Must work on mobile (responsive)
- Accessible (WCAG AA: keyboard nav, screen readers)
- Match existing design system (buttons, colors, fonts)

### **Technical Constraints**
- Must load in <2 seconds
- Support browsers: Chrome, Firefox, Safari (last 2 versions)
- Work offline? No (online required)

### **Business Constraints**
- Must ship by end of Q1 (March 31)
- No additional third-party services (use existing tech stack)

---

## Success Criteria (How do we know when to ship?)

### **Launch Readiness Checklist**

**Functionality**:
- [ ] User can complete entire flow
- [ ] All edge cases handled
- [ ] Progress saves and resumes correctly
- [ ] Email reminders send

**Quality**:
- [ ] No critical bugs
- [ ] Performance <2 sec load time
- [ ] Accessible (keyboard nav, screen readers)
- [ ] Works on mobile

**Documentation**:
- [ ] Help articles written
- [ ] Analytics instrumented
- [ ] Support team trained

**Go/No-Go**: If all checked, ship. If not, delay.

---

## Timeline & Milestones

**Week 1-2**: Design & Spec
- Designer creates mockups
- Engineering proposes technical approach
- PM finalizes requirements

**Week 3-4**: Build
- Engineering builds flow
- PM does weekly QA

**Week 5**: QA & Polish
- Full QA testing
- Bug fixes
- Design QA

**Week 6**: Launch
- Beta launch (10% of users)
- Monitor metrics
- Full rollout if successful

**Target ship date**: March 31, 2024

---

## Appendix: Research & Data

**User Research Summary**:
- 10 interviews with users who abandoned onboarding
- Key insight: Users didn't understand value or next steps
- Quote: "I signed up but didn't know what to do" (6/10 users said this)

**Analytics Data**:
- Signup to activation funnel: 100% → 60% → 40%
- Biggest drop-off: After signup (40% never return)
- Time to first [core action]: 45 minutes (median)

**Competitive Analysis**:
- Slack: Guided onboarding with team setup
- Notion: Template selection during onboarding
- Figma: Interactive tutorial showing core features

---

## Feedback & Collaboration

**This is a draft. I need your input:**

**For Engineering**:
- Is 6-week timeline realistic?
- What's the best technical approach (single-page vs. multi-page)?
- Are the open questions clear?

**For Design**:
- Does the user flow make sense?
- Any UX concerns?

**For Everyone**:
- What am I missing?
- What needs more clarity?

**How to give feedback**: Comment on this doc or ping me in Slack

---

## Why This PRD Works

**✅ Clear problem and user needs** (not just "build onboarding")  
**✅ Specific requirements** (R1-R9 with priority tiers)  
**✅ Success metrics defined upfront** (activation rate 40% → 60%)  
**✅ NOT prescriptive about HOW** (engineering chooses single-page vs. multi-page)  
**✅ Edge cases identified** (but asks engineering for input)  
**✅ Open questions explicitly called out** (invites collaboration)  
**✅ Out of scope defined** (prevents scope creep)  
**✅ Structured and scannable** (headers, bullets, 2 pages + appendix)

**Key balance**: 
- PM defines WHAT (requirements) and WHY (user needs, success metrics)
- Engineering defines HOW (technical approach, implementation)
- Open questions invite collaboration

This PRD gives engineers clarity without dictating solutions.
`,
  },
  {
    id: 2,
    question:
      "Write two versions of the same email announcing a product delay to stakeholders: (1) A 'bad' version that will frustrate stakeholders and damage trust, and (2) A 'good' version that maintains trust while delivering bad news. Then explain what makes each version effective or ineffective.",
    answer: `## Comprehensive Answer:

Let's see both versions and analyze what works and what doesn't.

---

## Version 1: BAD Email (Frustrating, damages trust)

**Subject**: Update

**Body**:

Hey everyone,

Just wanted to give you a heads up that we're running into some issues with the Salesforce integration project. There have been some technical challenges that we didn't anticipate, and the engineering team is telling me that things are taking longer than expected. We've been working really hard on this, but unfortunately it looks like we're not going to be able to hit our original timeline.

I know this is disappointing, and believe me, I'm just as frustrated as you are. We really thought we could deliver this on time, but it turns out there were some things we didn't account for in our initial planning. The engineering team has been doing their best, but some of these technical issues are just really complex and hard to solve.

At this point in time, we're currently looking at a timeline that extends beyond what we originally communicated. I don't have an exact date yet, but we're working on figuring that out and will let you know as soon as we have more information. We're doing everything we can to get this done as quickly as possible.

Let me know if you have any questions or concerns. I'm happy to discuss this further.

Thanks,
[Name]

---

### **Why This Email Is BAD**

**Problem #1: Vague subject line**
- "Update" tells me nothing
- I don't know if this is urgent or informational
- Won't stand out in crowded inbox

**Problem #2: Buried lead**
- Doesn't say "delayed" until halfway through
- Wastes reader's time with preamble
- Reader doesn't know impact until they read entire email

**Problem #3: No specific timeline**
- "beyond what we originally communicated" - how much longer?
- "I don't have an exact date yet" - when will I know?
- Can't plan around this information

**Problem #4: Blame-shifting language**
- "engineering team is telling me" - implies PM isn't accountable
- "technical issues" - vague, sounds like excuse
- "didn't anticipate" - sounds like poor planning

**Problem #5: No ownership**
- "I'm just as frustrated as you are" - sounds defensive
- Doesn't take responsibility
- Focuses on PM's feelings, not stakeholder impact

**Problem #6: Wordy and unclear**
- 5 paragraphs to say "We're delayed, don't know when it'll ship"
- Lots of filler ("at this point in time," "working really hard")
- No clear action items

**Problem #7: No mitigation plan**
- What are we doing about the delay?
- How are we preventing this in future?
- What's the path forward?

**Result**: Stakeholders are frustrated, don't trust PM, can't plan, and feel uninformed.

---

## Version 2: GOOD Email (Maintains trust, delivers bad news clearly)

**Subject**: Salesforce Integration Delayed to Q2 (Was Q1) - Action Needed

**Body**:

**Bottom line**: Salesforce integration delayed 6 weeks. New ship date: April 15 (was March 1).

**Why this matters**:
- Sales: Impacts 8 deals ($750K pipeline) - I'll work with you on customer communication
- Marketing: Need to adjust Q1 campaign - let's sync today
- Executives: Revenue impact discussed below

**Why we're delayed**:
We underestimated the complexity of bidirectional sync. Specific technical challenge: handling conflict resolution when same record is updated in both systems simultaneously. This wasn't discovered until Week 3 when we began integration testing.

**What I got wrong**: I should have done a technical spike before committing to the timeline. This is on me.

**New timeline**:
- March 1-31: Resolve technical challenges
- April 1-15: Complete testing and QA
- April 15: Ship to 100% of users

**Confidence level**: High (90%). We've now done deep technical work and understand remaining scope.

**What we're doing to prevent this**:
- Technical spikes before committing to timelines (1-2 week validation before planning)
- Weekly eng-PM check-ins on timeline confidence
- Buffer into all estimates (add 25% for unknowns)

**How this impacts you**:

**Sales** (Sarah):
- 8 deals need new timeline communication
- I'll join customer calls if helpful
- Let's sync today at 2 PM to plan customer messages

**Marketing** (Alex):
- Q1 campaign needs adjustment
- Can we pivot to Q2 launch campaign?
- Let's sync today at 3 PM

**Support** (Jamie):
- Training delayed to April 10
- I'll update docs with new timeline

**Actions needed from you**:
- [ ] Sales (Sarah): Confirm 2 PM sync time
- [ ] Marketing (Alex): Confirm 3 PM sync time
- [ ] Everyone: Reply by EOD if the new timeline doesn't work (we'll escalate to leadership)

I take full responsibility for this delay. I should have validated the technical complexity earlier. I'll keep you updated weekly on progress.

Reply with any questions. I'm here.

[Name]

---

### **Why This Email Is GOOD**

**✅ Clear subject line**
- "Salesforce Integration Delayed to Q2 (Was Q1)" - immediately clear
- Specific: Q2, not vague "delayed"
- "Action Needed" - signals urgency

**✅ Bottom line first**
- First sentence: "Delayed 6 weeks, new date April 15"
- Stakeholders know the news in 5 seconds
- Can decide whether to read details

**✅ Specific timeline**
- Old date: March 1
- New date: April 15
- Exact dates, not "soon" or "Q2ish"

**✅ Explains WHY clearly**
- "Bidirectional sync conflict resolution" - specific technical reason
- Not vague "technical issues"
- Demonstrates PM understands the problem

**✅ Takes ownership**
- "What I got wrong: I should have done technical spike"
- "This is on me"
- No blame-shifting to engineering
- Builds trust through accountability

**✅ High confidence in new timeline**
- "Confidence level: High (90%)"
- Explains why (we've now done deep technical work)
- Reassures this won't slip again

**✅ Prevention plan**
- Clear changes: technical spikes, weekly check-ins, buffer
- Shows PM learned from mistake
- Prevents future surprises

**✅ Stakeholder-specific impact**
- Tailored for Sales (8 deals impacted)
- Tailored for Marketing (campaign adjustment)
- Shows PM understands their concerns

**✅ Clear action items**
- Specific: "Confirm 2 PM sync time"
- Deadline: "Reply by EOD"
- Owners: Sarah, Alex, Everyone

**✅ Concise**
- Same information in half the words
- Scannable (headers, bullets, bold)
- Respects readers' time

**Result**: Stakeholders have clarity, know what to do, and trust PM is handling it.

---

## Side-by-Side Comparison

| Aspect | BAD Version | GOOD Version |
|--------|-------------|--------------|
| **Subject** | "Update" (vague) | "Delayed to Q2 (Was Q1) - Action Needed" (specific) |
| **Bottom line** | Buried in paragraph 2 | First sentence |
| **Timeline** | "Beyond original" (vague) | "April 15 (was March 1)" (specific) |
| **Ownership** | Blame-shifts | Takes responsibility |
| **Reason** | "Technical issues" (vague) | "Conflict resolution" (specific) |
| **Confidence** | Unclear | 90% confident |
| **Action items** | "Let me know" (vague) | Specific tasks, owners, deadlines |
| **Length** | 250 words | 300 words (but more information) |
| **Scannability** | Wall of text | Headers, bullets, bold |
| **Trust** | Damages | Maintains/builds |

---

## Key Principles for Delivering Bad News

**1. Lead with the bad news** (don't bury it)
- First sentence should be the news
- Don't make stakeholders hunt for it

**2. Be specific** (not vague)
- Exact dates, not "soon"
- Specific reasons, not "issues"

**3. Take ownership** (don't blame others)
- "I should have..." not "They didn't..."
- Accountability builds trust

**4. Explain impact** (show you understand consequences)
- How does this affect stakeholders?
- Demonstrate empathy and understanding

**5. Provide new plan** (not just "we're delayed")
- New timeline with confidence level
- Prevention plan (how we avoid this in future)

**6. Make it actionable** (clear next steps)
- Who does what by when?
- How can stakeholders help or respond?

**7. Be concise** (respect their time)
- Structure: headers, bullets, bold
- Scannable in 30 seconds

---

## When to Use This Approach

**Use this email style when**:
- Delivering bad news (delays, failures, issues)
- Asking for decisions under time pressure
- Communicating cross-functionally (sales, marketing, eng)
- Stakeholders are busy (executives, leadership)

**Don't use when**:
- Informal updates to team (Slack might be better)
- Celebrating good news (you can be more casual)
- One-on-one conversations (face-to-face or video call better for sensitive topics)

---

## The Ultimate Test

**Show both versions to a colleague**:
- Which would they rather receive?
- Which gives them what they need?
- Which builds or damages trust?

**100% will choose Version 2** (Good email).

**Why?** Because it respects their time, takes accountability, provides clarity, and enables action.

**Great PMs deliver bad news well**. It builds trust even when things go wrong.
`,
  },
  {
    id: 3,
    question:
      "You're writing a strategy document for your product's 2024 roadmap. Your CEO loves big visions but gets impatient with details. Your engineering team needs specifics to plan. Your designer wants to understand user problems. Write a 1-page strategy doc structure that satisfies all three audiences. Explain how you'd adapt the same information for each audience.",
    answer: `## Comprehensive Answer:

The challenge is one document, three audiences with different needs. Here's the solution:

---

# [Product Name] 2024 Strategy (1-Page Core + Appendices)

---

## Vision (For CEO: Big picture, inspiring)

**Where we're going**: Become the #1 [product category] for [target market] by 2025.

**2024 Goal**: Grow from 50K to 150K users, $5M to $15M ARR (3X growth).

**Why this matters**: We're at an inflection point. The market is growing 40% YoY. If we execute well in 2024, we capture market share before competitors scale. If we don't, we miss the window.

---

## Strategic Pillars (For everyone: Clear priorities)

We'll focus on 3 strategic pillars in 2024:

**1. Enterprise Readiness** (40% of engineering capacity)
- **Goal**: Win 20 enterprise deals ($10K+ ACV) = $200K+ ARR
- **Why**: Enterprise customers are where the growth is
- **What we're building**: SSO, advanced permissions, audit logs, Salesforce integration

**2. Product-Led Growth** (40% of engineering capacity)
- **Goal**: Double self-serve conversion from 10% to 20% = $3M ARR
- **Why**: Reduce customer acquisition cost, scale efficiently
- **What we're building**: Improved onboarding, viral loops, in-app upgrades

**3. Platform Stability** (20% of engineering capacity)
- **Goal**: 99.9% uptime, <2 sec load time, zero data loss
- **Why**: Enterprise customers demand reliability
- **What we're building**: Infrastructure improvements, monitoring, performance

---

## Key Results (For CEO: Measurable outcomes)

By end of 2024:

| Metric | Current | Target | Impact |
|--------|---------|--------|--------|
| ARR | $5M | $15M | 3X growth |
| Users | 50K | 150K | 3X growth |
| Enterprise deals | 5 | 25 | $500K ARR |
| Self-serve conversion | 10% | 20% | $3M ARR |
| Uptime | 99.5% | 99.9% | Enterprise-grade |

**Revenue breakdown**:
- Enterprise: $5M (from $1M)
- Self-serve: $10M (from $4M)
- Total: $15M ARR

---

## Quarterly Roadmap (For Engineering: Specific timeline)

**Q1**: Enterprise Foundations
- SSO/SAML
- Advanced permissions
- → Unlocks: 8 enterprise deals ($800K ARR)

**Q2**: PLG Engine
- Onboarding redesign
- In-app upgrade flows
- → Unlocks: Conversion 10% → 15%

**Q3**: Enterprise Scale
- Salesforce integration
- Audit logs
- → Unlocks: 12 more enterprise deals ($1.2M ARR)

**Q4**: Polish & Expand
- Performance improvements
- Platform stability
- → Unlocks: 99.9% uptime

---

## What We're NOT Doing (Critical: Scope management)

To achieve 3X growth, we're saying NO to:

❌ **Mobile app** (defer to 2025)  
❌ **New product lines** (focus on core)  
❌ **Small feature requests** (prioritize strategic initiatives)  
❌ **Geographic expansion** (stay focused on US market)

**Why**: Focus wins. If we build everything, we ship nothing well.

---

## Risks & Mitigations (For CEO: What could go wrong)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Enterprise sales slow | Medium | High | Hire sales engineer Q1, increase demos |
| Engineering capacity | High | High | Hire 3 engineers Q1-Q2, reduce scope if needed |
| Competitor launches similar | Medium | Medium | Monitor closely, differentiate on UX/quality |
| Churn increases | Low | High | Invest in customer success, monitor NPS weekly |

---

## Success Metrics (How we track progress)

**Monthly review metrics**:
- ARR growth (track toward $15M)
- Enterprise deals closed (track toward 25)
- Self-serve conversion (track toward 20%)
- Uptime (track toward 99.9%)

**Decision rule**: If we're <80% to target by Q2, we reassess strategy.

---

## Appendices (Click to expand)

- **Appendix A**: Detailed user research (for Designer)
- **Appendix B**: Technical architecture plan (for Engineering)
- **Appendix C**: Market analysis (for CEO)
- **Appendix D**: Competitive landscape (for everyone)

---

## Why This Structure Works

**✅ 1-page core** (everyone reads this)  
**✅ CEO gets vision and outcomes** (big picture)  
**✅ Engineering gets roadmap and timeline** (what to build when)  
**✅ Designer gets strategic context** (why these priorities)  
**✅ Appendices for deep dives** (optional, audience-specific)

---

# Adapting for Each Audience

Now let's see how to present this to each stakeholder:

---

## For CEO: Vision-Focused Version

**Meeting format**: 30-min 1-on-1  
**Presentation**: Slides with big visuals

### **Slide 1: Vision**
"We're capturing the enterprise market window in 2024. If we execute, we become the #1 player by 2025."

**Visual**: Chart showing market growth (40% YoY)

### **Slide 2: The Goal**
"3X growth: $5M → $15M ARR"

**Visual**: Revenue chart (current vs. target)

### **Slide 3: How We Get There**
"Three strategic pillars:
1. Enterprise Readiness → $5M ARR
2. Product-Led Growth → $10M ARR  
3. Platform Stability → Enterprise-grade"

**Visual**: Pie chart of engineering capacity allocation

### **Slide 4: Key Milestones**
"Q1: Enterprise foundations → 8 deals
Q2: PLG engine → 2X conversion
Q3: Enterprise scale → 12 more deals
Q4: Polish → 99.9% uptime"

**Visual**: Timeline with milestones

### **Slide 5: What We're NOT Doing**
"To achieve 3X, we say NO to: Mobile app, new products, small features, geo expansion"

**Visual**: List with ❌ marks

### **Slide 6: Risks**
"Biggest risks: Enterprise sales slow, engineering capacity
Mitigations: Hire sales engineer, hire 3 engineers Q1-Q2"

**Visual**: Risk matrix (probability vs. impact)

### **Slide 7: Ask**
"I need:
- Approval for 3 engineering hires
- Sales engineer budget
- Trust to say NO to off-strategy requests

Decision: Approve or adjust?"

**What CEO gets**:
- ✅ Big vision (inspiring)
- ✅ Clear outcomes ($15M ARR)
- ✅ Strategic clarity (3 pillars)
- ✅ Risk awareness (what could go wrong)
- ✅ Clear ask (approval for hires)

**CEO leaves saying**: "This is bold but achievable. Approved."

---

## For Engineering: Detail-Focused Version

**Meeting format**: 60-min team meeting  
**Presentation**: Detailed walkthrough + Q&A

### **Part 1: Context (10 min)**
"Here's WHY we're building these features:
- Enterprise customers = $10K+ ACV (vs. $100 self-serve)
- 20 enterprise deals = $200K ARR
- Market window is NOW (competitors scaling)"

**Visual**: Revenue breakdown (enterprise vs. self-serve)

### **Part 2: Quarterly Roadmap (20 min)**

**Q1 Deep Dive**:
- **SSO/SAML**: 4 weeks, 2 engineers
  - Why: Deal-breaker for 8 enterprise customers
  - User story: "As an IT admin, I want employees to sign in with company credentials"
  - Success: 8 deals close
  - Open questions: SAML 2.0 sufficient, or need OIDC too?

- **Advanced Permissions**: 3 weeks, 2 engineers  
  - Why: Enterprise customers need role-based access
  - User story: "As admin, I want to control what team members can access"
  - Success: Passes enterprise security review
  - Open questions: How granular? (feature-level, data-level, both?)

**Same format for Q2, Q3, Q4**

### **Part 3: Engineering Capacity (15 min)**

**Current capacity**: 5 engineers × 10 weeks = 50 eng-weeks per quarter

**Q1 allocation**:
- Enterprise: 20 weeks (SSO, permissions)
- PLG: 20 weeks (onboarding redesign)
- Platform: 10 weeks (monitoring, performance)

**Hiring plan**: +3 engineers by Q2 (8 engineers total → 80 eng-weeks)

**Question for team**: Is this allocation realistic, or should we adjust scope?

### **Part 4: Technical Architecture (10 min)**

**Key technical decisions needed**:
1. SSO: Build vs. buy (Auth0, Okta)?
2. Permissions: Role-based vs. attribute-based?
3. Platform: Monolith vs. microservices?

**Process**: Technical design reviews for each (you lead, I provide product context)

### **Part 5: Q&A (5 min)**

"What questions do you have? What am I missing?"

**What Engineering gets**:
- ✅ Clear roadmap (Q1-Q4)
- ✅ Specific timelines (4 weeks, 3 weeks)
- ✅ User stories (context for WHY)
- ✅ Capacity planning (50 eng-weeks allocation)
- ✅ Open questions (invites collaboration)

**Engineering leaves saying**: "This is clear. We know what to build and why."

---

## For Designer: User-Focused Version

**Meeting format**: 45-min 1-on-1 + co-working session  
**Presentation**: User research + collaborative discussion

### **Part 1: User Problems We're Solving (15 min)**

**Problem #1: Enterprise admins need control**
- User research quote: "I can't deploy this to my team without proper permissions"
- Interview insights: 5/8 enterprise customers mentioned this
- User need: Role-based access control

**Problem #2: New users don't activate**
- User research quote: "I signed up but didn't know what to do next"
- Interview insights: 60% abandon onboarding
- User need: Guided, clear onboarding flow

**Problem #3: Self-serve users don't upgrade**
- User research quote: "I didn't even know there was a paid plan"
- Interview insights: Users don't see upgrade prompts
- User need: Contextual upgrade suggestions

### **Part 2: Design Priorities (10 min)**

**Q1 Design Focus**: Enterprise permissions
- Challenge: Complex permissions, simple UI
- Inspiration: Notion, Figma permissions
- Success: Enterprise customers approve UX

**Q2 Design Focus**: Onboarding redesign
- Challenge: Guide without overwhelming
- Inspiration: Slack, Notion onboarding
- Success: Activation rate 40% → 60%

**Q3 Design Focus**: In-app upgrades
- Challenge: Upsell without being annoying
- Inspiration: Spotify, Notion upsells
- Success: Conversion 10% → 20%

### **Part 3: User Research Plan (10 min)**

**Together we'll**:
- Interview 5 enterprise admins (understand permissions needs)
- Test onboarding prototypes with 8 users (validate approach)
- Observe self-serve users (find upgrade moments)

**Timeline**: 2 weeks user research before each design phase

### **Part 4: Collaboration Model (10 min)**

**How we'll work**:
- Week 1-2: User research together
- Week 3-4: You explore design concepts, I provide feedback
- Week 5: Test prototypes with users together
- Week 6: Finalize designs, hand off to engineering

**I'll provide**: User research, product context, feedback, engineering coordination  
**You'll provide**: Design exploration, prototypes, visual quality, UX expertise

**What Designer gets**:
- ✅ User problems (not just "build permissions")
- ✅ Research insights (quotes, data)
- ✅ Design priorities (Q1-Q4)
- ✅ Collaboration model (how we work together)
- ✅ Creative freedom (explore concepts, not just execute)

**Designer leaves saying**: "I understand the user problems. Let's start research."

---

## Key Takeaways

**Same information, different framing**:

| Audience | Focus | Format | Depth | Tone |
|----------|-------|--------|-------|------|
| **CEO** | Vision, outcomes | Slides, visuals | High-level | Inspiring |
| **Engineering** | Roadmap, capacity | Detailed walkthrough | Specific | Technical |
| **Designer** | User problems, research | Collaborative | User-focused | Empathetic |

**The 1-page core doc** = single source of truth  
**The presentations** = tailored for each audience

**This approach ensures**:
- Everyone has same information (aligned)
- Each audience gets what they need (satisfied)
- One document to maintain (efficient)

**Great PMs adapt communication to audience while maintaining consistent strategy.**
`,
  },
];

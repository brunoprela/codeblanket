/**
 * Discussion questions for Working with Sales and Marketing
 * Product Management Fundamentals Module
 */

export const workingWithSalesMarketingQuiz = [
  {
    id: 1,
    question:
      "Your sales team brings you 15 different feature requests from customers, all claiming they're 'deal-breakers.' You can only build 2-3 features this quarter. Create a framework for prioritizing these requests. How do you distinguish between genuine patterns vs. one-off requests? How do you communicate your prioritization decisions back to sales without damaging the relationship?",
    answer: `## Comprehensive Answer:

Sales feature requests require systematic evaluation, not gut feelings. Let's create a framework.

### **Step 1: Create a Feature Request Evaluation Framework**

**For each of the 15 requests, collect these data points**:

| Request | Customers Asking | Revenue at Stake | Deal-Breaker? | Competitor Has It? | Strategic Fit | Effort |
|---------|------------------|------------------|---------------|-------------------|---------------|--------|
| Salesforce Integration | 8 customers | $750K | Yes (5 customers) | Yes (all) | High | 3 weeks |
| Custom PDF Export | 1 customer | $100K | Yes | No | Low | 2 weeks |
| SSO / SAML | 6 customers | $500K | Yes (4 customers) | Yes (all) | High | 4 weeks |
| Bulk Import | 4 customers | $200K | No | Yes (some) | Medium | 2 weeks |
| Advanced Analytics | 2 customers | $300K | No | Yes (most) | Medium | 6 weeks |
| Mobile App | 3 customers | $150K | No | No | Low | 12 weeks |
| ... | ... | ... | ... | ... | ... | ... |

### **Step 2: Apply Prioritization Framework**

**Question 1: Is this a pattern or one-off?**

**Pattern** (high priority):
- 5+ customers asking
- Multiple sales reps reporting it
- Appears in multiple deals

**One-off** (low priority):
- 1-2 customers
- Unique to specific use case
- Not mentioned by others

**From our table**:
- ✅ **Pattern**: Salesforce Integration (8 customers), SSO (6 customers), Bulk Import (4 customers)
- ❌ **One-off**: Custom PDF Export (1 customer), Mobile App (3 customers)

**Question 2: Is it truly a deal-breaker?**

**Real deal-breaker**:
- Customer explicitly says "no deal without this"
- We've lost deals because of it
- Sales rep confirms (not exaggerating)

**Nice-to-have**:
- Customer asks but will buy without it
- "Would be great to have"
- Not mentioned in final negotiations

**From our table**:
- ✅ **Deal-breaker**: Salesforce (5/8 customers), SSO (4/6 customers)
- ❌ **Nice-to-have**: Bulk Import, Advanced Analytics, Mobile App

**Question 3: Do competitors have it?**

**Competitive gap** (higher priority):
- All/most competitors have it
- We're losing deals because of it
- Table stakes for market

**Differentiation** (lower priority):
- Few competitors have it
- Not expected by market
- Nice innovation but not required

**From our table**:
- ✅ **Competitive gap**: Salesforce (all have it), SSO (all have it), Bulk Import (some have it)
- ❌ **Not critical**: Mobile App (no one has it), Custom PDF Export (no one has it)

**Question 4: Does it fit our product strategy?**

**Strategic fit** (higher priority):
- Aligns with where we're going
- Builds platform capabilities
- Serves target market

**Off-strategy** (lower priority):
- One-off feature for specific industry
- Doesn't serve target market
- Distracts from core mission

**From our table**:
- ✅ **Strategic fit**: Salesforce (we're building integrations), SSO (enterprise push)
- ❌ **Off-strategy**: Mobile App (web-first strategy), Custom PDF Export (not core)

**Question 5: What's the ROI?**

**ROI Formula**: (Revenue Impact) / (Engineering Effort)

**Example calculations**:
- Salesforce: $750K / 3 weeks = $250K per week → **High ROI**
- SSO: $500K / 4 weeks = $125K per week → **High ROI**
- Bulk Import: $200K / 2 weeks = $100K per week → **Medium ROI**
- Advanced Analytics: $300K / 6 weeks = $50K per week → **Medium ROI**
- Custom PDF Export: $100K / 2 weeks = $50K per week → **Medium ROI** (but one-off)
- Mobile App: $150K / 12 weeks = $12.5K per week → **Low ROI**

### **Step 3: Prioritization Decision**

**Scoring system** (0-5 points each):
- Pattern strength (5+ customers = 5 pts, 1 customer = 0 pts)
- Deal-breaker status (confirmed = 5 pts, nice-to-have = 0 pts)
- Competitive gap (all have = 5 pts, none have = 0 pts)
- Strategic fit (high = 5 pts, low = 0 pts)
- ROI (high = 5 pts, low = 0 pts)

**Total possible: 25 points**

**Scores**:
1. **Salesforce Integration**: 25/25 (8 customers=5, deal-breaker=5, competitive=5, strategic=5, ROI=5)
2. **SSO / SAML**: 24/25 (6 customers=4, deal-breaker=5, competitive=5, strategic=5, ROI=5)
3. **Bulk Import**: 16/25 (4 customers=3, nice-to-have=2, competitive=3, strategic=3, ROI=5)
4. **Advanced Analytics**: 12/25 (2 customers=1, nice-to-have=2, competitive=4, strategic=3, ROI=2)
5. **Mobile App**: 8/25 (3 customers=2, nice-to-have=1, competitive=0, strategic=0, ROI=0)
6. **Custom PDF Export**: 6/25 (1 customer=0, deal-breaker=2, competitive=0, strategic=0, ROI=4)

**Decision: Build top 2-3**
1. ✅ Salesforce Integration (Q1 priority)
2. ✅ SSO / SAML (Q1 priority)
3. ✅ Bulk Import (Q2 priority - if capacity allows)
4. ⏸️ Everything else: Defer to Q3+ or decline

### **Step 4: Communicate Decision to Sales**

**Don't just send an email saying "Here's what we're building."**

**Hold a Sales-Product Sync** (30-45 min):

**Agenda**:

**1. Thank sales for feedback** (5 min):
"Thank you for sharing these 15 customer requests. This feedback is incredibly valuable. I reviewed every single one and did analysis to prioritize."

**2. Share prioritization framework** (10 min):
"Here's how I evaluated these requests:
- Pattern strength (how many customers?)
- Deal-breaker status (truly required or nice-to-have?)
- Competitive gap (do we need this to compete?)
- Strategic fit (does it align with our direction?)
- ROI (revenue impact / engineering effort)

I scored each request on these dimensions."

**3. Share decisions with reasoning** (15 min):

**"What we're building (Q1)"**:
1. **Salesforce Integration** (3 weeks)
   - Why: 8 customers, $750K pipeline, deal-breaker for 5, all competitors have it, high strategic fit
   - Timeline: Shipping end of Q1
   - Impact: Unblocks 5 active deals

2. **SSO / SAML** (4 weeks)
   - Why: 6 customers, $500K pipeline, deal-breaker for 4, table stakes for enterprise
   - Timeline: Shipping mid-Q2
   - Impact: Opens enterprise market segment

**"What we're considering (Q2)"**:
3. **Bulk Import** (2 weeks)
   - Why: 4 customers, solid ROI, competitive gap
   - Decision: Will prioritize if Q1 ships on time

**"What we're deferring and why"**:
- **Custom PDF Export**: Only 1 customer, not a pattern yet. Will revisit if we see 5+ customers asking.
- **Mobile App**: Low ROI (12 weeks), not strategic priority (web-first), only 3 customers asking. Revisiting in 2026.
- **Advanced Analytics**: Medium priority, but SSO unblocks more revenue. Will consider Q3.

**4. How to handle customer conversations** (10 min):

**For customers asking for Salesforce/SSO**:
"Great news! We're building this. Expected delivery: [date]. Can we schedule a beta test with your customer?"

**For customers asking for deferred features**:
"We've heard this request and it's on our radar. Right now, we're prioritizing [Salesforce/SSO] which unblock larger revenue. We'll revisit [feature] in Q3. In the meantime, here's a workaround: [explain workaround]."

**5. Q&A and pushback** (10 min):

**Expected pushback**:

**Sales**: "But Customer X will walk if we don't build [Custom PDF Export]"

**PM Response**: "I understand. Let's look at the data:
- Only 1 customer asking (vs. 8 for Salesforce)
- $100K deal (vs. $750K for Salesforce)
- If we build PDF Export (2 weeks), we delay Salesforce (which unblocks 5 deals)
- Net: We'd close 1 deal but lose 5 deals

I'm optimizing for maximum revenue. Can we offer Customer X a workaround? Or a custom services engagement?"

**Sales**: "Why can't we build all of them?"

**PM Response**: "We have 3 engineers. Building everything would take 40+ weeks. We have 12 weeks this quarter. So we're prioritizing the 2-3 highest-impact features. The alternative is building everything poorly and shipping nothing on time."

### **Step 5: Follow Up**

**After the meeting**:

**1. Send summary email** (document decisions):
\`\`\`
Subject: Q1 Product Priorities - Based on Your Feedback

Team,

Thank you for the productive discussion today. Here's a summary:

BUILDING IN Q1:
✅ Salesforce Integration - Shipping end of Q1 (unblocks $750K pipeline)
✅ SSO / SAML - Shipping mid-Q2 (unblocks $500K pipeline)

CONSIDERING IN Q2:
⏸️ Bulk Import - Will prioritize if Q1 ships on time

DEFERRED:
❌ Custom PDF Export - Only 1 customer, not a pattern. Workaround: [explain]
❌ Mobile App - Low ROI, not strategic. Revisiting 2026.
❌ Advanced Analytics - Medium priority, deferred to Q3

HOW YOU CAN HELP:
- Let me know if you close deals with Salesforce/SSO commitments
- Share customer feedback on what's most valuable
- Flag if you see new patterns emerging

Thank you for your partnership. Let's close these deals!

[Your Name]
\`\`\`

**2. Monthly updates** (keep sales informed):
- "Salesforce Integration: 60% complete, on track for end of Q1"
- "SSO: Starting development next week"

**3. Celebrate wins together**:
When Salesforce ships: "Salesforce Integration is live! This unblocks 5 deals. Thank you for the customer feedback that informed this."

### **Key Principles for Sales Relationship**

**1. Be transparent** (share reasoning, not just decisions)
**2. Use data** (not opinions or feelings)
**3. Show you listened** (acknowledge every request)
**4. Explain trade-offs** (if we build X, we can't build Y)
**5. Follow through** (deliver what you committed)
**6. Celebrate together** (share credit when features close deals)

### **What NOT to Do**

❌ Ignore sales requests
❌ Say "No" without explanation
❌ Build everything they ask for (no strategy)
❌ Miss timelines repeatedly
❌ Dismiss their feedback as "just sales being sales"

### **The Long Game**

**First time you do this**: Sales might push back ("Why not my request?")

**After 2-3 quarters**: Sales learns:
- PM listens and acts on patterns
- PM makes data-driven decisions
- PM delivers on commitments
- PM is a partner, not blocker

**Result**: Trust builds, collaboration improves, better products ship.

**Success metric**: Sales says "PM gets it" instead of "PM doesn't listen."
`,
  },
  {
    id: 2,
    question:
      "Marketing wants to launch a big campaign announcing a feature that engineering says won't be ready for 6 more weeks. Marketing has already spent $50K on the campaign and doesn't want to delay. Engineering is firm on the timeline. You're caught in the middle. How do you handle this situation? What went wrong in the process, and how do you prevent it in the future?",
    answer: `## Comprehensive Answer:

This is a crisis situation requiring immediate damage control and long-term process fixes.

### **Step 1: Assess the Situation (Immediately)**

**Gather facts first** (don't react emotionally):

**To Marketing**:
- When is campaign launching?
- What exactly does it promise?
- How much has been spent? ($50K sunk cost)
- Can campaign be delayed? (What's the cost?)
- What channels? (Email, ads, PR, etc.)
- How many people will see it?

**To Engineering**:
- Why 6 weeks? (What's the blocker?)
- How confident is the estimate? (Could it be 4 weeks? 8 weeks?)
- Is there an MVP we could ship sooner?
- What's the risk of rushing? (Bugs? Technical debt?)
- Feature flags possible? (Ship to 10%, then scale?)

**Example findings**:
- Campaign launches in 3 days
- Promises feature to 50K customers via email + ads
- $50K spent (sunk cost)
- Delaying campaign costs $20K (ad spend already committed)
- Engineering needs 6 weeks (feature is complex)
- No MVP version possible (all-or-nothing feature)
- Rushing = high risk of bugs

**Conclusion**: We have a serious problem.

### **Step 2: Immediate Damage Control (Options)**

**Option A: Delay Campaign**
- **Pros**: Don't overpromise, maintain trust
- **Cons**: $20K wasted, momentum lost
- **When to choose**: If feature timeline very uncertain

**Option B: Launch Campaign, Delay Feature Access**
- **Pros**: Keep marketing momentum
- **Cons**: Customers expect feature now, will be frustrated
- **When to choose**: If we can communicate timeline clearly ("Coming in 6 weeks")

**Option C: Launch Campaign with Beta/Waitlist**
- **Pros**: Generate excitement, get early feedback
- **Cons**: Complex to manage, not all customers get access
- **When to choose**: If feature can be rolled out progressively

**Option D: Modify Campaign Promise**
- **Pros**: Align marketing with reality
- **Cons**: Less exciting campaign, may not achieve goals
- **When to choose**: If campaign can be quickly adjusted

**My recommendation: Option C (Beta/Waitlist)**

**Here's why**:
- Marketing campaign launches as planned (don't waste $50K)
- We don't overpromise (clear about beta timeline)
- We gather early feedback (makes feature better)
- Engineering has 6 weeks (no rushing, no bugs)
- Customers opt-in (self-select excited users)

### **Step 3: Execute the Solution**

**Modified launch plan**:

**Week 1 (now)**: 
- Marketing campaign launches with adjusted messaging:
  - OLD: "Feature X is now available!"
  - NEW: "Feature X is coming soon! Join the beta waitlist."
- Create waitlist landing page
- Set expectations: "Beta launches in 2 weeks, full rollout in 6 weeks"

**Week 2**:
- Engineering ships MVP to first 50 beta users
- PM + Marketing gather feedback
- Engineering iterates

**Week 4**:
- Beta expands to 500 users
- Monitor for issues

**Week 6**:
- Feature ships to all 50K customers
- Marketing announces general availability

**What this achieves**:
- Marketing campaign isn't wasted ($50K preserved)
- Engineering has time to ship quality (6 weeks)
- We don't overpromise (beta is clearly communicated)
- Early feedback improves feature

### **Step 4: Communication Plan**

**To Marketing** (1-on-1 with Marketing Lead):

"I understand you've invested $50K and don't want to delay. I also understand Engineering needs 6 weeks. Here's a solution that preserves your investment and ensures quality:

**Proposal: Beta/Waitlist Launch**
- Campaign launches on schedule (your $50K isn't wasted)
- Messaging changes to 'Join the Beta Waitlist' (not 'Available Now')
- We ship to beta users in 2 weeks (shows progress)
- Full rollout in 6 weeks (Engineering has time for quality)

**What I need from you**:
- Adjust campaign messaging (I'll help draft)
- Create waitlist landing page (I'll provide specs)
- Communicate beta timeline clearly (manage expectations)

**Why this works**:
- Your campaign generates leads and excitement
- We don't overpromise and lose customer trust
- Engineering ships quality (not rushed, buggy product)
- Beta feedback makes feature better

Does this work?"

**To Engineering** (1-on-1 with EM):

"Marketing is launching campaign in 3 days. I know you need 6 weeks for full feature. Here's my proposal:

**Proposal: Progressive Rollout**
- Marketing launches 'Beta Waitlist' campaign (not 'Available Now')
- You ship MVP to 50 beta users in 2 weeks (small scope, manageable)
- You have 4 more weeks to finish full feature (total 6 weeks)
- Full rollout after 6 weeks

**What I need from you**:
- Confirm 2-week MVP is feasible (even if limited)
- Define what's in MVP vs. full version
- Commit to 6-week timeline for full version

**Why this approach**:
- We don't overpromise to 50K customers
- You have time for quality (6 weeks total)
- Beta feedback informs final product
- Marketing campaign isn't wasted

Does this work?"

**To Leadership** (if escalation needed):

"We have a coordination issue: Marketing launching campaign in 3 days, Engineering needs 6 weeks. Here's how I'm resolving it:

**Immediate action**: Beta/Waitlist launch (preserves marketing investment, gives Engineering time)

**Root cause**: Marketing and Engineering weren't aligned on timeline

**Long-term fix**: New product launch process (see Step 5)

**Decision needed from you**: Approval for beta approach (vs. delaying campaign or rushing engineering)"

### **Step 5: Root Cause Analysis (What Went Wrong)**

**Ask yourself**:

**1. When did Marketing start planning this campaign?**
- If 3 months ago → PM should have been involved then
- If 2 weeks ago → Marketing moved too fast

**2. When did PM communicate timeline to Marketing?**
- If never → PM failed to communicate
- If 1 month ago but timeline changed → PM failed to update Marketing
- If PM communicated but Marketing ignored → Marketing problem

**3. When did Engineering commit to timeline?**
- If 3 months ago and haven't delivered → Engineering estimation problem
- If timeline changed recently → Scope creep or technical blockers

**Likely root causes**:
- ✅ **PM didn't establish launch process** (no coordination)
- ✅ **Marketing planned campaign without confirming timeline** (assumption)
- ✅ **Engineering timeline wasn't communicated clearly** (miscommunication)
- ✅ **No single source of truth for launch dates** (lack of process)

### **Step 6: Prevent This in the Future (Process Fix)**

**Create a Product Launch Process**:

**Phase 1: Planning (8-12 weeks before launch)**
- PM proposes feature for roadmap
- Engineering estimates effort
- PM sets target launch date (with buffer)
- PM communicates to Marketing: "Targeting Q2 launch"

**Phase 2: Kickoff (6-8 weeks before launch)**
- PM confirms timeline with Engineering: "Still on track for Q2?"
- PM holds Launch Kickoff with Marketing:
  - What are we launching?
  - When are we launching? (confirmed date)
  - Who's the audience?
  - What's the positioning?
- Marketing begins planning campaign (6 weeks out)

**Phase 3: Weekly Syncs (4-6 weeks before launch)**
- PM + Marketing + EM weekly sync (15 min)
- Status update: "Are we on track?"
- If timeline slips → Immediately adjust campaign plans
- No surprises

**Phase 4: Go/No-Go (2 weeks before launch)**
- PM holds Go/No-Go meeting:
  - Is feature ready?
  - Is marketing campaign ready?
  - Are we aligned on timeline?
- Decision: Launch or delay
- If delay → Marketing has 2 weeks to adjust (not 3 days)

**Phase 5: Launch (Launch week)**
- Coordinated launch (PM + Marketing + Eng aligned)
- Monitor for issues
- Celebrate together

**Key process elements**:

**1. Single source of truth** (shared launch calendar):
- Notion or Google Sheet: "Upcoming Launches"
- Columns: Feature, Target Date, Status, PM Owner, Marketing Owner, Eng Owner
- Updated weekly

**2. Weekly PM-Marketing-Eng sync** (30 min, every week):
- Review upcoming launches
- Flag timeline changes immediately
- Align on priorities

**3. Minimum 6 weeks notice** (rule):
- Marketing needs 6 weeks to plan campaigns
- PM confirms timeline 6+ weeks out (or doesn't commit)
- No "surprise launches"

**4. Go/No-Go meeting** (2 weeks before launch):
- Formal checkpoint
- Decision: Launch on time, delay, or adjust scope
- Forces honest assessment

### **Step 7: Post-Crisis Retro**

**After the crisis is resolved, hold a retro** (PM + Marketing + Eng):

**What went well?**
- (Probably not much in a crisis)

**What went wrong?**
- Marketing planned campaign without confirmed timeline
- Engineering timeline not communicated to Marketing
- PM didn't coordinate effectively
- No launch process existed

**What will we change?**
- Implement Product Launch Process (see Step 6)
- Weekly PM-Marketing-Eng syncs
- Shared launch calendar (single source of truth)
- No campaigns without Go/No-Go meeting

**Action items** (assign owners):
- [ ] Create launch calendar (PM, by [date])
- [ ] Set up weekly sync (PM, by [date])
- [ ] Document launch process (PM, by [date])
- [ ] Train Marketing on new process (PM, by [date])

### **Key Takeaways**

**Immediate crisis management**:
1. **Assess options** (delay, beta, modify)
2. **Choose solution that balances** marketing investment + engineering quality + customer expectations
3. **Communicate clearly** to all stakeholders
4. **Execute decisively**

**Long-term prevention**:
1. **Establish launch process** (planning → kickoff → syncs → go/no-go → launch)
2. **Create single source of truth** (shared calendar)
3. **Weekly coordination** (PM + Marketing + Eng)
4. **Minimum 6 weeks notice** (rule)
5. **Go/No-Go checkpoint** (2 weeks before launch)

**PM's role**:
- **Coordinator**: Align Marketing + Eng on timelines
- **Communicator**: Keep everyone informed
- **Decision-maker**: Choose best path forward in crisis
- **Process builder**: Prevent future crises

**Success metric**: After implementing process, no more "Marketing launching before product is ready" surprises.

**Remember**: This crisis happened because of lack of process, not bad people. Fix the process, not blame people.
`,
  },
  {
    id: 3,
    question:
      "Design a 'Go-to-Market (GTM) Playbook' for your product team. Include: how to decide which launches deserve marketing campaigns vs. quiet releases, the launch readiness checklist, coordination timeline (who does what when), stakeholder communication plan, success metrics, and post-launch review process. Make it detailed enough that a new PM could execute a launch using this playbook.",
    answer: `## Comprehensive Answer:

Here's a comprehensive Go-to-Market (GTM) Playbook:

---

# Go-to-Market (GTM) Playbook

## Part 1: Launch Tiering Framework

**Not all launches are equal. Decide launch tier first.**

### **Tier 1: Flagship Launch** (2-3 per year)

**What qualifies**:
- Major new product or feature
- Large revenue impact (>$500K)
- Significant competitive advantage
- Targets new market segment
- Strategic priority for company

**Examples**: New product launch, major platform capability, enterprise features

**Marketing investment**: High ($20K-50K+)
- PR campaign (press releases, media outreach)
- Paid advertising (Google, LinkedIn, etc.)
- Content marketing (blog posts, videos, webinars)
- Customer emails (multiple touchpoints)
- Sales enablement (training, materials, demos)
- Launch event (virtual or in-person)

**Timeline**: 8-12 weeks planning

---

### **Tier 2: Standard Launch** (6-8 per year)

**What qualifies**:
- Important feature for existing customers
- Medium revenue impact ($100K-500K)
- Addresses common customer requests
- Improves competitive position

**Examples**: Major feature additions, integrations, significant improvements

**Marketing investment**: Medium ($5K-20K)
- Blog post announcement
- Customer email (single touchpoint)
- In-app notification
- Social media posts
- Sales enablement (brief, materials)

**Timeline**: 4-6 weeks planning

---

### **Tier 3: Quiet Release** (monthly)

**What qualifies**:
- Minor feature or improvement
- Low revenue impact (<$100K)
- Nice-to-have, not critical
- Incremental improvement

**Examples**: UI improvements, small features, bug fixes, performance improvements

**Marketing investment**: Low ($0-5K)
- In-app notification
- Changelog entry
- Optional social media post
- No sales enablement

**Timeline**: 1-2 weeks planning

---

### **Decision Framework**

**Ask these questions to determine tier**:

| Question | Tier 1 | Tier 2 | Tier 3 |
|----------|--------|--------|--------|
| Revenue impact? | >$500K | $100-500K | <$100K |
| Strategic importance? | Critical | Important | Nice |
| Customer demand? | Very high | Medium | Low |
| Competitive advantage? | Significant | Moderate | Minor |
| Market expansion? | Yes | Maybe | No |
| Development effort? | >3 months | 1-3 months | <1 month |

**If 4+ answers point to Tier 1** → Flagship Launch
**If 3+ answers point to Tier 2** → Standard Launch  
**Otherwise** → Quiet Release

---

## Part 2: Launch Readiness Checklist

**Before ANY launch, complete this checklist.**

### **Product Readiness** (PM + Engineering)

- [ ] **Feature is built and tested**
  - All user stories complete
  - QA testing passed
  - Edge cases handled
  - Performance acceptable (<2 sec load time)

- [ ] **Feature flags configured** (for progressive rollout)
  - 1% rollout ready
  - 10% rollout ready
  - 100% rollout ready
  - Kill switch exists

- [ ] **Monitoring and alerting set up**
  - Key metrics tracked (usage, errors, performance)
  - Alerts for critical issues
  - Dashboard for monitoring

- [ ] **Rollback plan exists**
  - How to disable feature quickly
  - Communication plan if rollback needed
  - Testing of rollback procedure

- [ ] **Documentation written**
  - Help center articles
  - API documentation (if applicable)
  - FAQ document

### **Marketing Readiness** (PM + Marketing)

- [ ] **Positioning and messaging finalized**
  - One-sentence description
  - Target audience defined
  - Key benefits identified (3-5)
  - Differentiation vs. competitors

- [ ] **Visual assets created**
  - Screenshots (showing feature in action)
  - Demo video (2-3 min walkthrough)
  - GIFs for social media
  - Diagrams (if technical feature)

- [ ] **Marketing materials ready**
  - Blog post written (Tier 1 & 2)
  - Email copy ready (Tier 1 & 2)
  - Social media posts drafted
  - Press release (Tier 1 only)
  - Landing page (if applicable)

- [ ] **Launch timeline confirmed**
  - Launch date set
  - All stakeholders aligned
  - Go/No-Go meeting completed

### **Sales Readiness** (PM + Sales)

- [ ] **Sales enablement complete**
  - Product training delivered
  - One-pager created (1-page overview)
  - Demo script written
  - FAQ document for objections
  - Pricing confirmed (if applicable)

- [ ] **Demo environment ready**
  - Working sandbox with sample data
  - Admin access for sales team
  - Demo script tested

- [ ] **Customer proof** (Tier 1 & 2)
  - Beta customer testimonials
  - Case study (at least 1)
  - Usage statistics
  - ROI examples

### **Support Readiness** (PM + Support)

- [ ] **Support team trained**
  - Feature walkthrough completed
  - Common questions identified
  - Troubleshooting guide written

- [ ] **Help center updated**
  - Articles written
  - Screenshots added
  - Search keywords optimized

- [ ] **Support capacity planned**
  - Expect 2-3X support volume on launch day
  - On-call PM available for escalations

---

## Part 3: Coordination Timeline

### **Tier 1: Flagship Launch (8-12 weeks)**

**Weeks 12-10: Planning Phase**

**Week 12**:
- PM: Define feature scope and success metrics
- PM: Set tentative launch date
- PM: Create launch team (PM, Marketing, Sales, Support)

**Week 11**:
- PM + Eng: Technical design and estimation
- PM: Create project brief (problem, solution, success metrics)

**Week 10**:
- PM: Hold Launch Kickoff Meeting (PM, Marketing, Sales, Eng, Support)
  - What are we building?
  - Who's it for?
  - Why does it matter?
  - Target launch date
- Marketing: Begin planning campaign

---

**Weeks 9-5: Build Phase**

**Week 9**:
- Eng: Begin development
- Designer: Create mockups and user flows
- PM: Begin writing documentation

**Week 8**:
- PM + Marketing: Finalize positioning and messaging
- Marketing: Begin creating visual assets (screenshots, videos)
- Sales: Begin preparing sales materials

**Week 7**:
- PM: Weekly sync with Marketing, Sales, Support (status updates)
- Marketing: Draft blog post and email copy

**Week 6**:
- PM: Mid-point check: Are we on track?
- Eng: Feature 50% complete
- PM: Begin beta customer recruitment (for testimonials)

**Week 5**:
- Marketing: Finalize blog post, email, social posts
- Sales: Complete one-pager and demo script
- PM: Help center articles in draft

---

**Weeks 4-3: Preparation Phase**

**Week 4**:
- Eng: Feature code complete, begin testing
- PM: QA testing
- PM + Marketing: Review all marketing materials
- Sales: Product training for sales team

**Week 3**:
- PM: Beta launch to 10-20 customers
- PM: Gather beta feedback and testimonials
- Support: Support team training
- Marketing: Finalize all launch materials

---

**Week 2: Final Prep**

**Monday**: 
- PM: Go/No-Go meeting (PM, Eng, Marketing, Sales)
  - Is feature ready?
  - Is marketing ready?
  - Is sales ready?
  - Is support ready?
  - Decision: Go or Delay

**Wednesday** (if Go):
- PM: Final QA
- Marketing: Schedule all marketing campaigns
- Sales: Demo environment ready

**Friday**:
- Eng: Deploy to production (feature flagged off)
- PM: Final smoke testing
- All: Final confirmation of launch readiness

---

**Week 1: Launch Week**

**Monday (Pre-launch)**:
- PM: Enable feature for 1% of users (feature flag)
- PM: Monitor for issues (no major problems)

**Wednesday (Public launch)**:
- 8 AM: PM enables feature for 10% of users
- 9 AM: Marketing sends email to customers
- 10 AM: Marketing publishes blog post
- 11 AM: Marketing posts on social media
- 12 PM: PR sends press release (if applicable)
- Afternoon: Sales begins reaching out to prospects
- All day: PM + Support monitor for issues

**Thursday**:
- PM: Enable feature for 50% of users (if no issues)
- PM: Monitor metrics (usage, errors, support tickets)

**Friday**:
- PM: Enable feature for 100% of users
- PM: Send thank you to launch team

---

**Week 0: Post-Launch**

**1 week after launch**:
- PM: Gather initial metrics (adoption, engagement, support tickets)
- PM: Share results with launch team

**2 weeks after launch**:
- PM: Hold Post-Launch Review (see Part 6)
- PM: Document learnings

---

### **Tier 2: Standard Launch (4-6 weeks)**

**Simplified timeline** (same phases, compressed):
- Weeks 6-5: Planning
- Weeks 4-2: Build
- Week 1: Prep
- Week 0: Launch

**Key difference**: Less marketing investment, faster execution

---

### **Tier 3: Quiet Release (1-2 weeks)**

**Minimal timeline**:
- Week 2: Build
- Week 1: Test and prep
- Week 0: Release (no big announcement)

**Just**:
- In-app notification
- Changelog entry
- Basic documentation

---

## Part 4: Stakeholder Communication Plan

### **Before Launch: Building Awareness**

**8 weeks before launch**:
- **To Executive Team**: Email update
  - "We're planning [Feature X] launch for [Date]"
  - "This will [business impact]"
  - "I'll keep you updated"

**6 weeks before launch**:
- **To Sales Team**: Launch kickoff presentation
  - What we're building
  - Who it's for
  - How to sell it
  - Q&A

**4 weeks before launch**:
- **To Support Team**: Training session
  - Feature walkthrough
  - Common questions
  - Troubleshooting

**2 weeks before launch**:
- **To Executive Team**: Go/No-Go decision
  - "We're on track to launch [Date]"
  - "Here's what to expect"

**1 week before launch**:
- **To Entire Company**: All-hands preview
  - Demo the feature
  - Explain customer value
  - Ask for feedback

### **During Launch: Real-Time Updates**

**Launch day**:
- **To Executive Team** (8 AM): "Launching today"
- **To Sales Team** (9 AM): "Feature is live, here's how to sell it"
- **To Company** (10 AM in Slack): "We just launched [Feature]! Check it out: [link]"

**End of launch day**:
- **To Executive Team**: "Day 1 metrics: [X] users adopted, [Y] support tickets"

### **After Launch: Results**

**1 week after**:
- **To Executive Team**: Email with week 1 metrics
- **To Sales Team**: Update with early customer feedback

**2 weeks after**:
- **To Company**: Launch retrospective (what we learned)

---

## Part 5: Success Metrics

### **Define Success Metrics Before Launch**

**For every launch, define 3 types of metrics**:

**1. Adoption Metrics** (how many use it?):
- % of users who try feature (within 30 days)
- % of users who use regularly (weekly active)
- Time to first use (how quickly do users discover it?)

**Example targets**:
- 30% of users try feature within 30 days
- 15% become weekly active users

**2. Engagement Metrics** (how much do they use it?):
- Daily/Weekly active users
- Actions per session
- Time spent using feature

**Example targets**:
- 500 daily active users by end of month
- 5 actions per session on average

**3. Business Impact Metrics** (does it matter?):
- Revenue impact (new deals, expansion)
- Retention improvement (% churn reduction)
- User satisfaction (NPS, feedback)
- Support ticket reduction (if fixing pain point)

**Example targets**:
- $200K new revenue from feature-driven deals
- 5% reduction in churn
- NPS improvement from 40 to 45

### **Track Metrics Weekly for First Month**

**Create dashboard** (in Amplitude, Mixpanel, or custom):
- Adoption rate
- Engagement trends
- Business impact

**Share weekly** with stakeholders.

---

## Part 6: Post-Launch Review Process

### **Hold Post-Launch Review (2 weeks after launch)**

**Attendees**: PM, Marketing, Sales, Engineering, Support

**Duration**: 60 minutes

**Agenda**:

**1. Review Success Metrics** (15 min):
- Did we hit adoption targets?
- Did we hit engagement targets?
- Did we hit business impact targets?

**2. What Went Well?** (15 min):
- What worked in the launch?
- What should we repeat?
- Who should we thank?

**3. What Didn't Go Well?** (20 min):
- What didn't work?
- What surprised us (good or bad)?
- What would we change?

**4. Learnings and Action Items** (10 min):
- What did we learn?
- What will we do differently next time?
- Action items (assign owners and dates)

**Example Learnings**:
- ✅ "Beta customer testimonials were powerful in sales conversations → Do this for all Tier 1 launches"
- ⚠️ "Support team felt under-prepared → Do training 3 weeks before launch (not 1 week)"
- ⚠️ "Blog post didn't resonate → Work with Marketing earlier on messaging"

### **Document and Share**

**Create Launch Review Doc** (template):

\`\`\`markdown
# [Feature Name] Launch Review

## Success Metrics
- Adoption: [% of users who tried within 30 days]
- Engagement: [DAU, WAU]
- Business Impact: [Revenue, retention, NPS]

## What Went Well
- [List 3-5 things]

## What Didn't Go Well
- [List 3-5 things]

## Learnings
- [Key lessons]

## Action Items for Next Launch
- [ ] [Action 1] (Owner, Date)
- [ ] [Action 2] (Owner, Date)
\`\`\`

**Share with**:
- Launch team
- Executive team (summary)
- Archive for future reference

---

## Part 7: Example: Flagship Launch Execution

**Feature**: "Salesforce Integration"  
**Launch Tier**: Tier 1 (Flagship)  
**Timeline**: 12 weeks

**Week 12**: PM sets target launch date, creates launch team  
**Week 11**: Engineering estimates 8 weeks to build  
**Week 10**: Launch Kickoff (PM, Marketing, Sales, Eng, Support align)  
**Weeks 9-5**: Engineering builds, PM writes docs, Marketing creates materials  
**Week 4**: Beta launch (10 customers test)  
**Week 3**: Gather testimonials, finalize marketing  
**Week 2**: Go/No-Go (Decision: Go), final prep  
**Week 1**: Launch! Enable feature, Marketing campaigns, Sales sells  
**Week 0**: Post-Launch Review (assess results, document learnings)

**Result**: Successful launch, $500K pipeline unlocked, 40% adoption in 30 days.

---

## Key Takeaways

1. **Not all launches are equal** (use tier framework)
2. **Plan launches systematically** (8-12 weeks for Tier 1)
3. **Coordinate across teams** (PM, Marketing, Sales, Eng, Support)
4. **Use launch readiness checklist** (ensure nothing forgotten)
5. **Communicate proactively** (keep stakeholders informed)
6. **Define success metrics** (measure what matters)
7. **Learn and iterate** (post-launch review every time)

**This playbook ensures consistent, high-quality launches every time.**

---

**End of GTM Playbook**
`,
  },
];

/**
 * Section 8: Working with Sales and Marketing
 * Product Management Fundamentals Module
 */

import { ModuleSection } from '../../../types';

const content = `# Working with Sales and Marketing

## Introduction

Product Management sits at the intersection of product building (Engineering, Design) and product selling (Sales, Marketing). Great PMs understand both worlds and bridge the gap effectively. Poor PM relationships with Sales and Marketing lead to products that are hard to sell, misaligned messaging, and internal conflict.

This section covers how to build strong partnerships with Sales and Marketing teams.

## Understanding Sales and Marketing Perspectives

### **What Sales Values**

**1. Winning Deals**
- Features that close deals
- Competitive advantages
- Customer testimonials and case studies
- Product capabilities they can sell

**2. Customer Feedback**
- Sales hears customer pain points daily
- They know what customers ask for
- They understand buying objections
- They see competitive threats firsthand

**3. Predictability**
- Clear product roadmap (to set customer expectations)
- Advance notice of launches (to prepare sales materials)
- Reliable timelines (don't promise dates that slip)
- Understanding of product limitations (to set realistic expectations)

**4. Enablement**
- Product training (how does it work?)
- Sales materials (decks, one-pagers, demos)
- Demo environments (working sandbox)
- Customer success stories

### **What Marketing Values**

**1. Compelling Narratives**
- Clear product positioning ("what is it?")
- Differentiation ("why us vs. competitors?")
- Customer value propositions ("why should they care?")
- Brand consistency

**2. Launch Moments**
- Big, coordinated product launches
- Time to plan campaigns (not last-minute)
- Assets for marketing (screenshots, videos, testimonials)
- Measurable outcomes (leads, conversions)

**3. Market Insights**
- Understanding market trends
- Competitive intelligence
- Customer segments and personas
- Messaging that resonates

**4. Data and Results**
- Product metrics (usage, engagement, retention)
- Customer testimonials and case studies
- ROI stories ("Company X saved $100K")
- Launch performance data

### **Common Frustrations**

**Sales Complaints About PMs**:
- "PM doesn't understand what customers actually want"
- "PM builds features nobody asks for"
- "PM ignores our feature requests from customers"
- "PM's timelines are always wrong"
- "PM doesn't prioritize features that close deals"

**Marketing Complaints About PMs**:
- "PM launches products with no notice"
- "PM changes messaging constantly"
- "PM doesn't provide marketing materials"
- "PM launches features before we can build awareness"

**PM Complaints About Sales/Marketing**:
- "Sales just wants features for one customer"
- "Marketing doesn't understand the product"
- "Sales overpromises what product can do"
- "They don't respect prioritization"

**All of these are symptoms of poor cross-functional collaboration.**

## The PM-Sales Partnership

### **Best Practice #1: Regular Sales Feedback Loop**

**Create structured feedback channels** (not just ad-hoc complaints).

**Monthly Sales-Product Sync** (60 min):
**Attendees**: PM, Sales Leader, 2-3 sales reps

**Agenda**:
1. Product updates (15 min):
   - What shipped last month
   - What's coming next month
   - Roadmap highlights

2. Sales feedback (30 min):
   - Common customer requests
   - Feature gaps vs. competitors
   - Objections in deals
   - Recent wins and losses

3. Questions and discussion (15 min)

**Output**: Document of feedback, PM commits to reviewing for roadmap

**Key**: PM listens without being defensive. Thank sales for feedback.

### **Best Practice #2: Distinguish Between "One-off Requests" and "Patterns"**

**Sales will bring many customer requests. PM's job: Find patterns.**

**One-off request** (low priority):
- One customer wants custom integration
- Unique to their specific workflow
- Not valuable to broader market

**Pattern** (high priority):
- 5 customers asking for similar thing
- Common objection in deals
- Competitors have it
- Clearly addresses market need

**Framework for evaluating sales requests**:

**Question 1**: How many customers are asking for this?
- 1 customer = One-off
- 5+ customers = Pattern

**Question 2**: Is this a deal-breaker or nice-to-have?
- Nice-to-have = Lower priority
- Deal-breaker = Higher priority (if pattern)

**Question 3**: Does it fit our product strategy?
- No = Decline (even if requested)
- Yes = Prioritize based on impact

**Question 4**: What's the revenue impact?
- Low = Deprioritize
- High = Prioritize

**Example conversation**:

**Sales**: "Customer X wants Salesforce integration"

**PM**: "Thanks for flagging. Help me understand:
- How many customers have asked for Salesforce?
- Is this a deal-breaker or nice-to-have?
- How much revenue is at stake?
- What's the use case?"

**Sales**: "5 customers in pipeline, 2 existing customers. It's a deal-breaker for 3 of them. $500K in pipeline. They want to sync contacts."

**PM**: "Got it. This is a pattern (5 customers, deal-breaker, significant revenue). I'll prioritize this for Q3. I'll share timeline next week."

**Key**: PM treats sales requests seriously, asks clarifying questions, makes informed prioritization decisions.

### **Best Practice #3: Sales Enablement**

**PM's responsibility: Enable sales to sell effectively.**

**What sales needs**:

**1. Product Training** (monthly for new features):
- How does it work?
- What customer problems does it solve?
- How to demo it
- Common questions and objections

**2. Sales Materials**:
- One-pager (1 page feature overview)
- Deck (slides for customer presentations)
- Demo script (step-by-step demo guide)
- FAQ (common customer questions)
- Pricing and packaging

**3. Demo Environment**:
- Working sandbox environment
- Sample data pre-loaded
- Admin access for sales team
- Clear instructions

**4. Customer Stories**:
- Case studies (how customers use it)
- Testimonials (quotes from happy customers)
- ROI examples ("Company X saved $100K")

**PM doesn't have to create all of this** (Marketing helps), but PM ensures it exists.

### **Best Practice #4: Don't Build Everything Sales Asks For**

**Sales will ask for everything customers want. PM's job: Prioritize strategically.**

**Say NO to**:
- One-off custom requests (not valuable to broader market)
- Features that don't fit product strategy
- Requests from non-target customers
- "We need this ASAP" without clear justification

**How to say NO to sales**:

**Bad approach**:
- "No, that's not on the roadmap"
- "That's not a priority"
- Ignore the request

**Good approach**:
- Acknowledge the request: "Thanks for sharing this customer feedback"
- Explain the trade-off: "If we build this, we'd delay [other feature]. Here's why [other feature] impacts more customers..."
- Offer alternative: "Can we solve this with [workaround] in the meantime?"
- Set expectations: "I'll revisit this in Q4 based on demand"

**Example**:

**Sales**: "Customer wants custom PDF export. They'll close if we build it."

**PM**: "I appreciate you bringing this. Let me explain my thinking:
- This is the first customer to ask for PDF export (not a pattern yet)
- Our focus this quarter is [strategic initiative] which impacts 80% of users
- Building custom PDF would take 3 weeks and delay [feature X] which 50 customers requested
- Can the customer use our CSV export + convert to PDF as a workaround?
- If we see 5+ customers asking for PDF export, I'll prioritize it next quarter"

**Sales might push back**, but clear reasoning helps them understand trade-offs.

## The PM-Marketing Partnership

### **Best Practice #1: Involve Marketing Early**

**Bad timeline**:
- Week 1-8: PM + Eng build feature
- Week 9: PM tells Marketing "We're launching next week"
- Week 10: Marketing scrambles to create materials
- Result: Poor launch, low awareness

**Good timeline**:
- Week 1: PM tells Marketing "We're planning [feature] for Q2"
- Week 4: PM + Marketing sync: What is it? Who's it for? Why does it matter?
- Week 6: Marketing begins planning (messaging, assets, channels)
- Week 8: Marketing finalizes materials
- Week 10: Coordinated launch (PM + Marketing together)

**Key**: Involve Marketing at planning phase, not launch week.

**Minimum notice**: 4-6 weeks before launch for major features.

### **Best Practice #2: Collaborate on Messaging**

**PM and Marketing both care about positioning**, but have different perspectives.

**PM perspective** (product-focused):
- "We built [feature]"
- Technical capabilities
- How it works

**Marketing perspective** (customer-focused):
- "You can now [accomplish outcome]"
- Customer benefits
- Why it matters

**Best messaging combines both**:
- Start with customer benefit (Marketing)
- Explain how product delivers it (PM)

**Example**:

**PM's messaging** (product-focused):
"We added real-time sync with Salesforce"

**Marketing's messaging** (customer-focused):
"Never manually update CRM again—changes sync instantly"

**Collaborative messaging**:
"Never manually update CRM again. With real-time Salesforce sync, changes flow instantly between systems. Save 5 hours per week and eliminate data entry errors."

**Process**: PM drafts messaging, Marketing refines for audience, PM approves for accuracy.

### **Best Practice #3: Provide Marketing What They Need**

**For product launches, Marketing needs**:

**1. Positioning and Messaging** (PM drafts, Marketing refines):
- What is it? (1-sentence description)
- Who is it for? (target audience)
- What problem does it solve?
- Why choose us vs. competitors?
- Key benefits (3-5 bullets)

**2. Visual Assets** (Design creates, PM provides context):
- Screenshots (showing key features)
- Demo video (2-3 min walkthrough)
- Diagrams (how it works)
- Before/After comparisons

**3. Customer Proof** (PM + Sales + Marketing):
- Beta customer testimonials
- Case studies (early adopters)
- Usage statistics ("10X faster")
- ROI examples

**4. Technical Details** (PM provides):
- Feature documentation
- API details (if applicable)
- Integrations and compatibility
- Pricing and packaging

**Timeline**: Provide these 2-4 weeks before launch so Marketing can create campaigns.

### **Best Practice #4: Measure Launch Success Together**

**After launch, PM + Marketing review results**:

**Product Metrics** (PM tracks):
- Adoption rate (% of users using new feature)
- Engagement (daily/weekly active users)
- Retention impact (did it improve retention?)

**Marketing Metrics** (Marketing tracks):
- Campaign reach (impressions, clicks)
- Leads generated (sign-ups, trials)
- Conversion rate (trial → paid)

**Together, assess**:
- Did launch meet goals?
- What worked? What didn't?
- What to improve for next launch?

**This feedback loop improves future launches.**

## Common Conflicts and Resolutions

### **Conflict #1: Sales Wants Features for One Customer**

**Sales**: "Customer X wants [feature], $500K deal on the line"  
**PM**: "That's a one-off request, not valuable to broader market"

**Resolution**:
1. **Acknowledge**: "I understand closing this deal matters"
2. **Explain trade-off**: "Building this delays [feature Y] which 50 customers need"
3. **Explore alternatives**: "Can we offer [workaround] or [custom service]?"
4. **Make strategic call**: If one customer but huge revenue → Consider. If smaller deal → Decline.
5. **Set expectations**: "I'll revisit if we see pattern"

**Sometimes you'll build it** (if strategic), **sometimes you won't** (if not aligned with strategy).

### **Conflict #2: Marketing Launches Before Product is Ready**

**Marketing**: "We announced [feature] in our campaign"  
**PM**: "That feature won't ship for 2 more months"

**Resolution**:
1. **Prevent this**: Clear communication about launch dates
2. **If it happens**: Apologize to customers, communicate new timeline
3. **Post-mortem**: How did this happen? Fix the process.

**Prevention**: Weekly PM-Marketing syncs on upcoming launches and timelines.

### **Conflict #3: Sales Overpromises Product Capabilities**

**Sales**: "I told customer [feature] could do [X]"  
**PM**: "Our product doesn't do that"

**Resolution**:
1. **Short-term**: Work with Sales to clarify to customer (set realistic expectations)
2. **Medium-term**: Better sales training (PM trains sales on capabilities and limitations)
3. **Long-term**: Sales materials clearly state what product does and doesn't do

**PM's role**: Enable sales with accurate information so they don't accidentally overpromise.

## Key Takeaways

**Working with Sales**:
1. **Regular feedback loop** (monthly Sales-Product syncs)
2. **Find patterns** (not every customer request is a priority)
3. **Enable sales** (training, materials, demos, customer stories)
4. **Say NO strategically** (explain trade-offs, offer alternatives)
5. **Respect sales insights** (they talk to customers daily)

**Working with Marketing**:
1. **Involve early** (4-6 weeks before launch minimum)
2. **Collaborate on messaging** (customer benefits + product capabilities)
3. **Provide assets** (screenshots, videos, testimonials)
4. **Measure success together** (product + marketing metrics)
5. **Align on positioning** (what makes us different?)

**Overall**:
- **PM is the bridge** between product building and product selling
- **Balance stakeholder requests with strategy** (not everything can be prioritized)
- **Enable go-to-market** (Sales and Marketing need PM's support)
- **Communicate clearly and proactively** (no surprises)

## Practical Exercise

**Assess your PM-Sales-Marketing relationships**:

**Green flags**:
- [ ] Regular syncs with Sales and Marketing
- [ ] Sales feels heard when they share feedback
- [ ] Marketing has 4+ weeks notice for launches
- [ ] Clear process for evaluating feature requests
- [ ] Sales successfully sells your product
- [ ] Marketing creates compelling campaigns

**Red flags**:
- [ ] Sales constantly escalates to leadership
- [ ] Marketing surprised by launches
- [ ] Sales complains PM doesn't listen
- [ ] No structured feedback process
- [ ] Product features don't help close deals
- [ ] Marketing struggles to position product

**If red flags, take action**:
- Set up regular syncs
- Create feedback process
- Improve enablement
- Better launch communication

**Remember**: Sales and Marketing are partners in bringing products to market. Invest in these relationships.
`;

export const workingWithSalesMarketingSection: ModuleSection = {
  id: 'working-with-sales-marketing',
  title: 'Working with Sales and Marketing',
  content,
};

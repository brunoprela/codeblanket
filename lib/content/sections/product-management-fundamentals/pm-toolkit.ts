/**
 * Section 5: The PM Toolkit
 * Product Management Fundamentals Module
 */

import { ModuleSection } from '../../../types';

const content = `# The PM Toolkit

## Introduction

Product managers use dozens of tools daily to document, plan, analyze, design, and communicate. Choosing the right tools and using them effectively dramatically impacts your productivity and team collaboration. This section covers the essential PM toolkit and how to select tools for your context.

## The PM Tool Stack by Category

### **Category 1: Documentation & Knowledge Management**

**Purpose**: Writing PRDs, strategy docs, meeting notes, wikis

#### **Top Tools**:

**1. Notion** (Most popular)
- **Use case**: Team wiki, PRDs, meeting notes, project tracking
- **Strengths**: Flexible, beautiful, great for internal docs, databases
- **Weaknesses**: Not great for real-time collaboration (vs. Google Docs)
- **Best for**: Startups and mid-size companies
- **Cost**: Free for small teams, $8-10/user/month

**Example usage**:
- Product wiki with all PRDs
- Meeting notes database (searchable)
- Team roadmap (database view)
- OKRs tracking

**2. Confluence** (Enterprise standard)
- **Use case**: Company wiki, documentation, knowledge base
- **Strengths**: Integrates with Jira, enterprise features, permissions
- **Weaknesses**: Clunky UI, slow, hard to navigate
- **Best for**: Large companies already using Atlassian
- **Cost**: $5-10/user/month

**3. Google Docs** (Collaboration)
- **Use case**: Collaborative writing, drafts, stakeholder review
- **Strengths**: Real-time collaboration, commenting, version history
- **Weaknesses**: Not structured (no wiki), gets messy
- **Best for**: Documents needing multiple stakeholders
- **Free**

**4. Coda** (Notion alternative)
- **Use case**: Docs + spreadsheets + databases
- **Strengths**: More powerful than Notion (formulas, automations)
- **Weaknesses**: Steeper learning curve
- **Best for**: Power users, data-heavy teams

**When to use which**:
- **Draft PRD with stakeholders**: Google Docs (real-time collaboration)
- **Final PRD storage**: Notion/Confluence (searchable wiki)
- **Quick notes**: Notion (fast, organized)
- **Strategic memos**: Google Docs → Notion (draft → archive)

---

### **Category 2: Roadmapping & Prioritization**

**Purpose**: Planning what to build and when

#### **Top Tools**:

**1. ProductBoard** (Best in class)
- **Use case**: Roadmapping, feedback management, prioritization
- **Strengths**: 
  - Aggregates feedback from support, sales, users
  - Connects features to user insights
  - Beautiful roadmap views (Now-Next-Later)
  - Integrates with Jira
- **Weaknesses**: Expensive ($20-60/user/month)
- **Best for**: Product-led companies with PM teams

**Features**:
- Customer feedback portal
- Feature prioritization (RICE scoring built-in)
- Roadmap views (timeline, Kanban, list)
- Integration with analytics, CRM, support

**2. Aha!** (Strategy-focused)
- **Use case**: Strategic roadmapping, goal alignment
- **Strengths**: Strong strategy tools, OKR tracking, presentations
- **Weaknesses**: Complex, expensive, can be overkill
- **Best for**: Enterprise, strategy-heavy teams
- **Cost**: $59+/user/month

**3. Jira Product Discovery** (formerly Jira Product Management)
- **Use case**: Roadmapping for teams already using Jira
- **Strengths**: Native Jira integration, affordable
- **Weaknesses**: Basic features, Atlassian UI
- **Best for**: Engineering-heavy teams using Jira
- **Cost**: $10/user/month

**4. Notion/Airtable** (DIY approach)
- **Use case**: Custom roadmap databases
- **Strengths**: Flexible, cheap, customizable
- **Weaknesses**: Manual setup, no specialized features
- **Best for**: Startups, small teams, tight budgets

**When to use which**:
- **< 5 PMs**: Notion/Airtable (flexible, cheap)
- **5-20 PMs**: ProductBoard (worth the investment)
- **Enterprise**: Aha! (if strategy-focused) or ProductBoard (if feedback-focused)

---

### **Category 3: Analytics & Data**

**Purpose**: Measuring product performance, user behavior

#### **Top Tools**:

**1. Amplitude** (Best for product analytics)
- **Use case**: User behavior analysis, funnels, retention
- **Strengths**:
  - Event-based tracking (flexible)
  - Cohort analysis, retention curves
  - User journey mapping
  - Great for SaaS/mobile products
- **Weaknesses**: Expensive at scale, complex setup
- **Best for**: Product-led growth companies
- **Cost**: Free up to 10M events/month, then $$$

**Key features**:
- Behavioral cohorts (segment users by actions)
- Funnel analysis (where users drop off)
- Retention analysis (cohort retention curves)
- Pathfinder (common user journeys)

**2. Mixpanel** (Amplitude alternative)
- **Use case**: Similar to Amplitude, simpler
- **Strengths**: Easier to learn, good UI, cheaper
- **Weaknesses**: Less powerful than Amplitude
- **Best for**: Startups, small teams
- **Cost**: Free up to 1K users, then pricing scales

**3. Google Analytics 4** (Web analytics)
- **Use case**: Website traffic, marketing attribution
- **Strengths**: Free, integrates with Google Ads, standard
- **Weaknesses**: Not designed for product analytics
- **Best for**: Marketing-focused, web traffic
- **Free**

**4. PostHog** (Open-source)
- **Use case**: Product analytics, feature flags, session recording
- **Strengths**: Open-source, self-hosted option, affordable
- **Weaknesses**: Less mature than Amplitude
- **Best for**: Engineering-heavy teams, privacy-conscious
- **Cost**: Free self-hosted, cloud pricing

**5. Heap** (Auto-capture)
- **Use case**: Retroactive analytics (no manual events)
- **Strengths**: Captures all events automatically
- **Weaknesses**: Expensive, data overload
- **Best for**: Teams that struggle with instrumentation

**SQL Tools** (Essential for PMs):

**1. Mode Analytics**
- **Use case**: SQL queries + visualizations
- **Strengths**: Great for sharing reports, notebooks
- **Cost**: Free tier available

**2. Metabase** (Open-source)
- **Use case**: Self-serve analytics, dashboards
- **Strengths**: Free, easy to use, visual query builder

**3. Looker** (Enterprise)
- **Use case**: BI platform, complex data modeling
- **Strengths**: Powerful, scales well
- **Weaknesses**: Expensive, complex
- **Best for**: Large companies

**When to use which**:
- **Early stage (<Product-Market Fit)**: Google Analytics + SQL
- **Growth stage**: Amplitude or Mixpanel + SQL
- **Enterprise**: Amplitude + Looker + SQL
- **Every PM**: Learn SQL (query data yourself)

---

### **Category 4: User Research**

**Purpose**: Understanding users through research

#### **Top Tools**:

**1. UserTesting** (Moderated & unmoderated)
- **Use case**: Video recordings of users testing product
- **Strengths**: Large panel, fast results, video recordings
- **Weaknesses**: Expensive ($100+/test)
- **Best for**: Teams with budget, frequent testing

**2. Maze** (Prototype testing)
- **Use case**: Testing Figma prototypes with users
- **Strengths**: Quick setup, quantitative + qualitative
- **Weaknesses**: Limited to prototype testing
- **Best for**: Testing designs before building

**3. Dovetail** (Research repository)
- **Use case**: Storing, tagging, analyzing research
- **Strengths**: Searchable insights, tagging, highlights
- **Weaknesses**: Only useful if you do lots of research
- **Best for**: Research teams, frequent user studies

**4. Lookback** (Live user testing)
- **Use case**: Remote user interviews with screen recording
- **Strengths**: Live + recorded, good for interviews
- **Best for**: Remote user research

**5. Typeform/SurveyMonkey** (Surveys)
- **Use case**: Quantitative surveys (NPS, feature requests)
- **Strengths**: Easy to build, good response rates
- **Best for**: Quantitative validation

**DIY approach** (Budget-friendly):
- **User interviews**: Zoom + Otter.ai transcription
- **Survey**: Google Forms (free)
- **Prototype testing**: Share Figma link, watch via Zoom
- **Repository**: Notion database with tags

---

### **Category 5: Design & Prototyping**

**Purpose**: Creating designs and prototypes

#### **Top Tools**:

**1. Figma** (Industry standard)
- **Use case**: UI design, prototyping, collaboration
- **Strengths**: 
  - Real-time collaboration
  - Prototyping with interactions
  - Developer handoff
  - Web-based (no installs)
- **Why it won**: Replaced Sketch by being collaborative
- **Best for**: Everyone
- **Cost**: Free for 3 projects, $12/editor/month

**PM usage**:
- View designs (commenting)
- Test prototypes with users
- Create lo-fi wireframes (yes, PMs can do this!)
- Share prototypes with stakeholders

**2. Framer** (Advanced prototyping)
- **Use case**: High-fidelity, interactive prototypes
- **Strengths**: Code-based, realistic animations
- **Weaknesses**: Steeper learning curve
- **Best for**: Testing complex interactions

**3. Whimsical** (Low-fidelity)
- **Use case**: Wireframes, flowcharts, mind maps
- **Strengths**: Fast, simple, collaborative
- **Best for**: Early ideation, user flows

**4. Miro/Mural** (Whiteboarding)
- **Use case**: Brainstorming, workshops, journey maps
- **Strengths**: Infinite canvas, real-time collab
- **Best for**: Remote workshops, planning sessions

---

### **Category 6: Project Management & Delivery**

**Purpose**: Tracking development progress

#### **Top Tools**:

**1. Jira** (Enterprise standard)
- **Use case**: Sprint planning, bug tracking, agile
- **Strengths**: Powerful, flexible, integrations
- **Weaknesses**: Complex, overwhelming UI
- **Best for**: Engineering teams >10 people

**2. Linear** (Modern Jira alternative)
- **Use case**: Issue tracking, sprints, product development
- **Strengths**: Beautiful UI, fast, keyboard shortcuts
- **Weaknesses**: Fewer features than Jira
- **Best for**: Startups, design-conscious teams

**3. Asana** (General project management)
- **Use case**: Cross-functional projects, marketing
- **Strengths**: Easy to use, good for non-engineers
- **Weaknesses**: Not built for software development
- **Best for**: Marketing, operations, cross-functional

**4. ClickUp** (All-in-one)
- **Use case**: Tries to be everything
- **Strengths**: Flexible, many features
- **Weaknesses**: Can be overwhelming
- **Best for**: Teams wanting one tool for everything

**PM's relationship with these tools**:
- **Don't**: Micromanage tickets in Jira
- **Do**: Understand sprint progress, unblock issues
- **Don't**: Update every ticket yourself
- **Do**: Keep high-level view, trust EM for execution

---

### **Category 7: Communication**

**Purpose**: Staying aligned with team and stakeholders

#### **Top Tools**:

**1. Slack** (Standard)
- **Use case**: Team chat, quick updates, async
- **Strengths**: Integrations, searchable, channels
- **Weaknesses**: Can be distracting, noisy
- **Best practices**:
  - Status updates in channels (not DMs)
  - Use threads (keep conversations organized)
  - Turn off notifications after work
  - Create #product-updates channel

**2. Loom** (Async video)
- **Use case**: Product demos, feedback, explanations
- **Strengths**: Faster than meetings, shows screen + face
- **Weaknesses**: Can be overused
- **Best for**: Remote teams, async updates

**PM usage**:
- Demo features before launch
- Provide context for PRDs
- Give design feedback
- Share user research findings

**3. Email** (External)
- **Use case**: Stakeholder updates, external comms
- **When to use**: External partners, executives, formal updates
- **When not to use**: Internal team communication (use Slack)

---

### **Category 8: Feature Management**

**Purpose**: Controlling feature releases

#### **Top Tools**:

**1. LaunchDarkly** (Industry leader)
- **Use case**: Feature flags, progressive rollouts, A/B tests
- **Strengths**: Robust, reliable, many integrations
- **Weaknesses**: Expensive ($8,500+/year)
- **Best for**: Companies with complex releases

**2. Split.io** (Alternative)
- **Use case**: Feature flags + experimentation
- **Strengths**: Good for A/B testing
- **Cost**: Similar to LaunchDarkly

**3. Optimizely** (Experimentation)
- **Use case**: A/B testing platform
- **Strengths**: Experimentation focus
- **Weaknesses**: Very expensive
- **Best for**: Large companies, growth teams

**4. DIY** (Build your own)
- Many companies build internal feature flags
- Use for: Small startups, simple needs

**Why feature flags matter for PMs**:
- Release features anytime (decouple deploy from release)
- Test with 10% of users
- Kill switch if things break
- A/B test easily

---

## Tool Selection Framework

### **Questions to Ask When Choosing Tools**

**1. What problem are we solving?**
- Be specific (not "we need a PM tool")
- Example: "PMs can't track customer feedback systematically"

**2. What's our team size and stage?**
- < 5 people: Free/cheap tools, flexible
- 5-50 people: Specialized tools worth investment
- 50+ people: Enterprise tools, integrations matter

**3. What's our budget?**
- Startup: $50-200/month total
- Growth: $1,000-5,000/month
- Enterprise: $10,000+/month

**4. What's our technical sophistication?**
- Non-technical team: Easy UI (Notion, Asana)
- Engineering-heavy: Power tools (Jira, SQL, self-hosted)

**5. What integrations matter?**
- Example: If using Jira, Confluence integrates well
- Check: Does it integrate with our existing stack?

**6. Can we migrate later if needed?**
- Look for: Data export, open APIs
- Avoid: Tools that lock you in

### **The Recommended Starter Stack**

**For a startup (5-20 people)**:

| Category | Tool | Cost/month | Why |
|----------|------|------------|-----|
| Docs | Notion | $80 | Flexible, beautiful |
| Roadmap | Notion/ProductBoard | $0-400 | Start free, upgrade later |
| Analytics | Mixpanel | $0-300 | Free tier is generous |
| SQL | Mode/Metabase | $0 | Free options |
| Research | Zoom + Notion | $30 | DIY approach |
| Design | Figma | $120 | Industry standard |
| Project | Linear | $100 | Modern, fast |
| Comm | Slack | $65 | Standard |
| **Total** | | **$395-995** | Affordable |

**For growth stage (50-200 people)**:

| Category | Tool | Cost/month | Why |
|----------|------|------------|-----|
| Docs | Notion/Confluence | $500 | Scale both work |
| Roadmap | ProductBoard | $1,200 | Worth it now |
| Analytics | Amplitude | $2,000+ | Power features |
| SQL | Mode | $0-500 | Team plans |
| Research | UserTesting | $1,000 | Regular testing |
| Design | Figma | $1,000 | More designers |
| Project | Jira | $500 | Scales well |
| Comm | Slack | $650 | Standard |
| Feature Flags | LaunchDarkly | $750 | Critical now |
| **Total** | | **~$8,000** | Investment pays off |

---

## Best Practices for Using PM Tools

### **Practice #1: Document in Public**

**Bad**: PRDs in private Google Docs only PM can see  
**Good**: PRDs in shared Notion wiki everyone can access

**Why**: Transparency builds trust, reduces "what's coming?" questions

### **Practice #2: Single Source of Truth**

**Bad**: Roadmap in 3 places (Notion, Google Sheets, Confluence)  
**Good**: One canonical roadmap, others link to it

**Why**: Reduces confusion, easier to maintain

### **Practice #3: Integrate Tools**

**Example integrations**:
- Slack + Jira (updates in Slack)
- Amplitude + Slack (metric alerts)
- ProductBoard + Jira (push features to dev)
- Figma + Notion (embed designs)

**Why**: Reduces context-switching

### **Practice #4: Don't Overinvest in Tools**

**Trap**: Spending weeks setting up perfect tool  
**Better**: Use simple tool well > complex tool poorly

**Example**: Notion database > Custom-built internal tool

### **Practice #5: Revisit Tool Choices Annually**

**Why**: 
- Tools improve fast
- Your needs change
- Better options emerge

**Process**: Annual tool audit
- What's working?
- What's frustrating?
- What's missing?
- Research alternatives

---

## Common Tool Mistakes

**Mistake #1: Too Many Tools**
- Problem: 15 different tools, none used well
- Fix: Consolidate to 5-7 core tools

**Mistake #2: Enterprise Tools Too Early**
- Problem: Buying Aha! for 3-person team
- Fix: Start simple, upgrade when needed

**Mistake #3: No Tool Training**
- Problem: Team doesn't know how to use tools
- Fix: Onboarding, documentation, office hours

**Mistake #4: Tools as Crutch**
- Problem: Thinking tools solve process problems
- Fix: Fix process first, then find tool

**Mistake #5: Not Leveraging Integrations**
- Problem: Manual copy-paste between tools
- Fix: Use Zapier, native integrations, APIs

---

## Key Takeaways

1. **Tools enable processes, don't create them**: Fix process first
2. **Start simple, upgrade when needed**: Don't overbuy early
3. **Integrate tools**: Reduce context-switching
4. **Learn your tools deeply**: 20% of features = 80% of value
5. **Documentation tools are critical**: Notion or Confluence
6. **Analytics are essential**: Every PM needs SQL + analytics tool
7. **Feature flags unlock flexibility**: Worth the investment
8. **Roadmap tools scale impact**: ProductBoard worth it at 5+ PMs
9. **Design tools for PMs**: Learn basic Figma
10. **Review tools annually**: Landscape changes fast

## Practical Exercise

**Audit your current tool stack**:

1. List all PM tools you use
2. For each, rate:
   - How often used (daily/weekly/monthly)?
   - Friction level (easy/medium/hard)?
   - Value delivered (high/medium/low)?
3. Identify:
   - Tools to eliminate (low usage, low value)
   - Tools to learn better (high value, high friction)
   - Gaps (what's missing?)
4. Create action plan:
   - Learn: [Tool] better by [specific action]
   - Add: [Tool] to solve [problem]
   - Remove: [Tool] (redundant with [other tool])

**Remember**: Great PMs master their tools, they don't let tools master them.
`;

export const pmToolkitSection: ModuleSection = {
  id: 'pm-toolkit',
  title: 'The PM Toolkit',
  content,
};

/**
 * Discussion questions for The PM Toolkit
 * Product Management Fundamentals Module
 */

export const pmToolkitQuiz = [
  {
    id: 1,
    question:
      "Your startup just raised Series A and your CEO wants to invest in 'proper PM tools.' You currently use Google Docs, Sheets, and Jira. The CEO is considering: ProductBoard ($20K/year), Amplitude ($30K/year), UserTesting ($15K/year), and LaunchDarkly ($10K/year) = $75K total. You have a team of 3 PMs and 12 engineers. Evaluate this tool investment: which tools would you recommend adopting, which would you delay, and what's your rationale? Create a phased adoption plan.",
    answer: `## Comprehensive Answer:

$75K in tools for 3 PMs is significant. Let's analyze each tool and create a phased approach.

### **Tool-by-Tool Analysis**

**1. ProductBoard ($20K/year)**
- **Value**: Roadmapping, feedback aggregation
- **Current state**: Using Google Sheets
- **Need**: Medium (nice-to-have, not critical yet)
- **Recommendation**: DELAY
- **Rationale**: 
  - At 3 PMs, Notion/Airtable roadmap works fine
  - ProductBoard shines at 5+ PMs
  - $20K = 1/3 of a PM's salary
  - Invest in people > tools at this stage

**2. Amplitude ($30K/year)**
- **Value**: Product analytics, user behavior
- **Current state**: Probably basic analytics
- **Need**: HIGH (data-driven decisions critical)
- **Recommendation**: ADOPT (but negotiate)
- **Rationale**:
  - Analytics = competitive advantage
  - Enables data-driven prioritization
  - 3 PMs × $30K = $10K/PM (worth it)
  - Amplitude free tier first, upgrade when needed

**3. UserTesting ($15K/year)**
- **Value**: User research platform
- **Current state**: DIY (Zoom calls)
- **Need**: Low-Medium (can DIY effectively)
- **Recommendation**: DELAY
- **Rationale**:
  - User research is critical, but tool isn't
  - Zoom + manual recruiting works at this scale
  - $15K = 150 user tests ($100 each) - probably overkill
  - Invest in recruiting budget instead

**4. LaunchDarkly ($10K/year)**
- **Value**: Feature flags, progressive rollouts
- **Current state**: Probably no feature flags
- **Need**: MEDIUM-HIGH (enables safer releases)
- **Recommendation**: ADOPT
- **Rationale**:
  - Feature flags unlock confidence
  - Progressive rollouts reduce risk
  - $10K cheapest of the four
  - High ROI for engineering efficiency

### **My Recommendation**

**Year 1 Budget: $15K (not $75K)**
- ✅ Amplitude: $0-5K (start free tier, upgrade if needed)
- ✅ LaunchDarkly: $10K
- ❌ ProductBoard: Wait until 5+ PMs
- ❌ UserTesting: DIY with Zoom

**Savings: $60K** → Hire another PM or invest in growth

### **Phased Adoption Plan**

**Phase 1 (Q1): Analytics Foundation**
- Month 1: Implement Amplitude free tier
- Month 2: Train PMs on analytics
- Month 3: Evaluate if need paid tier
- Cost: $0-5K

**Phase 2 (Q2): Feature Flags**
- Month 4: Implement LaunchDarkly
- Month 5: Train eng team on feature flags
- Month 6: All new features use flags
- Cost: $10K/year

**Phase 3 (Q3-Q4): Evaluate Additions**
- If team grows to 5+ PMs → Consider ProductBoard
- If doing 20+ user tests/month → Consider UserTesting
- Otherwise: Continue with current tools

**Total Year 1 Investment: $15K**

### **Alternative Budget-Conscious Stack**

If even $15K feels like too much:

**$0/year option**:
- Analytics: Mixpanel free tier (up to 1K users)
- Feature flags: Build internally or use open-source
- Roadmap: Notion ($200/year)
- Research: Zoom + Google Forms
- **Total: $200/year**

### **Key Takeaway**

**Tools don't make great PMs**. At 3 PMs, invest in:
1. **People** (hire 4th PM with savings)
2. **Processes** (define good practices)
3. **Essential tools** (analytics, feature flags)
4. **Delay** nice-to-have tools until scale demands them
`,
  },
  {
    id: 2,
    question:
      "Your PM team uses 15 different tools and the engineers complain about fragmentation: 'I have to check 5 places to understand what we're building.' How would you audit and consolidate your tool stack? Create a framework for deciding what to keep, what to eliminate, and what to integrate.",
    answer: `## Comprehensive Answer:

Tool sprawl is a common problem. Let's systematically consolidate.

### **Step 1: Audit Current Tools**

List all 15 tools with:
1. **Purpose**: What it does
2. **Users**: Who uses it
3. **Frequency**: Daily/weekly/monthly/rarely
4. **Value**: High/medium/low
5. **Overlap**: Does another tool do this?

**Example audit**:

| Tool | Purpose | Users | Frequency | Value | Overlap |
|------|---------|-------|-----------|-------|---------|
| Jira | Sprint tracking | Eng, PM | Daily | High | None |
| Confluence | Docs | All | Weekly | Medium | Notion |
| Notion | Docs, PRDs | PM, Design | Daily | High | Confluence |
| Google Docs | Drafts | All | Daily | High | None |
| Slack | Communication | All | Daily | High | None |
| ProductBoard | Roadmap | PM | Daily | Medium | Notion |
| Aha! | Strategy | PM | Weekly | Low | ProductBoard |
| Amplitude | Analytics | PM | Daily | High | None |
| Mixpanel | Analytics | Growth PM | Daily | Low | Amplitude |
| Figma | Design | Design, PM | Daily | High | None |
| InVision | Prototypes | Design | Rarely | Low | Figma |
| Asana | Projects | PM | Weekly | Low | Jira |
| Monday.com | Marketing | Marketing | Weekly | Low | Asana |
| Trello | Random | Various | Rarely | Low | Jira |
| Zoom | Meetings | All | Daily | High | None |

### **Step 2: Consolidation Framework**

**Question 1: Is this tool used daily by >50% of team?**
- YES → Keep
- NO → Consider eliminating

**Question 2: Does another tool do 80% of this?**
- YES → Eliminate, migrate to other tool
- NO → Evaluate value

**Question 3: Can we integrate it with core tools?**
- YES → Keep + integrate
- NO → Consider eliminating

**Question 4: Does it solve a unique, high-value problem?**
- YES → Keep
- NO → Eliminate

### **Step 3: Consolidation Plan**

**KEEP (8 tools)**:
1. Jira (sprint tracking - core)
2. Notion (docs, PRDs, roadmap - core)
3. Google Docs (collaborative drafts - core)
4. Slack (communication - core)
5. Amplitude (analytics - core)
6. Figma (design - core)
7. Zoom (meetings - essential)
8. LaunchDarkly (feature flags - add this)

**ELIMINATE (7 tools)**:
1. ~~Confluence~~ → Migrate to Notion
2. ~~Aha!~~ → Consolidate to ProductBoard or Notion
3. ~~ProductBoard~~ → Use Notion (3 PMs don't justify cost)
4. ~~Mixpanel~~ → Standardize on Amplitude
5. ~~InVision~~ → Figma handles prototypes
6. ~~Asana~~ → Use Jira for everything
7. ~~Monday.com~~ → Marketing uses Jira or Asana
8. ~~Trello~~ → Jira

**Result: 15 tools → 8 tools**

### **Step 4: Migration Plan**

**Month 1: Document current state**
- Which tools have data to migrate?
- Who owns each tool?
- What workflows depend on them?

**Month 2: Confluence → Notion**
- Export all docs
- Recreate in Notion
- Train team
- Sunset Confluence

**Month 3: Analytics consolidation**
- Standardize on Amplitude
- Migrate Mixpanel dashboards
- Cancel Mixpanel

**Month 4: Project management**
- Move Asana/Trello to Jira
- Train teams
- Cancel redundant tools

### **Step 5: Create Single Source of Truth**

**Problem**: Engineers check 5 places

**Solution**: "Product Hub" in Notion

\`\`\`
Product Hub (Notion page)
├── Current Sprint (embed Jira)
├── Roadmap (Notion database)
├── PRDs (Notion wiki)
├── Metrics Dashboard (embed Amplitude)
├── Design Files (link to Figma)
└── Meeting Notes (Notion)
\`\`\`

**One place** engineers go to understand what we're building.

### **Key Takeaway**

**Tool consolidation**:
- Reduce cognitive load
- Increase adoption
- Save money
- Build clear workflows

**Rule**: If >2 tools solve same problem, pick one.
`,
  },
  {
    id: 3,
    question:
      "You're moving from startup (where you used free/cheap tools) to a growth-stage company. Your new VP of Product wants to 'upgrade to enterprise tools.' Compare the trade-offs between: (1) Notion vs. Confluence, (2) Linear vs. Jira, (3) Mixpanel vs. Amplitude. For each, explain when you'd choose the startup tool vs. the enterprise tool. At what team size/stage does the switch make sense?",
    answer: `## Comprehensive Answer:

Let's analyze each tool pair with clear switching thresholds.

### **1. Notion vs. Confluence**

**Notion (Startup tool)**:
- Pros: Beautiful UI, flexible, fast, affordable ($8/user)
- Cons: Less enterprise features (permissions, audit logs)
- Best for: 5-100 people, design-conscious teams

**Confluence (Enterprise tool)**:
- Pros: Enterprise features, Jira integration, robust permissions
- Cons: Clunky UI, slow, expensive ($10-20/user)
- Best for: 100+ people, heavy Jira users

**Switching threshold**: 100-200 people
- Below 100: Notion wins (better UX, easier adoption)
- Above 200: Confluence (enterprise features matter)

**Decision factors**:
- Using Jira heavily? → Confluence integrates better
- Design-conscious culture? → Stay with Notion
- Need complex permissions? → Confluence
- Budget-conscious? → Notion

**My recommendation**: Stay with Notion as long as possible. Switch to Confluence only when Notion's limitations hurt (complex permissions, Jira integration critical).

---

### **2. Linear vs. Jira**

**Linear (Startup tool)**:
- Pros: Beautiful UI, fast, modern, keyboard shortcuts ($8/user)
- Cons: Fewer features, less customizable, fewer integrations
- Best for: 10-100 people, engineering-focused teams

**Jira (Enterprise tool)**:
- Pros: Powerful, highly customizable, many integrations
- Cons: Complex, overwhelming, slow, expensive ($10-15/user)
- Best for: 100+ people, complex workflows

**Switching threshold**: 100-150 people
- Below 100: Linear wins (UX matters more than features)
- Above 150: Jira (need customization, integrations)

**Decision factors**:
- Simple workflows? → Linear
- Complex workflows (multiple teams, approval processes)? → Jira
- Using Atlassian ecosystem? → Jira
- Design-conscious? → Linear

**My recommendation**: Use Linear as long as you can. Jira's power comes with complexity cost. Only switch when Linear's limitations hurt.

**Counterintuitive insight**: Some 500-person companies use Linear successfully. Tool choice is culture, not just size.

---

### **3. Mixpanel vs. Amplitude**

**Mixpanel (Startup tool)**:
- Pros: Easier to learn, simpler UI, cheaper ($25-100/month)
- Cons: Less powerful, fewer features
- Best for: <50 people, simpler analytics needs

**Amplitude (Enterprise tool)**:
- Pros: More powerful, behavioral cohorts, better retention analysis
- Cons: Complex, expensive ($2K-10K+/month)
- Best for: 20+ people, sophisticated analytics needs

**Switching threshold**: 20-50 people
- Below 20: Mixpanel sufficient
- Above 50: Amplitude worth it

**Decision factors**:
- Simple tracking (page views, clicks)? → Mixpanel
- Complex behavioral analysis (cohorts, retention curves, funnels)? → Amplitude
- Budget <$1K/month? → Mixpanel
- Data-driven culture? → Amplitude

**My recommendation**: Switch to Amplitude earlier than other tools (at 20-30 people). Analytics ROI is high - better analytics = better decisions = faster growth.

---

### **Comparison Table**

| Tool Pair | Switch at Size | Key Decision Factor | Exception |
|-----------|----------------|---------------------|-----------|
| Notion → Confluence | 100-200 people | Jira integration needs | Stay Notion if possible |
| Linear → Jira | 100-150 people | Workflow complexity | Some stay Linear at 500+ |
| Mixpanel → Amplitude | 20-50 people | Analytics sophistication | Switch early for growth teams |

---

### **General Principles**

**When to stick with startup tools**:
1. Team <100 people
2. Simple workflows
3. Design-conscious culture
4. Budget-conscious
5. Tools are working well

**When to upgrade to enterprise tools**:
1. Team >100-200 people
2. Complex workflows, permissions, integrations
3. Enterprise features required (SSO, audit logs, compliance)
4. Tool limitations causing pain
5. Budget allows

**Key insight**: Team size isn't the only factor. Culture, complexity, and budget matter more.

**Rule of thumb**:
- Startup tools work longer than you think
- Don't upgrade just because "we're bigger now"
- Upgrade when current tools limit you

### **What Your VP of Product Should Hear**

"Let's upgrade tools strategically, not uniformly. Here's my recommendation:

**Keep startup tools** (working well):
- Notion (unless we hit 150+ people)
- Linear (unless workflows become complex)

**Upgrade to enterprise** (ROI is clear):
- Amplitude (analytics ROI is high)
- LaunchDarkly (feature flags critical at scale)

**Evaluate later** (not urgent):
- Confluence (only if Notion truly limits us)
- Jira (only if Linear can't handle complexity)

**Principle**: Upgrade when limited, not when 'enterprise.'"
`,
  },
];

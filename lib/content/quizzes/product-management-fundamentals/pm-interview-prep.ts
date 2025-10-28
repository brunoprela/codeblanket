/**
 * Discussion questions for PM Interview Preparation
 * Product Management Fundamentals Module
 */

export const pmInterviewPrepQuiz = [
  {
    id: 1,
    question:
      "You're in a PM interview and asked: 'Design a fitness app for busy professionals.' Walk through your complete answer using the CIRCLES framework. Show exactly what you'd say, including: clarifying questions, user identification, needs, solutions, trade-offs, and metrics. Time yourself - aim for 15-20 minutes.",
    answer: `## Comprehensive Answer:

Here's my complete answer using CIRCLES framework:

---

**[Interviewer asks: "Design a fitness app for busy professionals"]**

---

## C - Comprehend (Clarify) - 2 minutes

**Me**: "Great question. Let me ask a few clarifying questions to make sure I understand the scope:

1. **Platform**: Are we designing a mobile app, web app, or both? I'll assume mobile for this answer.

2. **Target user**: When you say 'busy professionals,' are we talking about any specific age range or type of professional? I'll assume 25-45 year olds with desk jobs.

3. **Goal**: Is the primary goal user acquisition, engagement, or monetization? I'll optimize for engagement (daily active usage).

4. **Existing context**: Is this a new app from scratch, or improving an existing fitness app?

I'll assume we're designing a new app from scratch. Does that sound good?"

**[Interviewer nods]**

---

## I - Identify Customer - 2 minutes

**Me**: "Let me identify potential user segments within 'busy professionals':

**Segment 1: Time-constrained executives** (35-45 years old)
- Work 60+ hours/week
- Travel frequently
- High income, willing to pay
- Want results with minimal time investment

**Segment 2: Young professionals** (25-35 years old)
- Work 50-60 hours/week
- Gym membership but rarely go
- Social, motivated by friends
- Want affordable, flexible solution

**Segment 3: Parents with careers** (30-45 years old)
- Work 40-50 hours/week + kids
- Exercise at home (no gym time)
- Need flexibility (workout when kids sleep)
- Value efficiency

For this answer, I'll focus on **Segment 2: Young professionals (25-35)** because:
- Largest addressable market
- High willingness to adopt new apps
- Viral potential (social, friend effects)
- Gateway to lifetime fitness habits

Does that segmentation make sense?"

**[Interviewer agrees]**

---

## R - Report Customer Needs - 3 minutes

**Me**: "Let me identify the key needs for young, busy professionals:

**Pain Points** (from hypothetical user research):

1. **Time**: 'I don't have time to go to the gym' (60-90 min total: travel, change, workout, shower)

2. **Consistency**: 'I sign up for gym membership but only go 2-3 times/month' (waste of $50/month)

3. **Motivation**: 'I lose motivation working out alone' (no accountability)

4. **Confusion**: 'I don't know what exercises to do or if I'm doing them right'

5. **Flexibility**: 'My schedule changes daily, can't commit to class times'

6. **Progress tracking**: 'I don't know if I'm making progress or just wasting time'

**Jobs to Be Done**:
- Get fit without spending 90 min at gym
- Stay consistent (workout 4-5x/week)
- Feel motivated and accountable
- Know what to do (workout plan + form)
- Fit workouts into unpredictable schedule
- See tangible progress

Which of these should we prioritize?"

**[Let interviewer weigh in, or continue]**

"I think the top 3 needs are:
1. **Time** (main pain point)
2. **Consistency** (behavioral challenge)
3. **Motivation** (accountability gap)

I'll design a solution addressing these three. Sound good?"

**[Interviewer nods]**

---

## C - Cut Through (Prioritize) - 1 minute

**Me**: "Given we're solving for time, consistency, and motivation, here's what I'll prioritize:

**Must Have** (V1):
- Quick workouts (15-30 min, no gym needed)
- Structured plans (tell user exactly what to do)
- Social accountability (workout with friends)

**Nice to Have** (V2):
- Personalization (AI-customized plans)
- Video coaching (form correction)
- Wearable integration (Apple Watch)

**Out of Scope** (not solving):
- Nutrition tracking (keep it simple, focus on fitness)
- Live classes (too expensive, not scalable)
- Complex equipment (bodyweight focus)

Does this prioritization resonate?"

---

## L - List Solutions - 5 minutes

**Me**: "Let me brainstorm 3-4 solution approaches:

**Solution A: Quick Workout Library**
- 15-30 min bodyweight workouts
- Filter by: time, equipment, focus area (abs, cardio, strength)
- Follow-along videos
- **Pro**: Simple, addresses time constraint
- **Con**: Doesn't solve motivation/consistency

**Solution B: Social Workout Challenges**
- Create workout challenges with friends (e.g., '30 pushups daily for 30 days')
- Leaderboards, progress tracking
- Push notifications if friends complete workouts
- **Pro**: Solves motivation via social pressure
- **Con**: Doesn't provide workout structure

**Solution C: Smart Workout Plans + Accountability**
- AI-generated workout plans based on goals (lose weight, build muscle, general fitness)
- Social element: Invite friends to same plan, see their progress
- Quick workouts (15-30 min, no equipment)
- Daily reminders + streak tracking
- **Pro**: Solves all three needs (time, consistency, motivation)
- **Con**: More complex to build

**Solution D: Micro-Workouts Throughout Day**
- 5-minute workouts you can do at desk, at home, anywhere
- Push notifications: 'Time for a quick workout!'
- Gamified (earn points for each micro-workout)
- **Pro**: Ultra-convenient, fits any schedule
- **Con**: Might not feel like 'real' workout, less effective

**My recommendation: Solution C** (Smart Workout Plans + Accountability)

**Why**:
- Addresses all three top user needs
- Differentiated (not just workout library like YouTube)
- Built-in retention (social accountability + plans)
- Scalable (AI-generated plans)
- Sticky (friend network effects)

Does this sound like the right direction?"

**[Interviewer asks to elaborate]**

---

## E - Evaluate Trade-offs - 2 minutes

**Me**: "Let me discuss trade-offs of Solution C:

**Pros**:
1. **Holistic solution**: Addresses time, consistency, motivation
2. **Viral growth**: Social features drive friend invites
3. **Retention**: Plans + accountability keep users coming back
4. **Differentiation**: Not just another workout library
5. **Monetization potential**: Premium plans, personal coaching

**Cons**:
1. **Complexity to build**: AI plans, social features = 3-4 months build time
2. **Cold start problem**: Social features weak until you have friends using it
3. **Content creation**: Need library of exercises (100+ exercises)
4. **Personalization quality**: AI plans need to be good or users churn

**Mitigations**:
- **MVP approach**: Launch with pre-built plans (beginner, intermediate, advanced), add AI personalization later
- **Cold start**: Invite friends during onboarding, offer solo mode too
- **Content**: Partner with fitness influencers for exercises
- **Quality**: Beta test with 100 users, iterate based on feedback

**Alternative I considered**: Solution D (micro-workouts) is easier to build but less differentiated and may not feel substantial enough.

I still think Solution C is right bet for long-term success, despite complexity."

---

## S - Summarize - 2 minutes

**Me**: "Let me summarize my recommendation:

**Product**: Fitness app for busy professionals (25-35) called 'FitSquad'

**Key Features**:
1. **Smart Workout Plans**: AI-generated plans based on goals (15-30 min, bodyweight-focused)
2. **Social Accountability**: Invite friends to same plan, see their progress, compete on leaderboards
3. **Daily Reminders**: Push notifications + streak tracking ('Don't break your 7-day streak!')
4. **Quick & Flexible**: Workouts anywhere, anytime, no equipment

**Why this works**:
- Solves top 3 user needs: time (15-30 min workouts), consistency (social accountability + streaks), motivation (friends + gamification)
- Differentiated vs. competitors (YouTube, Peloton, ClassPass)
- Built-in growth (friend invites)
- Sticky (network effects, streaks)

**Success Metrics**:
- **North Star**: Daily Active Users (% who complete workout)
  - Target: 40% DAU (vs. 10% for typical fitness apps)
- **Supporting**:
  - Activation: % of users who complete first workout within 24 hours (target: 60%)
  - Retention: D7, D30 retention (target: 40%, 20%)
  - Social: % of users who invite friends (target: 30%)
  - Engagement: Workouts completed per user per week (target: 4)

**Next Steps**:
1. Build MVP: Pre-built plans + social features (3 months)
2. Beta test: 100 users, gather feedback
3. Launch: Influencer partnerships for awareness
4. Iterate: Add AI personalization based on usage data

That's my recommendation. What questions do you have?"

**[End of answer]**

---

## Time Check

**Total time**: ~15-17 minutes (within target range)

---

## What Made This Answer Strong

✅ **Asked clarifying questions** (didn't jump to solution)  
✅ **Identified clear user segment** (25-35 young professionals)  
✅ **Listed specific user needs** (time, consistency, motivation)  
✅ **Brainstormed multiple solutions** (4 options)  
✅ **Explained trade-offs** (pros/cons, mitigations)  
✅ **Defined success metrics** (DAU, retention, engagement)  
✅ **Structured with framework** (CIRCLES)  
✅ **Showed strategic thinking** (differentiation, growth, monetization)  
✅ **Was concise** (15-17 min, not 45 min)

---

## What Interviewer Is Evaluating

✅ User empathy (understood busy professional pain)  
✅ Structured thinking (used framework)  
✅ Creativity (4 different solution approaches)  
✅ Prioritization (picked Solution C with clear reasoning)  
✅ Trade-off navigation (discussed pros/cons)  
✅ Metrics fluency (defined DAU, retention, activation)  
✅ Communication (clear, concise, engaging)

This answer would likely score 8-9/10 in a real PM interview.
`,
  },
  {
    id: 2,
    question:
      "You're in a final round interview and asked a behavioral question: 'Tell me about your biggest product failure.' Walk through a complete STAR answer for a real or hypothetical failure. Then explain: What made your answer effective? What would make it weak?",
    answer: `## Comprehensive Answer:

Here's a complete STAR answer:

---

**[Interviewer asks: "Tell me about your biggest product failure"]**

---

## STAR Answer

### **S - Situation (Context)** - 30 seconds

**Me**: "At my previous company, we had a B2B SaaS product with 50K users. Our biggest pain point was onboarding—only 40% of new signups ever activated (completed core workflow). This was crushing our growth because we were spending $50K/month on ads but losing 60% of users immediately."

**[Sets context: company, product, problem, stakes]**

---

### **T - Task (What I Was Trying to Achieve)** - 20 seconds

**Me**: "As PM for growth, my goal was to increase activation rate from 40% to 60% within one quarter. If successful, this would mean 3,000 more activated users per month, translating to roughly $300K annual recurring revenue based on our conversion rates."

**[Specific, measurable goal + business impact]**

---

### **A - Action (What I Did - BE SPECIFIC)** - 2 minutes

**Me**: "Here's the approach I took:

**1. Competitor research** (Week 1):
- I analyzed 5 competitor onboarding flows
- Found that all of them had interactive, guided onboarding
- Concluded: We need guided onboarding

**2. Designed solution** (Week 2):
- Created a 5-step guided onboarding flow
- Step 1: Welcome + role selection
- Step 2: Connect integration
- Step 3: Import data
- Step 4: Create first project
- Step 5: Invite team members
- Made it mandatory (couldn't skip)

**3. Got buy-in** (Week 3):
- Pitched to leadership: 'This will increase activation 40% → 60%'
- Showed competitor examples as validation
- Got approval + 2 engineers allocated

**4. Built it** (Week 4-9):
- Designer created mockups
- Engineers built the flow
- I wrote all copy and tooltips
- We shipped it to 100% of users

**What I DIDN'T do** (and this was the mistake):
- ❌ Didn't do user testing before building
- ❌ Didn't prototype and validate with real users
- ❌ Assumed competitors knew best
- ❌ Didn't question whether guided = better for our users

I was so confident this would work that I skipped validation. Big mistake."

**[Specific actions + what I got wrong]**

---

### **R - Result (What Happened - BE HONEST)** - 1 minute

**Me**: "We launched after 6 weeks of development. Here's what happened:

**The Bad**:
- Activation rate stayed at 40%—**zero improvement**
- Worse: User complaints increased 3X
- Feedback: 'Forced onboarding is annoying,' 'Let me explore on my own,' 'I skipped most steps just to get in'
- Time wasted: 6 weeks of engineering, design, PM time

**Why it failed**:
When I finally did user testing post-launch, I learned:
- Our users are technical (developers, engineers)—they prefer exploring
- Guided onboarding felt 'hand-holdy' and insulting
- They just wanted to get in and try the product
- Competitors target less technical users (so guided works for them)

**The lesson**: Our users ≠ competitor's users. I made a bad assumption."

**[Honest about failure + specific numbers + root cause analysis]**

---

### **Learning & What I Did Next** - 1 minute

**Me**: "Here's what I learned and how I fixed it:

**Key Learnings**:
1. **Test before you build**: Always validate with 5-10 users before writing code
2. **Competitors aren't always right**: Their users might be different from yours
3. **Assumptions are dangerous**: 'Guided = better' was assumption, not data
4. **User testing is cheap**: $500 for user testing vs. $50K engineering time wasted

**What I did next** (redemption):

**Month 2-3: The Fix**:
- This time, I prototyped 3 different onboarding concepts in Figma
- Tested all 3 with 10 users (developers like ours)
- **Found**: Simpler approach worked better
  - Short, optional 2-minute product tour
  - 'Skip' prominently displayed
  - Contextual tooltips (not forced flow)
- Built this instead

**Result**:
- Activation rate improved from 40% → 55% (vs. target of 60%)
- User complaints down 2X
- Net Promoter Score improved from 35 to 42

**Not perfect, but much better than the forced onboarding disaster.**"

**[Shows learning + redemption + improved outcome]**

---

## Why This Answer is STRONG

### **What Made It Effective**

**✅ Specific and detailed**:
- Not vague ("I launched a feature that didn't work")
- Specific: activation rate 40%, 6 weeks, 5-step flow, user complaints 3X

**✅ Takes full ownership**:
- "I didn't do user testing"
- "I made a bad assumption"
- No blame-shifting to designer or engineers

**✅ Explains the learning**:
- "Test before you build"
- "Competitors aren't always right"
- Shows self-awareness

**✅ Shows redemption**:
- Didn't just fail and move on
- Fixed it: prototyped, tested, iterated
- Result: 40% → 55% activation

**✅ Quantifies impact**:
- 40% → 40% (no improvement)
- User complaints 3X
- $50K engineering time wasted
- Shows you think in numbers

**✅ Demonstrates growth mindset**:
- Learned from failure
- Changed process (now always prototype first)
- Applied learning to future work

---

## What Would Make This Answer WEAK

### **Bad Version (What NOT to Say)**

**Weak Answer**:
"I launched an onboarding feature that didn't work. Users didn't like it. We iterated and made it better. I learned to test with users."

**Why it's weak**:
❌ **Vague**: "feature that didn't work" (what feature? why didn't it work?)  
❌ **No specifics**: No numbers, timeline, or details  
❌ **No ownership**: "Users didn't like it" (passive voice, no accountability)  
❌ **No depth**: What did you actually learn? How?  
❌ **No redemption**: What happened next?

---

### **Red Flags to Avoid**

**❌ Blame-shifting**:
- "Engineering didn't build it right"
- "Designer's mockups were confusing"
- "Users didn't understand it"
→ **Shows**: Lack of accountability

**❌ Choosing trivial failure**:
- "I missed a deadline by one day"
- "A button was the wrong color"
→ **Shows**: Not self-aware or hiding real failures

**❌ Not actually a failure**:
- "I launched a feature that succeeded"
- "Everything worked out fine"
→ **Shows**: Not answering the question

**❌ No learning**:
- "It failed, we moved on"
- No reflection on what you'd do differently
→ **Shows**: Don't learn from mistakes

**❌ Too much detail**:
- 15-minute story about minor failure
- Lost in the weeds
→ **Shows**: Can't be concise

**❌ Being defensive**:
- "It wasn't really my fault because..."
- Making excuses
→ **Shows**: Poor ownership

---

## The Interview Meta-Game

### **What Interviewer Is Actually Evaluating**

**Through this one question, they're assessing**:

1. **Self-awareness**: Do you recognize your mistakes?
2. **Accountability**: Do you take ownership or blame others?
3. **Learning ability**: Do you learn from failures?
4. **Growth mindset**: Do you improve over time?
5. **Resilience**: How do you handle setbacks?
6. **Honesty**: Are you authentic about weaknesses?

**A good failure story shows**:
- You're self-aware (recognized the mistake)
- You take ownership (it was your call)
- You learn (changed behavior going forward)
- You persevere (fixed it, didn't give up)

**Counterintuitive**: A good failure story can be STRONGER than a success story.

**Why**: Everyone fails. How you handle failure differentiates great PMs from good PMs.

---

## How to Prepare Your Own Answer

**Step 1: Pick the right failure**:
- Not trivial, not catastrophic
- You had clear ownership
- You learned something valuable
- Bonus: You fixed it later

**Step 2: Write it out** (STAR format):
- S: Context (company, role, situation)
- T: Goal (what you were trying to achieve)
- A: Actions (specific, honest about mistakes)
- R: Result (what happened, learning, redemption)

**Step 3: Practice out loud**:
- Time yourself (3-4 min max)
- Record yourself
- Refine until smooth

**Step 4: Prepare 2-3 follow-up variants**:
- "Tell me about a time you failed" (general)
- "Tell me about a product that didn't succeed" (product-specific)
- "Tell me about a time you made a wrong decision" (decision-specific)

---

## Final Tip

**The best failure stories have this structure**:

**1. Set up** (situation + stakes): Make them care  
**2. The mistake** (what you did wrong): Be specific  
**3. The learning** (what you took away): Be insightful  
**4. The redemption** (how you applied it): Show growth

**Example**: "I didn't test before building → wasted 6 weeks → learned to always prototype first → next project I tested and improved activation 55%"

**That's a compelling narrative arc that shows growth.**

---

**This STAR answer would score 9/10 in a real interview.** It's honest, specific, shows learning, and demonstrates growth. That's what great PMs do—they fail, learn, and get better.
`,
  },
  {
    id: 3,
    question:
      "Create a comprehensive 'PM Interview Prep Checklist' someone could use in the 2 weeks before their interviews. Include: daily study plan, practice questions to do, mock interview schedule, company research tasks, STAR stories to prepare, and day-before/day-of tips. Make it actionable enough that they could follow it without any other resources.",
    answer: `## Comprehensive Answer:

# 2-Week PM Interview Prep Checklist

---

## Overview

**Timeline**: 14 days before interviews  
**Daily time commitment**: 2-3 hours  
**Goal**: Be confident and ready for any PM interview question

---

## Week 1: Foundations (Days 1-7)

### **Day 1: Assess & Plan (Sunday)**

**Morning (1 hour)**:
- [ ] List all companies you're interviewing with
- [ ] Note interview dates and formats (phone, video, onsite)
- [ ] Identify interview types to expect (design, strategy, behavioral, technical)
- [ ] Set up practice schedule (mock interviews with friends)

**Afternoon (2 hours)**:
- [ ] Research each company:
  - What does the product do?
  - Who are the users?
  - What's the business model?
  - Recent news (funding, launches, press)?
- [ ] Use the product (sign up, explore for 30 min each)
- [ ] Read 3-5 articles about each company

**Evening (1 hour)**:
- [ ] Create doc: "PM Interview Prep Notes"
- [ ] Section 1: Company research notes
- [ ] Section 2: STAR stories (template)
- [ ] Section 3: Practice questions

---

### **Day 2: Learn Frameworks (Monday)**

**Morning (1 hour)**:
- [ ] Read about CIRCLES framework (product design)
  - C - Comprehend
  - I - Identify customer
  - R - Report needs
  - C - Cut through prioritization
  - L - List solutions
  - E - Evaluate trade-offs
  - S - Summarize
- [ ] Watch 2-3 example product design interviews on YouTube

**Afternoon (1.5 hours)**:
- [ ] Practice 2 product design questions using CIRCLES:
  1. "Design a fitness app for elderly"
  2. "Improve Google Maps for commuters"
- [ ] Time yourself (15-20 min each)
- [ ] Write out full answers

**Evening (30 min)**:
- [ ] Review your answers
- [ ] Identify what felt awkward
- [ ] Note areas to improve

---

### **Day 3: Strategy Questions (Tuesday)**

**Morning (1 hour)**:
- [ ] Learn strategy framework:
  - Clarify goal
  - Understand current state
  - Identify growth levers (acquisition, activation, retention, monetization)
  - Brainstorm tactics
  - Prioritize
  - Success metrics
- [ ] Watch 2 strategy interview examples

**Afternoon (1.5 hours)**:
- [ ] Practice 2 strategy questions:
  1. "How would you grow Spotify?"
  2. "Should Uber enter food delivery?"
- [ ] Write full answers (15-20 min each)

**Evening (30 min)**:
- [ ] Compare your answers to sample answers online
- [ ] Identify gaps in your thinking

---

### **Day 4: Metrics & Analytical (Wednesday)**

**Morning (1 hour)**:
- [ ] Learn metrics framework:
  - North Star Metric
  - Supporting metrics
  - Counter metrics
  - How to measure
  - Targets
- [ ] Study common metrics: DAU, WAU, MAU, retention, churn, LTV, CAC

**Afternoon (1.5 hours)**:
- [ ] Practice 2 metrics questions:
  1. "How would you measure success for Instagram Reels?"
  2. "DAU dropped 10% this week. Investigate."
- [ ] Write full answers

**Evening (30 min)**:
- [ ] Review SQL basics (if relevant for your interviews)
- [ ] Practice 3-5 simple SQL queries on Mode Analytics

---

### **Day 5: Behavioral Stories (Thursday)**

**Morning (1 hour)**:
- [ ] Learn STAR framework:
  - S - Situation
  - T - Task
  - A - Action
  - R - Result
- [ ] Understand what makes a good story

**Afternoon (2 hours)**:
- [ ] Write 8 STAR stories covering:
  1. **Success**: Launched impactful product
  2. **Failure**: Product/decision that didn't work
  3. **Conflict**: Disagreement with eng/design/stakeholder
  4. **Leadership**: Influenced without authority
  5. **Data-driven**: Used analytics to make decision
  6. **User research**: Discovered key insight
  7. **Technical**: Worked with engineering on complex problem
  8. **Prioritization**: Made tough trade-off

**Evening (30 min)**:
- [ ] Read your stories out loud
- [ ] Time each (should be 2-4 min)
- [ ] Refine to be more concise

---

### **Day 6: Technical Questions (Friday)**

**Morning (1 hour)**:
- [ ] Review technical concepts:
  - APIs (what they are, REST vs GraphQL)
  - Databases (SQL vs NoSQL)
  - System architecture (client-server, microservices)
  - Scale concepts (caching, load balancing, CDN)

**Afternoon (1.5 hours)**:
- [ ] Practice 2 technical questions:
  1. "How does Google Search work?"
  2. "How would you design the backend for Uber?"
- [ ] Draw diagrams (boxes and arrows)
- [ ] Explain out loud

**Evening (30 min)**:
- [ ] Research common technical questions for PM interviews
- [ ] Identify 5 you might be asked
- [ ] Prepare high-level answers

---

### **Day 7: Mock Interview #1 (Saturday)**

**Morning (2 hours)**:
- [ ] Schedule mock interview with friend/colleague
- [ ] Do full 45-60 min mock:
  - 1 product design question (20 min)
  - 1 strategy question (15 min)
  - 2 behavioral questions (10 min each)

**Afternoon (1 hour)**:
- [ ] Get feedback from interviewer
- [ ] Write down:
  - What went well?
  - What was awkward?
  - What to improve?
- [ ] Identify top 3 gaps

**Evening (Optional rest)**:
- Take evening off or review notes

---

## Week 2: Practice & Polish (Days 8-14)

### **Day 8: Intensive Practice Day (Sunday)**

**Morning (1.5 hours)**:
- [ ] Practice 3 product design questions:
  1. "Design a product for college students to find roommates"
  2. "Improve Airbnb for hosts"
  3. "Design a budgeting app for Gen Z"
- [ ] Time yourself (15-20 min each)
- [ ] Focus on weak areas from Day 7 mock

**Afternoon (1.5 hours)**:
- [ ] Practice 3 strategy questions:
  1. "How would you monetize TikTok?"
  2. "Should Netflix enter gaming?"
  3. "How would you grow LinkedIn?"
- [ ] Time yourself (15-20 min each)

**Evening (30 min)**:
- [ ] Review all practice answers
- [ ] Note patterns in your thinking
- [ ] Identify improvement areas

---

### **Day 9: Company Deep Dive (Monday)**

**Morning (1 hour)**:
- [ ] Deep research on Company #1:
  - Read latest earnings call / blog posts
  - Understand product strategy
  - Identify 3 product opportunities
  - Prepare 3 smart questions to ask interviewer

**Afternoon (1 hour)**:
- [ ] Deep research on Company #2
- [ ] Same process as above

**Evening (1 hour)**:
- [ ] Deep research on Company #3 (if applicable)
- [ ] Prepare company-specific STAR stories

---

### **Day 10: Polish STAR Stories (Tuesday)**

**Morning (1 hour)**:
- [ ] Re-read your 8 STAR stories
- [ ] Make them more concise (2-3 min each)
- [ ] Add specific numbers/metrics where possible

**Afternoon (1.5 hours)**:
- [ ] Practice telling each story out loud
- [ ] Record yourself (video or audio)
- [ ] Watch/listen back
- [ ] Note filler words (um, like, uh)

**Evening (30 min)**:
- [ ] Prepare answers to common questions:
  - "Why PM?"
  - "Why this company?"
  - "Why are you leaving current role?"
  - "Where do you see yourself in 5 years?"
  - "What's your biggest strength/weakness?"

---

### **Day 11: Mock Interview #2 (Wednesday)**

**Morning (2 hours)**:
- [ ] Do second mock interview
- [ ] Different questions than Day 7
- [ ] Ask for harder questions this time

**Afternoon (1 hour)**:
- [ ] Get feedback
- [ ] Compare to Day 7 feedback
- [ ] Measure improvement

**Evening (30 min)**:
- [ ] Practice 5 more product design questions (quick 10-min versions)
- [ ] Focus on thinking out loud

---

### **Day 12: Final Practice (Thursday)**

**Morning (1 hour)**:
- [ ] Practice 10 quick questions (5 min each):
  - 4 product design
  - 3 strategy
  - 3 metrics

**Afternoon (1 hour)**:
- [ ] Review all frameworks one more time:
  - CIRCLES (product design)
  - Growth levers (strategy)
  - Metrics framework
  - STAR (behavioral)

**Evening (1 hour)**:
- [ ] Prepare questions to ask interviewers (3-5 per company):
  - About product strategy
  - About team dynamics
  - About growth opportunities
  - About challenges

---

### **Day 13: Light Review & Rest (Friday)**

**Morning (30 min)**:
- [ ] Quick review of company research notes
- [ ] Skim through STAR stories

**Afternoon (30 min)**:
- [ ] Review frameworks (don't practice, just review)
- [ ] Watch 1-2 PM interview videos for inspiration

**Evening**:
- [ ] REST! No studying.
- [ ] Early bedtime

---

### **Day 14: Interview Day (Saturday/Sunday)**

**See "Day-Of Checklist" below**

---

## STAR Stories Template

**For each story, write**:

\`\`\`
Story #1: [Title - e.g., "Launched feature that grew users 20%"]

S (Situation - 30 sec):
- Company, role, context
- What was the problem/situation?
- Why did it matter?

T (Task - 20 sec):
- What were you trying to achieve?
- What was the goal/metric?

A (Action - 2 min):
- What specifically did you do? (Be detailed)
- What was your process?
- Who did you work with?
- What challenges did you face?

R (Result - 1 min):
- What happened? (Numbers!)
- What was the impact?
- What did you learn?
- (If failure) How did you fix it?

[Total time: 3-4 minutes]
\`\`\`

---

## Day-Before Checklist

**Evening before interview**:
- [ ] Confirm interview time and format (phone, video, in-person)
- [ ] Test video/audio setup (if virtual)
- [ ] Prepare outfit (if in-person or video)
- [ ] Print copy of resume
- [ ] Prepare notebook + pen
- [ ] Review company notes (15 min)
- [ ] Skim STAR stories (don't memorize, just refresh)
- [ ] Set 2 alarms for morning
- [ ] Early bedtime (8 hours sleep)

---

## Day-Of Checklist

**Morning**:
- [ ] Eat good breakfast (not too heavy)
- [ ] Drink coffee/tea if that's your routine (don't try something new)
- [ ] Arrive 10 min early (or log in 5 min early if virtual)
- [ ] Bring water bottle

**10 min before interview**:
- [ ] Review company name + interviewer name
- [ ] Quick glance at company notes
- [ ] Deep breaths
- [ ] Smile (even on video—it changes your voice)

**During interview**:
- [ ] Listen carefully to questions
- [ ] Ask clarifying questions
- [ ] Think out loud (show your process)
- [ ] Use frameworks (CIRCLES, STAR)
- [ ] Watch time (don't go over)
- [ ] Take notes on their questions
- [ ] Ask your prepared questions at end
- [ ] Be enthusiastic (energy matters!)

**After interview**:
- [ ] Send thank you email within 24 hours
- [ ] Personalize based on conversation
- [ ] Reiterate interest in role
- [ ] Reference specific discussion point

---

## Materials Needed

**Books** (optional but helpful):
- "Cracking the PM Interview" by Gayle McDowell

**Websites**:
- exponent.com (PM interview practice)
- YouTube: Search "PM interview examples"
- Mode Analytics (SQL practice)

**Tools**:
- Timer (for practice questions)
- Recording device (phone, laptop)
- Notebook for notes

---

## Success Metrics

**After 2 weeks, you should**:
- ✅ Have answered 30+ practice questions
- ✅ Have 8 polished STAR stories
- ✅ Be able to use CIRCLES framework smoothly
- ✅ Done 2-3 mock interviews
- ✅ Know each company's product deeply
- ✅ Feel confident, not panicked

---

## Final Tips

**The Night Before**:
- DON'T cram (you know enough)
- DO get 8 hours sleep
- DON'T drink alcohol
- DO relax and trust your prep

**During Interview**:
- Think out loud (let them see your process)
- It's okay to pause and think (5-10 seconds of silence is fine)
- If you don't know, say "I don't know, but here's how I'd approach it"
- Be enthusiastic (passion matters as much as answers)
- Remember: they're rooting for you to succeed

**After Interview**:
- Don't obsess over "mistakes"
- Send thank you email
- Move on to next interview or next company

---

**Good luck! You've got this. 🚀**
`,
  },
];

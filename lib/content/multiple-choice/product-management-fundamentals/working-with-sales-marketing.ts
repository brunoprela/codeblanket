/**
 * Multiple choice questions for Working with Sales and Marketing
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const workingWithSalesMarketingMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'When sales brings 15 customer feature requests, how should a PM distinguish between "patterns" worth building vs. "one-off requests" to defer?',
      options: [
        'Build everything sales requests to maintain good relationships',
        'Look for patterns: 5+ customers asking, deal-breaker status, competitive gaps, strategic fit',
        'Only build requests from the largest customers',
        'Defer all sales requests until next year',
      ],
      correctAnswer: 1,
      explanation:
        'PMs should evaluate requests systematically using criteria: (1) Pattern strength (5+ customers asking = pattern, 1-2 customers = one-off), (2) Deal-breaker status (truly required vs. nice-to-have), (3) Competitive gap (do competitors have it?), (4) Strategic fit (aligns with product direction?), and (5) ROI (revenue impact / engineering effort). For example, Salesforce integration requested by 8 customers with $750K pipeline is a clear pattern; custom PDF export requested by 1 customer is a one-off. PMs say NO to one-offs strategically while explaining trade-offs.',
    },
    {
      id: 'mc2',
      question:
        'How much advance notice should PM give Marketing for a major (Tier 1) product launch?',
      options: [
        '1 week (campaigns are fast to create)',
        '2 weeks (minimal planning needed)',
        '4-6 weeks (standard for major launches)',
        '6 months (for maximum preparation)',
      ],
      correctAnswer: 2,
      explanation:
        'Marketing needs 4-6 weeks advance notice for major (Tier 1) launches to plan campaigns, create assets, write content, and coordinate across channels. The GTM Playbook specifies: Week 6-8 before launch, PM confirms timeline with Marketing; Weeks 4-6, Marketing creates materials (blog posts, emails, videos); Week 2, Go/No-Go meeting confirms readiness. Less than 4 weeks leads to rushed campaigns or Marketing scrambling last-minute. The content explicitly states: "Minimum notice: 4-6 weeks before launch for major features." This prevents crises where Marketing launches campaigns before products are ready.',
    },
    {
      id: 'mc3',
      question:
        'What is the purpose of a "Go/No-Go meeting" in the launch process?',
      options: [
        'To decide if the feature is worth building at all',
        'To formally assess launch readiness 2 weeks before launch and decide to proceed or delay',
        'To get executive approval for marketing budget',
        'To announce the launch to the entire company',
      ],
      correctAnswer: 1,
      explanation:
        'The Go/No-Go meeting happens 2 weeks before planned launch with attendees (PM, Engineering, Marketing, Sales) to formally assess readiness. Checklist includes: Is feature built and tested? Is marketing campaign ready? Is sales enablement complete? Is support team trained? Decision: Go (proceed with launch) or No-Go (delay launch). This meeting forces honest assessment and provides Marketing enough time (2 weeks) to adjust if launch delays. It prevents scenarios where "Marketing launches campaign but engineering isn\'t ready." The content recommends this as a critical checkpoint in the GTM process.',
    },
    {
      id: 'mc4',
      question:
        'When Marketing has already spent $50K on a campaign launching in 3 days but Engineering says the feature needs 6 more weeks, what is the recommended PM approach?',
      options: [
        'Force Engineering to rush and ship in 3 days',
        'Cancel the campaign and waste the $50K investment',
        'Launch a beta/waitlist version to preserve marketing investment while giving Engineering time',
        'Blame Marketing for not checking timeline',
      ],
      correctAnswer: 2,
      explanation:
        'The content recommends a beta/waitlist approach that balances competing needs: (1) Marketing campaign launches as planned (preserving $50K investment), (2) Messaging changes to "Join Beta Waitlist" (not "Available Now"), (3) Engineering ships MVP to beta users in 2 weeks (shows progress), (4) Full rollout in 6 weeks (Engineering has time for quality). This solution prevents overpromising to customers, gives Engineering time for quality, and doesn\'t waste marketing investment. PM then fixes the root cause by implementing a launch process with weekly syncs, Go/No-Go meetings, and minimum 6-week notice rule to prevent future crises.',
    },
    {
      id: 'mc5',
      question:
        'According to the Launch Tiering Framework, which features deserve "Tier 1: Flagship Launch" treatment with high marketing investment ($20K-50K+)?',
      options: [
        'Every new feature to maximize awareness',
        'Major new products with >$500K revenue impact, strategic importance, and competitive advantage',
        'Only features requested by enterprise customers',
        'Features that are easiest to market',
      ],
      correctAnswer: 1,
      explanation:
        'Tier 1 Flagship Launches (2-3 per year) are reserved for features that meet multiple criteria: (1) Major new product or feature, (2) Large revenue impact (>$500K), (3) Significant competitive advantage, (4) Targets new market segment, (5) Strategic priority for company. These get full marketing treatment: PR campaigns, paid advertising, content marketing, launch events, 8-12 weeks planning. Tier 2 (Standard Launch) is for important features ($100K-500K impact), and Tier 3 (Quiet Release) for minor improvements. The framework prevents "marketing everything equally" which dilutes impact. Not all launches are equal - tier appropriately.',
    },
  ];

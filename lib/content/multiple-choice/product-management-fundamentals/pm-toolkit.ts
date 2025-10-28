/**
 * Multiple choice questions for The PM Toolkit
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const pmToolkitMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'According to the content, what is the recommended analytics tool for a startup with fewer than 20 people?',
    options: [
      'Amplitude (enterprise-grade)',
      'Google Analytics 4 + SQL',
      'Heap (auto-capture)',
      'Looker (BI platform)',
    ],
    correctAnswer: 1,
    explanation:
      "For early-stage startups (<20 people, pre-PMF), Google Analytics + SQL is recommended because: (1) GA4 is free, (2) SQL allows custom analysis, (3) Simple needs don't justify Amplitude's cost ($2K+/month), and (4) Focus should be on finding PMF, not sophisticated analytics. Once the team grows to 20-50 people and needs behavioral cohorts, funnel analysis, and retention curves, then upgrade to Amplitude or Mixpanel. The rule is: start simple, upgrade when current tools limit you.",
  },
  {
    id: 'mc2',
    question:
      'What is the primary benefit of feature flag tools like LaunchDarkly for PMs?',
    options: [
      'They make code run faster',
      'They allow deploying code anytime and controlling feature releases independently',
      'They automatically test features',
      'They replace the need for QA testing',
    ],
    correctAnswer: 1,
    explanation:
      'Feature flags decouple deployment from release. This means: (1) Deploy code anytime without exposing features, (2) Release features to 10% of users first (progressive rollout), (3) Kill switch if problems arise, (4) A/B test easily by showing different versions to different users. This dramatically reduces release risk and enables safer, faster iteration. Cost is $10K/year but ROI is high for engineering efficiency and reduced risk. Feature flags are one of the most valuable PM tools.',
  },
  {
    id: 'mc3',
    question:
      'According to the starter stack recommendation, what is the total monthly tool cost for a 5-20 person startup?',
    options: [
      '$100-200/month',
      '$395-995/month',
      '$2,000-3,000/month',
      '$5,000+/month',
    ],
    correctAnswer: 1,
    explanation:
      'The recommended starter stack costs $395-995/month including: Notion ($80), Mixpanel ($0-300), Mode/Metabase ($0), Zoom + Notion for research ($30), Figma ($120), Linear ($100), and Slack ($65). This provides essential PM tools while staying affordable. The key insight is that startups can be highly effective with sub-$1K/month tool budget. Expensive enterprise tools (ProductBoard $1,200/month, Amplitude $2K+/month) should wait until 50+ people or specific ROI justifies the cost.',
  },
  {
    id: 'mc4',
    question:
      'Why does the content recommend Notion over Confluence for teams under 100 people?',
    options: [
      'Notion has better enterprise features',
      'Notion has better Jira integration',
      'Notion has better UI, is faster, and is more affordable',
      'Confluence is discontinued',
    ],
    correctAnswer: 2,
    explanation:
      "Notion is recommended for teams <100 people because: (1) Beautiful, intuitive UI (higher adoption), (2) Fast and responsive (Confluence is notoriously slow), (3) More affordable ($8/user vs. $10-20/user), (4) Flexible (docs, databases, wikis in one tool), and (5) Better for modern, design-conscious teams. Confluence's advantages (enterprise permissions, Jira integration, audit logs) only matter at 100-200+ people. The principle is: better UX beats more features for smaller teams.",
  },
  {
    id: 'mc5',
    question:
      'What is the "single source of truth" principle in tool management?',
    options: [
      'Using only one tool for everything',
      'Having one canonical location for each type of information, with others linking to it',
      'Only the PM should know where information is',
      'All tools should sync automatically',
    ],
    correctAnswer: 1,
    explanation:
      'Single source of truth means having ONE canonical location for each type of information (e.g., roadmap lives in ProductBoard, PRDs in Notion), with other locations linking back to it. This prevents: (1) Conflicting information in multiple places, (2) Confusion about "which is the right version?", and (3) Maintenance burden of updating multiple locations. Example: Roadmap in Notion is source of truth; Slack links to it rather than copying it. This principle dramatically reduces miscommunication and makes information easier to find.',
  },
];

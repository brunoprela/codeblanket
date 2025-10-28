/**
 * Multiple choice questions for Writing for Product Managers
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const writingForPMsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'According to "Bottom Line First" principle, how should a PM structure important communications?',
    options: [
      'Start with background context, then analysis, then conclusion',
      'Lead with the key point/recommendation, then provide supporting details',
      'Save the recommendation for the end to build suspense',
      'Start with a question to engage the reader',
    ],
    correctAnswer: 1,
    explanation:
      'PMs should lead with the bottom line (conclusion, recommendation, key point) first, then provide supporting details. Example: "Recommendation: Build Salesforce integration in Q1. Why: Unblocks $750K pipeline. Details: [below]" vs. "We analyzed user feedback... after consideration... we recommend..." Busy executives read the first paragraph - make it count. This "newspaper article" structure respects readers\' time and ensures the key message lands even if they don\'t read the entire document. The content explicitly states: "Don\'t bury the lead. Lead with the conclusion."',
  },
  {
    id: 'mc2',
    question:
      'What is the key difference between how a PM should write the SAME feature announcement for engineers vs. executives?',
    options: [
      'Engineers and executives should receive identical communications',
      'Engineers need technical details and "why"; executives need business impact and bottom line',
      'Engineers only need "what"; executives only need "when"',
      'Engineers need longer documents; executives need shorter ones',
    ],
    correctAnswer: 1,
    explanation:
      'Different audiences need different framing. For engineers: technical details, "why" and "what" most important, specs detailed but not prescriptive, edge cases. For executives: bottom line upfront, business impact (revenue, retention, cost), strategic rationale, clear ask. Example: To engineers: "Building real-time Salesforce sync. Requirements: bidirectional, <5 sec latency, handle 10K objects/day." To executives: "Salesforce integration ships Q1, unlocking $750K pipeline. Investment: 3 engineering weeks. ROI: $250K per week." Same feature, different framing based on audience needs.',
  },
  {
    id: 'mc3',
    question:
      'According to the editing checklist, what should PM do after writing a first draft?',
    options: [
      'Send immediately - done is better than perfect',
      'Cut 30% of words, read aloud, check structure, get feedback',
      'Add more details to be thorough',
      'Rewrite from scratch to improve quality',
    ],
    correctAnswer: 1,
    explanation:
      'The editing process: (1) Write first draft (don\'t edit while writing), (2) Walk away (take a break), (3) Read aloud (catches awkward phrasing), (4) Cut 30% (remove unnecessary words), (5) Check structure (is it scannable?), (6) Get feedback (colleague review), (7) Ship it. The "cut 30%" rule forces conciseness. Example revision shown in content: 120 words → 30 words (75% shorter, equally clear). The content emphasizes: "Shorter is better. Respect readers\' time." Editing transforms first drafts into clear, concise communication.',
  },
  {
    id: 'mc4',
    question:
      'What is the problem with writing "Many users want this feature soon" in a PRD?',
    options: [
      "It's grammatically incorrect",
      "It's too concise and needs more words",
      'It\'s vague - should be specific like "50% of users (5,000 people) requested this. Shipping Q1"',
      "It's too formal and should be more casual",
    ],
    correctAnswer: 2,
    explanation:
      'Vague language like "many" and "soon" is a common writing mistake. Replace with specifics: "Many" → "50% of users (5,000 people)", "Soon" → "Q1" or "March 15". The content emphasizes "Show, Don\'t Tell" principle: use examples, data, and specifics. Weak: "Users are frustrated" (abstract). Strong: "Users abandon checkout 40% of the time. Quote: \'I couldn\'t figure out how to apply discount code.\'" Specificity enables better decision-making. Stakeholders can\'t plan around "many" or "soon" but can plan around "5,000 users" and "March 15."',
  },
  {
    id: 'mc5',
    question: 'What is the purpose of the "Out of Scope" section in a PRD?',
    options: [
      'To list features that are impossible to build',
      "To explicitly state what we're NOT building to prevent scope creep",
      'To criticize bad ideas from stakeholders',
      'To keep the PRD longer and more detailed',
    ],
    correctAnswer: 1,
    explanation:
      'The "Out of Scope" section explicitly lists what you\'re NOT building in this version to prevent scope creep and set clear boundaries. Example from content: "Out of Scope for V1: ❌ Video tutorials (defer to V2), ❌ Interactive tours (defer to V2), ❌ Mobile app onboarding (V1 = web only)." This prevents stakeholders from assuming everything is included and helps teams stay focused. The PRD template includes: Must Have, Should Have, Nice to Have, AND Out of Scope. Explicitly stating what\'s excluded is as important as stating what\'s included for managing expectations and maintaining focus.',
  },
];

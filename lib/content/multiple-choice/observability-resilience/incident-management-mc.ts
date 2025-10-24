/**
 * Multiple choice questions for Incident Management section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const incidentManagementMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is MTTR in incident management?',
    options: [
      'Mean Time To Respond',
      'Mean Time To Recovery (time from detection to resolution)',
      'Mean Time To Report',
      'Maximum Time To Restart',
    ],
    correctAnswer: 1,
    explanation:
      "MTTR (Mean Time To Recovery) is the average time from when an incident is detected to when it's resolved. Example: Alert fires at 10:00 → Issue resolved at 10:30 → MTTR = 30 minutes. Target: < 30 minutes for SEV-1 incidents. Other related metrics: MTTD (Mean Time To Detect) = time from incident start to detection, MTTA (Mean Time To Acknowledge) = time from alert to acknowledgment. Reducing MTTR is key to minimizing user impact.",
  },
  {
    id: 'mc2',
    question: 'What does "blameless post-mortem" mean?',
    options: [
      'Not writing a post-mortem at all',
      'Focusing on systems/processes that failed, not individuals, to enable learning',
      'Blaming everyone equally',
      'Only for minor incidents',
    ],
    correctAnswer: 1,
    explanation:
      'Blameless post-mortem focuses on systems and processes that failed, not individuals. Why: If you blame people, they hide mistakes → No learning → Same incidents recur. Instead: "Deployment lacked sufficient testing" (system issue) not "Bob deployed bad code" (blame). Action items fix systems: Add tests, improve monitoring, update runbooks. Creates psychological safety where people share mistakes openly, enabling organizational learning and preventing recurrence. Most effective post-mortems are blameless.',
  },
  {
    id: 'mc3',
    question: 'Who is the Incident Commander, and what is their primary role?',
    options: [
      'The person who fixes the incident',
      'The coordinator who makes decisions, delegates tasks, and communicates with stakeholders',
      'The most senior engineer available',
      'The person who caused the incident',
    ],
    correctAnswer: 1,
    explanation:
      "Incident Commander (IC) coordinates the response: Makes decisions (rollback vs hotfix), delegates to engineers (doesn't fix the issue themselves), communicates with stakeholders, declares incident resolved. IC focuses on coordination, not coding. Skills needed: Calm under pressure, clear communication, decisiveness. IC is NOT necessarily the most senior engineer - often a dedicated role trained in incident management. Engineers focus on fixing, IC focuses on coordinating.",
  },
  {
    id: 'mc4',
    question: 'How soon after an incident should a post-mortem be completed?',
    options: [
      'Immediately after resolution',
      'Within 48 hours while details are fresh',
      'Within 3 months',
      'Post-mortems are optional',
    ],
    correctAnswer: 1,
    explanation:
      "Post-mortems should be completed within 48 hours while details are fresh and team motivation is high. Delay leads to: Forgotten details, lower quality analysis, action items never completed, team moves on without learning. Process: Write document within 48 hours, hold post-mortem meeting within 1 week, assign action items with deadlines, track to completion. Critical: 80%+ action items should be completed. If action items don't get done, post-mortems lose value.",
  },
  {
    id: 'mc5',
    question:
      'What severity level (SEV) should be assigned to a complete service outage affecting all users?',
    options: [
      'SEV-4 (Low)',
      'SEV-3 (Medium)',
      'SEV-2 (High)',
      'SEV-1 (Critical)',
    ],
    correctAnswer: 3,
    explanation:
      'Complete service outage affecting all users is SEV-1 (Critical): Most/all users affected, significant revenue impact ($10K+/hour), requires immediate all-hands response, page on-call immediately. Other SEV levels: SEV-2 (High) = Significant degradation, subset of users affected, respond within 15 min. SEV-3 (Medium) = Minor degradation, small subset, respond within 1 hour. SEV-4 (Low) = Cosmetic, minimal impact, next business day. Proper severity classification ensures appropriate response urgency.',
  },
];

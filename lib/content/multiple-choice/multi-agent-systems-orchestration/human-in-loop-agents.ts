/**
 * Multiple choice questions for Human-in-the-Loop Agents section
 */

export const humaninloopagentsMultipleChoice = [
  {
    id: 'maas-hitl-mc-1',
    question:
      'An agent workflow has a human approval gate after Agent B. The human takes 30 minutes to respond. What is the PRIMARY impact on the workflow?',
    options: [
      'The workflow fails due to timeout',
      'The workflow blocks (waits) until human approval is received',
      'Agent B retries its task while waiting',
      'Subsequent agents (C, D) execute in parallel while waiting',
    ],
    correctAnswer: 1,
    explanation:
      "Human approval gates block execution—the workflow waits for the human to approve or reject before proceeding. This is the expected behavior. Option A would only occur if a timeout is configured. Options C and D don't represent typical approval gate behavior.",
  },
  {
    id: 'maas-hitl-mc-2',
    question:
      'You implement a confidence-based approval system: auto-approve if confidence >= 0.9, require human approval if confidence < 0.9. The agent always outputs confidence = 0.95 even when uncertain. What is this problem called?',
    options: [
      'Overconfident agent / miscalibration',
      'Underconfident agent',
      'Agent hallucination',
      'Agent bias',
    ],
    correctAnswer: 0,
    explanation:
      "This is overconfidence or miscalibration—the agent's confidence scores don't match its actual accuracy. Calibration involves comparing predicted confidence to actual approval rates. Options B and C don't match the scenario. Option D (bias) is too general.",
  },
  {
    id: 'maas-hitl-mc-3',
    question:
      'A human approver has 10 pending approvals. Which strategy BEST prioritizes their attention?',
    options: [
      'FIFO (First In First Out)—show approvals in the order they arrived',
      'Priority-based—show high-stakes or urgent approvals first',
      'Random—show approvals randomly to avoid bias',
      'LIFO (Last In First Out)—show most recent approvals first',
    ],
    correctAnswer: 1,
    explanation:
      "Priority-based sorting helps humans focus on what matters most (e.g., urgent, high-stakes, or complex decisions first). Option A (FIFO) might delay critical approvals. Options C and D are inefficient and don't respect importance.",
  },
  {
    id: 'maas-hitl-mc-4',
    question:
      'A progressive automation system starts at Level 0 (all decisions need human approval) and gradually moves to Level 3 (agent autonomy with spot checks). After reaching Level 3, the error rate spikes to 15%. What should happen?',
    options: [
      'Continue at Level 3 and investigate the cause of errors',
      'Immediately downgrade to Level 0 (full human oversight)',
      'Downgrade to Level 2 and investigate',
      'Increase automation to Level 4 to resolve errors faster',
    ],
    correctAnswer: 2,
    explanation:
      'Downgrade one level (to Level 2) to reduce blast radius while investigating. Level 0 is too extreme (full human oversight might not be needed). Option A is risky—continuing at Level 3 exposes users to errors. Option D (increase automation) makes no sense when errors are spiking.',
  },
  {
    id: 'maas-hitl-mc-5',
    question:
      'An approval gate shows the human: "Agent B recommends Action X. Approve or Reject?" The human approves without understanding why. What is the PRIMARY problem?',
    options: [
      'The human is not qualified to review the decision',
      "The approval UI lacks context (agent's reasoning, confidence, alternatives)",
      "The agent should not require approval if the human doesn't understand",
      'Human approval is unnecessary if the agent is confident',
    ],
    correctAnswer: 1,
    explanation:
      'Humans need rich context to make informed decisions: Why did the agent recommend X? What are alternatives? What is the confidence? Option A might be true but is not the primary UI problem. Option C is backwards—better UI solves this. Option D is incorrect—approval is there for a reason (safety).',
  },
];

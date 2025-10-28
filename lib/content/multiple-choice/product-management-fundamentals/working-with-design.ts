/**
 * Multiple choice questions for Working with Design
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const workingWithDesignMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'According to the PM-Design partnership model, when should designers be involved in the product development process?',
    options: [
      'After PM writes the PRD and defines requirements',
      'During solution specification after PM does user research',
      'At problem definition stage alongside PM during user research',
      'During build phase to create mockups for engineering',
    ],
    correctAnswer: 2,
    explanation:
      'Designers should be involved early at the problem definition stage, participating in user research alongside the PM. This builds shared understanding, allows designers to see user problems firsthand, and prevents the "PM hands finished requirements to designer" dynamic. When designers are involved early, they contribute to problem framing and feel ownership over the solution. The content explicitly states: "Involve designers early in problem definition, not just solution execution." Early involvement transforms designers from "order takers" to "partners."',
  },
  {
    id: 'mc2',
    question:
      'What is the key difference between a "prescriptive" PM brief and a "problem-focused" PM brief?',
    options: [
      'Prescriptive briefs are longer and more detailed',
      'Prescriptive briefs specify exact design solutions; problem-focused briefs define user problems and constraints',
      "Problem-focused briefs don't include success metrics",
      'Prescriptive briefs are better for experienced designers',
    ],
    correctAnswer: 1,
    explanation:
      'Prescriptive briefs specify exactly what to design (e.g., "Add Save for Later button, use blue color, place in sidebar"), removing designer autonomy. Problem-focused briefs define the user problem, needs, constraints, and success metrics, but let the designer explore solutions. Example: Instead of "Add bar chart showing spending by category," write "Problem: Users don\'t know where their money goes. User need: See spending patterns to identify wasteful spending." Problem-focused briefs give designers creative space while ensuring they understand the problem they\'re solving. This leads to better solutions and stronger partnership.',
  },
  {
    id: 'mc3',
    question:
      'How long should PM typically allocate for the complete design process (exploration, refinement, detailed design) for a complex feature?',
    options: [
      '1-2 days (designs should be quick)',
      '1 week (one sprint)',
      '3-6 weeks (depending on complexity)',
      '3-6 months (design requires perfection)',
    ],
    correctAnswer: 2,
    explanation:
      'The content recommends 3-6 weeks for complete design process for complex features: Exploration (1-2 weeks: research patterns, sketch concepts, create low-fi prototypes), Refinement (1-2 weeks: high-fidelity mockups, interactive prototypes, user testing), and Detailed Design (1-2 weeks: pixel-perfect specs, edge cases, documentation). Shorter timelines produce rushed designs. Much longer timelines are excessive for most features. PMs who don\'t plan design time often create "Can you mock this up by EOD?" situations that frustrate designers and produce lower-quality outcomes. Good PMs plan design time into project timelines upfront.',
  },
  {
    id: 'mc4',
    question:
      "When your CEO wants a full UI redesign (6 months) after seeing a competitor's interface, but users are happy with current UI, what should a PM do?",
    options: [
      'Immediately start the full redesign since CEO requested it',
      'Refuse the redesign and explain users are happy',
      'Gather data on user feedback and recommend targeted improvements instead',
      'Ask the designer to decide',
    ],
    correctAnswer: 2,
    explanation:
      'The PM should gather data (user feedback, support tickets, analytics, competitive analysis) to understand if this is a real user problem or CEO perception. If data shows users are satisfied (high task completion, few UI complaints, no lost deals), recommend targeted improvements (4-6 weeks) instead of full redesign (6 months). This balances CEO\'s concern with user needs and business priorities. Approach: "I hear your concern. Let me share user data and recommend targeted improvements that modernize key screens without halting feature development for 6 months." Good PMs make evidence-based recommendations that address stakeholder concerns while protecting user needs and business velocity.',
  },
  {
    id: 'mc5',
    question:
      "What is the best way for a PM to provide feedback when a designer's concept is beautiful but doesn't solve the core user need?",
    options: [
      'Tell the designer exactly how to fix the design',
      'Ask questions like "How does this design address [specific user need]?" to guide reflection',
      'Approve it anyway to avoid conflict',
      'Escalate to the design manager',
    ],
    correctAnswer: 1,
    explanation:
      'The content recommends asking questions that guide designer reflection rather than dictating solutions. Example: "Help me understand how this design addresses [user need]?" or "When users try to [accomplish task], how does this help them?" Questions invite designers to explain their thinking and often they\'ll identify gaps themselves. Follow up by revisiting the problem together: "Let me make sure I communicated the problem clearly" and using specific user scenarios: "When Sarah needs to track expenses across 3 accounts, what\'s her path?" This maintains designer ownership while realigning on user needs. It creates partnership, not hierarchy.',
  },
];

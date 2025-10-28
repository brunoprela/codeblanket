/**
 * Multiple choice questions for PM Core Competencies Framework
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const coreCompetenciesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does the "T-shaped" PM model mean?',
    options: [
      'PMs should focus only on their strongest competency and ignore others',
      'PMs should be equally expert at all 7 competencies',
      'PMs should have broad competency across all areas (horizontal bar) with deep expertise in 1-2 areas (vertical bar)',
      'PMs should master technical skills first, then develop other competencies',
    ],
    correctAnswer: 2,
    explanation:
      'The T-shaped model means PMs should be competent (Level 2-3) across all core areas while having deep expertise (Level 4-5) in 1-2 specific areas. The horizontal bar represents breadth (baseline competency in all areas), and the vertical bar represents depth (mastery in specific areas). For example, a Growth PM might be Level 5 in data analysis and execution while being Level 3 in other areas. This is realistic and more valuable than trying to be Level 5 at everything.',
  },
  {
    id: 'mc2',
    question:
      'According to the content, what should a junior PM (0-2 years) prioritize developing?',
    options: [
      'Strategic thinking and long-term vision',
      'Execution, user empathy, and communication fundamentals',
      'Leadership and stakeholder management',
      'Advanced technical skills and system architecture',
    ],
    correctAnswer: 1,
    explanation:
      'Junior PMs should focus on fundamentals: execution (shipping features reliably), user empathy (understanding user problems), and communication (clear requirements and updates). Strategic thinking, complex stakeholder management, and advanced technical skills come later with experience. Trying to develop everything simultaneously leads to burnout and mediocrity. The 80/20 rule applies: 80% of early PM success comes from mastering these three core competencies.',
  },
  {
    id: 'mc3',
    question:
      'Which competency is MOST critical for a Technical PM building developer tools (like Stripe API)?',
    options: [
      'User empathy',
      'Leadership',
      'Technical fluency',
      'Communication',
    ],
    correctAnswer: 2,
    explanation:
      'Technical fluency (Level 5) is the most critical competency for Technical PMs because: (1) their users are highly technical developers, (2) they need to design APIs and understand system architecture deeply, (3) technical credibility is essential for developer trust, and (4) they make complex technical trade-offs daily. While other competencies matter, technical depth is the differentiator. A Technical PM at Stripe needs to understand distributed systems, API design, idempotency, and performance at an expert level.',
  },
  {
    id: 'mc4',
    question:
      'What is the recommended approach to developing PM competencies to avoid burnout?',
    options: [
      'Work on all 7 competencies simultaneously to become well-rounded quickly',
      'Focus intensively on your weakest competency until it becomes a strength',
      'Sequential development: focus on 1-2 competencies per quarter, starting with strengths',
      'Only develop competencies through formal training programs and courses',
    ],
    correctAnswer: 2,
    explanation:
      'Sequential development is the anti-burnout approach: focus on 1-2 competencies per quarter (not all 7 simultaneously), start by building your strengths into the "vertical bar" of your T-shape, then gradually develop other areas to baseline competency. Most development happens on the job, not through separate studying. This focused approach with 3-5 hours/week of deliberate practice prevents overwhelm while ensuring steady progress. Trying to improve everything at once leads to mediocrity and burnout.',
  },
  {
    id: 'mc5',
    question:
      'For a Growth PM optimizing conversion funnels, which competency ranking is most accurate?',
    options: [
      'User Empathy #1, Technical Fluency #2, Data Analysis #3',
      'Data Analysis #1, Execution #2, Strategic Thinking #3',
      'Technical Fluency #1, Strategic Thinking #2, Leadership #3',
      'Communication #1, User Empathy #2, Execution #3',
    ],
    correctAnswer: 1,
    explanation:
      'For Growth PMs, Data Analysis is #1 (Level 5 required) because growth is fundamentally about numbers—SQL proficiency, A/B testing, funnel analysis, and statistical rigor are essential. Execution is #2 (Level 5) because Growth PMs ship constantly (many small experiments vs. few big bets). Strategic Thinking is #3 (Level 4) for identifying growth loops and prioritizing levers. Growth PMs rely more on quantitative data than qualitative user empathy, unlike Consumer PMs who prioritize empathy #1.',
  },
];

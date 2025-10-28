/**
 * Multiple choice questions for PM at Different Company Stages
 * Product Management Fundamentals Module
 */

import { MultipleChoiceQuestion } from '../../../types';

export const pmAtDifferentStagesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the typical equity range for a PM joining a Series A startup (10-50 people)?',
    options: [
      '0.001-0.01% (minimal equity)',
      '0.1-0.5% (meaningful equity)',
      '1-3% (significant equity)',
      '5-10% (founder-level equity)',
    ],
    correctAnswer: 1,
    explanation:
      'Series A startup (10-50 people, $2-10M funding) typically offers PMs 0.1-0.5% equity. This is meaningful but not founder-level. For comparison: Seed (0-10 people) offers 0.5-2%+, Series B (50-200 people) offers 0.01-0.1%, Growth (200-1000+) offers 0.001-0.01%, Public companies offer RSUs not percentage equity. Equity decreases as company matures because: (1) Less risk, (2) More employees to share with, (3) Company valuation higher. Series A is often the "sweet spot" - proven PMF (less risk) but still significant equity upside.',
  },
  {
    id: 'mc2',
    question:
      'What is the primary difference in PM day-to-day work between a seed-stage startup vs. public company?',
    options: [
      'Public companies pay more but work is identical',
      'Seed PM does everything (PM/designer/support); Public PM has narrow scope with lots of meetings and process',
      'Seed PM manages large teams; Public PM works alone',
      'No significant difference in responsibilities',
    ],
    correctAnswer: 1,
    explanation:
      'Seed PM (0-10 people): Wears all hats (PM + designer + marketer + support), talks to users constantly (5-10/week), ships daily, defines product vision with founder, no process. Public PM (1000+ people): Owns small piece of large product, optimizes metrics at scale, navigates bureaucracy, lots of meetings (20-30/week), follows established processes, launches take 6-12 months. Key differences: Pace (extreme vs slow), Scope (everything vs narrow), Autonomy (high vs low), Resources (minimal vs abundant), Process (create vs follow). Both are valid PM roles but dramatically different experiences.',
  },
  {
    id: 'mc3',
    question:
      'According to the content, what is the "sweet spot" company stage for most PMs?',
    options: [
      'Pre-seed (highest equity)',
      'Series A/B (balance of learning, mentorship, risk, and equity)',
      'Growth stage (highest stability)',
      'Public company (highest compensation)',
    ],
    correctAnswer: 1,
    explanation:
      'Series A/B is often the "sweet spot" because it balances: (1) Proven product-market fit (less risk than seed), (2) Still significant equity (0.1-0.5% vs 0.001% at public), (3) High learning opportunity (wear multiple hats, fast pace), (4) Mentorship available (senior PMs exist unlike seed), (5) Reasonable resources (not scrappy like seed, not bureaucratic like public). For early-career PMs especially, Series A/B offers best risk-reward balance: Learn rapidly, build generalist skills, meaningful equity upside, lower failure risk than seed. Can always join public company later, but harder to go public → startup → public.',
  },
  {
    id: 'mc4',
    question:
      'What is a key skill difference PMs develop at startups vs. public companies?',
    options: [
      'Startups develop breadth (0→1, scrappiness, generalist); Public develops depth (scale, systems, specialization)',
      'Startups develop technical skills; Public develops business skills',
      'Startups develop leadership; Public develops execution',
      'No skill difference - PMs learn the same things everywhere',
    ],
    correctAnswer: 0,
    explanation:
      'Startups develop BREADTH: 0→1 product building (from idea to first customers), scrappiness (do everything with zero resources), generalist skills (PM + designer + marketer), speed (ship daily), business sense (unit economics, fundraising). Public companies develop DEPTH: Scale (products with 100M+ users), systems thinking (complex interdependencies), data rigor (A/B testing, statistical significance), stakeholder management (navigate huge orgs), specialization (become expert in narrow area). Both skill sets are valuable. Career hack: Do startup early (build breadth), then public company (build depth), then leverage both.',
  },
  {
    id: 'mc5',
    question:
      'What should be the primary factor in choosing what company stage to join?',
    options: [
      'Always choose the highest equity',
      'Always choose the highest cash compensation',
      'Match to your risk tolerance, learning goals, financial needs, and life circumstances',
      'Always join the most prestigious company',
    ],
    correctAnswer: 2,
    explanation:
      'Choose based on personal factors, not absolute "best": (1) Risk tolerance (high → seed/A, low → growth/public), (2) Learning goals (breadth → startup, depth → public), (3) Financial needs (need cash → public, can take equity bet → startup), (4) Life circumstances (single/young → can take risk, family/mortgage → need stability). There\'s no universally "better" stage. Example: Age 25 single → Seed makes sense. Age 35 with kids → Public makes sense. Both choices are valid for those circumstances. The content emphasizes: "Choose based on your situation, not what sounds cooler."',
  },
];

/**
 * Multiple choice questions for Chaos Engineering section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const chaosEngineeringMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is Chaos Engineering?',
    options: [
      'Randomly breaking things in production',
      'Deliberately introducing failures to build confidence in system resilience',
      'A type of load testing',
      'Testing in development only',
    ],
    correctAnswer: 1,
    explanation:
      "Chaos Engineering is the discipline of deliberately introducing failures into systems to build confidence in their ability to withstand turbulent conditions. It's NOT about randomly breaking things - it's scientific experiments with hypotheses, controlled blast radius, and learning goals. Purpose: Find weaknesses before they cause outages, validate resilience mechanisms (circuit breakers, retries), practice incident response, build confidence. Example: Netflix Chaos Monkey randomly terminates instances to ensure services handle failures gracefully.",
  },
  {
    id: 'mc2',
    question:
      'Why should chaos experiments run in production rather than only staging?',
    options: [
      "They shouldn't, only run in staging",
      "Production has unique characteristics (real traffic, data, dependencies) that can't be replicated",
      'To save money on staging environments',
      "Staging doesn't exist",
    ],
    correctAnswer: 1,
    explanation:
      "Chaos experiments should run in production (with safeguards) because production has unique characteristics: Real traffic patterns (spikes, user behavior), real data volumes, real dependencies (third-party APIs), real failure modes. Staging can't replicate these. Safeguards: Start with 1% blast radius, business hours, kill switch, automatic rollback. Example: Staging works fine, but production discovers that 3AM batch job + instance failure causes cascade - only visible in production.",
  },
  {
    id: 'mc3',
    question: 'What is "blast radius" in chaos engineering?',
    options: [
      'The distance an explosion travels',
      'The maximum potential impact of an experiment (% of users/services affected)',
      'The number of servers',
      'The duration of an experiment',
    ],
    correctAnswer: 1,
    explanation:
      'Blast radius is the maximum potential impact of a chaos experiment - how many users, services, or systems could be affected if it goes wrong. Minimize by: Start small (canary → 1% → 10% → 100%), gradual expansion, feature flags, kill switch, automatic rollback if metrics degrade. Example: Phase 1: 1% traffic → Phase 2: 10% → Phase 3: 50% → Phase 4: 100%, only expanding if metrics stay stable. This limits risk while enabling learning.',
  },
  {
    id: 'mc4',
    question:
      'What is the difference between automated chaos engineering and Game Days?',
    options: [
      'They are the same thing',
      'Automated runs continuously testing technology, Game Days are scheduled exercises testing people/processes',
      'Game Days are for games, not chaos',
      'Automated is only for testing',
    ],
    correctAnswer: 1,
    explanation:
      'Automated Chaos (e.g., Chaos Monkey): Runs continuously or scheduled, tests technology (circuit breakers work, auto-scaling works), simple failures (kill instance, add latency). Game Days: Scheduled (monthly), entire team participates, tests people and processes (coordination, communication, runbooks), complex scenarios (region outage, multi-service failure). Both are valuable: Automated catches regressions, Game Days practice coordination and validate runbooks. Complement each other.',
  },
  {
    id: 'mc5',
    question:
      'What should be the first step in any chaos engineering experiment?',
    options: [
      'Inject the failure immediately',
      'Define steady state hypothesis (e.g., "System will maintain 99.9% availability despite 10% instance loss")',
      'Page the entire team',
      'Turn off monitoring',
    ],
    correctAnswer: 1,
    explanation:
      'First step is defining steady state hypothesis: measurable normal behavior and expectation during experiment. Example: "Despite killing 10% of instances, system will maintain: 99.9% success rate, p99 latency < 500ms, 10,000 req/s throughput." This enables objective evaluation: Did hypothesis hold? Why or why not? Without hypothesis, experiment has no clear success criteria and no structure for learning. Process: Define steady state → Form hypothesis → Inject failure → Observe metrics → Learn and improve.',
  },
];

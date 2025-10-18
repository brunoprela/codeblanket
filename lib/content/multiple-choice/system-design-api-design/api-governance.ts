/**
 * Multiple choice questions for API Governance section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apigovernanceMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'governance-q1',
    question: 'What is the purpose of an API catalog in API governance?',
    options: [
      'To store API keys and secrets',
      'To provide a central registry of all APIs with ownership, status, and documentation',
      'To automatically generate API documentation',
      'To monitor API performance',
    ],
    correctAnswer: 1,
    explanation:
      'API catalog is a central registry listing all APIs with: owner team, status (active/deprecated), documentation links, OpenAPI specs, and contact info. Essential for discovery and governance in organizations with many APIs.',
    difficulty: 'easy',
  },
  {
    id: 'governance-q2',
    question: 'Why implement automated linting of OpenAPI specifications?',
    options: [
      'To make APIs faster',
      'To enforce API design standards consistently across all APIs',
      'To reduce server costs',
      'To automatically fix API bugs',
    ],
    correctAnswer: 1,
    explanation:
      'Automated linting (e.g., Spectral) enforces design standards: naming conventions, required fields, error formats. Catches violations before merge, ensuring consistency across all APIs. Human reviews miss details; automation is consistent.',
    difficulty: 'medium',
  },
  {
    id: 'governance-q3',
    question: 'What is the design-first approach to API development?',
    options: [
      'Write code first, then document it',
      'Create OpenAPI specification before implementing the API',
      'Design the database schema before the API',
      'Create UI mockups before API design',
    ],
    correctAnswer: 1,
    explanation:
      'Design-first: Write OpenAPI spec before code. Benefits: (1) Review API design early, (2) Generate mocks for frontend, (3) Ensure docs match implementation, (4) Catch design issues before coding. Alternative: code-first (generate spec from code).',
    difficulty: 'easy',
  },
  {
    id: 'governance-q4',
    question: 'What should a proper API deprecation policy include?',
    options: [
      'Immediate removal of old versions',
      'Advance notice (6-12 months), sunset timeline, and migration guide',
      'Just update documentation',
      'Automatic redirects to new version',
    ],
    correctAnswer: 1,
    explanation:
      'API deprecation policy: (1) Announce 6-12 months early, (2) Set sunset date, (3) Provide migration guide, (4) Email affected clients, (5) Support N-1 versions. Never remove immediately; causes integration breakage.',
    difficulty: 'easy',
  },
  {
    id: 'governance-q5',
    question: 'Why have an API review board in large organizations?',
    options: [
      'To slow down API development',
      'To ensure APIs meet design standards, security requirements, and consistency',
      'To generate API documentation automatically',
      'To reduce the number of APIs',
    ],
    correctAnswer: 1,
    explanation:
      'API review board ensures: (1) Design consistency, (2) Security standards met, (3) No duplicate APIs, (4) Follow best practices. Reviews major changes before GA. Not to slow down, but ensure quality and consistency across organization.',
    difficulty: 'medium',
  },
];

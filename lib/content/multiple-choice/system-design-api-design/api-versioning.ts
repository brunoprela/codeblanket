/**
 * Multiple choice questions for API Versioning Strategies section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apiversioningMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'versioning-q1',
    question: 'What is the most common API versioning strategy?',
    options: [
      'Query parameter versioning (e.g., ?version=2)',
      'Header versioning (e.g., Accept-Version: v2)',
      'URL path versioning (e.g., /v1/users, /v2/users)',
      'Content negotiation (e.g., Accept: application/vnd.example.v2+json)',
    ],
    correctAnswer: 2,
    explanation:
      "URL path versioning (/v1/users) is most common because it's visible, simple, cacheable, and works with all clients. Header versioning is cleaner but less discoverable. Query params are messy. Content negotiation is most RESTful but complex. Most public APIs use URL path.",
  },
  {
    id: 'versioning-q2',
    question:
      'Which change is considered NON-BREAKING and safe to make without a version bump?',
    options: [
      'Removing a field from the response',
      'Adding a new optional field to the response',
      'Renaming an existing field',
      'Changing a field from string to integer',
    ],
    correctAnswer: 1,
    explanation:
      'Adding optional fields is non-breaking: clients ignore unknown fields. Removing fields breaks clients expecting them. Renaming fields breaks clients using old name. Type changes break parsing. Only additive, optional changes are safe without version bump.',
  },
  {
    id: 'versioning-q3',
    question:
      'What is the purpose of a deprecation period before removing an API version?',
    options: [
      'To test the new version in production',
      'To give clients time to migrate without disruption',
      'To reduce server costs gradually',
      'To collect user feedback',
    ],
    correctAnswer: 1,
    explanation:
      'Deprecation period (typically 6-12 months) gives clients time to migrate from old to new version without disruption. Removing versions immediately breaks existing integrations. Announce deprecation, warn users, then sunset after reasonable period.',
  },
  {
    id: 'versioning-q4',
    question:
      'In semantic versioning (MAJOR.MINOR.PATCH), when should you bump the MAJOR version?',
    options: [
      'When fixing any bug',
      'When adding new features',
      "When making breaking changes that aren't backward compatible",
      'Once per year',
    ],
    correctAnswer: 2,
    explanation:
      'MAJOR version bumps signal breaking changes (remove fields, rename, change behavior). MINOR is for backward-compatible features. PATCH is for bug fixes. Clients know MAJOR change requires code updates, while MINOR/PATCH are safe to upgrade.',
  },
  {
    id: 'versioning-q5',
    question:
      'What header can you use to warn clients that an API version will be sunset?',
    options: [
      'X-Deprecation-Date',
      'X-API-Sunset',
      'X-Version-End',
      'Deprecation',
    ],
    correctAnswer: 1,
    explanation:
      'X-API-Sunset header indicates when version will be removed (e.g., X-API-Sunset: 2024-12-31). Combine with X-API-Deprecated: true and link to migration guide. This gives programmatic way for clients to detect and plan migrations.',
  },
];

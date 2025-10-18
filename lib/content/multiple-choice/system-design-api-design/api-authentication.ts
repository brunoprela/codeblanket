/**
 * Multiple choice questions for API Authentication Methods section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apiauthenticationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'auth-q1',
    question:
      'Which authentication method is most appropriate for a mobile app accessing your API with per-user permissions?',
    options: [
      'API Keys (one per app install)',
      'Basic Authentication with username/password',
      'JWT with OAuth 2.0 PKCE flow',
      'Mutual TLS certificates',
    ],
    correctAnswer: 2,
    explanation:
      "JWT with OAuth 2.0 PKCE flow is designed for mobile apps: provides per-user auth, secure without client secret, supports token refresh. API keys don't differentiate users, Basic Auth sends credentials repeatedly, mTLS is complex for mobile.",
    difficulty: 'medium',
  },
  {
    id: 'auth-q2',
    question:
      'Why should JWT access tokens have short lifetimes (e.g., 15 minutes)?',
    options: [
      'To reduce server storage requirements',
      'To force users to log in frequently for security',
      'To limit damage window if token is compromised',
      'To improve API performance',
    ],
    correctAnswer: 2,
    explanation:
      "Short-lived access tokens limit the time window an attacker can use a stolen token. Use refresh tokens for long-term access. JWTs are stateless (not stored on server), and short lifetimes don't require frequent user logins (refresh tokens handle renewal).",
    difficulty: 'medium',
  },
  {
    id: 'auth-q3',
    question:
      'You need to allow a third-party app to access user data on behalf of users without receiving user passwords. Which approach?',
    options: [
      'Share API keys with third party',
      'Have users share their passwords with third party',
      'Implement OAuth 2.0 authorization',
      'Use Basic Authentication with temporary passwords',
    ],
    correctAnswer: 2,
    explanation:
      'OAuth 2.0 is specifically designed for delegation - allowing third parties to access resources on behalf of users without sharing credentials. Users grant permission, and third party receives scoped access tokens.',
    difficulty: 'easy',
  },
  {
    id: 'auth-q4',
    question: 'What is the main security advantage of mTLS over Bearer tokens?',
    options: [
      'mTLS tokens are shorter and faster',
      'mTLS provides cryptographic proof of identity from both sides',
      'mTLS tokens never expire',
      "mTLS doesn't require HTTPS",
    ],
    correctAnswer: 1,
    explanation:
      'mTLS (Mutual TLS) requires both client and server to present valid certificates, providing cryptographic proof of identity. Bearer tokens can be stolen and replayed. mTLS still requires TLS, and certificates do expire.',
    difficulty: 'hard',
  },
  {
    id: 'auth-q5',
    question:
      'Your API key was accidentally committed to a public GitHub repo. Best response?',
    options: [
      'Delete the commit immediately, problem solved',
      'Revoke the key, rotate to new key, audit usage logs',
      'Change GitHub repo to private',
      'Add .gitignore for future, keep current key',
    ],
    correctAnswer: 1,
    explanation:
      "Once exposed publicly, assume the key is compromised forever (GitHub history, scrapers, caches). Must revoke immediately, issue new key, and audit logs for unauthorized usage. Deleting commits doesn't remove from history.",
    difficulty: 'easy',
  },
];

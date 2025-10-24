import { MultipleChoiceQuestion } from '../../../types';

export const authenticationAuthorizationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'pllm-auth-mc-1',
      question:
        'What authentication method should you use for service-to-service API calls?',
      options: [
        'Username/password',
        'API keys with X-API-Key header',
        'No authentication',
        'Cookie-based sessions',
      ],
      correctAnswer: 1,
      explanation:
        'API keys are standard for service-to-service authentication: stateless, easy to rotate, rate-limitable per key, include in X-API-Key header.',
    },
    {
      id: 'pllm-auth-mc-2',
      question: 'How should you store API keys in your database?',
      options: [
        'Plain text',
        'Hashed with bcrypt or similar',
        'Encrypted',
        'In code',
      ],
      correctAnswer: 1,
      explanation:
        'Hash API keys with bcrypt before storage. Never store plain text. Compare hashes on authentication. Impossible to recover original if database compromised.',
    },
    {
      id: 'pllm-auth-mc-3',
      question:
        'What are the advantages of JWT tokens for user authentication?',
      options: [
        'Stateless authentication with claims encoded in token',
        'Require database lookup',
        'Cant expire',
        'More secure than sessions',
      ],
      correctAnswer: 0,
      explanation:
        'JWTs are stateless (no database lookup needed), contain claims (user_id, role), can be verified with secret key, support expiration.',
    },
    {
      id: 'pllm-auth-mc-4',
      question: 'How do you implement multi-tenant data isolation?',
      options: [
        'Trust client-provided tenant_id',
        'Extract tenant from API key/JWT, filter all queries by tenant_id, use RLS',
        'No isolation needed',
        'Separate databases',
      ],
      correctAnswer: 1,
      explanation:
        'Never trust client data. Extract tenant from verified source (API key/JWT), automatically filter all queries by tenant_id, use Row-Level Security.',
    },
    {
      id: 'pllm-auth-mc-5',
      question: 'How should you implement API key rotation?',
      options: [
        'Forced immediate rotation',
        'Dual-key system: primary and secondary both work during 30-day transition',
        'Never rotate',
        'Annual rotation',
      ],
      correctAnswer: 1,
      explanation:
        'Dual-key system allows zero-downtime rotation: generate secondary, test, switch, disable old primary. 30-day grace period for client migration.',
    },
  ];

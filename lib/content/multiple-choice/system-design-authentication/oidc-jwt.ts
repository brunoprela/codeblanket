/**
 * Multiple choice questions for OIDC (OpenID Connect) & JWT section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const oidcjwtMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the primary difference between OAuth 2.0 and OIDC?',
    options: [
      'OIDC is faster',
      'OIDC adds authentication (ID token) on top of OAuth 2.0 authorization',
      'OAuth 2.0 is newer',
      'They are the same',
    ],
    correctAnswer: 1,
    explanation:
      'OIDC (OpenID Connect) adds an authentication layer on top of OAuth 2.0. OAuth 2.0 handles authorization ("what can you access?"), while OIDC adds authentication ("who are you?") via ID tokens. OIDC = OAuth 2.0 + identity layer.',
  },
  {
    id: 'mc2',
    question: 'What is the purpose of the ID token in OIDC?',
    options: [
      'To access APIs',
      'To prove user identity to the client application',
      'To encrypt data',
      'To store session data',
    ],
    correctAnswer: 1,
    explanation:
      "ID token proves user identity to the client application. It's a JWT containing user claims (sub, email, name). Client validates signature and extracts user info. For API access, use access token. For authentication, use ID token.",
  },
  {
    id: 'mc3',
    question: 'What are the three parts of a JWT token?',
    options: [
      'Username, password, signature',
      'Header, payload, signature',
      'Public key, private key, data',
      'Token, refresh, access',
    ],
    correctAnswer: 1,
    explanation:
      'JWT has three parts: (1) Header - algorithm and token type, (2) Payload - claims (user data), (3) Signature - cryptographic proof of authenticity. Format: header.payload.signature. Each part is Base64URL encoded and separated by dots.',
  },
  {
    id: 'mc4',
    question: 'What triggers OIDC mode in an OAuth 2.0 flow?',
    options: [
      'Using HTTPS',
      'Including "openid" in the scope parameter',
      'Using POST instead of GET',
      'Including client_secret',
    ],
    correctAnswer: 1,
    explanation:
      'Including "openid" in the scope parameter triggers OIDC. Without it, it\'s plain OAuth 2.0 (authorization only). With "openid", you get ID token in addition to access token. Example: scope=openid email profile',
  },
  {
    id: 'mc5',
    question: 'Why is JWT signature verification critical?',
    options: [
      'To decrypt the payload',
      'To prove token authenticity and prevent tampering',
      'To compress the token',
      'To set expiration time',
    ],
    correctAnswer: 1,
    explanation:
      "Signature verification proves: (1) Token came from trusted IdP (has private key), (2) Token hasn't been tampered with. JWT payload is Base64 encoded, NOT encrypted - anyone can read it. Signature prevents forgery. Always verify signature before trusting token contents!",
  },
  {
    id: 'mc6',
    question: 'What is the purpose of the JWKS endpoint?',
    options: [
      'To store user data',
      'To publish public keys for JWT signature verification',
      'To authenticate users',
      'To generate tokens',
    ],
    correctAnswer: 1,
    explanation:
      "JWKS (JSON Web Key Set) endpoint publishes IdP's public keys in standard JSON format. Clients fetch public keys from JWKS to verify JWT signatures. Example: https://idp.com/.well-known/jwks.json. Supports key rotation - IdP can publish new keys without breaking clients.",
  },
];

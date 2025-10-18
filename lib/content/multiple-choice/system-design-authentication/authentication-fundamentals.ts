/**
 * Multiple choice questions for Authentication Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const authenticationfundamentalsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'What is the main difference between authentication and authorization?',
      options: [
        'Authentication is faster than authorization',
        'Authentication verifies identity, authorization determines permissions',
        'Authentication is for users, authorization is for applications',
        'There is no difference',
      ],
      correctAnswer: 1,
      explanation:
        'Authentication (AuthN) verifies WHO you are - proving identity. Authorization (AuthZ) determines WHAT you can do - checking permissions. Example: Login verifies you are John (authentication), then checks if John can view customer records (authorization).',
    },
    {
      id: 'mc2',
      question: 'What is the primary role of an Identity Provider (IdP)?',
      options: [
        'Store application data',
        'Authenticate users and issue tokens',
        'Host web applications',
        'Manage databases',
      ],
      correctAnswer: 1,
      explanation:
        'IdP is the authentication authority that authenticates users and issues tokens/assertions. Examples: Okta, Auth0, Google. The IdP knows who users are, verifies their credentials, and tells Service Providers "this user is authenticated".',
    },
    {
      id: 'mc3',
      question: 'In SSO architecture, what does the Service Provider (SP) do?',
      options: [
        'Stores user passwords',
        'Authenticates users directly',
        'Trusts IdP and validates authentication tokens',
        'Replaces the IdP',
      ],
      correctAnswer: 2,
      explanation:
        "Service Provider (SP) is the application users want to access. It trusts the IdP to handle authentication, validates tokens from IdP, and grants access based on those tokens. Critically, SP does NOT store passwords - that's the IdP's job.",
    },
    {
      id: 'mc4',
      question: 'Which protocol is best for modern mobile app authentication?',
      options: ['SAML', 'LDAP', 'OIDC', 'Kerberos'],
      correctAnswer: 2,
      explanation:
        "OpenID Connect (OIDC) is best for modern mobile apps. It's JSON-based (lightweight), has native mobile support, better security than SAML for mobile, and simpler to implement. SAML is XML-based and designed for browser redirects, making it clunky for mobile.",
    },
    {
      id: 'mc5',
      question: 'What is a key security benefit of SSO?',
      options: [
        'SSO is faster than traditional login',
        'Centralized MFA and instant deprovisioning when employees leave',
        'SSO does not require passwords',
        'SSO is cheaper to implement',
      ],
      correctAnswer: 1,
      explanation:
        'Key security benefits: (1) Enforce MFA in one place (IdP) instead of 100 apps. (2) When employee leaves, disable IdP access → instantly loses access to ALL apps. (3) Centralized audit logs. (4) Security team focuses on hardening one IdP instead of many apps.',
    },
    {
      id: 'mc6',
      question: 'What is the main risk of SSO?',
      options: [
        'SSO is too slow',
        'SSO is single point of failure - if IdP is compromised, all apps are at risk',
        'SSO does not work with mobile apps',
        'SSO cannot be used with databases',
      ],
      correctAnswer: 1,
      explanation:
        'The "master key problem": If IdP is compromised or goes down, all integrated apps are affected. This is why IdP security is critical - MFA, monitoring, high availability, zero-trust architecture. The trade-off is worth it because securing one IdP well is easier than securing 100 apps.',
    },
  ];

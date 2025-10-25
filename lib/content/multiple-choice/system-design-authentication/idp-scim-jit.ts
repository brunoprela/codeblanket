/**
 * Multiple choice questions for Identity Providers, SCIM & JIT Provisioning section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const idpscimjitMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the primary purpose of SCIM?',
    options: [
      'User authentication',
      'Automated user provisioning and deprovisioning across systems',
      'Token generation',
      'Password encryption',
    ],
    correctAnswer: 1,
    explanation:
      'SCIM (System for Cross-domain Identity Management) automates user lifecycle management - creating, updating, and deactivating user accounts across multiple applications. When HR adds/removes employee in IdP, SCIM pushes changes to all integrated apps automatically.',
  },
  {
    id: 'mc2',
    question: 'What is Just-In-Time (JIT) provisioning?',
    options: [
      'Provisioning users before they log in',
      'Automatically creating user accounts on first login using SSO attributes',
      'Scheduling user creation for later',
      'Batch importing users',
    ],
    correctAnswer: 1,
    explanation:
      'JIT provisioning creates user accounts automatically on first SSO login using attributes from SAML assertion or OIDC ID token. No pre-provisioning needed. User logs in → App receives attributes → App creates account → User has access.',
  },
  {
    id: 'mc3',
    question: 'What is the main security limitation of JIT compared to SCIM?',
    options: [
      'JIT is slower',
      'JIT does not handle deprovisioning - user accounts persist after being disabled at IdP',
      'JIT requires more code',
      'JIT does not support MFA',
    ],
    correctAnswer: 1,
    explanation:
      "JIT only handles provisioning (create on first login), not deprovisioning. When user is disabled at IdP, they can't log in via SSO, but their local account still exists. SCIM solves this by actively deactivating accounts when user is disabled at IdP.",
  },
  {
    id: 'mc4',
    question:
      'Which IdP is best suited for developers building a new SaaS application?',
    options: [
      'Microsoft Entra ID (Azure AD)',
      'Ping Identity',
      'Auth0',
      'Keycloak',
    ],
    correctAnswer: 2,
    explanation:
      "Auth0 is developer-focused with excellent SDKs, documentation, and API-first design. It\'s perfect for SaaS apps needing authentication (social login, email/password, SSO). Microsoft Entra ID is best for Microsoft ecosystems. Ping is for large enterprises. Keycloak is open-source but requires more setup.",
  },
  {
    id: 'mc5',
    question:
      'What should you use as the primary identifier for users in your application?',
    options: [
      'Email address',
      'Username',
      'Stable identifier like SAML NameID or OIDC sub claim',
      'Phone number',
    ],
    correctAnswer: 2,
    explanation:
      "Use stable identifier: SAML NameID or OIDC sub claim. Email addresses can change (user gets married, changes companies). Email should be secondary identifier. sub is guaranteed unique and stable across user's lifetime.",
  },
  {
    id: 'mc6',
    question: 'What happens in a SCIM deprovisioning flow?',
    options: [
      'User is deleted from IdP',
      'IdP calls SCIM API to set user active:false in all integrated apps',
      'User password is reset',
      'User is locked out of IdP only',
    ],
    correctAnswer: 1,
    explanation:
      'When user is deactivated at IdP (employee terminated), IdP calls SCIM PATCH endpoint for each integrated app, setting active:false. Apps immediately disable user accounts. This ensures instant, synchronized offboarding across all systems for security.',
  },
];

/**
 * Multiple choice questions for SAML (Security Assertion Markup Language) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const samldeepdiveMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What format does SAML use for assertions?',
    options: ['JSON', 'XML', 'YAML', 'Binary'],
    correctAnswer: 1,
    explanation:
      'SAML uses XML (Extensible Markup Language) for assertions. This is one of the main differences from modern protocols like OIDC which use JSON. XML is verbose but mature and well-established in enterprises.',
  },
  {
    id: 'mc2',
    question: 'What is the purpose of digitally signing SAML assertions?',
    options: [
      'To make them faster',
      'To prove authenticity and prevent tampering',
      'To compress them',
      'To encrypt them',
    ],
    correctAnswer: 1,
    explanation:
      "Digital signatures prove: (1) The assertion came from the IdP (authenticity) - only IdP has private key. (2) The assertion wasn't modified in transit (integrity) - any change breaks signature. This prevents attackers from forging or modifying assertions.",
  },
  {
    id: 'mc3',
    question:
      'In SP-initiated SAML flow, where does the authentication process begin?',
    options: [
      'At the IdP portal',
      'At the Service Provider (the app user wants to access)',
      "At the user's browser",
      'At the database',
    ],
    correctAnswer: 1,
    explanation:
      'SP-initiated flow starts when user accesses the Service Provider (e.g., visits salesforce.com). SP then redirects to IdP for authentication. This is the most common flow. Alternative is IdP-initiated where user starts at IdP portal and clicks app tile.',
  },
  {
    id: 'mc4',
    question: 'What is the typical validity window for a SAML assertion?',
    options: ['24 hours', '1 hour', '5 minutes', '1 second'],
    correctAnswer: 2,
    explanation:
      'SAML assertions typically have a 5-minute validity window (NotBefore to NotOnOrAfter). This short window prevents replay attacks where an attacker captures and reuses an old assertion. Short validity is safe because assertion is only used once during initial login to establish a longer session.',
  },
  {
    id: 'mc5',
    question: 'What is JIT (Just-In-Time) provisioning in SAML?',
    options: [
      'Faster authentication',
      'Automatic user account creation from SAML attributes on first login',
      'Real-time compilation',
      'Instant password reset',
    ],
    correctAnswer: 1,
    explanation:
      'JIT provisioning means SP automatically creates user account when they first log in via SAML, using attributes from the assertion (email, name, role). Eliminates need to pre-create accounts in SP. IdP assertion provides all info needed to create user.',
  },
  {
    id: 'mc6',
    question: 'What is the main disadvantage of SAML compared to OIDC?',
    options: [
      'SAML is less secure',
      'SAML is not standardized',
      'SAML is XML-based, more complex, and has poor mobile support',
      'SAML cannot do SSO',
    ],
    correctAnswer: 2,
    explanation:
      "SAML's disadvantages: (1) XML is verbose and complex to parse. (2) Poor mobile app support (designed for browser redirects). (3) Complex developer experience. (4) Large payload size. OIDC addresses these with JSON, native mobile support, and simpler implementation. However, SAML remains dominant in enterprise due to legacy support.",
  },
];

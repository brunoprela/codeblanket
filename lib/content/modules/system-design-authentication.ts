/**
 * System Design: Authentication & SSO Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { authenticationfundamentalsSection } from '../sections/system-design-authentication/authentication-fundamentals';
import { samldeepdiveSection } from '../sections/system-design-authentication/saml-deep-dive';
import { oauth2Section } from '../sections/system-design-authentication/oauth2';
import { oidcjwtSection } from '../sections/system-design-authentication/oidc-jwt';
import { idpscimjitSection } from '../sections/system-design-authentication/idp-scim-jit';

// Import quizzes
import { authenticationfundamentalsQuiz } from '../quizzes/system-design-authentication/authentication-fundamentals';
import { samldeepdiveQuiz } from '../quizzes/system-design-authentication/saml-deep-dive';
import { oauth2Quiz } from '../quizzes/system-design-authentication/oauth2';
import { oidcjwtQuiz } from '../quizzes/system-design-authentication/oidc-jwt';
import { idpscimjitQuiz } from '../quizzes/system-design-authentication/idp-scim-jit';

// Import multiple choice
import { authenticationfundamentalsMultipleChoice } from '../multiple-choice/system-design-authentication/authentication-fundamentals';
import { samldeepdiveMultipleChoice } from '../multiple-choice/system-design-authentication/saml-deep-dive';
import { oauth2MultipleChoice } from '../multiple-choice/system-design-authentication/oauth2';
import { oidcjwtMultipleChoice } from '../multiple-choice/system-design-authentication/oidc-jwt';
import { idpscimjitMultipleChoice } from '../multiple-choice/system-design-authentication/idp-scim-jit';

export const systemDesignAuthenticationModule: Module = {
  id: 'system-design-authentication',
  title: 'System Design: Authentication & SSO',
  description:
    'Master authentication concepts including SSO, SAML, OAuth, OIDC, JWT, identity providers, and modern authentication patterns',
  category: 'System Design',
  difficulty: 'Medium',
  estimatedTime: '2-3 hours',
  prerequisites: [],
  icon: '🔐',
  keyTakeaways: [
    'SSO (Single Sign-On) enables one login to access multiple applications',
    'IdP (Identity Provider) authenticates users and issues tokens; SP (Service Provider) trusts IdP',
    'SAML uses XML-based assertions for enterprise SSO (legacy but dominant)',
    'OAuth 2.0 is for authorization (delegated access), NOT authentication',
    'OIDC adds authentication layer on OAuth 2.0 with ID tokens (modern standard)',
    'JWT tokens are self-contained, stateless, and scalable but cannot be revoked before expiration',
    'PKCE (Proof Key for Code Exchange) secures OAuth flows for mobile apps and SPAs',
    'SCIM automates user provisioning and deprovisioning across systems',
    'JIT provisioning creates users on first login but does NOT handle deprovisioning',
    'Always use stable identifiers (sub/NameID) as primary key, not email',
    'Digital signatures prevent token forgery - always verify signatures',
    'Identity providers: Okta (enterprise), Auth0 (developer-friendly), Azure AD (Microsoft), Keycloak (open-source)',
    'For enterprise security, use SCIM for instant deprovisioning when employees leave',
    'Modern trend: OIDC replacing SAML for new applications',
  ],
  learningObjectives: [
    'Understand the difference between authentication and authorization',
    'Explain SSO architecture and the roles of IdP and SP',
    'Describe the complete SAML SSO flow (SP-initiated and IdP-initiated)',
    'Understand SAML security mechanisms (signatures, replay prevention)',
    'Explain OAuth 2.0 authorization flows (Authorization Code, PKCE, Client Credentials)',
    'Understand OIDC and the role of ID tokens in authentication',
    'Implement JWT token validation and understand JWT structure',
    'Compare different identity providers (Okta, Auth0, Azure AD, Keycloak)',
    'Design and implement SCIM provisioning and deprovisioning',
    'Implement secure JIT provisioning with proper attribute mapping',
    'Build multi-tenant authentication supporting customer IdPs',
    'Design secure offboarding processes for enterprise applications',
  ],
  sections: [
    {
      ...authenticationfundamentalsSection,
      quiz: authenticationfundamentalsQuiz,
      multipleChoice: authenticationfundamentalsMultipleChoice,
    },
    {
      ...samldeepdiveSection,
      quiz: samldeepdiveQuiz,
      multipleChoice: samldeepdiveMultipleChoice,
    },
    {
      ...oauth2Section,
      quiz: oauth2Quiz,
      multipleChoice: oauth2MultipleChoice,
    },
    {
      ...oidcjwtSection,
      quiz: oidcjwtQuiz,
      multipleChoice: oidcjwtMultipleChoice,
    },
    {
      ...idpscimjitSection,
      quiz: idpscimjitQuiz,
      multipleChoice: idpscimjitMultipleChoice,
    },
  ],
};

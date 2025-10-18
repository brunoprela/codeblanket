/**
 * Quiz questions for Authentication Fundamentals section
 */

export const authenticationfundamentalsQuiz = [
  {
    id: 'q1',
    question:
      'Explain why SSO improves security despite creating a single point of failure.',
    sampleAnswer:
      "SSO improves security by centralizing security controls. Instead of 100 apps with varying password policies and security practices, you have ONE highly-secured IdP. Benefits: (1) Enforce strong MFA at IdP - covers all apps automatically. (2) Instant deprovisioning - employee leaves, one click disables all access (vs. manually disabling 100 accounts). (3) Strong password policy enforced once. (4) Security team can focus resources on hardening the IdP. (5) Centralized monitoring and anomaly detection. (6) Reduces password reuse (users don't need 100 passwords). While IdP is a single point of failure, it's a HARDENED single point with: redundancy, monitoring, MFA, zero-trust controls. The risk is mitigated by making the IdP extremely secure and highly available (multi-region deployment). Alternative is 100 apps with inconsistent security, which is objectively less secure.",
    keyPoints: [
      'Centralized security controls in one hardened IdP',
      'Enforce strong MFA once, covers all apps',
      'Instant deprovisioning on employee termination',
      'Centralized monitoring and anomaly detection',
      'Risk mitigated through HA and strong IdP security',
    ],
  },
  {
    id: 'q2',
    question:
      'How does SSO improve the user experience in enterprise environments?',
    sampleAnswer:
      'SSO dramatically improves UX: (1) One password instead of dozens - users remember one strong password vs. 50 weak passwords written on sticky notes. (2) Single MFA device - approve one push notification vs. multiple SMS codes. (3) Seamless access - click link to Salesforce, already logged in via SSO session. (4) Reduced password reset tickets - forgot one password vs. forgot 10 passwords. (5) Consistent login experience across all apps. (6) Faster onboarding - new employee gets access to all apps by being added to IdP once. (7) Mobile-friendly - modern OIDC works great on phones. Real impact: Studies show SSO reduces password-related help desk tickets by 50%+ and improves employee productivity. Users spend less time logging in, more time working.',
    keyPoints: [
      'One password instead of dozens',
      'Single MFA device for all apps',
      'Seamless access without repeated logins',
      'Faster onboarding for new employees',
      '50%+ reduction in password reset tickets',
    ],
  },
  {
    id: 'q3',
    question:
      "What happens when an employee leaves a company that uses SSO vs. one that doesn't?",
    sampleAnswer:
      "WITH SSO: Admin disables user account in IdP (Okta, Auth0) → User immediately loses access to ALL integrated applications (Salesforce, Slack, GitHub, AWS, etc.) in real-time. One action, complete coverage. WITHOUT SSO: Admin must manually disable accounts in each application: (1) Disable AD account. (2) Remove from Salesforce. (3) Remove from Slack. (4) Revoke GitHub access. (5) Disable AWS IAM user... (6-50) 45 more apps. This takes hours or days, creating a security window where ex-employees still have access. Often accounts are missed. Security nightmare! Real incident: Ex-employee at Capital One retained AWS access after termination, leading to major data breach. SSO prevents this by centralizing access control. This instant deprovisioning is one of SSO's biggest security benefits.",
    keyPoints: [
      'SSO: One action disables all app access instantly',
      'Non-SSO: Manual process across dozens of apps',
      'Security window where ex-employees retain access',
      'Real breaches from incomplete offboarding',
      'Instant deprovisioning is major security benefit',
    ],
  },
];

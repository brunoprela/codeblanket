/**
 * Quiz questions for SAML (Security Assertion Markup Language) section
 */

export const samldeepdiveQuiz = [
  {
    id: 'q1',
    question:
      'Explain the complete SP-initiated SAML SSO flow and the role of each component.',
    sampleAnswer:
      "SP-initiated flow: (1) User visits Service Provider (e.g., salesforce.com) without being authenticated. (2) SP generates SAML AuthnRequest - an XML document requesting authentication from IdP. (3) SP redirects browser to IdP with AuthnRequest encoded in URL (HTTP Redirect binding) or form (HTTP POST binding). (4) Browser presents AuthnRequest to IdP. (5) IdP checks if user already has SSO session - if yes, skip login; if no, show login form. (6) User authenticates at IdP (password + MFA). (7) IdP generates SAML Response containing signed SAML Assertion with user identity and attributes. (8) IdP redirects browser back to SP (via HTTP POST) with SAML Response. (9) SP's Assertion Consumer Service (ACS) endpoint receives SAML Response. (10) SP validates: signature using IdP public key, timestamps (NotBefore/NotOnOrAfter), audience matches SP, assertion ID not replayed. (11) SP extracts user identity and attributes from assertion. (12) SP creates local session, sets session cookie. (13) User is logged in to SP! Key insight: User never entered credentials at SP. SP completely trusts IdP's signed assertion. This is the foundation of SSO.",
    keyPoints: [
      'User visits SP, SP generates AuthnRequest',
      'SP redirects to IdP with AuthnRequest',
      'User authenticates at IdP (password + MFA)',
      'IdP generates signed SAML assertion',
      'SP validates assertion signature and creates session',
    ],
  },
  {
    id: 'q2',
    question:
      'What are SAML attributes and how are they used in enterprise SSO scenarios?',
    sampleAnswer:
      'SAML attributes are name-value pairs included in SAML assertions that convey user information from IdP to SP. Common attributes: email (john@company.com), firstName, lastName, groups (["Engineering", "Admins"]), employeeId, role, department, manager. How they\'re used: (1) User Matching: SP uses email or employeeId to match SAML identity to local user account. (2) JIT Provisioning: SP creates new user account automatically using attributes (firstName, lastName, email, role). (3) Authorization: SP grants permissions based on group or role attribute (if role=admin, grant admin access). (4) Profile Updates: SP updates user profile when attributes change in IdP (employee changes department). (5) Audit Logging: SP records attributes for compliance. Example: Employee logs into Salesforce via SAML. Assertion includes role=Sales_Manager. Salesforce maps this to local "Sales Manager" profile, granting appropriate permissions. When employee is promoted to VP, IdP admin updates role. Next login, new SAML assertion has role=VP, Salesforce auto-updates permissions. This eliminates manual provisioning/deprovisioning in each app.',
    keyPoints: [
      'Name-value pairs conveying user info from IdP to SP',
      'Used for user matching and JIT provisioning',
      'Enable authorization based on groups/roles',
      'Auto-update user profiles when attributes change',
      'Eliminate manual provisioning in each app',
    ],
  },
  {
    id: 'q3',
    question:
      'Why is SAML still dominant in enterprises despite being older and more complex than OIDC?',
    sampleAnswer:
      'SAML remains dominant in enterprises for several reasons: (1) Legacy Applications: Thousands of enterprise SaaS apps (Salesforce, Workday, ServiceNow, SAP) have supported SAML for 10-20 years. Switching is costly. (2) Investment Protection: Enterprises invested millions in SAML infrastructure (AD FS, Ping, Okta SAML). It works reliably. (3) Feature Maturity: SAML has 20+ years of enterprise features: attribute-based access control, delegation, complex attribute mapping. (4) Compliance Certifications: Many SAML implementations are certified for SOC 2, HIPAA, FedRAMP. Re-certification is expensive. (5) Vendor Support: Enterprise IdPs (Okta, Ping, Microsoft) maintain strong SAML support because customers require it. (6) Network Effects: Once all your apps use SAML, new apps must support SAML to fit ecosystem. (7) Risk Aversion: Enterprises are conservative - "nobody got fired for choosing SAML." That said, OIDC is gaining: new SaaS startups support OIDC first, mobile apps require OIDC, modern IdPs (Auth0) push OIDC.',
    keyPoints: [
      'Legacy app support - 10-20 years of SAML integration',
      'Massive investment in SAML infrastructure',
      '20+ years of mature enterprise features',
      'Compliance certifications (SOC 2, HIPAA, FedRAMP)',
      'Network effects - ecosystem lock-in',
    ],
  },
];

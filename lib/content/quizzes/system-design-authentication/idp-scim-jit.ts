/**
 * Quiz questions for Identity Providers, SCIM & JIT Provisioning section
 */

export const idpscimjitQuiz = [
  {
    id: 'q1',
    question:
      'Compare SCIM and JIT provisioning. When would you use each, and why?',
    sampleAnswer:
      'SCIM vs JIT comparison: SCIM (System for Cross-domain Identity Management): (1) Provisioning: Users created BEFORE first login. IdP pushes users to app via SCIM API. (2) Deprovisioning: IdP actively deactivates users when they leave. SCIM PATCH sets active:false. (3) Synchronization: User attribute changes (name, role, email) sync in real-time. (4) Security: Strong - immediate deprovisioning. (5) Complexity: Higher - app must implement SCIM endpoints, IdP must support SCIM client. (6) Best for: Enterprise apps, security-critical systems, apps requiring pre-provisioning. JIT (Just-In-Time): (1) Provisioning: Users created ON first login. App extracts attributes from SAML assertion / OIDC ID token. (2) Deprovisioning: ❌ None. User disabled at IdP cannot log in (SSO fails), but local account persists. (3) Synchronization: Attributes update on each login (not real-time). (4) Security: Weaker - orphaned accounts exist. (5) Complexity: Lower - just parse SSO attributes. (6) Best for: Internal tools, B2C apps, rapid SSO implementation. Use SCIM for enterprise B2B SaaS and security-sensitive apps. Use JIT for B2C apps and rapid MVP implementation.',
    keyPoints: [
      'SCIM: Pre-provisioning + real-time sync + active deprovisioning',
      'JIT: On-login provisioning, no deprovisioning',
      'SCIM best for enterprise security-critical apps',
      'JIT best for B2C apps and rapid implementation',
      'Many apps use hybrid: JIT provision + SCIM deprovision',
    ],
  },
  {
    id: 'q2',
    question:
      'Design a comprehensive offboarding process for an enterprise using Okta as IdP with 50+ integrated applications.',
    sampleAnswer:
      'Enterprise offboarding with Okta: AUTOMATED FLOW: (1) HR marks employee terminated in HRIS (Workday). (2) Workday pushes termination to Okta via API. (3) Okta triggers deactivation workflow: Terminates all active SSO sessions, Deactivates Okta account, Pushes SCIM deactivation to 40/50 apps with SCIM (PATCH /scim/v2/Users/{id} {active:false}), Revokes all OAuth/OIDC tokens, Calls webhook to on-premise systems (Active Directory, VPN, Wi-Fi). (4) Secondary systems: Email converted to shared mailbox, Mobile device wiped via MDM, Physical badge deactivated. (5) Manual cleanup for 10 apps without SCIM. (6) Okta generates audit report: Apps deprovisioned, Sessions terminated, Timestamp of each action. RESULTS: < 1 minute for automated deprovisioning, comprehensive audit trail, no orphaned accounts. Before Okta/SCIM: 2-3 days, manual, accounts missed. Key success factors: SCIM integration (90%+ apps), automation, audit trail, exception handling, regular testing.',
    keyPoints: [
      'HRIS triggers Okta → Okta pushes to all apps via SCIM',
      'Terminates sessions, deactivates accounts, revokes tokens',
      'Automated deprovisioning in < 1 minute',
      'Comprehensive audit trail for compliance',
      'Manual process for apps without SCIM',
    ],
  },
  {
    id: 'q3',
    question:
      'You are building a B2B SaaS product. How would you implement multi-tenant authentication supporting both your own IdP and customer IdPs (Okta, Azure AD)?',
    sampleAnswer:
      "Multi-tenant B2B SaaS auth: ARCHITECTURE: Database tables: tenants (id, name, domain), idp_connections (id, tenant_id, protocol, metadata_url, client_id), users (id, email, tenant_id, external_id). IDP DISCOVERY: User enters email → Extract domain → Query tenants by domain → Find tenant's IdP connection → Redirect to customer IdP. MULTI-TENANT OIDC: Store per-tenant: authorization_endpoint, token_endpoint, jwks_uri, client_secret. User logs in → Discover tenant → Redirect to tenant's Okta → Okta callback → Exchange code using tenant's client_secret → Validate ID token using tenant's JWKS → Extract sub → Create/lookup user with tenant_id → Create session with tenant_id. TENANT ISOLATION: Every query includes tenant_id filter. Session stores tenant_id. Middleware enforces tenant isolation. CUSTOMER SELF-SERVICE: Admin portal for customers to configure SSO. For SAML: Upload IdP metadata. For OIDC: Provide discovery URL + client credentials. Test SSO button. SCIM: Generate per-tenant SCIM token. Implement SCIM endpoints: /scim/v2/{tenant_id}/Users. Customer configures Okta SCIM with base URL + token. Example: Slack - each workspace has own IdP.",
    keyPoints: [
      'Store per-tenant IdP config in database',
      'IdP discovery: email domain → tenant → IdP',
      'Tenant isolation: filter all queries by tenant_id',
      'Customer self-service admin portal for SSO config',
      'SCIM with per-tenant endpoints and tokens',
    ],
  },
];

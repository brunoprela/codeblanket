/**
 * Authentication Fundamentals Section
 */

export const authenticationfundamentalsSection = {
  id: 'authentication-fundamentals',
  title: 'Authentication Fundamentals',
  content: `Authentication is the process of verifying **who you are**. It\'s one of the most critical aspects of system design and security.

## Authentication vs Authorization

**Authentication (AuthN)**: *Who are you?*
- Proves identity
- Username/password, biometrics, tokens
- Answers: "Is this person who they claim to be?"

**Authorization (AuthZ)**: *What can you do?*
- Determines permissions
- Roles, access control lists (ACLs)
- Answers: "Does this person have permission to do X?"

**Example**: 
- Authentication: You log into Salesforce with your email/password → Proven you're John Smith
- Authorization: Salesforce checks if John Smith can view customer records → Permission granted

---

## Traditional Authentication Flow

### 1. Username/Password (Basic Auth)

**How it works**:
1. User submits credentials (username + password)
2. Server validates against database
3. Server creates session, returns session ID cookie
4. Browser sends cookie with subsequent requests
5. Server validates session on each request

\`\`\`
User                    Server                  Database
 |                        |                        |
 |-- POST /login -------->|                        |
 |   {user, pass}         |                        |
 |                        |-- Check credentials -->|
 |                        |<-- User valid ---------|
 |                        |                        |
 |                        |-- Create session ----->|
 |<-- Set-Cookie ---------|<-- Session ID ---------|
 |   sessionId=abc123     |                        |
 |                        |                        |
 |-- GET /profile ------->|                        |
 |   Cookie: abc123       |-- Verify session ----->|
 |                        |<-- Session valid ------|
 |<-- Profile data -------|                        |
\`\`\`

**Problems at scale**:
- ❌ Sessions stored on single server (doesn't scale horizontally)
- ❌ Every request requires database lookup
- ❌ Doesn't work across multiple applications
- ❌ Password fatigue (users have 100+ accounts)

---

## The SSO Revolution

**Single Sign-On (SSO)**: Log in once, access multiple applications.

**Why SSO Matters**:
- ✅ **User experience**: One password instead of dozens
- ✅ **Security**: Centralized security controls, MFA in one place
- ✅ **IT efficiency**: One place to manage users, instant deprovisioning
- ✅ **Compliance**: Centralized audit logs

**Real-world example**: You log into Google once, then access Gmail, YouTube, Drive, Calendar without logging in again.

**Enterprise example**: Log into Okta once, access Salesforce, Slack, GitHub, AWS, Zoom without re-authenticating.

---

## SSO Architecture Components

### Identity Provider (IdP)

The **authentication authority** that knows who users are.

**Responsibilities**:
- Store user identities (directory)
- Authenticate users (verify passwords, MFA)
- Issue authentication assertions/tokens
- Maintain session state

**Examples**: 
- **Enterprise**: Okta, Auth0, Microsoft Entra ID (Azure AD), Ping Identity
- **Social**: Google, Facebook, GitHub (for "Sign in with Google")
- **On-premise**: Active Directory + ADFS

### Service Provider (SP)

The **application** users want to access.

**Responsibilities**:
- Trust the IdP
- Validate authentication tokens/assertions
- Grant access based on IdP assertions
- Does NOT store passwords!

**Examples**: Salesforce, Slack, AWS Console, Gmail, your company's internal apps

### Key Insight

**With SSO**: SP delegates authentication to IdP. SP trusts IdP to verify identity.

\`\`\`
Before SSO:
User → App A (password 1)
User → App B (password 2)  
User → App C (password 3)
→ 3 passwords to remember, 3 places to hack

With SSO:
User → IdP (one password + MFA)
      ↓
    [Token]
      ↓
App A, App B, App C (trust IdP's token)
→ 1 password, 1 place to secure, 1 MFA device
\`\`\`

---

## Authentication Flows

### Browser-based SSO Flow

1. **User accesses SP**: User clicks "Sign in to Salesforce"
2. **SP redirects to IdP**: Salesforce redirects to Okta login
3. **User authenticates at IdP**: User enters credentials at Okta, does MFA
4. **IdP creates token**: Okta creates signed assertion "User X is authenticated"
5. **IdP redirects back to SP**: Browser redirected to Salesforce with token
6. **SP validates token**: Salesforce validates Okta\'s signature
7. **SP grants access**: Salesforce creates session, user is logged in

**Key point**: User never enters password at Salesforce. Salesforce trusts Okta's assertion.

---

## SSO Protocols

There are three main protocols for implementing SSO:

### 1. SAML (Security Assertion Markup Language)
- **Age**: Created in 2001, mature and stable
- **Format**: XML-based
- **Use case**: Enterprise B2B SSO
- **Best for**: Traditional enterprise applications
- **Example**: "Sign in with corporate credentials"

### 2. OAuth 2.0
- **Age**: Standardized in 2012
- **Format**: JSON-based
- **Use case**: **Authorization** delegated access
- **Best for**: API access, third-party app permissions
- **Example**: "Allow Spotify to access your Facebook friends"

### 3. OpenID Connect (OIDC)
- **Age**: Standardized in 2014
- **Format**: JSON-based, built on OAuth 2.0
- **Use case**: **Authentication** modern SSO
- **Best for**: Modern web/mobile apps
- **Example**: "Sign in with Google"

---

## When to Use Which?

| Scenario | Protocol | Why |
|----------|----------|-----|
| Employee accessing enterprise apps | SAML | Legacy support, mature enterprise features |
| Consumer "Sign in with Google" | OIDC | Modern, mobile-friendly, simple |
| App accessing API on user's behalf | OAuth 2.0 | Designed for authorization/delegation |
| New modern B2B SaaS | OIDC | Better dev experience than SAML |
| Mobile app authentication | OIDC | Native support, better security |

**Modern trend**: OIDC is replacing SAML for new implementations. SAML still dominant in enterprises due to existing integrations.

---

## Security Considerations

### Why SSO is More Secure

1. **Centralized MFA**: One place to enforce multi-factor authentication
2. **Instant deprovisioning**: Employee leaves → one click disables all access
3. **Password policies**: Enforce strong passwords in one place
4. **Audit logging**: All authentications logged centrally
5. **Security team focus**: Secure one IdP instead of 100 apps

### SSO Risks

1. **Single point of failure**: IdP down = all apps inaccessible
2. **Master key problem**: IdP compromise = all apps compromised
3. **Session management**: Long-lived SSO sessions can be risky

**Mitigation**: 
- High availability for IdP (multi-region)
- Strong IdP security (MFA, monitoring, zero-trust)
- Adaptive authentication (re-auth for sensitive operations)
- Session timeouts and continuous verification

---

## Real-World Architecture

### Typical Enterprise Setup

\`\`\`
                     ┌─────────────────┐
                     │   Identity      │
                     │   Provider      │
                     │   (Okta)        │
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              │               │               │
      ┌───────▼─────┐  ┌──────▼──────┐  ┌────▼──────┐
      │  Salesforce │  │    Slack    │  │   AWS     │
      │     (SP)    │  │    (SP)     │  │   (SP)    │
      └─────────────┘  └─────────────┘  └───────────┘

User logs in once to Okta → Accesses all SPs
\`\`\`

### Integration Requirements

**IdP Requirements**:
- User directory (LDAP, Active Directory, or native)
- Authentication methods (password, MFA, biometrics)
- Token signing keys (cryptographic)
- Metadata endpoint (public key, configuration)

**SP Requirements**:
- Trust configuration (IdP metadata)
- Token validation (verify signatures)
- Session management (local sessions after SSO)
- Attribute mapping (IdP fields → app fields)

---

## Key Takeaways

1. **SSO** = Single Sign-On = One login for many apps
2. **IdP** = Knows who you are, issues tokens
3. **SP** = App you want to use, trusts IdP
4. **Three protocols**: SAML (enterprise), OAuth 2.0 (authorization), OIDC (modern auth)
5. **Security**: SSO is more secure when done right (centralized MFA, instant deprovisioning)
6. **Trade-off**: Convenience vs. single point of failure (mitigate with HA and strong IdP security)`,
};

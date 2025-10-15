import { Module } from '@/lib/types';

export const systemDesignAuthenticationModule: Module = {
  id: 'system-design-authentication',
  title: 'System Design: Authentication & SSO',
  description:
    'Master authentication concepts including SSO, SAML, OAuth, OIDC, JWT, identity providers, and modern authentication patterns',
  icon: '🔐',
  category: 'System Design',
  difficulty: 'Medium',
  estimatedTime: '2-3 hours',
  sections: [
    {
      id: 'authentication-fundamentals',
      title: 'Authentication Fundamentals',
      content: `Authentication is the process of verifying **who you are**. It's one of the most critical aspects of system design and security.

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
6. **SP validates token**: Salesforce validates Okta's signature
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
      quiz: [
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
      ],
      multipleChoice: [
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
          question:
            'In SSO architecture, what does the Service Provider (SP) do?',
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
          question:
            'Which protocol is best for modern mobile app authentication?',
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
      ],
    },
    {
      id: 'saml-deep-dive',
      title: 'SAML (Security Assertion Markup Language)',
      content: `SAML is the **veteran** of SSO protocols. Created in 2001, it's XML-based and remains the dominant protocol in enterprise B2B scenarios.

## What is SAML?

**SAML**: Security Assertion Markup Language

**Purpose**: Enable SSO by allowing IdP to pass authentication and authorization assertions to SP.

**Key Concept**: IdP creates a digitally-signed XML message (assertion) that says "User X is authenticated" and SP trusts it.

---

## SAML Components

### 1. Assertions

XML documents containing authentication and authorization statements.

**Three types**:

1. **Authentication Assertion**: "User successfully authenticated at time T"
2. **Attribute Assertion**: "User has email=john@company.com, role=admin"
3. **Authorization Decision Assertion**: "User is allowed to access Resource Y"

**Example SAML Assertion** (simplified):

\`\`\`xml
<saml:Assertion>
  <saml:Issuer>https://idp.okta.com</saml:Issuer>
  <saml:Subject>
    <saml:NameID>john.smith@company.com</saml:NameID>
  </saml:Subject>
  <saml:Conditions NotBefore="2024-01-15T10:00:00Z" 
                   NotOnOrAfter="2024-01-15T10:05:00Z"/>
  <saml:AuthnStatement AuthnInstant="2024-01-15T10:00:00Z"/>
  <saml:AttributeStatement>
    <saml:Attribute Name="email">
      <saml:AttributeValue>john.smith@company.com</saml:AttributeValue>
    </saml:Attribute>
    <saml:Attribute Name="role">
      <saml:AttributeValue>admin</saml:AttributeValue>
    </saml:Attribute>
  </saml:AttributeStatement>
  <Signature>...</Signature>  <!-- Digital signature -->
</saml:Assertion>
\`\`\`

### 2. Protocols

Define how SAML messages are exchanged.

**Key protocols**:
- **Authentication Request Protocol**: SP requests authentication
- **Single Logout Protocol**: Logout from all SPs at once

### 3. Bindings

Define HOW SAML messages are transported.

**Common bindings**:
- **HTTP Redirect**: SAML message in URL query parameter (for small messages)
- **HTTP POST**: SAML message in HTML form (for larger messages)
- **HTTP Artifact**: Reference instead of full message (most secure)

### 4. Profiles

Define complete use cases combining assertions, protocols, and bindings.

**Most important**: **Web Browser SSO Profile** - the standard browser-based SSO flow.

---

## SAML SSO Flow (SP-Initiated)

Most common flow: User starts at Service Provider.

### Step-by-Step

**Setup** (one-time):
- SP and IdP exchange metadata (certificates, endpoints)
- Trust relationship established

**Runtime flow**:

1. **User accesses SP**: User visits \`https://salesforce.com\`

2. **SP generates AuthnRequest**: Salesforce creates SAML authentication request
   \`\`\`xml
   <samlp:AuthnRequest ID="_abc123" IssueInstant="2024-01-15T10:00:00Z">
     <saml:Issuer>https://salesforce.com</saml:Issuer>
     <samlp:NameIDPolicy Format="email"/>
   </samlp:AuthnRequest>
   \`\`\`

3. **SP redirects to IdP**: Browser redirected to Okta with AuthnRequest
   \`\`\`
   https://idp.okta.com/sso?SAMLRequest=<base64_encoded_request>
   \`\`\`

4. **User authenticates at IdP**: User enters username/password, completes MFA

5. **IdP generates SAML Response**: Okta creates signed assertion
   \`\`\`xml
   <samlp:Response>
     <saml:Assertion>
       <saml:Subject>john@company.com</saml:Subject>
       <saml:AttributeStatement>...</saml:AttributeStatement>
       <Signature>...</Signature>
     </saml:Assertion>
   </samlp:Response>
   \`\`\`

6. **IdP redirects to SP**: Browser redirected to Salesforce with SAML Response (HTTP POST)

7. **SP validates assertion**: 
   - Verify signature using IdP's public key
   - Check NotBefore and NotOnOrAfter times
   - Verify Audience matches SP
   - Check assertion is not replayed

8. **SP creates session**: Salesforce creates local session, user is logged in!

### Sequence Diagram

\`\`\`
User Browser          Service Provider          Identity Provider
     |                      |                           |
     |--- (1) Access App -->|                           |
     |                      |                           |
     |                      |--- (2) Generate --------->|
     |                      |    AuthnRequest           |
     |<--- (3) Redirect ----|                           |
     |                      |                           |
     |-------- (4) Present AuthnRequest --------------->|
     |                                                   |
     |<------- (5) Login Page --------------------------|
     |                                                   |
     |-------- (6) Credentials + MFA ------------------>|
     |                                                   |
     |                                  [Verify Identity]|
     |                                                   |
     |<------- (7) SAML Response (POST) ----------------|
     |       + signed assertion                         |
     |                                                   |
     |--- (8) Submit SAML Response -->|                 |
     |                                |                 |
     |                     [Validate signature, times]  |
     |                                |                 |
     |<--- (9) Access Granted --------|                 |
     |    + session cookie            |                 |
\`\`\`

---

## IdP-Initiated Flow

Alternative flow: User starts at Identity Provider.

**Flow**:
1. User logs into IdP portal (e.g., Okta dashboard)
2. User clicks app tile (e.g., "Salesforce")
3. IdP generates SAML Response (no AuthnRequest needed)
4. IdP redirects browser to SP with assertion
5. SP validates and grants access

**Use case**: Enterprise portals where users pick apps from dashboard.

**Security note**: SP-initiated is more secure (can verify AuthnRequest ID), but IdP-initiated is more convenient.

---

## SAML Security

### Digital Signatures

**Critical feature**: Assertions are digitally signed by IdP.

**How it works**:
1. IdP has private key (secret)
2. IdP has public key (shared in metadata)
3. IdP signs assertion with private key
4. SP verifies signature with public key

**This proves**: 
- Assertion came from IdP (authenticity)
- Assertion wasn't modified (integrity)

**Attack prevented**: Man-in-the-middle cannot forge assertions without IdP's private key.

### Replay Attack Prevention

**Problem**: Attacker captures valid SAML Response, replays it later.

**Mitigations**:
1. **Assertion ID**: Each assertion has unique ID, SP tracks used IDs
2. **Timestamps**: \`NotBefore\` and \`NotOnOrAfter\` limit validity window (typically 5 minutes)
3. **Audience restriction**: Assertion specifies which SP it's for
4. **TLS**: Encrypt communication (HTTPS)

### XML Signature Wrapping

**Attack**: Manipulate XML structure to bypass signature verification.

**Example**: Attacker wraps signed assertion in malicious assertion, some parsers only check outer assertion.

**Mitigation**: Use robust SAML libraries (don't roll your own parser!), validate structure carefully.

---

## SAML Metadata

Configuration document exchanged between IdP and SP.

**IdP Metadata** (SP needs this):
\`\`\`xml
<EntityDescriptor entityID="https://idp.okta.com">
  <IDPSSODescriptor>
    <KeyDescriptor use="signing">
      <KeyInfo>
        <X509Data>
          <X509Certificate>MII...</X509Certificate>  <!-- Public key -->
        </X509Data>
      </KeyInfo>
    </KeyDescriptor>
    <SingleSignOnService Binding="HTTP-POST"
                         Location="https://idp.okta.com/sso"/>
  </IDPSSODescriptor>
</EntityDescriptor>
\`\`\`

**SP Metadata** (IdP needs this):
\`\`\`xml
<EntityDescriptor entityID="https://salesforce.com">
  <SPSSODescriptor>
    <AssertionConsumerService Binding="HTTP-POST"
                              Location="https://salesforce.com/saml/acs"
                              index="0"/>
  </SPSSODescriptor>
</EntityDescriptor>
\`\`\`

---

## SAML Attributes

IdP can pass user attributes to SP in assertions.

**Common attributes**:
- Email: \`user@company.com\`
- First name / Last name
- Groups: \`["Engineering", "Admins"]\`
- Employee ID: \`12345\`
- Role: \`admin\`, \`user\`

**Attribute Mapping**: SP maps SAML attributes to local user fields.

**Example**: 
- SAML: \`<Attribute Name="firstName">John</Attribute>\`
- SP maps to: \`user.first_name = "John"\`

**Just-In-Time (JIT) Provisioning**: SP automatically creates user account from SAML attributes on first login. No pre-provisioning needed!

---

## Single Logout (SLO)

**Problem**: User logs out of one app, but still logged into others.

**SAML SLO**: Logout from IdP → IdP notifies all SPs → User logged out everywhere.

**Flow**:
1. User clicks logout in SP
2. SP sends \`LogoutRequest\` to IdP
3. IdP sends \`LogoutRequest\` to all other SPs with active sessions
4. Each SP terminates local session
5. IdP terminates SSO session
6. User redirected to logout confirmation page

**Challenge**: If any SP is down, SLO might fail (user still logged in there).

**Real-world**: Many implementations skip SLO due to complexity. Instead, use short session timeouts.

---

## SAML vs. OAuth/OIDC

| Aspect | SAML | OAuth 2.0 | OIDC |
|--------|------|-----------|------|
| **Format** | XML | JSON | JSON |
| **Primary use** | SSO | Authorization/Delegation | SSO |
| **Age** | 2001 (23 years) | 2012 (12 years) | 2014 (10 years) |
| **Mobile support** | Poor | Good | Excellent |
| **Complexity** | High | Medium | Low |
| **Enterprise adoption** | Very high | High | Growing |
| **Developer experience** | Difficult | Good | Great |
| **Token type** | SAML Assertion (XML) | Access Token | ID Token (JWT) |
| **Best for** | Enterprise B2B | API access | Modern SSO |

---

## When to Use SAML

✅ **Use SAML when**:
- Integrating with enterprise applications (they probably only support SAML)
- Customers require SAML (common in enterprise sales)
- Existing SAML infrastructure
- Need mature, battle-tested protocol

❌ **Avoid SAML for**:
- Mobile apps (use OIDC)
- New projects with choice (OIDC is easier)
- API authorization (use OAuth 2.0)
- Consumer applications (use OIDC)

---

## SAML Implementation Checklist

### For Service Providers (SP)

- [ ] Generate SP metadata
- [ ] Obtain IdP metadata
- [ ] Validate SAML assertion signatures
- [ ] Check NotBefore and NotOnOrAfter timestamps
- [ ] Implement replay attack prevention (track assertion IDs)
- [ ] Map SAML attributes to user fields
- [ ] Handle JIT provisioning (if needed)
- [ ] Implement session management
- [ ] Test with multiple IdPs
- [ ] Use established SAML library (don't roll your own!)

### For Identity Providers (IdP)

- [ ] Generate IdP metadata
- [ ] Securely store signing certificate private key
- [ ] Implement authentication (username/password, MFA)
- [ ] Generate and sign SAML assertions
- [ ] Support SP-initiated flow
- [ ] Support IdP-initiated flow (optional)
- [ ] Implement Single Logout (optional)
- [ ] Handle attribute release policies
- [ ] Monitor for suspicious activity

---

## Common SAML Errors

**"Invalid signature"**: SP cannot verify IdP signature
- **Cause**: Wrong public key, expired certificate, signature wrapping attack
- **Fix**: Ensure IdP metadata is up-to-date

**"Assertion expired"**: Timestamps outside validity window
- **Cause**: Clock skew between IdP and SP
- **Fix**: Sync clocks with NTP, allow clock skew tolerance (e.g., ±5 minutes)

**"Audience restriction"**: Assertion not intended for this SP
- **Cause**: Wrong EntityID in SP configuration
- **Fix**: Ensure SP EntityID matches assertion Audience

**"No assertion found"**: SAML Response doesn't contain assertion
- **Cause**: IdP error, authentication failed
- **Fix**: Check IdP logs

---

## Real-World SAML Scenario

**Scenario**: Acme Corp uses Okta (IdP) for SSO. Employees access Salesforce (SP) via SAML.

**Setup**:
1. Okta admin configures Salesforce app in Okta
2. Okta admin downloads Salesforce SP metadata
3. Salesforce admin uploads Okta IdP metadata to Salesforce
4. Salesforce admin enables SAML SSO
5. Test user logs in to verify

**Daily use**:
1. Employee visits \`acme.salesforce.com\`
2. Salesforce redirects to Okta (\`acme.okta.com\`)
3. Employee enters Okta password + approves Okta Verify push notification (MFA)
4. Okta generates signed SAML assertion with employee email and role
5. Okta redirects to Salesforce with assertion
6. Salesforce validates signature, creates session
7. Employee accesses Salesforce without ever entering Salesforce password

**Benefits**:
- IT admin disables Okta account → employee loses Salesforce access instantly
- Employee only remembers one password (Okta)
- MFA enforced at Okta, covers Salesforce automatically
- Audit log in Okta shows all Salesforce logins`,
      quiz: [
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
      ],
      multipleChoice: [
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
      ],
    },
    {
      id: 'oauth2',
      title: 'OAuth 2.0 - Authorization Framework',
      content: `OAuth 2.0 is **NOT an authentication protocol** - it's an **authorization framework**. This is the most important distinction to understand.

## OAuth 2.0 Purpose

**OAuth 2.0**: Allows users to grant third-party applications **limited access** to their resources without sharing passwords.

**Key concept**: **Delegated authorization** - "I authorize App B to access my data in App A"

**Example**: "Allow Spotify to see your Facebook friends"
- You don't give Spotify your Facebook password
- You grant Spotify limited, specific access
- You can revoke access anytime
- Facebook never tells Spotify your password

---

## The Problem OAuth Solves

### Before OAuth

**Scenario**: You want to print your Gmail contacts using a printing service.

**Old approach**:
1. Give printing service your Gmail username and password
2. Printing service logs into Gmail as you
3. Printing service accesses ALL your Gmail data

**Problems**:
- ❌ Password sharing (massive security risk)
- ❌ Printing service has full access (could read emails, send emails, delete account)
- ❌ Can't revoke access without changing password (breaks legitimate apps)
- ❌ No way to know what printing service is doing with your account

### With OAuth 2.0

1. Printing service redirects you to Gmail
2. You log into Gmail (printing service never sees password)
3. Gmail asks: "Allow Printing Service to read contacts?" (specific scope)
4. You approve
5. Gmail gives printing service an **access token** with limited permissions (read contacts only)
6. Printing service uses token to access contacts (cannot access emails)
7. You can revoke token anytime without changing password

**Benefits**:
- ✅ No password sharing
- ✅ Limited scope (read contacts only, not emails)
- ✅ Revocable (revoke printing service, keep other apps)
- ✅ Audit trail (Gmail logs what printing service accessed)

---

## OAuth 2.0 Roles

### 1. Resource Owner

The **user** who owns the data.

**Example**: You (owner of Facebook profile, Gmail contacts, GitHub repos)

### 2. Client

The **third-party application** requesting access.

**Example**: Spotify (wants to access Facebook), Printing Service (wants Gmail contacts), Netlify (wants GitHub repos)

**Types**:
- **Confidential clients**: Can keep secrets (web server apps)
- **Public clients**: Cannot keep secrets (mobile apps, SPAs)

### 3. Authorization Server

The server that **authenticates resource owner** and **issues access tokens**.

**Example**: Facebook OAuth server, Google OAuth server, GitHub OAuth server

**Note**: Often combined with Resource Server in same service.

### 4. Resource Server

The server that **hosts protected resources** and accepts access tokens.

**Example**: Facebook Graph API, Gmail API, GitHub API

---

## OAuth 2.0 Tokens

### Access Token

**Purpose**: Grants access to protected resources.

**Properties**:
- Opaque string (client doesn't need to understand it): \`ya29.a0AfH6SMB...\`
- Short-lived (typically 1 hour)
- Specific scope (read:contacts, write:posts)
- Presented to Resource Server with each request

**Usage**:
\`\`\`http
GET /api/contacts HTTP/1.1
Host: gmail.com
Authorization: Bearer ya29.a0AfH6SMB...
\`\`\`

**Security**: If stolen, attacker has limited access for limited time.

### Refresh Token

**Purpose**: Obtain new access tokens without user interaction.

**Properties**:
- Opaque string, longer than access token
- Long-lived (days, weeks, or indefinite)
- Must be kept highly secure
- Only for confidential clients

**Why needed**: Access tokens expire quickly for security. Refresh token lets client get new access token without bothering user.

**Usage**:
\`\`\`http
POST /oauth/token HTTP/1.1
Host: authorization-server.com
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token&
refresh_token=tGzv3JOkF0XG5Qx2TlKWIA&
client_id=s6BhdRkqt3
\`\`\`

**Response**:
\`\`\`json
{
  "access_token": "new_access_token_here",
  "token_type": "Bearer",
  "expires_in": 3600
}
\`\`\`

### Token Lifecycle

\`\`\`
User authorizes → Client gets Access Token (1hr) + Refresh Token
                     ↓
                Access Token used for API calls
                     ↓
                Access Token expires (1hr later)
                     ↓
                Client uses Refresh Token to get new Access Token
                     ↓
                Repeat until Refresh Token expires or revoked
\`\`\`

---

## OAuth 2.0 Grant Types (Flows)

Different flows for different scenarios.

### 1. Authorization Code Flow ⭐ Most Secure

**For**: Confidential clients (web apps with backend)

**Flow**:

1. **Client redirects to Authorization Server**:
   \`\`\`
   https://auth-server.com/oauth/authorize?
     response_type=code&
     client_id=abc123&
     redirect_uri=https://myapp.com/callback&
     scope=read:contacts&
     state=random_state_xyz
   \`\`\`

2. **User authenticates and approves**

3. **Authorization Server redirects to Client with code**:
   \`\`\`
   https://myapp.com/callback?code=AUTH_CODE&state=random_state_xyz
   \`\`\`

4. **Client exchanges code for tokens** (backend call):
   \`\`\`http
   POST /oauth/token HTTP/1.1
   Host: auth-server.com
   Content-Type: application/x-www-form-urlencoded

   grant_type=authorization_code&
   code=AUTH_CODE&
   client_id=abc123&
   client_secret=SECRET&
   redirect_uri=https://myapp.com/callback
   \`\`\`

5. **Authorization Server returns tokens**:
   \`\`\`json
   {
     "access_token": "eyJhbGciOi...",
     "refresh_token": "tGzv3JOk...",
     "token_type": "Bearer",
     "expires_in": 3600
   }
   \`\`\`

**Why secure**: 
- Authorization code is one-time use
- Access token never passes through browser
- Client must authenticate with secret to exchange code
- If code is intercepted, attacker can't use it without client secret

### 2. Authorization Code Flow with PKCE

**For**: Public clients (mobile apps, SPAs) - most common for modern apps

**PKCE** = Proof Key for Code Exchange

**Problem it solves**: Public clients can't keep \`client_secret\` safe (anyone can decompile mobile app or inspect SPA code).

**How PKCE works**:

1. **Client generates random \`code_verifier\`**: \`dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk\`

2. **Client hashes it to create \`code_challenge\`**: \`SHA256(code_verifier)\` → \`E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM\`

3. **Client sends \`code_challenge\` in auth request**:
   \`\`\`
   https://auth-server.com/oauth/authorize?
     response_type=code&
     client_id=abc123&
     code_challenge=E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM&
     code_challenge_method=S256
   \`\`\`

4. **Auth server stores \`code_challenge\` with authorization code**

5. **Client exchanges code + \`code_verifier\`**:
   \`\`\`http
   POST /oauth/token HTTP/1.1
   
   grant_type=authorization_code&
   code=AUTH_CODE&
   client_id=abc123&
   code_verifier=dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk
   \`\`\`

6. **Auth server verifies \`SHA256(code_verifier) == code_challenge\`**

7. **If match, returns tokens**

**Why secure**: Even if attacker intercepts authorization code, they don't have \`code_verifier\` (it never leaves client), so they can't exchange code for tokens.

### 3. Client Credentials Flow

**For**: Machine-to-machine (no user involved)

**Use case**: Backend service accessing API

**Flow**:
\`\`\`http
POST /oauth/token HTTP/1.1
Host: auth-server.com
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&
client_id=service_abc&
client_secret=SERVICE_SECRET&
scope=read:data
\`\`\`

**Response**:
\`\`\`json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "Bearer",
  "expires_in": 3600
}
\`\`\`

**Note**: No refresh token (client can request new access token anytime with credentials).

### 4. Implicit Flow (Deprecated ❌)

**Previously for**: SPAs (single-page apps)

**Flow**: Returns access token directly in URL fragment (no code exchange)

**Why deprecated**: 
- Access token exposed in browser history
- No refresh token
- Less secure than PKCE

**Modern replacement**: Authorization Code + PKCE

### 5. Resource Owner Password Credentials Flow (Avoid ❌)

**Flow**: Client collects username/password directly, sends to auth server

**Why avoid**: 
- Client sees user's password (defeats purpose of OAuth!)
- Only acceptable for highly-trusted first-party apps

---

## OAuth 2.0 Scopes

**Scope**: Defines what access is granted.

**Examples**:
- \`read:contacts\` - Read contacts only
- \`write:posts\` - Create posts
- \`read:email\` - Read email
- \`admin\` - Full access

**How it works**:
1. Client requests scopes: \`scope=read:contacts write:posts\`
2. User sees permission dialog: "Allow Printing Service to read contacts and create posts?"
3. User can approve or deny
4. Access token is limited to approved scopes
5. Resource server checks token scopes before allowing access

**Best practice**: Request minimum necessary scope (principle of least privilege).

---

## OAuth 2.0 Security

### State Parameter

**Purpose**: Prevent CSRF attacks

**How**: 
1. Client generates random \`state\`: \`state=xyz789\`
2. Client includes in auth request
3. Auth server returns same \`state\` in callback
4. Client verifies \`state\` matches

**Attack prevented**: Attacker can't trick user into authorizing malicious app.

### Redirect URI Validation

**Security requirement**: Authorization server MUST validate \`redirect_uri\` matches registered URI.

**Attack prevented**: Attacker can't steal authorization code by redirecting to their server.

### Token Storage

**Best practices**:
- **Web apps**: Store tokens in backend session, never in cookies/localStorage
- **Mobile apps**: Use secure storage (iOS Keychain, Android Keystore)
- **SPAs**: Store in memory (lost on refresh) or secure httpOnly cookies

---

## OAuth 2.0 vs SAML

| Aspect | OAuth 2.0 | SAML |
|--------|-----------|------|
| **Purpose** | Authorization (delegated access) | Authentication (SSO) |
| **Use case** | "Allow app to access my data" | "Log me into app" |
| **Format** | JSON | XML |
| **Tokens** | Access token + refresh token | SAML assertion |
| **User experience** | Permission dialog | Login redirect |
| **Example** | "Allow Spotify to access Facebook" | "Sign into Salesforce with Okta" |

**Key difference**: OAuth says "what can you access?", SAML says "who are you?"

**They solve different problems!** You can use both: SAML for authentication, OAuth for API access.

---

## Common OAuth Misconceptions

❌ **"OAuth is for authentication"**
- OAuth is for authorization. Don't use OAuth access token as proof of identity.
- Use OIDC for authentication (built on OAuth)

❌ **"Access token identifies the user"**
- Access token grants access to resources. It may or may not identify user.
- For user identity, use OIDC ID token

❌ **"OAuth is secure without PKCE for SPAs"**
- Implicit flow (without PKCE) is deprecated
- Always use Authorization Code + PKCE for SPAs

❌ **"Longer expiration is more convenient"**
- Short access token expiration is security feature
- Use refresh tokens for convenience without sacrificing security

---

## OAuth 2.0 in the Real World

### Example: Netlify Deploying from GitHub

**Scenario**: Netlify needs to read your GitHub repos to deploy your site.

**OAuth flow**:
1. You click "Deploy from GitHub" on Netlify
2. Netlify (client) redirects you to GitHub (authorization server)
3. GitHub asks: "Allow Netlify to read your repositories?"
4. You approve
5. GitHub gives Netlify an access token scoped to \`repo:read\`
6. Netlify uses token to fetch your repo and deploy

**Benefits**:
- Netlify never knows your GitHub password
- Netlify can only read repos (can't delete, can't access other resources)
- You can revoke Netlify's access anytime in GitHub settings
- If Netlify is compromised, attacker only gets limited repo access

### Example: Google Calendar on Mobile App

**Scenario**: Third-party calendar app wants to access your Google Calendar.

**OAuth flow (PKCE)**:
1. App generates \`code_verifier\` and \`code_challenge\`
2. App opens browser to Google OAuth with \`code_challenge\`
3. You log into Google, approve calendar access
4. Browser redirects back to app with authorization code
5. App exchanges code + \`code_verifier\` for access token
6. App uses token to fetch calendar events

**Security**: Even if malware intercepts authorization code, it can't exchange it without \`code_verifier\`.`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the complete Authorization Code Flow with PKCE and why each step is necessary for security.',
          sampleAnswer:
            'Authorization Code Flow with PKCE for mobile/SPA: (1) Client generates random code_verifier (high-entropy random string). This proves the client making the authorization request is the same client exchanging the code. (2) Client hashes code_verifier with SHA256 to create code_challenge. Hash is one-way - can verify but not reverse. (3) Client redirects to authorization server with code_challenge (not code_verifier). If network is intercepted, attacker gets code_challenge but cannot derive code_verifier. (4) User authenticates and approves. (5) Authorization server stores code_challenge associated with authorization code. (6) Authorization server redirects to client with authorization code. Code is one-time use, short expiration (5 min). (7) Client sends authorization code + code_verifier (kept secret on client) to token endpoint. (8) Authorization server verifies SHA256(code_verifier) matches stored code_challenge. This proves the client has the original code_verifier. (9) If verified, server returns access token + refresh token. Why each step: code_challenge prevents authorization code interception (attacker cannot exchange without code_verifier). State parameter prevents CSRF. Redirect URI validation prevents code theft. Client can be public (no client_secret) because code_verifier provides proof.',
          keyPoints: [
            'Client generates code_verifier and hashes it to code_challenge',
            'Authorization server stores code_challenge with auth code',
            'Client proves identity by providing code_verifier',
            'Prevents authorization code interception attacks',
            'Solves public client problem without client_secret',
          ],
        },
        {
          id: 'q2',
          question:
            'When would you use OAuth 2.0 vs SAML in a real-world enterprise scenario? Give specific examples.',
          sampleAnswer:
            'Use cases: OAuth 2.0 for authorization/API access. SAML for authentication/SSO. Concrete examples: (1) Use SAML: Employee logging into Salesforce with corporate credentials. Employee clicks Salesforce link, redirects to Okta (IdP), authenticates once, SAML assertion grants access. SAML is authentication - proving who you are. (2) Use OAuth 2.0: Salesforce app wants to access employee\'s Google Drive files. User authorizes Salesforce to access specific Google Drive folders. OAuth grants Salesforce limited API access without Salesforce knowing user\'s Google password. OAuth is authorization - granting access to resources. (3) Use both: Employee logs into internal portal with SAML (authentication). Portal needs to call external API (GitHub). Portal uses OAuth to get access token for GitHub API (authorization). (4) Use OIDC (built on OAuth): New mobile app for employee directory. Use OIDC for login (authentication) instead of SAML because OIDC is mobile-friendly. Key distinction: SAML/OIDC answer "who is this person?" OAuth answers "what can they access?"',
          keyPoints: [
            'SAML for authentication/SSO (who you are)',
            'OAuth for authorization/API access (what you can access)',
            'Use both together for complete auth system',
            'OIDC for modern mobile authentication',
            'Enterprises often use hybrid SAML/OIDC approach',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the security implications of access token and refresh token lifespans, and how do you balance security with user experience?',
          sampleAnswer:
            'Token lifespan trade-offs: SHORT ACCESS TOKEN (security): (1) If stolen, attacker has limited time to exploit (e.g., 1 hour). (2) Frequent rotation limits damage from compromised token. (3) Revocation is easier - just wait for expiration. LONG REFRESH TOKEN (convenience): (1) User stays logged in for days/weeks without re-authentication. (2) Background token refresh provides seamless experience. (But) if stolen, attacker has long-term access. BALANCE STRATEGY: (1) Access token: 15 min - 1 hour. Short enough for security, client refreshes automatically. (2) Refresh token: 7-90 days depending on sensitivity. Banking: 7 days. Social media: 90 days. (3) Sliding window: Reset refresh token expiration on each use. Active users stay logged in indefinitely. (4) Rotation: Issue new refresh token with each access token refresh. Old refresh token invalidated. (5) Device binding: Bind refresh token to device fingerprint. (6) Anomaly detection: Detect suspicious token use, revoke automatically. (7) Sensitive operations: Require step-up authentication. Real-world: Google uses 1-hour access tokens, 6-month refresh tokens with rotation.',
          keyPoints: [
            'Short access tokens (15-60 min) limit damage from theft',
            'Long refresh tokens (7-90 days) maintain user experience',
            'Sliding window + rotation for active users',
            'Device binding and anomaly detection',
            'Balance: Google uses 1h access, 6mo refresh',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the primary purpose of OAuth 2.0?',
          options: [
            'User authentication',
            'Delegated authorization - granting limited access without sharing passwords',
            'Data encryption',
            'Session management',
          ],
          correctAnswer: 1,
          explanation:
            'OAuth 2.0 is an authorization framework, not authentication. Its purpose is delegated authorization - allowing users to grant third-party apps limited access to their resources without sharing passwords. Example: "Allow Spotify to see your Facebook friends" without giving Spotify your Facebook password.',
        },
        {
          id: 'mc2',
          question:
            'What is the difference between access token and refresh token?',
          options: [
            'Access token is for authentication, refresh token is for authorization',
            'Access token grants API access (short-lived), refresh token obtains new access tokens (long-lived)',
            'They are the same thing',
            'Access token is encrypted, refresh token is plain text',
          ],
          correctAnswer: 1,
          explanation:
            'Access token grants access to protected resources and is short-lived (typically 1 hour) for security. Refresh token is long-lived and used to obtain new access tokens without user interaction. This separation allows frequent token rotation (security) while maintaining user convenience.',
        },
        {
          id: 'mc3',
          question: 'Why is PKCE important for mobile apps and SPAs?',
          options: [
            'PKCE makes apps faster',
            'PKCE prevents authorization code interception attacks since public clients cannot keep client_secret safe',
            'PKCE is required by law',
            'PKCE reduces server load',
          ],
          correctAnswer: 1,
          explanation:
            'PKCE (Proof Key for Code Exchange) is critical for public clients (mobile apps, SPAs) because they cannot securely store client_secret - anyone can decompile the app or inspect code. PKCE uses dynamically-generated code_verifier that never leaves the client, preventing attackers from using intercepted authorization codes.',
        },
        {
          id: 'mc4',
          question: 'What is the purpose of "scope" in OAuth 2.0?',
          options: [
            'To identify the user',
            'To define what specific access/permissions are granted',
            'To encrypt the token',
            'To set token expiration time',
          ],
          correctAnswer: 1,
          explanation:
            'Scope defines the specific permissions granted by the access token. Examples: read:contacts, write:posts, admin. This implements principle of least privilege - app requests only what it needs. User sees scope in permission dialog: "Allow App to read contacts?" and can approve/deny.',
        },
        {
          id: 'mc5',
          question: 'Which OAuth 2.0 flow should modern mobile apps use?',
          options: [
            'Implicit Flow',
            'Password Flow',
            'Authorization Code Flow with PKCE',
            'Client Credentials Flow',
          ],
          correctAnswer: 2,
          explanation:
            'Authorization Code Flow with PKCE is recommended for mobile apps. Implicit Flow is deprecated (insecure), Password Flow requires app to see user password (bad), Client Credentials is for machine-to-machine (no user). PKCE solves the public client problem securely.',
        },
        {
          id: 'mc6',
          question:
            'Why should you NOT use OAuth 2.0 access tokens for authentication?',
          options: [
            'Access tokens are too slow',
            'Access tokens grant authorization (what you can access), not authentication (who you are)',
            'Access tokens expire too quickly',
            'Access tokens are too large',
          ],
          correctAnswer: 1,
          explanation:
            "OAuth 2.0 is designed for authorization, not authentication. Access token grants access to resources but doesn't reliably identify the user. For authentication, use OIDC (OpenID Connect) which provides ID token specifically designed to identify users. This is a common critical mistake.",
        },
      ],
    },
    {
      id: 'oidc-jwt',
      title: 'OIDC (OpenID Connect) & JWT',
      content: `OpenID Connect (OIDC) is the **modern standard** for authentication. It's built on top of OAuth 2.0 and uses JWT tokens.

## What is OIDC?

**OpenID Connect (OIDC)**: An authentication layer built on OAuth 2.0.

**Key insight**: OAuth 2.0 is for authorization, OIDC adds authentication on top.

\`\`\`
OAuth 2.0: "What can you access?"
OIDC: "Who are you?" + "What can you access?"
\`\`\`

**Why OIDC exists**: OAuth 2.0 was designed for delegation, not authentication. People misused OAuth access tokens for authentication (bad practice). OIDC standardizes authentication with proper ID tokens.

---

## OIDC vs OAuth 2.0

| Feature | OAuth 2.0 | OIDC |
|---------|-----------|------|
| **Purpose** | Authorization | Authentication + Authorization |
| **Primary token** | Access token (for API access) | ID token (for identity) |
| **UserInfo** | Not defined | Standardized /userinfo endpoint |
| **Use case** | "Allow app to access my photos" | "Sign in with Google" |
| **Protocol type** | Authorization framework | Authentication protocol |

**Key difference**: OIDC adds **ID token** that contains user identity information.

---

## OIDC Tokens

### ID Token

**Purpose**: Proves user identity to client application.

**Format**: JSON Web Token (JWT) - a signed, encoded JSON object.

**Contents**:
- \`sub\`: Subject (user identifier)
- \`iss\`: Issuer (IdP URL)
- \`aud\`: Audience (client ID)
- \`exp\`: Expiration time
- \`iat\`: Issued at time
- User claims: email, name, picture, etc.

**Example ID Token** (decoded):
\`\`\`json
{
  "iss": "https://accounts.google.com",
  "sub": "110169484474386276334",
  "aud": "your-client-id.apps.googleusercontent.com",
  "exp": 1716234000,
  "iat": 1716230400,
  "email": "john@example.com",
  "email_verified": true,
  "name": "John Smith",
  "picture": "https://photo.jpg",
  "given_name": "John",
  "family_name": "Smith"
}
\`\`\`

**Usage**: Client validates ID token signature, extracts user info, creates session.

**Important**: ID token is for client, not for API calls. Use access token for APIs.

### Access Token

Same as OAuth 2.0 - for accessing APIs.

**In OIDC**: Can be JWT or opaque. If JWT, may contain user info.

### Refresh Token

Same as OAuth 2.0 - get new access/ID tokens without user interaction.

---

## JWT (JSON Web Token)

**JWT**: A compact, self-contained way to represent information as a JSON object.

### Structure

JWT has three parts separated by dots:

\`\`\`
header.payload.signature
\`\`\`

**Example**:
\`\`\`
eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.
eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gU21pdGgiLCJpYXQiOjE1MTYyMzkwMjJ9.
SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
\`\`\`

#### 1. Header

Describes the token:
\`\`\`json
{
  "alg": "RS256",  // Signing algorithm
  "typ": "JWT",    // Token type
  "kid": "abc123"  // Key ID (which public key to use)
}
\`\`\`

Base64URL encoded → \`eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9\`

#### 2. Payload

Contains claims (statements about user):
\`\`\`json
{
  "sub": "1234567890",
  "name": "John Smith",
  "email": "john@example.com",
  "iat": 1516239022,  // Issued at
  "exp": 1516242622   // Expires
}
\`\`\`

Base64URL encoded → \`eyJzdWIiOi...\`

#### 3. Signature

Cryptographic signature proving authenticity:

\`\`\`javascript
signature = RSA-SHA256(
  base64UrlEncode(header) + "." + base64UrlEncode(payload),
  private_key
)
\`\`\`

**Verification**: Anyone with public key can verify signature, confirming token hasn't been tampered with.

### JWT Claims

**Standard claims** (registered):
- \`iss\`: Issuer (who created token)
- \`sub\`: Subject (user identifier)
- \`aud\`: Audience (who token is for)
- \`exp\`: Expiration time (Unix timestamp)
- \`iat\`: Issued at time
- \`nbf\`: Not before time
- \`jti\`: JWT ID (unique identifier)

**Custom claims** (your app-specific data):
- \`email\`, \`name\`, \`role\`, \`permissions\`, etc.

**Important**: JWT payload is Base64 encoded, NOT encrypted. Anyone can decode and read it. Signature prevents tampering, not reading.

### Why JWT?

**Traditional sessions**:
\`\`\`
Client                    Server                  Database
  |-- Request + cookie -->|                          |
  |                       |-- Lookup session ------->|
  |                       |<-- Session data ---------|
  |<-- Response ----------|
\`\`\`
- Database lookup every request (slow)
- Hard to scale (sticky sessions or shared session store)

**JWT**:
\`\`\`
Client                    Server
  |-- Request + JWT ----->|
  |                       [Verify signature locally]
  |                       [Extract user from payload]
  |<-- Response ----------|
\`\`\`
- No database lookup (self-contained)
- Stateless (easy to scale horizontally)
- Faster (no DB roundtrip)

**Trade-off**: Cannot revoke JWT before expiration (it's valid until \`exp\`). Mitigation: short expiration (15 min) + refresh tokens.

---

## OIDC Flow

OIDC uses OAuth 2.0 flows with additional ID token.

### Authorization Code Flow (Most Common)

1. **Client redirects to Authorization Server**:
   \`\`\`
   https://accounts.google.com/o/oauth2/v2/auth?
     response_type=code&
     client_id=YOUR_CLIENT_ID&
     redirect_uri=https://yourapp.com/callback&
     scope=openid email profile&  // "openid" = OIDC!
     state=random_state
   \`\`\`

   **Note**: \`scope=openid\` triggers OIDC. Without it, it's just OAuth 2.0.

2. **User authenticates at IdP** (Google login page)

3. **IdP redirects with authorization code**:
   \`\`\`
   https://yourapp.com/callback?code=AUTH_CODE&state=random_state
   \`\`\`

4. **Client exchanges code for tokens**:
   \`\`\`http
   POST /token HTTP/1.1
   Host: oauth2.googleapis.com
   Content-Type: application/x-www-form-urlencoded

   grant_type=authorization_code&
   code=AUTH_CODE&
   client_id=YOUR_CLIENT_ID&
   client_secret=YOUR_SECRET&
   redirect_uri=https://yourapp.com/callback
   \`\`\`

5. **IdP returns tokens**:
   \`\`\`json
   {
     "access_token": "ya29.a0AfH6...",
     "id_token": "eyJhbGciOiJSUzI1NiI...",  // ← ID token!
     "refresh_token": "1//0gFz7...",
     "token_type": "Bearer",
     "expires_in": 3600
   }
   \`\`\`

6. **Client validates ID token**:
   - Verify signature using IdP's public key (from JWKS endpoint)
   - Check \`iss\` matches IdP
   - Check \`aud\` matches client ID
   - Check \`exp\` hasn't passed
   - Extract user info from claims

7. **Client creates session**: User is logged in!

8. **Optional - Get more user info**:
   \`\`\`http
   GET /userinfo HTTP/1.1
   Host: openidconnect.googleapis.com
   Authorization: Bearer ya29.a0AfH6...
   \`\`\`

   Returns:
   \`\`\`json
   {
     "sub": "110169484474386276334",
     "email": "john@example.com",
     "email_verified": true,
     "name": "John Smith",
     "picture": "https://photo.jpg"
   }
   \`\`\`

---

## OIDC Scopes

**\`openid\`** (required): Triggers OIDC, returns ID token with \`sub\` claim.

**\`profile\`**: Returns profile info (name, picture, birthdate, etc.)

**\`email\`**: Returns email and email_verified.

**\`address\`**: Returns postal address.

**\`phone\`**: Returns phone number.

**Example**:
\`\`\`
scope=openid profile email
\`\`\`
Returns ID token with \`sub\`, \`name\`, \`picture\`, \`email\`, \`email_verified\`.

---

## OIDC Discovery

**Problem**: How does client know IdP's endpoints and configuration?

**Solution**: OIDC Discovery - a standardized metadata endpoint.

**Discovery URL**: \`https://idp.com/.well-known/openid-configuration\`

**Example** (Google):
\`\`\`
https://accounts.google.com/.well-known/openid-configuration
\`\`\`

**Response**:
\`\`\`json
{
  "issuer": "https://accounts.google.com",
  "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
  "token_endpoint": "https://oauth2.googleapis.com/token",
  "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
  "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
  "scopes_supported": ["openid", "email", "profile"],
  "response_types_supported": ["code", "token", "id_token"],
  "grant_types_supported": ["authorization_code", "refresh_token"]
}
\`\`\`

**Benefit**: Client can auto-configure by fetching this one URL. No hard-coding endpoints.

---

## JWKS (JSON Web Key Set)

**Problem**: How does client get IdP's public key to verify JWT signatures?

**Solution**: JWKS endpoint - publishes public keys in standard format.

**Example** (Google):
\`\`\`
GET https://www.googleapis.com/oauth2/v3/certs
\`\`\`

**Response**:
\`\`\`json
{
  "keys": [
    {
      "kid": "abc123",
      "kty": "RSA",
      "alg": "RS256",
      "use": "sig",
      "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM...",
      "e": "AQAB"
    },
    {
      "kid": "def456",
      "kty": "RSA",
      ...
    }
  ]
}
\`\`\`

**Usage**:
1. Client receives ID token with \`kid: "abc123"\` in header
2. Client fetches JWKS
3. Client finds key with matching \`kid\`
4. Client verifies signature using that public key

**Key rotation**: IdP can rotate keys by publishing new keys to JWKS. Clients automatically fetch updated keys.

---

## ID Token Validation Checklist

Always validate ID tokens!

- [ ] Verify signature using public key from JWKS
- [ ] Check \`iss\` (issuer) matches expected IdP
- [ ] Check \`aud\` (audience) matches your client ID
- [ ] Check \`exp\` (expiration) hasn't passed
- [ ] Check \`iat\` (issued at) is not in future
- [ ] If using \`nonce\`, verify it matches expected value
- [ ] Extract user info from claims only after validation

**Security note**: NEVER trust ID token without signature verification! Attacker can forge unsigned JWT.

---

## OIDC vs SAML

| Aspect | OIDC | SAML |
|--------|------|------|
| **Format** | JSON/JWT | XML |
| **Based on** | OAuth 2.0 | Custom XML protocol |
| **Mobile support** | Excellent | Poor |
| **Complexity** | Simple | Complex |
| **Developer experience** | Great | Difficult |
| **Enterprise adoption** | Growing fast | Dominant (legacy) |
| **Token format** | JWT (self-contained) | XML assertion |
| **Discovery** | Standardized (.well-known) | Manual metadata exchange |
| **Token validation** | Verify JWT signature | Verify XML signature |
| **Best for** | Modern apps, mobile | Enterprise legacy apps |

**Trend**: OIDC is the future. New apps should use OIDC unless enterprise customer requires SAML.

---

## OIDC in the Real World

### Example: "Sign in with Google"

**Scenario**: Your app offers "Sign in with Google" button.

**Implementation**:
1. Register app with Google Cloud Console
2. Get client ID and secret
3. User clicks "Sign in with Google"
4. App redirects to Google with \`scope=openid email profile\`
5. User logs into Google (or already has session)
6. Google redirects back with authorization code
7. App exchanges code for ID token + access token
8. App validates ID token signature
9. App extracts \`email\`, \`name\`, \`picture\` from ID token
10. App creates user account (or matches existing) using \`sub\` (unique Google user ID)
11. App creates session, user logged in!

**Benefits**:
- No password management for your app
- Leverages Google's security (2FA, anomaly detection)
- Better UX (one-click login for Gmail users)
- Google handles password resets, account recovery

### Example: Enterprise SaaS with OIDC

**Scenario**: Your B2B SaaS offers OIDC SSO for enterprise customers.

**Setup**:
1. Customer uses Okta as IdP
2. Customer configures your app in Okta (OIDC client)
3. Okta provides: Authorization endpoint, Token endpoint, JWKS URL, Issuer
4. Your app stores these per customer (multi-tenant)

**Login**:
1. User clicks "Sign in with Okta" on your app
2. Your app redirects to customer's Okta authorization endpoint
3. User authenticates at Okta
4. Okta redirects with code
5. Your app exchanges code for ID token
6. Your app validates token, extracts email
7. Your app matches email to customer tenant, creates session

**Benefits**:
- Customer's IT controls access (add/remove employees in Okta)
- Leverage customer's MFA, password policies
- Easier enterprise sales (OIDC SSO is checkbox requirement)

---

## Common OIDC Pitfalls

❌ **Skipping signature verification**
- Always verify ID token signature!
- Libraries: \`jsonwebtoken\` (Node.js), \`jose\` (modern), \`PyJWT\` (Python)

❌ **Using ID token for API calls**
- ID token proves identity to client, not for API access
- Use access token for API calls

❌ **Not checking \`aud\` claim**
- Attacker could use ID token from different app
- Always verify \`aud\` matches your client ID

❌ **Storing tokens insecurely**
- Never store tokens in localStorage (XSS risk)
- Use httpOnly cookies or secure backend session

❌ **Not handling token expiration**
- ID tokens expire (typically 1 hour)
- Use refresh tokens to get new tokens
- Handle gracefully (refresh in background)

---

## OIDC Implementation Libraries

**Don't roll your own!** Use battle-tested libraries.

**JavaScript/Node.js**:
- \`oidc-client-ts\` (client-side)
- \`openid-client\` (server-side)
- \`passport-openidconnect\` (with Passport.js)

**Python**:
- \`authlib\` (full-featured)
- \`python-jose\` (JWT validation)

**Java**:
- Spring Security OAuth2

**C# / .NET**:
- Microsoft.AspNetCore.Authentication.OpenIdConnect

**Libraries handle**:
- Discovery (fetch .well-known/openid-configuration)
- JWKS fetching and caching
- JWT validation
- Token refresh
- PKCE for public clients`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the structure of a JWT token and how signature verification works to ensure security.',
          sampleAnswer:
            'JWT structure has three Base64URL-encoded parts separated by dots: HEADER.PAYLOAD.SIGNATURE. (1) Header: Algorithm and token type. Example: {"alg":"RS256","typ":"JWT"}. Specifies signing algorithm (RS256 = RSA with SHA-256). (2) Payload: Claims about user and token metadata. Example: {"sub":"user123","email":"john@acme.com","iss":"https://auth.acme.com","aud":"app-client-id","exp":1234567890,"iat":1234564290}. Standard claims: sub (subject/user ID), iss (issuer), aud (audience), exp (expiration), iat (issued at). (3) Signature: Cryptographic signature of header + payload. For RS256: signature = RSA-sign(base64(header) + "." + base64(payload), private_key). Only IdP has private key. VERIFICATION PROCESS: (1) Client receives JWT from IdP. (2) Client fetches IdP\'s public key from JWKS endpoint (/.well-known/jwks.json). (3) Client splits JWT into header, payload, signature. (4) Client recomputes: expected_signature = RSA-verify(base64(header) + "." + base64(payload), public_key, signature). (5) If signatures match → token is authentic (came from IdP with private key, wasn\'t modified). (6) Check exp claim → ensure token not expired. (7) Check aud claim → ensure token intended for this app. Why this works: Attacker cannot forge signature without private key. Attacker cannot modify payload without breaking signature. Public key cryptography: anyone can verify (public key) but only IdP can sign (private key). This is foundation of JWT security.',
          keyPoints: [
            'JWT = Header.Payload.Signature (Base64URL-encoded)',
            'Signature created with IdP private key, verified with public key',
            'Client fetches public key from JWKS endpoint',
            'Verification ensures authenticity and integrity',
            'Must also validate exp and aud claims',
          ],
        },
        {
          id: 'q2',
          question:
            'How does OIDC Discovery work and why is it important for enterprise SSO?',
          sampleAnswer:
            'OIDC Discovery standardizes how clients find IdP endpoints. Instead of manually configuring 5+ URLs, client just needs issuer URL. Discovery process: (1) Client knows issuer: https://auth.acme.com. (2) Client makes GET request to /.well-known/openid-configuration: https://auth.acme.com/.well-known/openid-configuration. (3) IdP returns JSON metadata document with all endpoints and capabilities. (4) Client automatically configures itself using this metadata. Why this matters for enterprise: (1) Self-Service: Enterprise customer can set up SSO by just providing issuer URL. Sales engineer doesn\'t need hour-long config call. (2) Automatic Updates: If IdP changes endpoint URLs, clients automatically discover new URLs. No manual reconfiguration. (3) Multi-Tenant SaaS: Your B2B SaaS serves 1000 enterprise customers with different IdPs. Discovery lets you support them all with minimal config per customer. (4) Standards Compliance: Discovery is part of OIDC spec. IdPs that support it are standards-compliant. (5) Reduced Errors: Manual endpoint configuration is error-prone. Discovery eliminates "typo in token endpoint URL" support tickets. Real-world example: Slack supports OIDC SSO for Enterprise Grid. Customer provides issuer URL, Slack uses discovery to auto-configure everything. Customer IT self-serves SSO setup in 5 minutes instead of days.',
          keyPoints: [
            'Discovery endpoint: /.well-known/openid-configuration',
            'Returns all IdP endpoints and capabilities in one JSON doc',
            'Enables self-service SSO setup for enterprise customers',
            'Auto-updates when IdP changes endpoints',
            'Critical for multi-tenant B2B SaaS applications',
          ],
        },
        {
          id: 'q3',
          question:
            'Compare stateless JWT-based authentication versus traditional session-based authentication. When would you use each?',
          sampleAnswer:
            "STATELESS JWT: Server generates JWT, signs it, sends to client. Client includes JWT in every request (Authorization: Bearer header). Server validates signature and extracts claims. No session storage needed. PROS: (1) Scalability: No session store needed. Any server can validate JWT. Perfect for distributed systems. (2) Microservices: JWT can be passed between services. Each service validates independently. (3) Cross-domain: JWT works across different domains. (4) Mobile-friendly: Simple token storage on device. CONS: (1) Cannot revoke: JWT valid until expiration. Logout doesn't work server-side. (2) Larger payload: Every request includes full JWT. (3) Secrets in token: Claims are readable (only signed, not encrypted). TRADITIONAL SESSIONS: Server creates session, stores in Redis/DB, sends session ID cookie to client. Client sends cookie with each request. Server looks up session. PROS: (1) Revocation: Delete session = instant logout. (2) Smaller payload: Cookie just contains session ID. (3) Secure storage: Session data stays on server. (4) Fine-grained control: Can track last access, IP address, device. CONS: (1) Scalability: Need shared session store (Redis). (2) Server-side storage: Millions of users = millions of sessions. (3) Not stateless: Harder to scale horizontally. WHEN TO USE EACH: Use JWT: (1) Microservices architecture (services need to validate independently). (2) Third-party API access (GitHub API, Google API). (3) Short-lived tokens (5-15 min) with refresh token. (4) Mobile apps (simpler than cookie management). Use Sessions: (1) Traditional web apps with server-side rendering. (2) Need instant logout/revocation (banking, healthcare). (3) Long-lived sessions (days/weeks). (4) Need to track detailed session metadata. HYBRID APPROACH (best of both worlds): (1) Use JWT for authentication with 15-min expiration. (2) Store session server-side with refresh token. (3) Client refreshes JWT automatically in background. (4) Logout revokes refresh token server-side. This gives JWT benefits (stateless validation) with session benefits (revocation). Used by Auth0, Okta.",
          keyPoints: [
            'JWT: Stateless, scalable, no revocation',
            'Sessions: Revocable, server-stored, harder to scale',
            'JWT best for microservices and mobile apps',
            'Sessions best for traditional web apps needing revocation',
            'Hybrid: Short JWT + refresh token session',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the primary difference between OAuth 2.0 and OIDC?',
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
      ],
    },
    {
      id: 'idp-scim-jit',
      title: 'Identity Providers, SCIM & JIT Provisioning',
      content: `Modern authentication relies on **Identity Providers (IdPs)** to manage user identities. Let's explore the major players and how user provisioning works.

## Identity Providers (IdPs)

### What is an IdP?

**Identity Provider**: A system that creates, maintains, and manages identity information while providing authentication services.

**Core functions**:
1. **User directory**: Store user identities
2. **Authentication**: Verify credentials (password, MFA, biometrics)
3. **Token issuance**: Generate SAML assertions, OIDC tokens, JWT
4. **Session management**: Maintain SSO sessions
5. **Security policies**: Password policies, MFA enforcement, conditional access

---

## Major Identity Providers

### 1. Okta ⭐ Enterprise Leader

**Type**: Cloud-native, enterprise-focused

**Key features**:
- Comprehensive SSO (SAML, OIDC, WS-Federation)
- Universal Directory (centralized user store)
- Multi-factor authentication (Okta Verify, SMS, hardware tokens)
- Lifecycle Management (user provisioning/deprovisioning)
- API Access Management (OAuth 2.0 authorization server)
- Adaptive authentication (risk-based auth)

**Use case**: Enterprise SSO for SaaS applications
**Example**: Large company uses Okta as central IdP for 200+ applications

**Pricing**: $2-15+ per user/month

**When to use**:
- Enterprise with 100+ employees
- Need to integrate many SaaS apps
- Require advanced features (adaptive auth, lifecycle management)
- Compliance requirements (SOC 2, HIPAA)

### 2. Auth0 (by Okta) ⭐ Developer-Friendly

**Type**: Developer-focused, API-first

**Key features**:
- Easy integration (SDKs for every platform)
- Universal Login (hosted login page)
- Social connections (Google, Facebook, GitHub, etc.)
- Passwordless authentication (email/SMS magic links)
- Custom databases (bring your own user store)
- Rules and Actions (custom logic in auth flow)
- Excellent documentation

**Use case**: SaaS application needing authentication
**Example**: B2C app with "Sign in with Google" and email/password

**Pricing**: Free tier available, $23+ per month

**When to use**:
- Building a new application
- Need social login + traditional auth
- Want developer-friendly API
- B2C or B2B2C scenarios
- Rapid development

**Auth0 vs Okta**:
- **Auth0**: Developer experience, API-first, code-centric, better for product auth
- **Okta**: Enterprise features, admin UI-centric, better for workforce identity

### 3. Microsoft Entra ID (Azure AD)

**Type**: Enterprise, Microsoft ecosystem

**Key features**:
- Deep Microsoft 365 integration
- Azure resource access control
- Conditional Access (location, device, risk-based policies)
- B2C and B2B capabilities
- On-premise Active Directory sync
- Free tier with Azure

**Use case**: Microsoft-centric enterprises
**Example**: Company using Office 365 + Azure, Entra ID provides SSO

**Pricing**: Free tier, $6-9 per user/month for premium

**When to use**:
- Already using Microsoft 365 / Azure
- Need Windows/Azure integration
- On-premise Active Directory migration
- Government/large enterprise

### 4. Google Cloud Identity / Workspace

**Type**: Cloud-native, Google ecosystem

**Key features**:
- Gmail / Google Workspace integration
- Google as identity provider (OIDC)
- Cloud Identity for non-Gmail users
- Mobile device management
- Context-aware access

**Use case**: Google Workspace organizations
**Example**: Startup using Google Workspace, adds Cloud Identity for SSO

**Pricing**: Included with Workspace, standalone $6 per user/month

**When to use**:
- Using Google Workspace
- Need Google as IdP for SSO
- Small to mid-size organizations

### 5. Ping Identity

**Type**: Enterprise, hybrid (cloud + on-premise)

**Key features**:
- PingFederate (federation server)
- PingOne (cloud IdP)
- Strong on-premise capabilities
- API intelligence
- Government/defense focus

**Use case**: Large enterprises with complex hybrid infrastructure
**Example**: Bank with on-premise systems + cloud apps

**Pricing**: Enterprise (contact sales)

**When to use**:
- Large enterprise (10,000+ employees)
- Hybrid cloud + on-premise
- Highly regulated industries
- Complex federation requirements

### 6. Keycloak (Open Source)

**Type**: Open-source, self-hosted

**Key features**:
- Free and open-source
- SAML, OIDC, OAuth 2.0
- User federation (LDAP, Active Directory)
- Social login
- Multi-tenancy
- Admin console

**Use case**: Organizations wanting full control
**Example**: Company self-hosting IdP for cost savings or data sovereignty

**Pricing**: Free (self-host), support available

**When to use**:
- Budget constraints
- Data sovereignty requirements (can't use cloud IdP)
- Full control over IdP
- Custom modifications needed

**Trade-off**: Must maintain infrastructure, less features than commercial IdPs

---

## User Provisioning

**Problem**: When employee joins, admin must manually create accounts in 50 applications. When they leave, must manually disable 50 accounts.

**Solution**: Automated provisioning from IdP to applications.

### SCIM (System for Cross-domain Identity Management)

**SCIM**: An open standard for automating user identity management across systems.

**Purpose**: Automate user provisioning and deprovisioning.

**How it works**: IdP (source of truth) pushes user changes to applications (targets) via SCIM API.

#### SCIM Protocol

**RESTful API** with standardized endpoints:

**Create user**:
\`\`\`http
POST /scim/v2/Users
Content-Type: application/scim+json

{
  "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
  "userName": "john.smith@company.com",
  "name": {
    "givenName": "John",
    "familyName": "Smith"
  },
  "emails": [{
    "value": "john.smith@company.com",
    "primary": true
  }],
  "active": true
}
\`\`\`

**Response**:
\`\`\`json
{
  "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
  "id": "2819c223-7f76-453a-919d-413861904646",
  "userName": "john.smith@company.com",
  "meta": {
    "resourceType": "User",
    "created": "2024-01-15T10:00:00.000Z",
    "location": "https://app.com/scim/v2/Users/2819c223..."
  }
}
\`\`\`

**Update user**:
\`\`\`http
PATCH /scim/v2/Users/2819c223-7f76-453a-919d-413861904646
Content-Type: application/scim+json

{
  "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
  "Operations": [{
    "op": "replace",
    "path": "active",
    "value": false
  }]
}
\`\`\`

**Deprovision user**: Set \`active: false\` or DELETE

#### SCIM Workflow

**Scenario**: New employee John joins company

1. **HR adds John to HRIS** (Workday, BambooHR)
2. **HRIS syncs to IdP** (Okta) via SCIM or custom integration
3. **IdP pushes John to all apps** via SCIM:
   - POST /scim/v2/Users to Salesforce
   - POST /scim/v2/Users to Slack
   - POST /scim/v2/Users to GitHub
   - POST /scim/v2/Users to AWS SSO
4. **John has access** to all apps on day one

**Scenario**: John leaves company

1. **HR marks John as terminated in HRIS**
2. **HRIS syncs to IdP**
3. **IdP pushes deactivation** via SCIM:
   - PATCH /scim/v2/Users/{john_id} \`active: false\` to all apps
4. **John loses access** to all apps instantly

#### SCIM Benefits

- ✅ **Instant provisioning**: New employee has access on day one
- ✅ **Instant deprovisioning**: Ex-employee loses access immediately (security!)
- ✅ **No manual work**: IT doesn't create/delete accounts manually
- ✅ **Synchronized**: User changes (name, email, role) sync automatically
- ✅ **Audit trail**: Central log of provisioning actions

#### SCIM Architecture

\`\`\`
┌──────────┐         ┌──────────┐         ┌─────────────────┐
│   HRIS   │  sync   │   IdP    │  SCIM   │  Applications   │
│ (Workday)│-------->│  (Okta)  │-------->│  - Salesforce   │
└──────────┘         │          │-------->│  - Slack        │
                     │          │-------->│  - GitHub       │
                     └──────────┘-------->│  - AWS          │
                                          └─────────────────┘

Flow:
1. HR adds user to Workday
2. Workday → Okta (API or CSV import)
3. Okta → Apps (SCIM push)
\`\`\`

---

### JIT (Just-In-Time) Provisioning

**JIT**: Automatically create user account on first login (no pre-provisioning needed).

**How it works**: Application receives SAML assertion or OIDC ID token with user attributes, creates account if doesn't exist.

#### JIT Flow

**First login**:
1. User logs in via SSO (SAML or OIDC)
2. IdP sends assertion/token with user attributes (email, name, role)
3. Application receives assertion
4. Application checks if user exists in database
5. **User doesn't exist** → Application creates user account using attributes
6. Application grants access

**Subsequent logins**:
1-4. Same
5. **User exists** → Application updates attributes (if changed)
6. Application grants access

#### JIT Example (SAML)

**SAML assertion**:
\`\`\`xml
<saml:Assertion>
  <saml:Subject>
    <saml:NameID>john.smith@company.com</saml:NameID>
  </saml:Subject>
  <saml:AttributeStatement>
    <saml:Attribute Name="email">
      <saml:AttributeValue>john.smith@company.com</saml:AttributeValue>
    </saml:Attribute>
    <saml:Attribute Name="firstName">
      <saml:AttributeValue>John</saml:AttributeValue>
    </saml:Attribute>
    <saml:Attribute Name="lastName">
      <saml:AttributeValue>Smith</saml:AttributeValue>
    </saml:Attribute>
    <saml:Attribute Name="role">
      <saml:AttributeValue>Sales</saml:AttributeValue>
    </saml:Attribute>
  </saml:AttributeStatement>
</saml:Assertion>
\`\`\`

**Application logic**:
\`\`\`python
def handle_saml_login(assertion):
    # Validate assertion (signature, timestamps, etc.)
    attributes = extract_attributes(assertion)
    email = attributes['email']
    
    user = User.find_by_email(email)
    
    if not user:
        # JIT: Create user on first login
        user = User.create(
            email=email,
            first_name=attributes['firstName'],
            last_name=attributes['lastName'],
            role=attributes['role']
        )
        log_audit("JIT created user", user.id)
    else:
        # Update attributes on subsequent logins
        user.update(
            first_name=attributes['firstName'],
            last_name=attributes['lastName'],
            role=attributes['role']
        )
    
    # Create session
    session['user_id'] = user.id
    return redirect('/dashboard')
\`\`\`

#### JIT vs SCIM

| Aspect | JIT | SCIM |
|--------|-----|------|
| **When provisioned** | First login | Before login |
| **IdP requirement** | Attributes in assertion | SCIM API support |
| **App requirement** | Parse assertions | SCIM endpoint |
| **Deprovisioning** | ❌ No (user still exists after disabled at IdP) | ✅ Yes (instant) |
| **Offboarding security** | ❌ Risk: User account persists | ✅ Secure: Account disabled |
| **Complexity** | Simple | More complex |
| **Best for** | Small apps, fast setup | Enterprise, security-critical |

**Key difference**: JIT doesn't handle deprovisioning! User is disabled at IdP (can't log in via SSO) but local account still exists.

**Recommendation**: Use SCIM for enterprise applications where security is critical. Use JIT for internal tools where user cleanup isn't urgent.

---

## Provisioning Best Practices

### 1. Use SCIM for Deprovisioning

Don't rely on JIT alone. Implement SCIM \`active: false\` handling to disable users.

### 2. Attribute Mapping

Map IdP attributes to app fields carefully:
- IdP \`email\` → App \`email\`
- IdP \`firstName\` + \`lastName\` → App \`full_name\`
- IdP \`department\` → App \`team\`
- IdP \`role\` → App \`permissions\`

### 3. Primary Identifier

Use stable identifier as primary key:
- ✅ SAML \`NameID\` or \`sub\` claim
- ✅ OIDC \`sub\` claim
- ❌ NOT email (users change emails)

### 4. Audit Logging

Log all provisioning events:
- User created (JIT)
- User updated (attribute sync)
- User deactivated (SCIM)
- Login attempts
- Permission changes

### 5. Error Handling

Handle provisioning errors gracefully:
- Email conflict (user exists with different identifier)
- Invalid attributes (missing required fields)
- SCIM endpoint down (retry with exponential backoff)

### 6. Testing

Test provisioning scenarios:
- Create user
- Update user attributes
- Deactivate user
- Reactivate user
- Delete user
- Duplicate email handling

---

## Real-World Integration

### Example: Okta → Salesforce (SCIM)

**Setup**:
1. Salesforce admin enables SCIM API, generates API token
2. Okta admin configures Salesforce app in Okta
3. Okta admin enters Salesforce SCIM endpoint + API token
4. Okta admin maps attributes:
   - Okta \`email\` → Salesforce \`Username\`
   - Okta \`firstName\` → Salesforce \`FirstName\`
   - Okta \`profile\` → Salesforce \`ProfileId\`

**Runtime**:
1. HR adds John to Okta Universal Directory
2. Okta assigns Salesforce app to John
3. Okta calls Salesforce SCIM API: POST /services/scim/v2/Users
4. Salesforce creates John's account
5. John logs into Salesforce via Okta SSO
6. John is already provisioned, immediate access

**Offboarding**:
1. HR marks John as terminated in Okta
2. Okta calls Salesforce SCIM API: PATCH /services/scim/v2/Users/{john_id} \`active: false\`
3. John's Salesforce account disabled instantly
4. John's Okta SSO session terminated
5. John loses access to all apps

**Security win**: Complete offboarding in < 1 minute vs. manual process taking hours/days.

### Example: Auth0 → Custom App (JIT)

**Setup**:
1. Configure OIDC in Auth0
2. App implements OIDC login
3. App parses ID token for user attributes

**Runtime**:
1. User clicks "Sign in with Google" on app
2. Auth0 handles Google OAuth, returns ID token
3. App validates ID token
4. App extracts \`sub\`, \`email\`, \`name\` from ID token
5. App checks if user with this \`sub\` exists
6. User doesn't exist → App creates user (JIT)
7. User logged in

**Benefits**:
- Zero pre-provisioning
- Users self-onboard via social login
- Great for B2C apps`,
      quiz: [
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
      ],
      multipleChoice: [
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
          question:
            'What is the main security limitation of JIT compared to SCIM?',
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
            "Auth0 is developer-focused with excellent SDKs, documentation, and API-first design. It's perfect for SaaS apps needing authentication (social login, email/password, SSO). Microsoft Entra ID is best for Microsoft ecosystems. Ping is for large enterprises. Keycloak is open-source but requires more setup.",
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
      ],
    },
  ],
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
};

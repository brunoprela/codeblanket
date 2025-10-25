/**
 * SAML (Security Assertion Markup Language) Section
 */

export const samldeepdiveSection = {
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
   https://idp.okta.com/sso? SAMLRequest=<base64_encoded_request>
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
};

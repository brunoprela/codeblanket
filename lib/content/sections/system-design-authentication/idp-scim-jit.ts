/**
 * Identity Providers, SCIM & JIT Provisioning Section
 */

export const idpscimjitSection = {
  id: 'idp-scim-jit',
  title: 'Identity Providers, SCIM & JIT Provisioning',
  content: `Modern authentication relies on **Identity Providers (IdPs)** to manage user identities. Let\'s explore the major players and how user provisioning works.

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
def handle_saml_login (assertion):
    # Validate assertion (signature, timestamps, etc.)
    attributes = extract_attributes (assertion)
    email = attributes['email']
    
    user = User.find_by_email (email)
    
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
4. Salesforce creates John\'s account
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
};

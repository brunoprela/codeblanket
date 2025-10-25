/**
 * OIDC (OpenID Connect) & JWT Section
 */

export const oidcjwtSection = {
  id: 'oidc-jwt',
  title: 'OIDC (OpenID Connect) & JWT',
  content: `OpenID Connect (OIDC) is the **modern standard** for authentication. It\'s built on top of OAuth 2.0 and uses JWT tokens.

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
  base64UrlEncode (header) + "." + base64UrlEncode (payload),
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
- Leverages Google\'s security (2FA, anomaly detection)
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
};

/**
 * OAuth 2.0 - Authorization Framework Section
 */

export const oauth2Section = {
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
};

/**
 * API Authentication Methods Section
 */

export const apiauthenticationSection = {
  id: 'api-authentication',
  title: 'API Authentication Methods',
  content: `Authentication verifies who is making an API request. Choosing the right method depends on your use case, security requirements, and client types.

## Overview of Authentication Methods

### **1. API Keys**
Simple string tokens identifying the application.

\`\`\`http
GET /api/users
X-API-Key: sk_live_abc123xyz
\`\`\`

**Pros**: Simple, good for identifying applications
**Cons**: Not per-user, hard to rotate, easily leaked if not HTTPS

**Use Cases**: Public APIs with usage tracking, server-to-server communication

### **2. Bearer Tokens (JWT)**
JSON Web Tokens containing encoded claims.

\`\`\`http
GET /api/users
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
\`\`\`

**Structure**:
\`\`\`
header.payload.signature
\`\`\`

**Pros**: Stateless, contains user info, self-contained
**Cons**: Can't revoke easily (until expiry), size overhead

**Use Cases**: Modern web/mobile apps, microservices

### **3. OAuth 2.0**
Delegation protocol for third-party access.

**Flows**:
- Authorization Code: Web apps
- Client Credentials: Server-to-server
- Implicit: Legacy SPAs (deprecated)
- PKCE: Mobile/SPA (recommended)

**Use Cases**: Social login, third-party integrations

### **4. Basic Authentication**
Username:password in Base64.

\`\`\`http
Authorization: Basic dXNlcjpwYXNzd29yZA==
\`\`\`

**Pros**: Simple, built into HTTP
**Cons**: Credentials in every request, no built-in expiry

**Use Cases**: Simple internal APIs, dev/test environments

### **5. Mutual TLS (mTLS)**
Both client and server present certificates.

**Pros**: Very secure, cryptographic authentication
**Cons**: Complex setup, certificate management

**Use Cases**: High-security microservices, banking APIs

## API Keys Deep Dive

**Generation**:
\`\`\`javascript
const apiKey = 'sk_' + crypto.randomBytes(32).toString('hex');
\`\`\`

**Storage**: Hash before storing (like passwords)

**Best Practices**:
- Prefix keys (\`sk_\` for secret, \`pk_\` for public)
- Allow multiple keys per account
- Track last used date
- Enable rotation
- Rate limit by key

## JWT Best Practices

**Claims**:
\`\`\`json
{
  "sub": "user_123",
  "iat": 1640000000,
  "exp": 1640003600,
  "role": "admin"
}
\`\`\`

**Short-lived Access Tokens**:
- Access token: 15 minutes
- Refresh token: 7 days
- Rotate refresh tokens

**Validation**:
1. Check signature
2. Verify expiration
3. Validate issuer/audience
4. Check revocation list (if needed)

## OAuth 2.0 in Practice

**Authorization Code Flow**:
\`\`\`
1. Client → Authorization Server: Request auth code
2. User logs in, grants permission
3. Authorization Server → Client: Auth code
4. Client → Authorization Server: Exchange code for token
5. Authorization Server → Client: Access token + refresh token
\`\`\`

**Scopes**:
\`\`\`
read:users write:users admin:all
\`\`\`

## Security Best Practices

1. **Always use HTTPS**: Prevents token interception
2. **Short token lifetimes**: Limit damage if compromised
3. **Implement rate limiting**: Prevent brute force
4. **Log authentication attempts**: Detect attacks
5. **Support token revocation**: For compromised tokens
6. **Use secure storage**: HttpOnly cookies or secure storage
7. **Implement CORS properly**: Prevent unauthorized origins`,
};

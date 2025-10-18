/**
 * Quiz questions for API Authentication Methods section
 */

export const apiauthenticationQuiz = [
  {
    id: 'auth-d1',
    question:
      'Design an authentication system for an API serving web app, mobile app, and third-party integrations. What methods would you use for each and why?',
    sampleAnswer: `Different clients have different security requirements and constraints:

**Web App (SPA)**:
- OAuth 2.0 Authorization Code + PKCE
- Short-lived JWT access tokens (15 min)
- Refresh tokens in HttpOnly cookies
- Why: Secure for browser environment, PKCE prevents token interception, HttpOnly prevents XSS

**Mobile App**:
- OAuth 2.0 PKCE flow
- JWT access + refresh tokens in secure storage
- Biometric auth locally
- Why: No client secret security, native secure storage, good UX with biometrics

**Third-Party Integrations**:
- OAuth 2.0 Authorization Code (for user delegation)
- API Keys for server-to-server
- Scoped permissions (read:users, write:posts)
- Why: Users control what third parties can access, revocable, scoped

**Internal Microservices**:
- mTLS or service-to-service JWT
- Service mesh for automatic cert management
- Why: High security, no user involvement, automatic rotation

**Implementation**:
- Single Authorization Server (like Auth0, Keycloak)
- Multiple authentication flows supported
- Centralized token validation
- Audit logs for all auth events

Trade-offs: Complexity vs security. Multiple methods increase implementation and maintenance but provide appropriate security for each use case.`,
    keyPoints: [
      'Different clients need different auth methods',
      'OAuth 2.0 with PKCE for user-facing clients',
      'API keys for server-to-server integration',
      'mTLS for high-security internal services',
      'Centralized auth server for consistency',
    ],
  },
  {
    id: 'auth-d2',
    question:
      'JWTs are stateless, which makes them scalable but difficult to revoke. How would you handle immediate token revocation (e.g., user logs out or account compromised)?',
    sampleAnswer: `JWT revocation challenge: They're valid until expiration by design. Solutions:

**1. Short-Lived Tokens + Refresh Token Rotation** (Recommended):
- Access token: 15 minutes
- Refresh token: Stored in database, can be revoked
- On compromise: Revoke refresh token, access token expires soon

**2. Token Blacklist (Allow List)**:
\`\`\`javascript
// Check on every request
const isRevoked = await redis.get(\`revoked:\${jti}\`);
\`\`\`
Pros: Immediate revocation
Cons: Requires state (contradicts stateless), network call each request

**3. Token Versioning**:
\`\`\`json
{ "sub": "user_123", "version": 5 }
\`\`\`
Store current version in cache. Revoke all by incrementing version.
Pros: One cache lookup per request
Cons: Still requires state

**4. Short Expiry + Frequent Refresh**:
- Ultra-short access tokens (5 min)
- Check permissions on refresh
- Effective revocation within 5 minutes

**5. Event-Driven Revocation**:
- Broadcast revocation events to all servers
- Local in-memory cache of revoked tokens
- Reduces latency vs database check

**Best Practice Approach**:
- Short-lived access tokens (15 min) 
- Revocable refresh tokens in database
- Allow list for critical revocations (admin lockout)
- Accept eventual consistency for most cases

**Trade-off**: Balance between immediate revocation and scalability. Most apps can tolerate 15-minute window. High-security apps need blacklist despite scalability cost.`,
    keyPoints: [
      'JWT statelessness makes immediate revocation challenging',
      'Short-lived access tokens + revocable refresh tokens is standard',
      'Blacklist/allow list provides immediate revocation but adds state',
      'Token versioning reduces database load',
      'Accept eventual consistency for better scalability',
    ],
  },
  {
    id: 'auth-d3',
    question:
      'A third-party integration stores API keys in their database unencrypted, then gets hacked. How would you design your API key system to limit damage from such incidents?',
    sampleAnswer: `Defense in depth for API key security:

**1. Hash Keys Before Storage** (Like Passwords):
\`\`\`javascript
// Generation
const apiKey = 'sk_' + randomBytes(32).toString('hex');
const hash = bcrypt.hash(apiKey, 10);
// Store: id, hash, prefix 'sk_...xyz' (last 4), metadata

// Validation
const match = bcrypt.compare(providedKey, storedHash);
\`\`\`
Benefit: Even if YOUR database is compromised, keys can't be recovered

**2. Scope Limitations**:
\`\`\`json
{
  "key_id": "key_123",
  "scopes": ["read:public_data"],
  "rate_limit": "1000/hour"
}
\`\`\`
Benefit: Compromised key has limited permissions

**3. IP Whitelisting**:
\`\`\`json
{"key_id": "key_123", "allowed_ips": ["52.12.34.56"]}
\`\`\`
Benefit: Stolen key useless from other IPs

**4. Key Rotation**:
- Support multiple active keys
- Force rotation every 90 days
- One-click rotation in dashboard

**5. Usage Monitoring**:
- Alert on unusual patterns (new IP, high volume, off-hours)
- Automatic suspension on suspicious activity
- Anomaly detection

**6. Environment Segregation**:
- Test vs production keys
- Different prefixes (\`sk_test_\`, \`sk_live_\`)
- Test keys can't access production data

**7. Revocation & Audit**:
- Instant revocation capability
- Audit logs of all API key usage
- Track: timestamp, IP, endpoint, response code

**Implementation Example (Stripe-style)**:
- Publishable keys (\`pk_\`) for client-side (limited)
- Secret keys (\`sk_\`) for server-side (full access)
- Restricted keys with custom permissions
- Test mode keys for development

**Response to Breach**:
1. Notify customer immediately
2. Automatic suspension option in dashboard
3. Audit logs show what was accessed
4. Easy key rotation (generate new, test, revoke old)

**Trade-off**: More security features increase complexity but necessary for production APIs. Start simple, add features as you grow.`,
    keyPoints: [
      'Hash API keys before storage (unrecoverable if DB compromised)',
      'Scope limitations reduce damage from compromised keys',
      'IP whitelisting, rate limiting, and monitoring detect misuse',
      'Support multiple keys per account for easy rotation',
      'Audit logs essential for breach investigation',
    ],
  },
];

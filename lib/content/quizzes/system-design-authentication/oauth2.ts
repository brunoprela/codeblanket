/**
 * Quiz questions for OAuth 2.0 - Authorization Framework section
 */

export const oauth2Quiz = [
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
];

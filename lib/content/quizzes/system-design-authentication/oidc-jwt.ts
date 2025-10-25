/**
 * Quiz questions for OIDC (OpenID Connect) & JWT section
 */

export const oidcjwtQuiz = [
  {
    id: 'q1',
    question:
      'Explain the structure of a JWT token and how signature verification works to ensure security.',
    sampleAnswer:
      'JWT structure has three Base64URL-encoded parts separated by dots: HEADER.PAYLOAD.SIGNATURE. (1) Header: Algorithm and token type. Example: {"alg":"RS256","typ":"JWT"}. Specifies signing algorithm (RS256 = RSA with SHA-256). (2) Payload: Claims about user and token metadata. Example: {"sub":"user123","email":"john@acme.com","iss":"https://auth.acme.com","aud":"app-client-id","exp":1234567890,"iat":1234564290}. Standard claims: sub (subject/user ID), iss (issuer), aud (audience), exp (expiration), iat (issued at). (3) Signature: Cryptographic signature of header + payload. For RS256: signature = RSA-sign (base64(header) + "." + base64(payload), private_key). Only IdP has private key. VERIFICATION PROCESS: (1) Client receives JWT from IdP. (2) Client fetches IdP\'s public key from JWKS endpoint (/.well-known/jwks.json). (3) Client splits JWT into header, payload, signature. (4) Client recomputes: expected_signature = RSA-verify (base64(header) + "." + base64(payload), public_key, signature). (5) If signatures match → token is authentic (came from IdP with private key, wasn\'t modified). (6) Check exp claim → ensure token not expired. (7) Check aud claim → ensure token intended for this app. Why this works: Attacker cannot forge signature without private key. Attacker cannot modify payload without breaking signature. Public key cryptography: anyone can verify (public key) but only IdP can sign (private key). This is foundation of JWT security.',
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
];

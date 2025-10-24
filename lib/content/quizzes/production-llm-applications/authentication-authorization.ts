export const authenticationAuthorizationQuiz = [
  {
    id: 'pllm-q-10-1',
    question:
      'Design a complete authentication and authorization system for a multi-tenant LLM SaaS product. Include API keys, OAuth2, role-based access control, and how you prevent unauthorized access to other tenants data.',
    sampleAnswer:
      'Multi-layer security: 1) API key auth for service-to-service (stateless, X-API-Key header, rate-limited per key), 2) OAuth2 with JWT for user auth (stateful, supports refresh tokens), 3) RBAC with roles (admin, member, viewer) per tenant. Tenant isolation: Every query includes tenant_id filter, middleware extracts tenant from subdomain or API key, Row-Level Security (RLS) in PostgreSQL enforces tenant_id filter automatically, separate encryption keys per tenant. Implementation: API keys: generate with secrets.token_urlsafe(32), store hashed (bcrypt), prefix with sk_ for identification, rotate quarterly. OAuth2: Use Auth0 or implement with authlib, JWT tokens with 15min expiry, refresh tokens with 30 day expiry, include tenant_id and role in claims. RBAC: Permissions matrix (admin: all, member: read/write own, viewer: read only), check on every request with @require_permission decorator, cache permissions in Redis (5min TTL). Prevention: Never trust client-provided tenant_id (always from API key/token), use parameterized queries (prevent SQL injection), encrypt API keys at rest, log all auth failures, implement rate limiting on auth endpoints (prevent brute force), require 2FA for admins, rotate keys on user removal.',
    keyPoints: [
      'Multi-layer auth: API keys for services, OAuth2/JWT for users',
      'Tenant isolation via RLS and middleware-enforced filtering',
      'RBAC with cached permissions and comprehensive security measures',
    ],
  },
  {
    id: 'pllm-q-10-2',
    question:
      'Explain how you would implement session management for an LLM application including session storage, expiration, renewal, and secure cookie handling. How do you handle concurrent sessions and session hijacking?',
    sampleAnswer:
      'Session creation: On login, generate session_id (secrets.token_urlsafe(32)), store {user_id, tenant_id, created_at, last_activity, ip, user_agent} in Redis with 24hr TTL, set httpOnly secure sameSite=lax cookie. Session validation: On each request, extract session_id from cookie, load from Redis, check expiration and update last_activity, extend TTL if activity <1hr ago, validate IP/user_agent (optional), allow if valid else return 401. Renewal: Sliding expiration (extend TTL on activity), fixed expiration for sensitive ops (require re-auth after 8hrs regardless), explicit refresh endpoint for mobile apps. Concurrent sessions: Allow up to 5 concurrent sessions per user, store list in Redis, oldest evicted when limit reached, show active sessions in dashboard with revoke button. Security: httpOnly prevents JavaScript access, secure requires HTTPS, sameSite=lax prevents CSRF, include CSRF token in forms, rotate session_id on privilege escalation, implement session fixation protection (new session after login). Prevent hijacking: IP address validation (warn on change, optional block), user agent validation, monitor for impossible travel (login from NY then London 1min later), require re-auth for sensitive operations, automatic logout after 30min inactivity, implement device fingerprinting. Logout: Delete session from Redis, clear cookie, optionally revoke all sessions.',
    keyPoints: [
      'Redis-backed sessions with sliding/fixed expiration strategies',
      'Secure cookie configuration: httpOnly, secure, sameSite',
      'Hijacking prevention: IP/user-agent validation, impossible travel detection',
    ],
  },
  {
    id: 'pllm-q-10-3',
    question:
      'How would you implement API key rotation for an LLM service without causing downtime for clients? Include the rotation process, grace periods, and migration strategy.',
    sampleAnswer:
      'Two-key system: Each tenant has primary and secondary keys, both work simultaneously during transition. Rotation process: 1) User initiates rotation in dashboard, 2) Generate new secondary key, old primary stays active, 3) Grace period: 30 days where both keys work, 4) User updates applications to use new key, tests in staging, 5) After verification, new key becomes primary, old key disabled, 6) Can immediately generate new secondary for next rotation. Implementation: API_Keys table (tenant_id, key_hash, type PRIMARY/SECONDARY, created_at, last_used_at, expires_at), authentication checks both keys, log which key used, email warnings at 7/3/1 days before expiration. Forced rotation: On security incident, immediately disable old keys, force generate new ones, email all affected users with instructions, provide migration guide. Migration strategy: 1) Announce 30 days in advance, 2) Email with new key and deadline, 3) Show banner in dashboard, 4) Provide test endpoint to verify new key works, 5) Warning emails at 7/3/1 days, 6) Automatic rotation with notification (optional). Monitor: Track key usage (last_used_at), alert on deprecated key usage after deadline, provide dashboard showing key age and usage. Best practices: Rotate every 90 days, never expose keys in logs/responses, encrypt in database, support key prefixes for identification, implement emergency revocation.',
    keyPoints: [
      'Dual-key system with primary and secondary for zero-downtime rotation',
      '30-day grace period with warnings and verification steps',
      'Comprehensive monitoring and emergency revocation capabilities',
    ],
  },
];

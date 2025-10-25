import { MultipleChoiceQuestion } from '@/lib/types';

export const authenticationJwtOauth2MultipleChoice = [
  {
    question: 'Why should passwords be hashed with bcrypt instead of SHA256?',
    options: [
      'bcrypt is faster than SHA256',
      'bcrypt is intentionally slow and has built-in salt, making brute force attacks infeasible',
      'SHA256 produces longer hashes',
      'bcrypt is easier to implement',
    ],
    correctAnswer: 1,
    explanation:
      'bcrypt is intentionally computationally expensive (~100ms per hash), making brute force attacks infeasible. It also has built-in salting (prevents rainbow table attacks). SHA256 is fast (~1ms) which is BAD for passwords—attacker can try billions of passwords per second! bcrypt: 100ms × 10 attempts = 1 second. SHA256: 0.001ms × 1 billion attempts = 1 second. Attacker can try 100 million times more passwords with SHA256! bcrypt also has cost factor (adjustable work factor) to stay secure as hardware improves. Never use MD5, SHA1, or plain SHA256 for passwords. Always use bcrypt, scrypt, or argon2.',
  },
  {
    id: 'fastapi-auth-mc-2',
    question: 'What is the purpose of the "exp" claim in a JWT token?',
    options: [
      'To encrypt the token',
      'To specify when the token expires (Unix timestamp)',
      'To identify the user',
      'To specify the token issuer',
    ],
    correctAnswer: 1,
    explanation:
      '"exp" (expiration time) is a registered JWT claim containing Unix timestamp when token expires. Example: "exp": 1700000000 means token expires at that Unix time. Server validates: if datetime.utcnow().timestamp() > exp: raise HTTPException(401, "Token expired"). This limits damage from stolen tokens—even if attacker gets token, it\'s only valid for duration specified. Typical values: access tokens 15-30 minutes, refresh tokens 7-30 days. Other JWT claims: "iat" (issued at), "sub" (subject/user ID), "iss" (issuer), "aud" (audience).',
  },
  {
    id: 'fastapi-auth-mc-3',
    question:
      'What is the difference between access tokens and refresh tokens?',
    options: [
      'There is no difference—they are interchangeable',
      'Access tokens are short-lived for API access; refresh tokens are long-lived to get new access tokens',
      'Refresh tokens are encrypted; access tokens are not',
      'Access tokens can be revoked; refresh tokens cannot',
    ],
    correctAnswer: 1,
    explanation:
      'Access tokens: short-lived (15-30 min), used for API requests, sent with every request (Authorization: Bearer <token>), not stored server-side (stateless). Refresh tokens: long-lived (7-30 days), used ONLY to get new access tokens, sent to /token/refresh endpoint, stored in database (for revocation). Flow: 1. Login → get both tokens, 2. Use access token for API calls, 3. When access token expires → use refresh token to get new access token, 4. Repeat. Benefits: Compromised access token has short window (15 min), refresh token can be revoked server-side, balance security (short access) and UX (long refresh, no re-login).',
  },
  {
    id: 'fastapi-auth-mc-4',
    question: 'What is the purpose of OAuth2PasswordBearer in FastAPI?',
    options: [
      'To hash passwords',
      'To extract the token from the Authorization header',
      'To generate JWT tokens',
      'To store user sessions',
    ],
    correctAnswer: 1,
    explanation:
      'OAuth2PasswordBearer extracts the token from the Authorization header. Usage: oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token"). As dependency: def get_token(token: str = Depends(oauth2_scheme)). It looks for: Authorization: Bearer <token> header, extracts <token> part, returns token string, raises 401 if header missing. Does NOT validate token—just extracts it. You still need to decode/validate the JWT yourself. The tokenUrl parameter tells OpenAPI docs where to get tokens (for "Try it out" feature in /docs). Essential for FastAPI authentication flow.',
  },
  {
    id: 'fastapi-auth-mc-5',
    question:
      'Why should JWT secret keys be stored in environment variables, not code?',
    options: [
      'Environment variables are faster to access',
      'Prevents secret exposure if code is pushed to version control or shared',
      'Environment variables are encrypted automatically',
      'It makes the code run faster',
    ],
    correctAnswer: 1,
    explanation:
      'Hardcoding secrets in code is CRITICAL security vulnerability. If code is: pushed to GitHub (even private repos—can be leaked), shared with contractors, deployed (logs may expose it), anyone with code access can forge tokens! Environment variables: not in version control (.env in .gitignore), different per environment (dev/staging/prod keys separate), can be rotated without code changes, stored in secure secret managers (AWS Secrets Manager, HashiCorp Vault). Usage: SECRET_KEY = os.getenv("JWT_SECRET_KEY") or settings.jwt_secret_key. Never: SECRET_KEY = "hardcoded-secret-key" in code. If secret exposed: Rotate immediately, invalidate all tokens, audit for unauthorized access.',
  },
].map(({ id, ...q }, idx) => ({ id: `fastapi-mc-${idx + 1}`, ...q }));

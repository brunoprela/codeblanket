/**
 * Multiple choice questions for OAuth 2.0 - Authorization Framework section
 */

export const oauth2MultipleChoice = [
  {
    id: 'mc1',
    question: 'What is the primary purpose of OAuth 2.0?',
    options: [
      'User authentication',
      'Delegated authorization - granting limited access without sharing passwords',
      'Data encryption',
      'Session management',
    ],
    correctAnswer: 1,
    explanation:
      'OAuth 2.0 is an authorization framework, not authentication. Its purpose is delegated authorization - allowing users to grant third-party apps limited access to their resources without sharing passwords. Example: "Allow Spotify to see your Facebook friends" without giving Spotify your Facebook password.',
  },
  {
    id: 'mc2',
    question: 'What is the difference between access token and refresh token?',
    options: [
      'Access token is for authentication, refresh token is for authorization',
      'Access token grants API access (short-lived), refresh token obtains new access tokens (long-lived)',
      'They are the same thing',
      'Access token is encrypted, refresh token is plain text',
    ],
    correctAnswer: 1,
    explanation:
      'Access token grants access to protected resources and is short-lived (typically 1 hour) for security. Refresh token is long-lived and used to obtain new access tokens without user interaction. This separation allows frequent token rotation (security) while maintaining user convenience.',
  },
  {
    id: 'mc3',
    question: 'Why is PKCE important for mobile apps and SPAs?',
    options: [
      'PKCE makes apps faster',
      'PKCE prevents authorization code interception attacks since public clients cannot keep client_secret safe',
      'PKCE is required by law',
      'PKCE reduces server load',
    ],
    correctAnswer: 1,
    explanation:
      'PKCE (Proof Key for Code Exchange) is critical for public clients (mobile apps, SPAs) because they cannot securely store client_secret - anyone can decompile the app or inspect code. PKCE uses dynamically-generated code_verifier that never leaves the client, preventing attackers from using intercepted authorization codes.',
  },
  {
    id: 'mc4',
    question: 'What is the purpose of "scope" in OAuth 2.0?',
    options: [
      'To identify the user',
      'To define what specific access/permissions are granted',
      'To encrypt the token',
      'To set token expiration time',
    ],
    correctAnswer: 1,
    explanation:
      'Scope defines the specific permissions granted by the access token. Examples: read:contacts, write:posts, admin. This implements principle of least privilege - app requests only what it needs. User sees scope in permission dialog: "Allow App to read contacts?" and can approve/deny.',
  },
  {
    id: 'mc5',
    question: 'Which OAuth 2.0 flow should modern mobile apps use?',
    options: [
      'Implicit Flow',
      'Password Flow',
      'Authorization Code Flow with PKCE',
      'Client Credentials Flow',
    ],
    correctAnswer: 2,
    explanation:
      'Authorization Code Flow with PKCE is recommended for mobile apps. Implicit Flow is deprecated (insecure), Password Flow requires app to see user password (bad), Client Credentials is for machine-to-machine (no user). PKCE solves the public client problem securely.',
  },
  {
    id: 'mc6',
    question:
      'Why should you NOT use OAuth 2.0 access tokens for authentication?',
    options: [
      'Access tokens are too slow',
      'Access tokens grant authorization (what you can access), not authentication (who you are)',
      'Access tokens expire too quickly',
      'Access tokens are too large',
    ],
    correctAnswer: 1,
    explanation:
      "OAuth 2.0 is designed for authorization, not authentication. Access token grants access to resources but doesn't reliably identify the user. For authentication, use OIDC (OpenID Connect) which provides ID token specifically designed to identify users. This is a common critical mistake.",
  },
];

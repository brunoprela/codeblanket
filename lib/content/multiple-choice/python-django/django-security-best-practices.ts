import { MultipleChoiceQuestion } from '@/lib/types';

export const DjangoSecurityBestPracticesMultipleChoice = [
  {
    id: 1,
    question:
      'Which setting should you enable to force all connections to use HTTPS in production?',
    options: [
      'A) FORCE_HTTPS = True',
      'B) SECURE_SSL_REDIRECT = True',
      'C) HTTPS_ONLY = True',
      'D) REQUIRE_SSL = True',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) SECURE_SSL_REDIRECT = True**

\`\`\`python
# settings.py
SECURE_SSL_REDIRECT = True  # Redirects HTTP to HTTPS
SECURE_HSTS_SECONDS = 31536000  # Enable HSTS
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
\`\`\`

This ensures all traffic uses encrypted HTTPS connections.
      `,
  },
  {
    question: 'How does Django prevent SQL injection attacks?',
    options: [
      'A) By validating all user input',
      'B) By automatically parameterizing ORM queries',
      'C) By escaping SQL keywords',
      'D) By using prepared statements manually',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) By automatically parameterizing ORM queries**

\`\`\`python
# Safe - parameterized
User.objects.filter(username=user_input)

# Safe - raw with params
User.objects.raw('SELECT * FROM users WHERE id = %s', [user_id])

# UNSAFE
User.objects.raw(f'SELECT * FROM users WHERE id = {user_id}')
\`\`\`

Django ORM parameterizes queries automatically, preventing SQL injection.
      `,
  },
  {
    question: 'What is the purpose of CORS (Cross-Origin Resource Sharing)?',
    options: [
      'A) To encrypt data transmission',
      'B) To control which domains can access your API',
      'C) To prevent SQL injection',
      'D) To manage user sessions',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) To control which domains can access your API**

\`\`\`python
CORS_ALLOWED_ORIGINS = [
    'https://app.example.com',
]

# Allows only specified domains to make API requests
\`\`\`

CORS prevents unauthorized domains from accessing your API from browsers.
      `,
  },
  {
    question: 'Which password hasher is considered most secure in Django?',
    options: [
      'A) MD5PasswordHasher',
      'B) PBKDF2PasswordHasher',
      'C) Argon2PasswordHasher',
      'D) SHA1PasswordHasher',
    ],
    correctAnswer: 2,
    explanation: `
**Correct Answer: C) Argon2PasswordHasher**

\`\`\`python
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.Argon2PasswordHasher',  # Most secure
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
]
\`\`\`

Argon2 won the Password Hashing Competition and is most resistant to attacks.
      `,
  },
  {
    question: 'How do you protect Django templates from XSS attacks?',
    options: [
      'A) Use {% safe %} tag',
      'B) Django auto-escapes template variables by default',
      'C) Use escape() function manually',
      'D) Install XSS protection package',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Django auto-escapes template variables by default**

\`\`\`django
{{ user_input }}  <!-- Auto-escaped -->

{{ trusted_html|safe }}  <!-- Mark safe when needed -->
\`\`\`

Django templates automatically escape variables, preventing XSS attacks.
      `,
  },
].map(({ id, ...q }, idx) => ({ id: `django-mc-${idx + 1}`, ...q }));

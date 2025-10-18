/**
 * Multiple choice questions for HTTP/HTTPS Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const httphttpsfundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'http-idempotent',
    question:
      'Which of the following HTTP methods is both safe and idempotent?',
    options: ['POST', 'GET', 'PATCH', 'All of the above'],
    correctAnswer: 1,
    explanation:
      "GET is both safe (doesn't modify server state) and idempotent (multiple identical requests have the same effect as one). POST is neither safe nor idempotent. PATCH modifies state so it's not safe, and its idempotency depends on implementation.",
  },
  {
    id: 'http-status-codes',
    question:
      "A user is authenticated but tries to access a resource they don't have permission for. What status code should the API return?",
    options: [
      '401 Unauthorized',
      '403 Forbidden',
      '404 Not Found',
      '400 Bad Request',
    ],
    correctAnswer: 1,
    explanation:
      "403 Forbidden is correct because the user is authenticated but not authorized. 401 Unauthorized means authentication is required (user hasn't logged in). This is a common interview question!",
  },
  {
    id: 'http2-benefit',
    question:
      'What is the primary advantage of HTTP/2 multiplexing over HTTP/1.1?',
    options: [
      'It uses UDP instead of TCP',
      'It eliminates the need for SSL/TLS',
      'It allows multiple requests/responses simultaneously on one connection',
      'It compresses the request body',
    ],
    correctAnswer: 2,
    explanation:
      'HTTP/2 multiplexing allows multiple requests and responses to be sent simultaneously over a single TCP connection, eliminating head-of-line blocking at the HTTP layer. HTTP/2 still uses TCP (not UDP), requires TLS, and header compression (not body compression) is a separate feature.',
  },
  {
    id: 'https-tls',
    question:
      'During the TLS handshake, which key type is used to encrypt the session key exchange?',
    options: [
      "The server's private key",
      "The server's public key (from the certificate)",
      'A symmetric session key',
      "The client's private key",
    ],
    correctAnswer: 1,
    explanation:
      "The client uses the server's public key (from the certificate) to encrypt the pre-master secret. The server then decrypts it using its private key. This allows secure key exchange over an insecure channel. After this exchange, both parties derive symmetric session keys for efficient encryption.",
  },
  {
    id: 'http-caching',
    question:
      'Which Cache-Control directive ensures a resource is NEVER cached, even in the browser?',
    options: [
      'Cache-Control: no-cache',
      'Cache-Control: private',
      'Cache-Control: no-store',
      'Cache-Control: max-age=0',
    ],
    correctAnswer: 2,
    explanation:
      'Cache-Control: no-store tells browsers and intermediary caches to never store the response. "no-cache" means you must revalidate before using cached version (it still caches). "private" means only browser can cache (not CDNs). "max-age=0" means expired immediately but still cached.',
  },
];

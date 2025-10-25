import { MultipleChoiceQuestion } from '@/lib/types';

export const middlewareCorsMultipleChoice = [
  {
    id: 1,
    question:
      'In FastAPI middleware, why does the order in which middleware is added matter?',
    options: [
      'Middleware executes in the order added for requests (outer to inner) and reverse order for responses (inner to outer), like an onion',
      'Only the first middleware in the list will execute',
      'Middleware order determines which endpoints they apply to',
      "Order doesn't matter - all middleware execute in parallel",
    ],
    correctAnswer: 0,
    explanation:
      "Middleware forms layers like an onion. Request flow: outer middleware → inner middleware → endpoint → inner middleware → outer middleware (response). Example with 3 middleware (A, B, C added in that order): Request: A (before) → B (before) → C (before) → endpoint → C (after) → B (after) → A (after). Why order matters: 1) Exception handler should be outermost to catch errors from all other middleware, 2) Request ID should be early so logging can use it, 3) Authentication should be inner (close to routes) so it sees processed request, 4) Rate limiting before auth prevents credential discovery attacks. Bad order example: If logging is before request ID generation, logs won't have request IDs. If exception handler is innermost, it won't catch errors from outer middleware. Production pattern: Exception handling → Request ID → Logging → CORS → Security headers → Rate limiting → Auth → Routes. Middleware don't execute in parallel (option 4), they execute sequentially. All middleware execute (option 2), not just the first. Order doesn't determine endpoint applicability (option 3).",
  },
  {
    id: 2,
    question:
      'What is the Same-Origin Policy and why does it necessitate CORS configuration?',
    options: [
      'Browsers block JavaScript from making requests to different origins (protocol+domain+port), requiring servers to explicitly allow cross-origin requests via CORS headers',
      'The Same-Origin Policy prevents servers from accepting requests from different domains',
      'CORS is only needed for POST requests, not GET requests',
      'The Same-Origin Policy is a server-side security feature',
    ],
    correctAnswer: 0,
    explanation:
      "Same-Origin Policy (SOP) is a browser security mechanism that restricts JavaScript from accessing responses from different origins. Origin = protocol + domain + port. Examples of DIFFERENT origins: https://example.com vs https://api.example.com (different subdomain), https://example.com vs http://example.com (different protocol), https://example.com:443 vs https://example.com:8000 (different port). Without SOP, malicious site evil.com could use your session cookies to make requests to bank.com and steal your data. How it works: Browser sends request with Origin header, server responds with Access-Control-Allow-Origin, browser checks if origin is allowed, if not allowed browser BLOCKS the response (request still happens, but JavaScript can't see response). CORS is the mechanism to selectively relax SOP. Important: SOP is client-side (browser enforcement), not server-side (option 2). CORS applies to ALL methods including GET (option 3). SOP is browser feature (option 4), servers don't enforce it. Production: Configure CORS to allow your frontend domains while blocking malicious sites.",
  },
  {
    id: 3,
    question:
      'What is a CORS preflight request and when does the browser send it?',
    options: [
      'An OPTIONS request sent before the actual request to check if the cross-origin request is allowed, triggered for non-simple requests',
      'A request that checks if the server is online before sending data',
      'A security scan that runs on all requests',
      'Preflight is only sent for file uploads',
    ],
    correctAnswer: 0,
    explanation:
      "Preflight is an OPTIONS request the browser automatically sends before certain cross-origin requests to check permissions. Triggers for non-simple requests: 1) Methods other than GET, HEAD, POST (e.g., PUT, DELETE, PATCH), 2) Custom headers (e.g., Authorization), 3) Content-Type other than application/x-www-form-urlencoded, multipart/form-data, or text/plain. Preflight flow: 1) Browser sends OPTIONS request with Origin, Access-Control-Request-Method, Access-Control-Request-Headers, 2) Server responds with Access-Control-Allow-Origin, Allow-Methods, Allow-Headers, 3) Browser checks if request is allowed, 4) If allowed, browser sends actual request, 5) If not, browser blocks and shows CORS error. Example: DELETE /users/1 with Authorization header → preflight OPTIONS request → server allows → actual DELETE request. Simple requests (GET/POST with simple content-type, no custom headers) skip preflight. Performance: Preflight can be cached with Access-Control-Max-Age: 3600 (1 hour), reducing OPTIONS requests. Preflight isn't for server health checks (option 2), security scans (option 3), or only file uploads (option 4).",
  },
  {
    id: 4,
    question:
      "Why can't you use Access-Control-Allow-Origin: * with Access-Control-Allow-Credentials: true?",
    options: [
      "It's a security risk - allowing all origins with credentials would let any malicious site access authenticated endpoints using users' cookies",
      'It causes performance issues with too many allowed origins',
      "The wildcard doesn't work with credentials for technical reasons",
      'This combination is allowed and commonly used',
    ],
    correctAnswer: 0,
    explanation:
      'Allowing all origins (*) with credentials (cookies, Authorization header) is a critical security vulnerability. Attack scenario: 1) User logs into bank.com (sets cookie), 2) User visits evil.com, 3) evil.com JavaScript: fetch("https://bank.com/api/transfer", {credentials: "include", body: {to: "attacker", amount: 1000}}), 4) If bank.com used * with credentials=true, browser would send cookies and transfer money! The security fix: Browsers block * with credentials. You must specify exact origins: Access-Control-Allow-Origin: https://app.example.com with Access-Control-Allow-Credentials: true. Why: This ensures only YOUR frontend can make authenticated requests, not evil.com. Production pattern: maintain allowlist of your domains, dynamically return requesting origin if in allowlist. Options: * with credentials is BLOCKED by browsers (option 3), not just a performance issue (option 2), and definitely not allowed/common (option 4). This is fundamental CORS security. Never use * for authenticated APIs - always specify exact origins.',
  },
  {
    id: 5,
    question:
      'What is the purpose of rate limiting middleware and where should it be positioned in the middleware stack?',
    options: [
      'Rate limiting prevents abuse by limiting requests per time window; it should be before authentication to prevent credential discovery attacks through timing',
      'Rate limiting is for load balancing across multiple servers',
      'Rate limiting should be after authentication to identify premium users',
      'Rate limiting is only needed for public APIs, not authenticated ones',
    ],
    correctAnswer: 0,
    explanation:
      "Rate limiting prevents abuse: DDoS attacks, credential stuffing, API scraping, resource exhaustion. Implementation: Track requests per identifier (IP for anonymous, user_id for authenticated), limit to N requests per time window (e.g., 100/minute), return 429 Too Many Requests when exceeded. Position in middleware stack: BEFORE authentication. Why before auth: Prevents attackers from discovering valid credentials through timing attacks - if auth is first, attacker can make 1000 login attempts to find valid usernames/passwords before rate limit applies. If rate limit is first, attacker is blocked at 10 attempts regardless of auth outcome. Example: Attacker tries user@example.com with 1000 passwords → rate limit blocks after 10 attempts → protects auth system. Rate limiting isn't for load balancing (option 2 - that's load balancer's job), and shouldn't be after auth (option 3 - defeats security purpose). Rate limiting is crucial for both public AND authenticated APIs (option 4) - authenticated users can still abuse. Common limits: Anonymous: 10-100/min, Free users: 100-1000/min, Premium: 10,000/min. Implementation: Use Redis for distributed rate limiting across multiple API servers.",
  },
].map(({ id, ...q }, idx) => ({ id: `fastapi-mc-${idx + 1}`, ...q }));

/**
 * Quiz questions for Microservices Security section
 */

export const microservicessecurityQuiz = [
  {
    id: 'q1-security',
    question:
      'Why is security more challenging in microservices compared to monoliths? What additional measures are needed?',
    sampleAnswer:
      'Microservices increase attack surface: (1) All communication over network (vs in-process function calls) - can be intercepted, (2) Multiple entry points (vs single perimeter), (3) Authentication needed at each service, (4) More services to secure and monitor. Additional measures: (1) mTLS for service-to-service authentication, (2) API Gateway for centralized auth, (3) Network policies to restrict communication, (4) Service mesh for automatic encryption, (5) Distributed audit logging, (6) Secrets management (Vault) for credentials. Defense in depth is critical - multiple layers of security so if one fails, others protect the system.',
    keyPoints: [
      'More attack surface (network communication, multiple services)',
      'Need mTLS for service-to-service authentication',
      'API Gateway centralizes user authentication',
      'Network policies restrict which services can communicate',
      'Defense in depth: multiple security layers',
    ],
  },
  {
    id: 'q2-security',
    question:
      'Explain mTLS (Mutual TLS). How does it differ from regular TLS? Why is it important for microservices?',
    sampleAnswer:
      "Regular TLS: Only server authenticates itself (client verifies server certificate). Example: Your browser verifies bank.com is really the bank. mTLS (Mutual TLS): BOTH client and server authenticate each other with certificates. Both parties present and verify certificates. Important for microservices because: (1) Services need to verify each other (Order Service verifies it's really talking to Payment Service, not an attacker), (2) Prevents spoofing attacks, (3) Encrypts all communication, (4) Works automatically with service mesh (Istio). Without mTLS, attacker could pretend to be a service. Service mesh makes mTLS automatic - provisions certificates, rotates them, handles all crypto without code changes.",
    keyPoints: [
      'mTLS: both parties authenticate (vs TLS: only server)',
      'Both present and verify certificates',
      'Prevents service spoofing attacks',
      'Service mesh (Istio) makes it automatic',
      'Certificates rotate automatically',
    ],
  },
  {
    id: 'q3-security',
    question:
      'What security vulnerabilities should you watch for when accepting user input? How do you prevent them?',
    sampleAnswer:
      'Vulnerabilities: (1) SQL Injection - attacker injects SQL code (email="\' OR \'1\'=\'1"). Prevention: parameterized queries/prepared statements. (2) NoSQL Injection - attacker sends {email: {"$ne": null}}. Prevention: validate input types with schema validator (Joi, Zod). (3) XSS (Cross-Site Scripting) - attacker injects JavaScript. Prevention: escape HTML output, use Content-Security-Policy header. (4) Command Injection - attacker injects shell commands. Prevention: avoid exec(), validate inputs. General rule: NEVER trust user input. Always validate (whitelist approach, not blacklist), sanitize output, use parameterized queries, and implement input length limits. Use validation libraries (Joi, Zod) rather than custom regex.',
    keyPoints: [
      'SQL Injection: use parameterized queries',
      'NoSQL Injection: validate input types with schema',
      'XSS: escape HTML output, CSP headers',
      'Never trust user input - always validate',
      'Use validation libraries (Joi, Zod)',
    ],
  },
];

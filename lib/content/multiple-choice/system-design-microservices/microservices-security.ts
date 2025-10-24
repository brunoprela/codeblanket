/**
 * Multiple choice questions for Microservices Security section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const microservicessecurityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-security-1',
    question:
      'What is the primary purpose of mTLS (Mutual TLS) in microservices?',
    options: [
      'To make services faster',
      'To authenticate both client and server, encrypting all communication',
      'To reduce network latency',
      'To store passwords securely',
    ],
    correctAnswer: 1,
    explanation:
      "mTLS (Mutual TLS) ensures both the client and server authenticate each other using certificates, and encrypts all communication. This prevents service spoofing (attacker pretending to be Payment Service) and eavesdropping. Regular TLS only authenticates the server. In microservices, services need to verify they're talking to legitimate services, not attackers. Service mesh (Istio) can automate mTLS setup. Option 1 is wrong (mTLS adds slight overhead). Option 3 is wrong (mTLS adds latency). Option 4 is unrelated (that's hashing/encryption).",
  },
  {
    id: 'mc-security-2',
    question:
      'Why should you use parameterized queries instead of string concatenation for database queries?',
    options: [
      'Parameterized queries are faster',
      'To prevent SQL injection attacks',
      'To reduce database load',
      'For better error messages',
    ],
    correctAnswer: 1,
    explanation:
      "Parameterized queries (prepared statements) prevent SQL injection by separating SQL code from data. With string concatenation, attacker can inject malicious SQL: email=\"' OR '1'='1\" becomes SELECT * FROM users WHERE email=' OR '1'='1' (returns all users!). Parameterized queries treat user input as data, not code: db.query(\"SELECT * FROM users WHERE email=$1\", [email]). The database escapes input automatically. This is the #1 defense against SQL injection. Option 1 is sometimes true but not the main reason. Options 3 and 4 are not primary benefits.",
  },
  {
    id: 'mc-security-3',
    question:
      'What is the principle of "defense in depth" in microservices security?',
    options: [
      'Use only one very strong security measure',
      'Multiple layers of security so if one fails, others protect the system',
      'Deep encryption algorithms',
      'Hiding services from the internet',
    ],
    correctAnswer: 1,
    explanation:
      'Defense in depth means multiple security layers: network firewalls, API Gateway auth, service mesh mTLS, service-level authorization, encryption, audit logging. If one layer is breached, others still protect. Example: Attacker bypasses API Gateway → still blocked by network policies → still can\'t decrypt mTLS → still blocked by service-level auth. Single security measure (Option 1) is risky - single point of failure. Option 3 misunderstands "depth". Option 4 is security by obscurity (bad practice).',
  },
  {
    id: 'mc-security-4',
    question:
      'Where should you store sensitive secrets (API keys, passwords) in Kubernetes?',
    options: [
      'Hardcoded in application code',
      'In Kubernetes Secrets or external secret management (HashiCorp Vault)',
      'In environment variables in Dockerfile',
      'In source control (Git)',
    ],
    correctAnswer: 1,
    explanation:
      'Use Kubernetes Secrets (base64 encoded, access-controlled) or external secret management systems (HashiCorp Vault, AWS Secrets Manager) for sensitive data. These provide encryption, access control, audit logging, and secret rotation. Option 1 is extremely dangerous (secrets in code). Option 3 exposes secrets in container images. Option 4 is worst (secrets in Git history forever). Never commit secrets to source control. Use secret management tools with proper encryption and access controls.',
  },
  {
    id: 'mc-security-5',
    question:
      'What is the purpose of Network Policies in Kubernetes for microservices security?',
    options: [
      'To make network faster',
      'To restrict which pods can communicate with each other',
      'To encrypt data at rest',
      'To load balance traffic',
    ],
    correctAnswer: 1,
    explanation:
      "Network Policies restrict pod-to-pod communication at network level. Example: Only Order Service can call Payment Service, Payment Service can only access its database. This implements principle of least privilege and limits blast radius if a service is compromised. Without network policies, any pod can talk to any pod (dangerous). Option 1 is wrong (no performance benefit). Option 3 is wrong (that's encryption at rest). Option 4 is wrong (that's Services/Ingress).",
  },
];

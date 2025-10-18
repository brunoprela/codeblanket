/**
 * Multiple choice questions for Service Mesh section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const servicemeshMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-mesh-1',
    question: 'What is the primary purpose of a service mesh?',
    options: [
      'To replace Kubernetes',
      'To handle service-to-service communication with observability, security, and traffic management via sidecar proxies',
      'To store data in a distributed database',
      'To replace the need for load balancers',
    ],
    correctAnswer: 1,
    explanation:
      "Service mesh handles service-to-service (east-west) communication via sidecar proxies. It provides observability (tracing, metrics), security (mTLS, authorization), resilience (circuit breakers, retries), and traffic management (canary, A/B testing) without code changes. Option 1 is wrong (service mesh runs ON Kubernetes). Option 3 is unrelated. Option 4 is wrong (service mesh includes load balancing but doesn't replace external LBs).",
  },
  {
    id: 'mc-mesh-2',
    question:
      'How does a service mesh provide mTLS encryption without changing application code?',
    options: [
      'It modifies the application binary at runtime',
      'Services must import a special library',
      'Sidecar proxies handle encryption/decryption, services talk to localhost',
      'It uses a special compiler',
    ],
    correctAnswer: 2,
    explanation:
      'Service mesh uses sidecar proxy pattern: each service has a proxy container. Service talks to its localhost proxy, which handles mTLS encryption with the destination proxy. Control plane provisions certificates to proxies automatically. Service code unchanged - just calls http://payment-service, proxy intercepts and encrypts. This works with any programming language. Option 1 is wrong (no binary modification). Option 2 is wrong (no libraries needed). Option 4 makes no sense.',
  },
  {
    id: 'mc-mesh-3',
    question:
      'What are the two main components of a service mesh architecture?',
    options: [
      'Frontend and Backend',
      'Master and Worker nodes',
      'Data Plane (proxies) and Control Plane (management)',
      'Load Balancer and Database',
    ],
    correctAnswer: 2,
    explanation:
      'Service mesh has two planes: (1) Data Plane - sidecar proxies (Envoy) that handle all network traffic, enforce policies, collect metrics, (2) Control Plane - management layer (Istio/Linkerd) that configures proxies, collects telemetry, provides APIs. Data plane does the work, control plane tells it what to do. Options 1, 2, and 4 are unrelated concepts.',
  },
  {
    id: 'mc-mesh-4',
    question: 'When should you consider using a service mesh?',
    options: [
      'Always use service mesh from day one',
      'Only for external APIs',
      'When you have 15+ microservices, polyglot environment, strict security needs, or complex traffic management',
      "Never, it's deprecated technology",
    ],
    correctAnswer: 2,
    explanation:
      'Use service mesh when: (1) 15+ microservices (complexity justifies overhead), (2) Polyglot environment (Java, Node, Go), (3) Strict security requirements, (4) Need advanced traffic management, (5) Running on Kubernetes. For < 10 services, simpler solutions often suffice. Option 1 is wrong (adds complexity for small deployments). Option 2 confuses service mesh with API gateway. Option 4 is wrong (service mesh is actively used by major companies).',
  },
  {
    id: 'mc-mesh-5',
    question:
      'What is the typical latency overhead of a service mesh per request?',
    options: [
      'No overhead',
      '2-5 milliseconds (proxies add small latency)',
      '500+ milliseconds',
      '10+ seconds',
    ],
    correctAnswer: 1,
    explanation:
      "Service mesh adds 2-5ms latency per hop because each request goes through 2 proxies (source sidecar and destination sidecar). Linkerd is faster (~1-2ms), Istio ~2-5ms. For most applications, this is negligible compared to benefits. For ultra-low-latency systems (HFT, real-time gaming), this might matter. Option 1 is wrong (there's always some overhead). Options 3 and 4 are way too high (that would be unacceptable).",
  },
];

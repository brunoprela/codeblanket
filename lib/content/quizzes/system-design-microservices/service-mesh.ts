/**
 * Quiz questions for Service Mesh section
 */

export const servicemeshQuiz = [
  {
    id: 'q1-mesh',
    question:
      'What is a service mesh and what problems does it solve? How is it different from an API gateway?',
    sampleAnswer:
      'A service mesh is an infrastructure layer that handles service-to-service communication (east-west traffic) via sidecar proxies. It solves cross-cutting concerns: (1) Observability - automatic tracing, metrics, logging, (2) Security - automatic mTLS encryption, authorization policies, (3) Resilience - circuit breakers, retries, timeouts, (4) Traffic management - canary deployments, A/B testing. All WITHOUT code changes. API Gateway handles north-south traffic (client to services) and provides single entry point, authentication, rate limiting. Service mesh handles east-west (service-to-service) and provides per-service proxies. They complement each other: API Gateway at edge, service mesh for internal communication.',
    keyPoints: [
      'Service mesh: infrastructure layer for service-to-service (east-west) communication',
      'Sidecar proxies provide observability, security, resilience, traffic management',
      'No code changes needed (vs libraries)',
      'API Gateway: edge (north-south), Service Mesh: internal (east-west)',
      'They complement each other',
    ],
  },
  {
    id: 'q2-mesh',
    question:
      'When would you choose NOT to use a service mesh? What are the trade-offs?',
    sampleAnswer:
      "Skip service mesh when: (1) < 5-10 microservices - overhead not justified, simpler solutions work, (2) Performance-critical applications - service mesh adds 2-5ms latency per hop, (3) Team lacks operational maturity - service mesh adds significant complexity, (4) Simple communication patterns - don't need advanced traffic management. Trade-offs: PROS: mTLS, circuit breakers, retries, tracing, rate limiting without code changes, consistent policies across polyglot services. CONS: adds latency (2-5ms), operational complexity (another thing to manage), resource overhead (proxy containers), learning curve. Start with libraries (Resilience4j) or API gateway, graduate to service mesh as complexity grows.",
    keyPoints: [
      'Skip for: < 10 services, performance-critical, simple patterns, immature ops team',
      'Pros: powerful features without code changes, consistent policies',
      'Cons: latency (+2-5ms), operational complexity, resource overhead',
      'Trade-off: features vs complexity',
      'Start simple, graduate to mesh as needed',
    ],
  },
  {
    id: 'q3-mesh',
    question:
      'Explain the sidecar proxy pattern. How does it enable mTLS without code changes?',
    sampleAnswer:
      "Sidecar proxy pattern: Each service instance has a companion proxy container (Envoy) deployed alongside it. Service communicates with localhost:port (its sidecar), sidecar handles all network communication. For mTLS: (1) Control plane (Istio) provisions certificates to each sidecar, (2) When Order Service calls Payment Service, it calls localhost:15001 (its sidecar), (3) Order\'s sidecar establishes mTLS connection with Payment's sidecar, (4) Payment's sidecar decrypts and forwards to localhost:8080 (Payment Service), (5) Certificates rotate automatically. Service code unchanged - just calls http://payment-service, proxy intercepts and encrypts. This is powerful because: no library dependencies, works with any language, centralized certificate management, automatic rotation.",
    keyPoints: [
      'Sidecar proxy: companion container per service instance',
      'Service talks to localhost (proxy), proxy handles network',
      'Control plane provisions certificates to proxies',
      'Proxies establish mTLS between themselves',
      'Service code unchanged (no library, works with any language)',
    ],
  },
];

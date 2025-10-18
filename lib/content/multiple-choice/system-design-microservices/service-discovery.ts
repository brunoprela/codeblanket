/**
 * Multiple choice questions for Service Discovery & Registry section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const servicediscoveryMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-discovery-1',
    question:
      'What is the main advantage of client-side service discovery over server-side discovery?',
    options: [
      'Simpler client implementation',
      'No need for health checks',
      'Lower latency (no extra network hop) and client controls load balancing',
      'Works with any programming language',
    ],
    correctAnswer: 2,
    explanation:
      'Client-side discovery has lower latency because clients call service instances directly without going through a load balancer (no extra hop). Clients also have full control over the load balancing algorithm. However, this comes at the cost of more complex client implementation. Option 1 is wrong (client-side is actually more complex). Option 2 is wrong (health checks are still needed). Option 4 is wrong (server-side discovery is more language-agnostic).',
  },
  {
    id: 'mc-discovery-2',
    question:
      'Your payment service autoscales from 2 to 20 instances during Black Friday. How does service discovery handle this?',
    options: [
      'Manual configuration update required for all clients',
      'New instances self-register with registry; clients automatically discover them',
      'Need to restart all client services',
      'Load balancer must be manually reconfigured',
    ],
    correctAnswer: 1,
    explanation:
      'Service discovery automatically handles dynamic scaling. New instances self-register with the service registry on startup (or are registered by an orchestrator like Kubernetes). Clients query the registry and automatically get the updated list of instances. No manual intervention needed. This is the whole point of service discovery - handling dynamic infrastructure. Options 1, 3, and 4 all describe manual processes that service discovery eliminates.',
  },
  {
    id: 'mc-discovery-3',
    question:
      'What happens if a service instance fails but remains registered in the service registry?',
    options: [
      'Nothing - the system continues to work normally',
      'Clients receive errors when trying to call the dead instance',
      'The service registry automatically detects and removes it after N failed health checks',
      'All instances of that service must be restarted',
    ],
    correctAnswer: 2,
    explanation:
      'Health checks prevent routing to dead instances. The service registry periodically checks instance health (heartbeat or active polling). After N consecutive failed health checks, the registry marks the instance as DOWN and stops returning it in discovery queries. Clients never see the dead instance. Option 2 can happen temporarily, but proper health checking minimizes this window. Option 1 is wrong (dead instances cause problems). Option 4 is unnecessary overkill.',
  },
  {
    id: 'mc-discovery-4',
    question: 'In Kubernetes, how does built-in service discovery work?',
    options: [
      'Kubernetes uses Netflix Eureka internally',
      'Services must manually register with etcd',
      'Kubernetes creates DNS entries for Services; pods are automatically discovered via label selectors',
      'Each pod must implement health check endpoints',
    ],
    correctAnswer: 2,
    explanation:
      "Kubernetes has built-in service discovery: You create a Service resource with label selectors. Kubernetes automatically discovers all pods matching those labels and adds them to the Service. CoreDNS creates DNS entries (service-name.namespace.svc.cluster.local) that resolve to the Service's ClusterIP, which load balances to healthy pods. It's completely automatic - no manual registration needed. Option 1 is false (Kubernetes doesn't use Eureka). Option 2 is false (automatic via label matching). Option 4 is good practice but not required for basic discovery.",
  },
  {
    id: 'mc-discovery-5',
    question:
      'Why is DNS-based service discovery considered simpler but more limited than registry-based discovery?',
    options: [
      'DNS is slower than registries',
      'DNS requires special client libraries',
      'DNS has caching issues (TTL) and limited load balancing options, but is universally supported',
      'DNS only works with HTTP services',
    ],
    correctAnswer: 2,
    explanation:
      "DNS-based discovery is simpler because every programming language has built-in DNS support - no special libraries needed. However, it's limited: (1) DNS caching (TTL) means clients may get stale data, (2) DNS provides limited load balancing (usually just round-robin A records), (3) DNS doesn't remove unhealthy instances quickly. Registry-based discovery (Eureka, Consul) provides real-time updates, sophisticated load balancing, and immediate health-based routing. Option 1 is debatable. Option 2 is backwards (DNS needs no libraries). Option 4 is false (DNS works with any protocol).",
  },
];

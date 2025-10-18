/**
 * Multiple choice questions for Inter-Service Communication section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interservicecommunicationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc-communication-1',
      question:
        'For an e-commerce checkout flow, you need to check inventory, process payment, and send confirmation email. Which communication pattern is BEST?',
      options: [
        'All synchronous REST calls (inventory, payment, email)',
        'All asynchronous message queue (inventory, payment, email)',
        'Synchronous for inventory and payment, asynchronous for email',
        'Asynchronous for inventory and payment, synchronous for email',
      ],
      correctAnswer: 2,
      explanation:
        "Option 3 is correct because: (1) Inventory check and payment processing are critical to checkout success and need immediate results - user is waiting and can't proceed without confirmation, so these should be synchronous. (2) Email notification can be sent asynchronously - user doesn't need to wait for email to be sent, and if email service is temporarily down, it shouldn't block the order. Option 1 is worse because email failure would block order. Option 2 is wrong because user needs immediate confirmation of inventory/payment. Option 4 makes no sense (why would email be synchronous but payment async?).",
    },
    {
      id: 'mc-communication-2',
      question:
        'Your Order Service calls User Service, which calls Product Service, which calls Inventory Service in sequence. What is the main problem?',
      options: [
        'Too many services - should be combined into one',
        'Sequential network calls create latency waterfall and tight coupling',
        'REST is the wrong protocol - should use gRPC',
        'Services need to share a database instead',
      ],
      correctAnswer: 1,
      explanation:
        "Sequential (synchronous chain) communication creates two major problems: (1) Latency adds up in a waterfall (if each call is 10ms, total is 40ms), and (2) Tight coupling - if any service in the chain is down or slow, the entire flow fails. Solutions include: making calls in parallel where possible, using an aggregation service (BFF pattern), or redesigning to use asynchronous events. Option 1 (merge services) defeats microservices benefits. Option 3 (gRPC) would be slightly faster but doesn't solve fundamental problem. Option 4 (shared database) is an anti-pattern.",
    },
    {
      id: 'mc-communication-3',
      question:
        'When an order is placed, 5 different services need to be notified (Payment, Inventory, Shipping, Analytics, Notification). What pattern should you use?',
      options: [
        'Synchronous REST calls from Order Service to all 5 services in sequence',
        'Synchronous REST calls from Order Service to all 5 services in parallel',
        'Order Service publishes "OrderPlaced" event to message broker, services subscribe',
        'All 5 services poll Order Service database every minute for new orders',
      ],
      correctAnswer: 2,
      explanation:
        'Pub/Sub (publish/subscribe) pattern is ideal when multiple services need to react to the same event. Order Service publishes a single "OrderPlaced" event, and each service subscribes independently. Benefits: (1) Loose coupling - Order Service doesn\'t know/care who subscribes, (2) Easy to add new subscribers without changing Order Service, (3) Parallel processing - all services process simultaneously, (4) Each service can fail/retry independently. Option 1 (sequential REST) creates tight coupling and latency. Option 2 (parallel REST) is better but still couples Order Service to all 5 services. Option 4 (polling) is inefficient and adds latency.',
    },
    {
      id: 'mc-communication-4',
      question:
        'What is the main benefit of a service mesh like Istio for microservices communication?',
      options: [
        'Service mesh replaces HTTP with faster binary protocol',
        'Service mesh handles cross-cutting concerns (retries, circuit breakers, mTLS, tracing) without changing service code',
        'Service mesh eliminates the need for asynchronous communication',
        'Service mesh combines all microservices into a single deployment',
      ],
      correctAnswer: 1,
      explanation:
        "Service mesh's primary value is handling cross-cutting concerns at the infrastructure layer via sidecar proxies: automatic retries, circuit breakers, timeouts, mutual TLS encryption, distributed tracing, and load balancing - all without developers writing code for these in each service. This is huge for polyglot environments where services are written in different languages. Option 1 is false (service mesh works with HTTP/gRPC). Option 3 is false (you still need async for some use cases). Option 4 is completely wrong (service mesh is for distributed microservices).",
    },
    {
      id: 'mc-communication-5',
      question:
        'Which API change is backward-compatible and does NOT require API versioning?',
      options: [
        'Removing the "phoneNumber" field from User response',
        'Changing "amount" field from number to string',
        'Adding a new optional field "email" to User response',
        'Renaming "userId" field to "id"',
      ],
      correctAnswer: 2,
      explanation:
        'Adding optional fields is backward-compatible because: (1) Old clients will ignore the new field (JSON parsers skip unknown fields), (2) New clients can use the new field, (3) No existing functionality breaks. Options 1, 2, and 4 are all breaking changes: removing fields breaks clients expecting them, changing types breaks parsers, renaming fields breaks all references. Backward-compatible changes include: adding optional fields, adding new endpoints, making required fields optional, adding enum values. Breaking changes require versioning (/v1/ vs /v2/) or deprecation periods.',
    },
  ];

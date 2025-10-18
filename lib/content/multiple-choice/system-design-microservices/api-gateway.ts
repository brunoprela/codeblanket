/**
 * Multiple choice questions for API Gateway Pattern section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apigatewayMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-gateway-1',
    question: 'What is the primary purpose of an API Gateway in microservices?',
    options: [
      'Store user session data',
      'Single entry point that handles routing, authentication, and request aggregation',
      'Replace the need for load balancers',
      'Store business logic to keep services simple',
    ],
    correctAnswer: 1,
    explanation:
      'The API Gateway serves as a single entry point for all client requests. It handles cross-cutting concerns like routing to appropriate services, authentication/authorization, rate limiting, request aggregation, and protocol translation. It simplifies the client by hiding backend complexity. Option 1 is wrong (gateways are usually stateless). Option 3 is wrong (gateways often work WITH load balancers). Option 4 is an anti-pattern (business logic belongs in services, not gateway).',
  },
  {
    id: 'mc-gateway-2',
    question:
      'Your mobile app loads the home screen slowly because it makes 8 separate API calls. What API Gateway pattern should you use?',
    options: [
      'Rate Limiting',
      'Circuit Breaker',
      'API Composition (aggregate multiple calls into one)',
      'Service Discovery',
    ],
    correctAnswer: 2,
    explanation:
      "API Composition (also called request aggregation or BFF pattern) solves this problem. The gateway makes multiple backend calls internally and returns a single combined response to the client. This reduces network round trips from 8 to 1, dramatically improving load times on mobile networks. Option 1 (Rate Limiting) prevents abuse but doesn't help with performance. Option 2 (Circuit Breaker) handles failures but doesn't reduce calls. Option 4 (Service Discovery) is for services finding each other, not client optimization.",
  },
  {
    id: 'mc-gateway-3',
    question: 'Which of these is an API Gateway anti-pattern?',
    options: [
      'Implementing authentication in the gateway',
      'Routing requests to different microservices',
      'Implementing complex business logic (tax calculation, discount rules) in the gateway',
      'Caching responses to reduce backend load',
    ],
    correctAnswer: 2,
    explanation:
      'Putting business logic in the gateway is an anti-pattern called "Smart Gateway, Dumb Services". Business logic should live in services, not the gateway. If you add another gateway or bypass it, the logic is missing. Gateway should handle cross-cutting concerns (auth, routing, caching) but not business rules. Options 1, 2, and 4 are legitimate gateway responsibilities. Keep the gateway focused on infrastructure concerns, not business logic.',
  },
  {
    id: 'mc-gateway-4',
    question: 'What is the Backend for Frontend (BFF) pattern?',
    options: [
      'Having a single API Gateway that serves all clients equally',
      'Creating separate API Gateways optimized for each client type (mobile, web, IoT)',
      'Using a load balancer in front of services',
      'Implementing circuit breakers in the frontend',
    ],
    correctAnswer: 1,
    explanation:
      "BFF pattern creates dedicated gateways for each client type. Mobile BFF returns minimal data optimized for bandwidth; Web BFF returns richer data with more details. Each BFF can aggregate different calls, transform responses differently, and evolve independently. This provides better user experience at the cost of maintaining multiple gateways. Option 1 is wrong (that's a single gateway anti-pattern when clients have very different needs). Option 3 is unrelated. Option 4 makes no sense (circuit breakers are backend, not frontend).",
  },
  {
    id: 'mc-gateway-5',
    question:
      'API Gateway increases latency by adding an extra network hop. When might you skip using one?',
    options: [
      'When you have 50+ microservices that need centralized auth',
      'When you have mobile, web, and partner clients with different needs',
      'When you have a simple monolith with one client and latency is critical',
      'When you need rate limiting and request aggregation',
    ],
    correctAnswer: 2,
    explanation:
      "Skip API Gateway for simple applications (monolith or few services) with one client type, especially when latency is critical. The operational complexity isn't justified. Start simple and add gateway later if needed. Options 1, 2, and 4 are perfect use cases FOR an API Gateway (many services, multiple clients, cross-cutting concerns). Don't over-engineer when you don't need it (YAGNI principle).",
  },
];

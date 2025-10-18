/**
 * Quiz questions for API Gateway Pattern section
 */

export const apigatewayQuiz = [
  {
    id: 'q1-api-gateway',
    question:
      'Your mobile app makes 5 separate API calls when loading the home screen, causing slow load times. How would you solve this using API Gateway? What pattern would you use?',
    sampleAnswer:
      "I would implement the API Composition pattern in the API Gateway. Create a new endpoint like GET /api/home that internally makes all 5 backend calls in parallel using Promise.all() and returns a single combined response. This follows the BFF (Backend for Frontend) pattern, where the gateway is optimized for the mobile client's needs. This reduces network round trips from 5 to 1, dramatically improving performance on mobile networks. The gateway should make the backend calls in parallel, not sequentially, and handle partial failures gracefully (e.g., if notifications service is down, still return the rest of the data).",
    keyPoints: [
      'Use API Composition pattern to aggregate multiple backend calls',
      'Create dedicated BFF endpoint optimized for mobile',
      'Make backend calls in parallel (Promise.all) not sequential',
      'Reduces round trips from N to 1 (critical for mobile)',
      'Handle partial failures gracefully',
    ],
  },
  {
    id: 'q2-api-gateway',
    question:
      'What are the trade-offs of using an API Gateway? When might you NOT want to use one?',
    sampleAnswer:
      'API Gateway trade-offs: Adds latency (extra network hop), becomes a single point of failure (mitigated by horizontal scaling and health checks), increases operational complexity (another component to monitor), and can become a bottleneck if not scaled properly. Skip API Gateway when: (1) Simple monolith with one client type - no need for the complexity, (2) Team too small to maintain it - operational overhead not justified, (3) Latency-critical applications where every millisecond matters, (4) Internal services only - might not need centralized auth/rate limiting. Start without it and add later if needed (YAGNI principle).',
    keyPoints: [
      'Adds latency (extra network hop)',
      'Single point of failure (needs HA setup)',
      'Operational complexity',
      'Can become bottleneck',
      'Skip for simple apps, small teams, or latency-critical systems',
    ],
  },
  {
    id: 'q3-api-gateway',
    question:
      'Explain the Backend for Frontend (BFF) pattern. Why would you use separate gateways for mobile and web?',
    sampleAnswer:
      'BFF pattern creates dedicated API Gateways for each client type (mobile BFF, web BFF, IoT BFF, etc.). Mobile needs: minimal data (bandwidth constraints), aggregated responses (fewer round trips), optimized images (thumbnails not full res). Web needs: richer data, multiple images, detailed pagination. By having separate BFFs, each can be optimized for its client without compromising others. Mobile BFF might aggregate 5 calls into 1 and return 100KB, while Web BFF returns 1MB with full details. BFFs can also handle client-specific auth (mobile uses OAuth, web uses cookies, partners use API keys). The trade-off is more code to maintain, but better user experience and separation of concerns.',
    keyPoints: [
      'Separate gateway per client type (mobile, web, IoT)',
      'Mobile BFF: minimal data, aggregated calls, optimized for bandwidth',
      'Web BFF: richer data, multiple images, detailed responses',
      'Each can evolve independently',
      'Trade-off: more code vs better UX and separation of concerns',
    ],
  },
];

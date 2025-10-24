/**
 * Multiple choice questions for Rate Limiting for Safety section
 */

export const ratelimitingsafetyMultipleChoice = [
  {
    id: 'rate-limit-mc-1',
    question:
      'A user makes 100 requests in 1 minute. Your limit is 60/minute. The token bucket algorithm has capacity=60, fill_rate=1/second. What happens?',
    options: [
      'First 60 requests allowed, remaining 40 blocked',
      'All 100 requests blocked',
      'All 100 requests allowed',
      'Requests distributed evenly over the minute',
    ],
    correctAnswer: 0,
    explanation:
      'Token bucket starts with 60 tokens (capacity). First 60 requests consume all tokens and are allowed. Remaining 40 requests find no tokens and are blocked. Tokens refill at 1/second, but not fast enough for the burst. Option D describes rate smoothing, not token bucket behavior.',
  },
  {
    id: 'rate-limit-mc-2',
    question:
      'You detect 50 prompt injection attempts from one user. What is the BEST immediate action?',
    options: [
      'Send user a warning email',
      'Reduce their rate limit by 50%',
      'Block the user immediately',
      'Flag for human review',
    ],
    correctAnswer: 2,
    explanation:
      '50 injection attempts is a clear attack—block immediately. This is not accidental. Option A (warning) allows continued attacks. Option B (reduce limit) still allows attacks, just slower. Option D (review) is too slow for an active attack. Block first, review later.',
  },
  {
    id: 'rate-limit-mc-3',
    question:
      'Your rate limiter uses user reputation: trusted users get 10x higher limits. A trusted user is compromised. What is the risk?',
    options: [
      'No risk—rate limiting will still stop attacks',
      'Attacker can abuse higher limits before detection',
      'Reputation score will immediately drop to zero',
      'Higher limits are only for legitimate use',
    ],
    correctAnswer: 1,
    explanation:
      "If a trusted account is compromised, the attacker inherits the higher limits and can cause more damage before being detected. This is a real risk of reputation-based systems. Option A is wrong—higher limits mean more attack surface. Option C is wrong—reputation doesn't instantly update. Monitoring and anomaly detection are crucial.",
  },
  {
    id: 'rate-limit-mc-4',
    question:
      'You implement sliding window rate limiting: 100 requests per hour. A user makes 100 requests at 1:59 PM. Can they make more requests at 2:01 PM?',
    options: [
      "Yes—it's a new hour",
      'No—must wait until 3:00 PM',
      'Yes—requests from 1:01 PM have expired',
      'No—must wait 60 minutes from last request',
    ],
    correctAnswer: 2,
    explanation:
      'Sliding window means "100 requests in any 60-minute window". At 2:01 PM, requests made before 1:01 PM have fallen out of the window, so user has capacity again. This differs from fixed windows (option A) which reset at clock hours. Option B would be overly restrictive.',
  },
  {
    id: 'rate-limit-mc-5',
    question:
      'Your system has rate limits: Free=10/min, Paid=100/min. A free user complains about hitting limits. What is the BEST response?',
    options: [
      'Increase free tier limit to 20/min',
      'Explain limits and suggest paid upgrade',
      'Make an exception for this user',
      'Remove rate limiting for verified users',
    ],
    correctAnswer: 1,
    explanation:
      'The best response is to explain the limits and offer the paid tier upgrade. Rate limits exist for a reason (abuse prevention, cost control). Option A (increase free limit) sets bad precedent and defeats the purpose. Option C (exception) is not scalable. Option D (remove limits) is dangerous.',
  },
];

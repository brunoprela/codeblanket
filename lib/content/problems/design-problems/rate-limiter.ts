/**
 * Design Rate Limiter
 * Problem ID: rate-limiter
 * Order: 10
 */

import { Problem } from '../../../types';

export const rate_limiterProblem: Problem = {
  id: 'rate-limiter',
  title: 'Design Rate Limiter',
  difficulty: 'Medium',
  topic: 'Design Problems',
  description: `Design a rate limiter that limits the number of requests a user can make to an API within a certain time window.

Implement the \`RateLimiter\` class:

- \`RateLimiter(int maxRequests, int timeWindow)\` Initializes the rate limiter with maximum \`maxRequests\` allowed in \`timeWindow\` seconds.
- \`boolean shouldAllow(int userId, int timestamp)\` Returns \`true\` if the user with \`userId\` is allowed to make a request at the given \`timestamp\`, otherwise returns \`false\`.

The rate limiter should handle multiple users, and timestamps are given in chronological order (monotonically increasing).`,
  hints: [
    'Track request timestamps per user',
    'Remove timestamps outside the time window',
    'Deque allows O(1) removal from front',
    'Alternative: Fixed window with counter (simpler but less accurate)',
    'Token bucket algorithm for smooth rate limiting',
  ],
  approach: `## Intuition

Rate limiting ensures users don't exceed a request quota within a time window.

**Challenge**: Efficiently track and count requests in sliding time window.

---

## Approach 1: Sliding Window Log (Exact)

For each user, store request timestamps in deque:

\`\`\`python
def shouldAllow(userId, timestamp):
    # Remove old requests (outside window)
    while requests[userId] and requests[userId][0] <= timestamp - timeWindow:
        requests[userId].popleft()
    
    # Check if under limit
    if len(requests[userId]) < maxRequests:
        requests[userId].append(timestamp)
        return True
    return False
\`\`\`

**Pros**: Exact count, no boundary issues  
**Cons**: O(N) memory per user

---

## Approach 2: Fixed Window Counter (Simple)

Track count per time window:

\`\`\`python
def shouldAllow(userId, timestamp):
    window = timestamp // timeWindow
    
    if user_windows[userId] != window:
        # New window, reset
        user_windows[userId] = window
        user_counts[userId] = 0
    
    if user_counts[userId] < maxRequests:
        user_counts[userId] += 1
        return True
    return False
\`\`\`

**Pros**: O(1) memory per user  
**Cons**: Boundary spike problem (2x rate at boundaries)

---

## Approach 3: Token Bucket (Industry Standard)

Each user has bucket that refills at constant rate:

\`\`\`python
def shouldAllow(userId, timestamp):
    # Refill tokens based on time elapsed
    elapsed = timestamp - last_refill[userId]
    tokens[userId] = min(capacity, tokens[userId] + elapsed * refill_rate)
    last_refill[userId] = timestamp
    
    if tokens[userId] >= 1:
        tokens[userId] -= 1
        return True
    return False
\`\`\`

**Pros**: Smooth rate limiting, allows bursts  
**Cons**: Slightly more complex

---

## Time Complexity:
- Approach 1: O(1) amortized per request
- Approach 2: O(1) per request
- Approach 3: O(1) per request

## Space Complexity: O(U) where U = number of users`,
  testCases: [
    {
      input: [
        ['RateLimiter', 3, 60],
        ['shouldAllow', 1, 0],
        ['shouldAllow', 1, 10],
        ['shouldAllow', 1, 20],
        ['shouldAllow', 1, 30],
        ['shouldAllow', 1, 70],
      ],
      expected: [null, true, true, true, false, true],
    },
  ],
  solution: `from collections import defaultdict, deque

# Approach 1: Sliding Window Log (Exact Count)
class RateLimiter:
    def __init__(self, maxRequests: int, timeWindow: int):
        self.maxRequests = maxRequests
        self.timeWindow = timeWindow
        self.user_requests = defaultdict(deque)  # userId -> deque of timestamps
    
    def shouldAllow(self, userId: int, timestamp: int) -> bool:
        """Check if request should be allowed"""
        requests = self.user_requests[userId]
        
        # Remove requests outside time window
        while requests and requests[0] <= timestamp - self.timeWindow:
            requests.popleft()
        
        # Check if under limit
        if len(requests) < self.maxRequests:
            requests.append(timestamp)
            return True
        return False


# Approach 2: Fixed Window Counter (Simple)
class RateLimiter:
    def __init__(self, maxRequests: int, timeWindow: int):
        self.maxRequests = maxRequests
        self.timeWindow = timeWindow
        self.user_windows = {}  # userId -> current window number
        self.user_counts = defaultdict(int)  # userId -> count in window
    
    def shouldAllow(self, userId: int, timestamp: int) -> bool:
        """Check if request should be allowed"""
        window = timestamp // self.timeWindow
        
        if self.user_windows.get(userId) != window:
            # New window - reset count
            self.user_windows[userId] = window
            self.user_counts[userId] = 0
        
        if self.user_counts[userId] < self.maxRequests:
            self.user_counts[userId] += 1
            return True
        return False


# Approach 3: Token Bucket (Industry Standard)
class RateLimiter:
    def __init__(self, maxRequests: int, timeWindow: int):
        self.capacity = maxRequests
        self.refill_rate = maxRequests / timeWindow  # tokens per second
        self.user_tokens = {}  # userId -> tokens
        self.user_last_refill = {}  # userId -> last refill timestamp
    
    def shouldAllow(self, userId: int, timestamp: int) -> bool:
        """Check if request should be allowed"""
        # Initialize user if first request
        if userId not in self.user_tokens:
            self.user_tokens[userId] = self.capacity
            self.user_last_refill[userId] = timestamp
        
        # Refill tokens based on time elapsed
        elapsed = timestamp - self.user_last_refill[userId]
        self.user_tokens[userId] = min(
            self.capacity,
            self.user_tokens[userId] + elapsed * self.refill_rate
        )
        self.user_last_refill[userId] = timestamp
        
        # Check if enough tokens
        if self.user_tokens[userId] >= 1:
            self.user_tokens[userId] -= 1
            return True
        return False

# Example usage:
# limiter = RateLimiter(3, 60)  # 3 requests per 60 seconds
# limiter.shouldAllow(1, 0)    # True
# limiter.shouldAllow(1, 10)   # True
# limiter.shouldAllow(1, 20)   # True
# limiter.shouldAllow(1, 30)   # False (limit reached)
# limiter.shouldAllow(1, 70)   # True (new window)`,
  timeComplexity:
    'O(1) amortized for sliding window, O(1) for fixed window and token bucket',
  spaceComplexity: 'O(U) where U is number of users',
  patterns: ['Deque', 'Sliding Window', 'Design', 'Hash Table'],
  companies: ['Amazon', 'Google', 'Microsoft', 'Stripe', 'Cloudflare'],
};

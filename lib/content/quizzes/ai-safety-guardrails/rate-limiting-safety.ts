/**
 * Quiz questions for Rate Limiting for Safety section
 */

export const ratelimitingsafetyQuiz = [
  {
    id: 'rate-limit-q-1',
    question:
      'Design an adaptive rate limiting system that adjusts limits based on user behavior. A trusted user with 6 months of good history should have higher limits than a new user. How do you implement reputation-based rate limiting while preventing abuse?',
    hint: 'Consider reputation scoring, graduated limits, and monitoring for reputation manipulation.',
    sampleAnswer:
      "**Reputation-Based Rate Limiting:** **Reputation Score (0-1):** New user: 0.5 (default). Good behavior: +0.05 per week. Bad behavior (injection attempts, violations): -0.2 immediately. Capped at 1.0 (trusted) and 0.0 (blocked). **Rate Limits by Reputation:** def get_rate_limits(user): reputation = get_reputation(user.id). if reputation >= 0.9: # Trusted, return {capacity: 1000, fill_rate: 50}. elif reputation >= 0.7: # Good, return {capacity: 100, fill_rate: 5}. elif reputation >= 0.5: # New/neutral, return {capacity: 10, fill_rate: 1}. else: # Suspicious, return {capacity: 1, fill_rate: 0.1}. **Preventing Abuse:** (1) Reputation increase is slow (+0.05/week), decrease is fast (-0.2/incident). (2) Cap on increase: Can't go from 0.5 to 0.9 in one day. (3) Monitor sudden changes: If reputation drops 0.3 in one day, investigate. (4) Require verification for high reputation: Email verified, Payment method, Age of account. **Implementation:** reputation = 0.8, limits = get_rate_limits(user), rate_limiter = TokenBucket(capacity=limits[capacity], fill_rate=limits[fill_rate]). **Result:** Trusted users: 100x higher limits. Abuse: Quickly detected and limited.",
    keyPoints: [
      'Reputation score based on behavior history',
      'Adaptive rate limits tied to reputation',
      'Slow reputation increases, fast decreases',
      'Monitor for reputation manipulation attempts',
    ],
  },
  {
    id: 'rate-limit-q-2',
    question:
      'Your system detects 150 prompt injection attempts in 1 hour from 3 IP addresses. Design an automated response system. What actions do you take immediately, and what follow-up is needed?',
    hint: 'Consider blocking, investigation, and updating defenses.',
    sampleAnswer:
      '**Immediate Actions (First 5 Minutes):** (1) Block attacking IPs: for ip in attacking_ips: firewall.block(ip, duration="24 hours"). (2) Rate limit all users temporarily (prevent further spread): global_rate_limit = 5 requests/minute  # Down from 60. (3) Alert security team: send_alert("Security", "150 injection attempts detected from 3 IPs"). **Investigation (First Hour):** (1) Analyze attack patterns: patterns = [extract_pattern(attempt) for attempt in injection_attempts]. most_common = Counter(patterns).most_common(5). (2) Check if any attempts succeeded: for attempt in injection_attempts: if not blocked_by_defenses(attempt): CRITICAL_ALERT("Successful injection!"). (3) Identify attack type: New technique? Known technique we failed to catch? (4) Check other users from same IPs: related_users = find_users_from_ips(attacking_ips). Review their activity. **Defense Updates (Same Day):** (1) Add new patterns to detector: for pattern in most_common: injection_detector.add_pattern(pattern). (2) Update tests: add_test_case(injection_attempt, expected=BLOCKED). (3) Deploy updated detector: deploy_to_production(injection_detector_v2). **Follow-Up (Next Week):** (1) Review logs for similar patterns from other IPs. (2) Red team test with captured patterns. (3) Document attack in incident log. (4) Share findings with team in security review. **Result:** Attack stopped within 5 minutes. Defenses improved within 24 hours.',
    keyPoints: [
      'Immediate: Block IPs, rate limit globally, alert team',
      'Investigate: Analyze patterns, check for successful attempts',
      'Update defenses: Add patterns, update tests, redeploy',
      'Follow-up: Document, share learnings, improve',
    ],
  },
  {
    id: 'rate-limit-q-3',
    question:
      'You implement rate limiting but 20% of blocked requests are from legitimate power users hitting limits. Design a solution that accommodates power users without compromising security. How do you distinguish power users from attackers?',
    hint: 'Consider API keys, subscription tiers, and behavioral analysis.',
    sampleAnswer:
      '**Problem:** Power users hitting limits, but can\'t just raise limits (security risk). **Solution: Multi-Tier System:** **Tier 1: Free Users** - Limits: 10 req/min, 1000 req/day. Authentication: Email only. Reputation: Starts at 0.5. **Tier 2: Verified Users** - Limits: 100 req/min, 10,000 req/day. Requirements: Email verified + phone verified. Reputation: Starts at 0.7. **Tier 3: Paid Users** - Limits: 1,000 req/min, 100,000 req/day. Requirements: Payment method. Dedicated API key. Reputation: Starts at 0.8. **Tier 4: Enterprise** - Limits: Custom (negotiated). Dedicated infrastructure. Isolated rate limiters. Contract with SLA. **Distinguishing Power Users from Attackers:** Power User Indicators: (1) Consistent usage patterns (same time daily). (2) Clean history (no violations). (3) API key usage (not web UI). (4) Payment method on file. (5) Registered business email domain. Attacker Indicators: (1) Sudden spike in usage. (2) Injection attempts in history. (3) Accessing from VPN/Tor. (4) Disposable email. (5) New account. **Behavioral Analysis:** def classify_user(user): score = 0. if user.verified_email and user.verified_phone: score += 20. if user.payment_method: score += 30. if user.clean_history: score += 25. if user.consistent_usage: score += 15. if user.api_key_user: score += 10. if score >= 70: return "trusted_power_user". elif score >= 40: return "legitimate_user". else: return "potentially_suspicious". **Automated Tier Upgrades:** If free user hits limits repeatedly AND has high trust score: Suggest upgrade: "You\'ve hit your limits. Upgrade for higher limits?" Auto-upgrade trial: 7-day trial of next tier. **Result:** Power users get appropriate limits. Security maintained for untrusted users.',
    keyPoints: [
      'Multi-tier system with different limits per tier',
      'Behavioral analysis to identify legitimate power users',
      'Verification requirements (email, phone, payment)',
      'Automated tier suggestions for users hitting limits',
    ],
  },
];

/**
 * Quiz questions for Alerting Strategies section
 */

export const alertingStrategiesQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between "alerting on symptoms" vs "alerting on causes." Provide examples of each and explain why symptom-based alerting is preferred.',
    sampleAnswer:
      'Symptom-based alerting focuses on what users experience, while cause-based alerting focuses on infrastructure metrics. **Symptom-Based Alerts (✅ Preferred)**: Alert on user impact. Examples: (1) "API error rate > 1%" - Users experiencing failed requests. (2) "p99 latency > 500ms" - Users experiencing slow responses. (3) "Checkout conversion rate dropped 50%" - Users unable to complete purchases. Why these are good: Direct user impact, clear action needed (fix the error/latency), aligns with SLOs. **Cause-Based Alerts (❌ Often Poor)**: Alert on infrastructure. Examples: (1) "CPU > 80%" - High CPU, but users might be fine. (2) "Memory usage > 70%" - High memory, but service might be caching effectively. (3) "Disk I/O wait > 30%" - Disk busy, but not necessarily impacting users. Why these are problematic: (1) May not impact users (high CPU during batch job is fine). (2) Unclear action (should I scale up? optimize? ignore?). (3) Creates alert fatigue (many alerts with no user impact). **Real Example**: Cause-based: "CPU 90%" → Engineer investigates → Discovers it\'s a scheduled batch job, completely expected, wasted time. Symptom-based: "Error rate 5%" → Engineer investigates → Discovers database connection pool exhausted → Fixes immediately, users benefit. **Exception - Predictive Causes**: Some cause-based alerts predict future symptoms: "Disk 95% full" → Will cause service failure soon. "Memory leak detected" → Will cause crash eventually. "Error rate trending up 5%/hour" → Will exceed SLO in 2 hours. These are acceptable because they predict imminent user impact. **Best Practice**: Default to symptom-based (error rate, latency), add select predictive cause-based alerts (disk full), review and remove cause-based alerts that don\'t correlate with user impact.',
    keyPoints: [
      'Symptoms = user impact (error rate, latency), Causes = infrastructure (CPU, memory)',
      'Symptom alerts correlate with user problems and SLO violations',
      "Cause alerts often don't impact users (high CPU might be fine)",
      'Exception: Predictive causes that forecast user impact (disk 95% full)',
      'Best practice: Alert on symptoms, selectively add predictive causes',
    ],
  },
  {
    id: 'q2',
    question:
      'What is alert fatigue, what causes it, and how do you prevent it? If a team is receiving 50+ alerts per day, how would you fix the problem?',
    sampleAnswer:
      'Alert fatigue occurs when engineers receive so many alerts that they ignore them, missing critical issues. **Causes**: (1) Too Many Alerts: Alerting on everything "just in case." (2) Non-Actionable: Alerts for FYI information or things that don\'t need immediate action. (3) False Positives: Alerts fire but no real problem (static thresholds on noisy metrics). (4) No Aggregation: Same issue triggers 100 separate alerts. (5) Wrong Severity: Everything marked P1/Critical. **Consequences**: Real alerts buried in noise, delayed response to incidents, team burnout, "boy who cried wolf" syndrome. **Fixing 50+ Alerts/Day**: (1) **Alert Audit (Week 1)**: Review every alert that fired last month. For each alert ask: "Was action taken?" If no action 80% of the time → delete alert. "Could user have noticed first?" If yes → delete (monitoring should detect before users). "Is this actionable?" If unclear what to do → delete or improve. Result: Typically eliminates 50% of alerts. (2) **Aggregation (Week 2)**: Group related alerts. Instead of 20 alerts for "pod-1 down, pod-2 down, ..., pod-20 down," one alert: "50% of pods unhealthy." Use alert grouping in PagerDuty/Opsgenie. Result: Reduces alert count 3-5x. (3) **Threshold Tuning (Week 3)**: For remaining alerts, tune thresholds based on data. Change "CPU > 80%" to "CPU > 90% for 10 minutes" (filters transient spikes). Use dynamic thresholds/anomaly detection for noisy metrics. Result: Reduces false positives 50%. (4) **Severity Classification (Week 4)**: Re-classify alerts by actual urgency: P0 (immediate page): Service down, data loss, security breach. P1 (page during business hours): Degraded performance. P2 (ticket): Non-critical issues. P3 (dashboard only): Info/trends. Result: Most alerts become P2/P3, not pages. **Target**: <5 pages per week per engineer. **Measure Success**: Track alert quality metrics: % of alerts where action was taken, time to acknowledge, alert-to-incident ratio.',
    keyPoints: [
      'Alert fatigue: Too many alerts → engineers ignore them',
      'Causes: Too many, non-actionable, false positives, no aggregation',
      'Fix: Delete non-actionable alerts, aggregate related alerts, tune thresholds',
      'Re-classify by true urgency (P0-P3), most should be P2/P3',
      'Target: <5 pages/week per engineer, >80% alerts are actionable',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain alert duration and frequency configuration (initialDelay, period, failureThreshold). How would you tune these for a production API to avoid false positives while maintaining fast detection?',
    sampleAnswer:
      'Alert configuration parameters control when alerts fire and how quickly to detect issues, with trade-offs between sensitivity and false positives. **Key Parameters**: (1) **Duration**: How long condition must be true before alerting. (2) **Period**: Interval between checks. (3) **Failure Threshold**: Consecutive failures before alerting. **Example Configuration**: "Alert if error_rate > 5% for 5 minutes, checking every 1 minute, with 3 consecutive failures." **Tuning Trade-offs**: **Too Aggressive**: Period: 10 seconds, Duration: 30 seconds, Threshold: 1 failure. Result: Alerts on transient blips (single slow request, brief network hiccup). Alert fatigue from false positives. **Too Lenient**: Period: 5 minutes, Duration: 30 minutes, Threshold: 10 failures. Result: Issue affects users for 30 minutes before alert. Slow Mean Time To Detect (MTTD). **Balanced for Production API**: (1) **Error Rate Alert**: Check every: 1 minute (frequent for user-facing), Duration: 5 minutes (filters transient spikes), Threshold: 3 consecutive (requires sustained issue), Condition: error_rate > 1%. Rationale: 5-minute duration catches real issues, filters deploy blips and single errors. 3 consecutive readings = sustained problem worth investigating. (2) **Latency Alert**: Check every: 1 minute, Duration: 3 minutes (faster than errors, latency more variable), Threshold: 2 consecutive, Condition: p99_latency > 1s. Rationale: Shorter duration because latency impacts users immediately. 2 consecutive = likely real issue. (3) **Availability Alert**: Check every: 30 seconds (critical metric), Duration: 2 minutes (fast detection), Threshold: 4 consecutive, Condition: success_rate < 99%. Rationale: More frequent checks for critical availability. 2 minutes tolerance before paging. **Special Cases**: Deploy window: Increase duration to 10 minutes (expect brief errors during deploys). Low traffic (nights): Increase threshold (fewer data points, more noise). **Validation**: Simulate issues in staging with same config. Track false positive rate (<10% target).',
    keyPoints: [
      'Duration: How long condition true before alerting (filters transient blips)',
      'Period: Check frequency (1 min typical for APIs)',
      'Failure Threshold: Consecutive failures needed (prevents single-point false positives)',
      'Balanced: 5 min duration, 1 min period, 3 consecutive for error rate',
      'Trade-off: Sensitivity (fast detection) vs false positives (alert fatigue)',
    ],
  },
];

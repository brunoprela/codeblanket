export const systemMonitoringAlertingMC = [
  {
    id: 'system-monitoring-alerting-mc-1',
    question: 'What is a "p99 latency" metric?',
    options: [
      '99% of requests have latency below this value',
      '99% of requests have latency above this value',
      'The average latency of the fastest 99% of requests',
      'The latency 99 seconds ago',
    ],
    correctAnswer: 0,
    explanation:
      'Answer: 99% of requests have latency below this value.\n\n' +
      'Example:\n' +
      '- 100 orders with latencies: 10ms, 15ms, ..., 150ms (sorted)\n' +
      '- p50 (median): 50th percentile = 50ms\n' +
      '- p99: 99th percentile = 150ms (only 1% of orders slower)\n\n' +
      'Why p99 matters more than average:\n' +
      '- Average: 20ms (looks good)\n' +
      '- p99: 500ms (1% of orders are very slow!)\n' +
      '- Users experience the p99, not the average\n\n' +
      'Trading impact: If p99 order latency is 500ms, 1% of orders are delayed significantly (could miss opportunities).',
  },
  {
    id: 'system-monitoring-alerting-mc-2',
    question:
      'Your trading system sends 100 alerts per day. What is the problem?',
    options: [
      'Not enough alerts - should send more',
      'Too many alerts (alert fatigue) - thresholds too sensitive',
      'Perfect amount of alerts',
      'Alerts are too slow',
    ],
    correctAnswer: 1,
    explanation:
      'Answer: Too many alerts (alert fatigue).\n\n' +
      'Alert fatigue:\n' +
      '- 100 alerts/day = 4 alerts/hour during trading hours\n' +
      '- Engineers start ignoring alerts (false positives)\n' +
      '- Real critical alerts get missed\n\n' +
      'Best practice:\n' +
      '- **Critical alerts**: <5 per week (page on-call)\n' +
      '- **Warning alerts**: <10 per day (email/Slack)\n' +
      '- **Info**: Unlimited (logs only, no notification)\n\n' +
      'Fix:\n' +
      '1. Increase thresholds (e.g., alert on >100ms latency for 10 minutes, not 1 minute)\n' +
      '2. Combine related alerts (batch similar issues)\n' +
      '3. Auto-resolve non-critical alerts\n\n' +
      'Real-world: Top trading firms aim for <1 page per week. More pages = broken alerting.',
  },
  {
    id: 'system-monitoring-alerting-mc-3',
    question: 'What is the BEST metric to alert on for trading system health?',
    options: [
      'CPU usage >80%',
      'Memory usage >90%',
      'Order success rate <95%',
      'Disk space <10% free',
    ],
    correctAnswer: 2,
    explanation:
      'Answer: Order success rate <95%.\n\n' +
      'Why order success rate:\n' +
      '- Directly measures business impact (are orders executing?)\n' +
      "- High CPU/memory doesn't mean orders are failing\n" +
      '- Low disk space is important but not as critical as orders failing\n\n' +
      'Order success rate:\n' +
      '```\n' +
      'success_rate = (successful_orders / total_orders) * 100%\n' +
      '```\n' +
      '- Target: >99% (industry standard)\n' +
      '- Alert if <95% for 5 minutes\n' +
      '- Page on-call if <90%\n\n' +
      'Why not other metrics:\n' +
      '- **CPU 80%**: Acceptable during market spikes\n' +
      '- **Memory 90%**: May be normal for in-memory caching\n' +
      '- **Disk 10%**: Important but orders can still execute\n\n' +
      'Production: Alert on business metrics (orders, fills, P&L), not just infrastructure.',
  },
  {
    id: 'system-monitoring-alerting-mc-4',
    question:
      'Your monitoring system collects metrics every second. How long should you retain high-resolution data?',
    options: [
      'Forever (never delete)',
      '7 days (then downsample to 1-minute resolution)',
      '1 day only',
      '1 year at full resolution',
    ],
    correctAnswer: 1,
    explanation:
      'Answer: 7 days (then downsample to 1-minute resolution).\n\n' +
      'Data retention strategy:\n\n' +
      '1. **High-resolution** (1-second): Last 7 days\n' +
      '   - Use for: Real-time debugging, recent incident investigation\n' +
      '   - Storage: 1 metric × 1 value/sec × 86,400 sec/day × 7 days = 604,800 data points\n\n' +
      '2. **Medium-resolution** (1-minute): 7-90 days\n' +
      '   - Downsampled from 1-second data\n' +
      '   - Storage: 60x less (10,080 data points)\n\n' +
      '3. **Low-resolution** (1-hour): 90-365 days\n' +
      '   - For long-term trends\n' +
      '   - Storage: 3600x less (8,760 data points)\n\n' +
      'Total storage: ~100MB per metric per year (vs 3GB if keeping 1-second forever).\n\n' +
      'Real-world: Prometheus default is 15 days at full resolution, then downsample.',
  },
  {
    id: 'system-monitoring-alerting-mc-5',
    question:
      'At 2 AM, you receive a page: "Portfolio P&L dropped below -$50K". What should you do FIRST?',
    options: [
      'Immediately liquidate all positions',
      'Check if the alert is accurate (verify P&L calculation)',
      'Go back to sleep (it will fix itself)',
      'Call the CEO',
    ],
    correctAnswer: 1,
    explanation:
      'Answer: Check if the alert is accurate.\n\n' +
      'Alert response procedure:\n\n' +
      '1. **Verify alert** (first 2 minutes):\n' +
      '   - Is P&L actually -$50K or is it a bug?\n' +
      '   - Check positions table: SELECT SUM(unrealized_pnl) FROM positions\n' +
      '   - Check recent fills: Any large unexpected fills?\n' +
      '   - Check market data: Are prices stale/incorrect?\n\n' +
      '2. **If alert is accurate** (minutes 2-10):\n' +
      '   - Review positions: Which symbol(s) caused the loss?\n' +
      '   - Check if position is still actively trading\n' +
      '   - Decide: Liquidate, hold, or hedge?\n\n' +
      '3. **If alert is false** (minutes 2-5):\n' +
      '   - Fix bug (e.g., stale prices)\n' +
      '   - Silence alert\n' +
      '   - Document for post-mortem\n\n' +
      'Why not other options:\n' +
      "- **Liquidate**: May lock in losses if it's a temporary dip\n" +
      '- **Ignore**: Could lead to larger losses\n' +
      '- **Call CEO**: Only after verifying the issue is real\n\n' +
      'Real-world: 50% of pages are false positives (stale data, bug). Always verify before acting.',
  },
];

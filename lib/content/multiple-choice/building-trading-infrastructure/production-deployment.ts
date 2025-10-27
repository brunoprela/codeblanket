export const productionDeploymentMC = [
  {
    id: 'production-deployment-mc-1',
    question: 'What is the PRIMARY advantage of Blue-Green deployment?',
    options: [
      'It costs less than other deployment strategies',
      'It allows instant rollback with zero downtime',
      'It requires less testing',
      'It deploys faster than other strategies',
    ],
    correctAnswer: 1,
    explanation:
      'Answer: It allows instant rollback with zero downtime.\n\n' +
      'Blue-Green deployment:\n' +
      '- **Green** has new version\n' +
      '- **Blue** has old version (still running)\n' +
      '- If green has bugs → Switch back to blue (takes 1 second)\n\n' +
      'Rollback comparison:\n' +
      '- **Blue-Green**: 1 second (just switch traffic)\n' +
      '- **Rolling**: 10-30 minutes (redeploy old version to all servers)\n' +
      '- **Canary**: 5-10 minutes (shift traffic back to old version)\n\n' +
      'Why other options are wrong:\n' +
      '- **Cost**: Actually more expensive (2x infrastructure)\n' +
      '- **Testing**: Still requires same testing\n' +
      '- **Speed**: Same deployment speed as others\n\n' +
      'Real-world: Netflix uses Blue-Green for instant rollback capability.',
  },
  {
    id: 'production-deployment-mc-2',
    question:
      'During a canary deployment, you start with 5% traffic to the new version. What should you monitor FIRST?',
    options: [
      'CPU usage',
      'Disk space',
      'Error rate comparison (new vs old version)',
      'Number of servers',
    ],
    correctAnswer: 2,
    explanation:
      'Answer: Error rate comparison (new vs old version).\n\n' +
      'Canary monitoring priority:\n\n' +
      '1. **Error rate** (most important):\n' +
      '   - New version: 0.5% errors\n' +
      '   - Old version: 0.1% errors\n' +
      '   - **Alert**: New version has 5x higher error rate → Rollback\n\n' +
      '2. **Latency**:\n' +
      '   - p99 latency new vs old\n' +
      '   - If new version is >2x slower → Rollback\n\n' +
      '3. **Success rate**:\n' +
      '   - Order success rate new vs old\n' +
      '   - If new <95% → Rollback\n\n' +
      'Why other metrics are less important:\n' +
      "- **CPU/Disk**: May vary but don't directly indicate bugs\n" +
      "- **Server count**: Doesn't change during canary\n\n" +
      'Monitoring dashboard:\n' +
      '```\n' +
      'Error Rate:        Old: 0.1%    New: 0.5%    🚨 ALERT\n' +
      'Latency (p99):     Old: 50ms    New: 55ms    ✓ OK\n' +
      'Success Rate:      Old: 99.5%   New: 99.0%   ✓ OK\n' +
      '```',
  },
  {
    id: 'production-deployment-mc-3',
    question: 'What is the BEST time to deploy a trading system update?',
    options: [
      '9:30 AM (market open)',
      '12:00 PM (lunch time, lower volume)',
      '3:59 PM (market close)',
      '6:00 PM (after market hours)',
    ],
    correctAnswer: 3,
    explanation:
      'Answer: 6 PM (after market hours).\n\n' +
      'Deployment timing for trading systems:\n\n' +
      '**Best: 6:00 PM (After Market Close)**\n' +
      '- Market is closed (no active trading)\n' +
      '- Zero impact on live orders\n' +
      '- Time to test and rollback if needed\n' +
      '- Can deploy at a leisurely pace\n\n' +
      '**Acceptable: 12:00 PM (Lunch)**\n' +
      '- Lower trading volume\n' +
      '- But still risky (market is open)\n' +
      '- Use Blue-Green for zero downtime\n\n' +
      '**BAD: 9:30 AM (Market Open)**\n' +
      '- Highest volume (peak trading)\n' +
      '- Maximum impact if bugs\n' +
      '- Never deploy at market open\n\n' +
      '**BAD: 3:59 PM (Market Close)**\n' +
      '- Still high volume (closing auctions)\n' +
      '- Too close to EOD processes\n\n' +
      'Production rule: Deploy after market close (6 PM) unless emergency.',
  },
  {
    id: 'production-deployment-mc-4',
    question:
      'Your deployment checklist includes "rollback plan documented". Why is this important?',
    options: [
      'SEC regulatory requirement',
      'Ensures you can quickly undo changes if something goes wrong',
      'Helps write better code',
      'Makes deployment faster',
    ],
    correctAnswer: 1,
    explanation:
      'Answer: Ensures you can quickly undo changes if something goes wrong.\n\n' +
      'Rollback plan importance:\n\n' +
      '**Why document rollback:**\n' +
      '1. **Speed**: During incident, no time to figure out how to rollback\n' +
      '2. **Correctness**: Rollback can be complex (database migrations, config changes)\n' +
      '3. **Team coordination**: Any engineer should be able to execute rollback\n\n' +
      '**Rollback plan template:**\n' +
      '```\n' +
      '1. Switch load balancer back to blue environment\n' +
      '2. Rollback database migrations:\n' +
      '   $ psql -c "SELECT rollback_migration(\'2025_10_26_add_order_type\')"\n' +
      '3. Revert config changes:\n' +
      '   $ kubectl rollout undo deployment/oms\n' +
      '4. Verify: Check order success rate >99%\n' +
      '5. Timeline: Should complete in <5 minutes\n' +
      '```\n\n' +
      'Real incident:\n' +
      '- 2:00 PM: Deploy new version\n' +
      '- 2:05 PM: Bug discovered (order success rate 90%)\n' +
      '- 2:06 PM: Execute rollback plan\n' +
      '- 2:08 PM: Back to old version, success rate 99%\n' +
      '- Total impact: 3 minutes',
  },
  {
    id: 'production-deployment-mc-5',
    question:
      'After deployment, your monitoring shows: Error rate 0.5% (was 0.1%), Latency p99 110ms (was 50ms), Success rate 98% (was 99%). Should you rollback?',
    options: [
      'No - metrics are acceptable',
      'Yes - error rate increased 5x',
      'Maybe - monitor for another hour',
      'No - only rollback if success rate <95%',
    ],
    correctAnswer: 1,
    explanation:
      'Answer: Yes - error rate increased 5x.\n\n' +
      'Rollback decision matrix:\n\n' +
      '**Error rate: 0.1% → 0.5% (5x increase) 🚨**\n' +
      '- Absolute threshold: >1% = critical\n' +
      '- Relative threshold: >2x increase = warning\n' +
      '- 5x increase = **ROLLBACK**\n\n' +
      '**Latency: 50ms → 110ms (2.2x increase) 🚨**\n' +
      '- Target: <100ms\n' +
      '- Actual: 110ms (exceeds target)\n' +
      '- **ROLLBACK**\n\n' +
      '**Success rate: 99% → 98% (1% drop) 🚨**\n' +
      '- Target: >99%\n' +
      '- Actual: 98% (below target)\n' +
      '- **ROLLBACK**\n\n' +
      '**Verdict: ROLLBACK IMMEDIATELY**\n' +
      '- All three metrics degraded\n' +
      '- Error rate 5x increase is critical\n' +
      "- Don't wait for more data\n\n" +
      'Production rule:\n' +
      '- ANY metric exceeds threshold → Rollback\n' +
      '- Multiple metrics degraded → Rollback immediately\n' +
      '- Better to rollback and investigate than let issues persist',
  },
];

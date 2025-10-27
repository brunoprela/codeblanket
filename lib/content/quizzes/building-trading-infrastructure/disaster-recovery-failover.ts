export const disasterRecoveryFailoverQuiz = [
  {
    id: 'disaster-recovery-failover-q-1',
    question:
      'Your trading system primary server crashes at 10:00 AM. Your DR plan has RTO=5 minutes and RPO=1 second. Explain the failover process and how you meet these targets.',
    sampleAnswer:
      'Failover Process (RTO=5 min, RPO=1 sec):\n\n' +
      '**Architecture:**\n' +
      '- Primary server (active) + Secondary server (passive)\n' +
      '- Database streaming replication (1-second lag)\n' +
      '- Load balancer for automatic routing\n\n' +
      '**Failover Timeline:**\n\n' +
      '**10:00:00** - Primary server crashes\n' +
      '**10:00:03** - Health check detects failure (3 missed heartbeats)\n' +
      '**10:00:05** - Failover monitor triggers failover\n' +
      '**10:00:10** - Secondary database promoted to primary (pg_promote)\n' +
      '**10:00:15** - Secondary trading server starts accepting orders\n' +
      '**10:00:20** - Load balancer routes traffic to secondary\n' +
      '**10:00:30** - DNS updated to point to secondary\n' +
      '**10:04:00** - Full failover complete (monitoring confirms)\n\n' +
      '**RTO: 4 minutes ✓** (within 5-minute target)\n' +
      '**RPO: 1 second ✓** (streaming replication lag)',
    keyPoints: [
      'Architecture: Active-passive servers, database streaming replication (1-sec lag), load balancer for routing',
      'Detection: Health check detects failure in 3 seconds (3 missed heartbeats)',
      'Failover: Promote passive database to primary (pg_promote), start accepting orders, route traffic',
      'RTO: 4 minutes (within 5-minute target), from detection to full operation',
      'RPO: 1 second (streaming replication lag), last 1 second of data may be lost',
    ],
  },
  {
    id: 'disaster-recovery-failover-q-2',
    question:
      'Explain the difference between RTO (Recovery Time Objective) and RPO (Recovery Point Objective). Why do trading systems typically require RTO<5min and RPO<1sec?',
    sampleAnswer:
      'RTO vs RPO:\n\n' +
      '**RTO (Recovery Time Objective):**\n' +
      '- How long system can be down before recovering\n' +
      '- Example: RTO = 5 minutes means system must be back online within 5 minutes of failure\n' +
      '- Trading impact: Every minute of downtime = missed trading opportunities\n\n' +
      '**RPO (Recovery Point Objective):**\n' +
      '- How much data can be lost in a disaster\n' +
      '- Example: RPO = 1 second means last 1 second of data may be lost\n' +
      '- Trading impact: Lost orders = regulatory violations, customer complaints\n\n' +
      '**Why trading requires RTO<5min:**\n' +
      '- Market moves fast (S&P 500 can move 1% in 5 minutes)\n' +
      '- Miss opportunities during downtime\n' +
      '- Institutional clients require 99.99% uptime\n\n' +
      '**Why trading requires RPO<1sec:**\n' +
      '- Orders must not be lost (regulatory requirement)\n' +
      '- Customer trust (lost orders = lawsuits)\n' +
      '- At 100 orders/sec, 1 second RPO = max 100 lost orders',
    keyPoints: [
      'RTO: Downtime tolerance (5 min = system must recover within 5 min), trading misses opportunities during downtime',
      'RPO: Data loss tolerance (1 sec = last 1 sec of data may be lost), lost orders = regulatory violations',
      'Trading RTO<5min: Market moves fast (1% in 5 min), miss opportunities, institutional clients require 99.99% uptime',
      'Trading RPO<1sec: Orders must not be lost (regulatory), customer trust, 100 orders/sec × 1 sec = max 100 lost orders',
      'Implementation: Active-passive servers (RTO), streaming replication (RPO)',
    ],
  },
  {
    id: 'disaster-recovery-failover-q-3',
    question:
      'Design a disaster recovery test plan. How often should you test DR, and what should you test?',
    sampleAnswer:
      'DR Test Plan:\n\n' +
      '**Testing Frequency:**\n' +
      '1. **Monthly**: Automated failover test (non-market hours)\n' +
      '2. **Quarterly**: Full DR drill (entire team)\n' +
      '3. **Annually**: Disaster simulation (entire company)\n\n' +
      '**Monthly Automated Test (2 AM on first Sunday):**\n' +
      '- Simulate primary server failure\n' +
      '- Verify secondary takes over automatically\n' +
      '- Check RTO/RPO metrics\n' +
      '- Restore primary and failback\n' +
      '- Document: 30 minutes\n\n' +
      '**Quarterly Full DR Drill (Saturday):**\n' +
      '- Entire trading team participates\n' +
      '- Simulate primary datacenter failure\n' +
      '- Verify all systems failover correctly\n' +
      '- Test order flow end-to-end\n' +
      '- Document: 4 hours\n\n' +
      '**What to Test:**\n' +
      '- Database failover\n' +
      '- Application failover\n' +
      '- Network routing\n' +
      '- Monitoring/alerting\n' +
      '- Order flow (send test orders)\n' +
      '- Team communication (who gets paged?)',
    keyPoints: [
      'Testing frequency: Monthly automated (2 AM Sunday), quarterly full drill (Saturday), annual disaster simulation',
      'Monthly test: Simulate primary failure, verify automatic failover, check RTO/RPO, restore and failback (30 min)',
      'Quarterly drill: Entire team, simulate datacenter failure, test end-to-end order flow (4 hours)',
      'What to test: Database failover, application failover, network routing, monitoring, order flow, team communication',
      'Documentation: Record RTO/RPO achieved, issues found, improvements needed for next test',
    ],
  },
];

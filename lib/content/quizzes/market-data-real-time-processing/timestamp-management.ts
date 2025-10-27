export const timestampManagementQuiz = [
  {
    id: 'timestamp-management-q-1',
    question:
      'Your trading system clock drifts +50ms over 24 hours. Using NTP, design synchronization: (1) How often to sync? (2) Detect unacceptable drift (> 100ms)? (3) Handle sync during market hours? (4) Measure impact on strategy latency?',
    sampleAnswer:
      'NTP synchronization strategy: (1) Sync frequency: Every 1 hour (3600s). More frequent = unnecessary network load, less frequent = higher drift risk. At +50ms/day drift rate, hourly sync keeps drift < 2.1ms. (2) Drift detection: After each sync, check offset. If |offset| > 100ms, alert and force immediate resync. Log all offsets for trend analysis. (3) Market hours: Sync during off-hours (pre-market 4-9:30am, lunch 12-1pm, after-hours). Avoid sync during active trading (9:30am-4pm) as brief clock jump could cause strategy errors. (4) Latency: NTP adds 10-50ms per sync (query + apply). During sync, timestamp() calls may block briefly (< 1ms). Measure by comparing strategy latency before/after sync.',
    keyPoints: [
      'Sync frequency: Every 1 hour balances drift control (< 2.1ms between syncs) vs network load',
      'Drift detection: Alert if |offset| > 100ms, log all offsets, track trend over days',
      'Market hours: Sync during off-hours (pre-market, lunch, after-hours) to avoid disrupting active trading',
      'Latency impact: NTP sync = 10-50ms network delay, brief < 1ms clock adjustment pause',
      'Drift rate: +50ms/day = +2.1ms/hour, hourly sync keeps system within 5ms of atomic time',
    ],
  },
  {
    id: 'timestamp-management-q-2',
    question:
      'Compare NTP vs PTP for HFT market making with 10μs latency target. Analyze: (1) Accuracy, (2) Cost, (3) Complexity, (4) When to use each?',
    sampleAnswer:
      'NTP vs PTP: (1) Accuracy: NTP = 1-10ms, PTP = <1μs. For 10μs strategy latency target, timestamp accuracy must be < 1μs (10% of total), so PTP required. NTP 1ms >> 10μs (unacceptable). (2) Cost: NTP = free (software only). PTP = $1K-5K (PTP-capable NIC) + $10K+ (PTP grandmaster clock) + network switches. (3) Complexity: NTP = trivial (ntpd daemon). PTP = complex (hardware, ptp4l daemon, network config, grandmaster sync). (4) Use cases: NTP for non-HFT (> 1ms latency OK, low cost). PTP for HFT (< 1ms required, cost justified by trading profits). Example: Market making with 100μs latency target → PTP essential (1μs accuracy = 1% of budget). Swing trading with 100ms latency → NTP sufficient (5ms accuracy = 5% of budget).',
    keyPoints: [
      'Accuracy: NTP 1-10ms vs PTP <1μs (1000× better), PTP required for <100μs strategies',
      'Cost: NTP free vs PTP $10K-20K (NIC + grandmaster + switches), PTP only for HFT',
      'Complexity: NTP trivial (software) vs PTP complex (hardware + network config)',
      'Rule: Timestamp accuracy should be <10% of strategy latency target (10μs strategy → <1μs timestamps → PTP)',
      'Decision: NTP for retail/institutional (>1ms OK), PTP for HFT market making (<1ms required)',
    ],
  },
  {
    id: 'timestamp-management-q-3',
    question:
      'Regulate (SEC CAT) requires microsecond timestamps within 100ms of atomic clock. You use NTP (10ms accuracy). Are you compliant? If not, fix it.',
    sampleAnswer:
      'Compliance analysis: (1) Requirement: Microsecond PRECISION (can represent 1μs granularity) AND within 100ms ACCURACY of NIST atomic clock. (2) NTP provides: 10ms accuracy (OK, < 100ms threshold ✓) but Python datetime only has microsecond precision ✓. So technically compliant. (3) However, 10ms accuracy is close to 100ms limit - no safety margin. Better: Upgrade NTP to pool.ntp.org tier-1 servers for 1-2ms accuracy (10× safety margin). (4) Best: Use GPS-synchronized NTP server in datacenter (<1ms) or PTP for <1μs. (5) Compliance check: Daily report clock offset vs NIST. Alert if offset > 50ms (early warning before 100ms violation). (6) Audit trail: Log all timestamps with source (exchange, system, NTP server) for regulatory inspection. Verdict: Technically compliant but risky. Upgrade to tier-1 NTP for safety.',
    keyPoints: [
      'CAT requirements: Microsecond PRECISION (1μs granularity) + 100ms ACCURACY (vs atomic clock)',
      'NTP compliance: 10ms accuracy < 100ms limit ✓, but no safety margin (risky)',
      'Solution: Upgrade to tier-1 NTP servers (1-2ms accuracy = 50× safety margin)',
      'Monitoring: Daily report offset vs NIST, alert if > 50ms (early warning)',
      'Best practice: GPS-synchronized NTP (<1ms) or PTP (<1μs) for HFT',
    ],
  },
];

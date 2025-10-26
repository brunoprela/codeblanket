export const systemMonitoringAlertingQuiz = [
    {
        id: 'system-monitoring-alerting-q-1',
        question:
            'Design a monitoring system for a trading platform executing 10,000 orders/day. What metrics would you track and what alert thresholds would you set?',
        sampleAnswer:
            'Trading Monitoring System:\n\n' +
            '**Key Metrics:**\n\n' +
            '1. **Order Metrics:**\n' +
            '   - Orders/sec (rate)\n' +
            '   - Order latency p50/p99 (histogram)\n' +
            '   - Order rejection rate (%)\n' +
            '   - Fill rate (%)\n\n' +
            '2. **Position/P&L Metrics:**\n' +
            '   - Portfolio P&L (gauge)\n' +
            '   - Position concentration (% in single symbol)\n' +
            '   - Margin usage (%)\n\n' +
            '3. **System Metrics:**\n' +
            '   - CPU usage (%)\n' +
            '   - Memory usage (%)\n' +
            '   - Network latency (ms)\n' +
            '   - Database query time (ms)\n\n' +
            '**Alert Thresholds:**\n' +
            '- Order latency >100ms for 5 minutes → Warning\n' +
            '- Order latency >1s → Critical (page on-call)\n' +
            '- Order rejection rate >10% → Warning\n' +
            '- P&L < -$50K → Critical\n' +
            '- CPU >80% for 10 minutes → Warning',
        keyPoints: [
            'Order metrics: Orders/sec, latency (p50/p99), rejection rate, fill rate',
            'Position/P&L: Portfolio P&L, position concentration, margin usage',
            'System metrics: CPU, memory, network latency, database query time',
            'Alert thresholds: Latency >100ms (warning), >1s (critical), rejection >10%, P&L <-$50K',
            'Alerting tiers: Warning (email), Critical (page on-call), Auto-remediation (restart service)',
        ],
    },
    {
        id: 'system-monitoring-alerting-q-2',
        question:
            'Your order latency suddenly spikes from 10ms to 500ms. How would you investigate and diagnose the root cause?',
        sampleAnswer:
            'Latency Spike Investigation:\n\n' +
            '**Step 1: Check system metrics (first 1 minute):**\n' +
            '- CPU: Is it maxed out (100%)?\n' +
            '- Memory: Is swap being used?\n' +
            '- Network: Is there packet loss?\n' +
            '- Database: Are queries slow?\n\n' +
            '**Step 2: Check trading metrics:**\n' +
            '- Orders/sec: Did volume spike?\n' +
            '- Queue depth: Is message queue backing up?\n' +
            '- Broker latency: Is broker responding slowly?\n\n' +
            '**Step 3: Check logs:**\n' +
            '- Error rate: Increased errors?\n' +
            '- GC pauses: Garbage collection (if Java/Python)?\n' +
            '- Lock contention: Thread waiting on locks?\n\n' +
            '**Common causes:**\n' +
            '1. Load spike (solution: scale horizontally)\n' +
            '2. Database slow query (solution: add index)\n' +
            '3. Network congestion (solution: check broker connectivity)\n' +
            '4. GC pause (solution: tune GC settings)\n\n' +
            'Resolution: Fix root cause and monitor for 30 minutes',
        keyPoints: [
            'Check system metrics: CPU maxed out, memory swap, network packet loss, database slow queries',
            'Check trading metrics: Order volume spike, message queue backlog, broker latency increase',
            'Check logs: Error rate, GC pauses (Java/Python), lock contention (threads waiting)',
            'Common causes: Load spike (scale), database slow query (add index), network congestion, GC pause (tune settings)',
            'Resolution: Fix root cause, monitor for 30 minutes to confirm, document incident for post-mortem',
        ],
    },
    {
        id: 'system-monitoring-alerting-q-3',
        question:
            'What is the difference between monitoring and alerting? Why do you need both?',
        sampleAnswer:
            'Monitoring vs Alerting:\n\n' +
            '**Monitoring:**\n' +
            '- Continuous data collection (orders/sec, latency, P&L)\n' +
            '- Visualized on dashboards (Grafana)\n' +
            '- Used for: Performance analysis, capacity planning, debugging\n\n' +
            '**Alerting:**\n' +
            '- Triggered when thresholds exceeded\n' +
            '- Notifies on-call engineer (PagerDuty)\n' +
            '- Used for: Incident response, preventing downtime\n\n' +
            '**Why both:**\n' +
            '- Monitoring: Proactive (see trends before problems)\n' +
            '- Alerting: Reactive (notified when problems occur)\n\n' +
            'Example: Monitor order latency continuously. Alert if latency >100ms for 5 minutes.',
        keyPoints: [
            'Monitoring: Continuous data collection, dashboards (Grafana), proactive performance analysis',
            'Alerting: Threshold-based notifications, on-call paging (PagerDuty), reactive incident response',
            'Why both: Monitoring shows trends before problems, alerting notifies when thresholds exceeded',
            'Example: Monitor order latency continuously, alert if >100ms for 5 minutes',
            'Production: 90% of time spent monitoring (optimization), 10% responding to alerts (incidents)',
        ],
    },
];


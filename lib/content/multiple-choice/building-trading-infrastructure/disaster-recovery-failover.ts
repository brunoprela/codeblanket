export const disasterRecoveryFailoverMC = [
    {
        id: 'disaster-recovery-failover-mc-1',
        question:
            'What is "active-passive" failover?',
        options: [
            'Both servers actively process orders simultaneously',
            'One server (active) processes orders, another (passive) is on standby',
            'Servers alternate processing orders every hour',
            'All servers are passive until manually activated',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: One server (active) processes orders, another (passive) is on standby.\n\n' +
            'Active-Passive pattern:\n' +
            '- **Active server**: Handles all production traffic\n' +
            '- **Passive server**: Standby, does not handle traffic\n' +
            '- **Failover**: If active fails, passive is promoted to active\n\n' +
            'Advantages:\n' +
            '- Simple (only one server active at a time)\n' +
            '- No split-brain problem (two servers thinking they\'re active)\n' +
            '- Data consistency (single source of truth)\n\n' +
            'Disadvantages:\n' +
            '- Passive server is idle (wasted resources)\n' +
            '- RTO typically 1-5 minutes (time to promote passive)\n\n' +
            'Alternative - Active-Active: Both servers handle traffic (better resource utilization, but more complex).',
    },
    {
        id: 'disaster-recovery-failover-mc-2',
        question:
            'Your RTO is 5 minutes but actual failover takes 15 minutes. What is the risk?',
        options: [
            'No risk - 15 minutes is acceptable',
            'Risk of missing trading opportunities and breaching SLAs with clients',
            'Risk of data loss',
            'Risk of too many servers running',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Risk of missing trading opportunities and breaching SLAs.\n\n' +
            'RTO breach impact:\n\n' +
            '**Business impact:**\n' +
            '- 15 minutes downtime during market hours\n' +
            '- Miss trading opportunities (market could move 0.5-1%)\n' +
            '- Potential loss: $100K+ for medium-sized fund\n\n' +
            '**Client impact:**\n' +
            '- SLA breach (contract says 5 min RTO)\n' +
            '- Client trust damaged\n' +
            '- Could lose institutional clients\n\n' +
            '**Regulatory impact:**\n' +
            '- SEC may require explanation for extended outage\n' +
            '- Must document why RTO was exceeded\n\n' +
            'Fix:\n' +
            '1. Automate failover (reduce manual steps)\n' +
            '2. Pre-warm passive server (keep it ready)\n' +
            '3. Test DR monthly (find bottlenecks)\n\n' +
            'Real-world: NYSE requires member firms to have <15 minute RTO. Most HFT firms target <1 minute.',
    },
    {
        id: 'disaster-recovery-failover-mc-3',
        question:
            'What is database "streaming replication"?',
        options: [
            'Streaming market data into a database',
            'Real-time replication of database changes to a replica',
            'Backing up database to tape',
            'Compressing database for faster queries',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Real-time replication of database changes to a replica.\n\n' +
            'PostgreSQL streaming replication:\n' +
            '1. Primary database writes changes to Write-Ahead Log (WAL)\n' +
            '2. WAL is streamed to replica(s) in real-time\n' +
            '3. Replica applies WAL entries (typically <1 second lag)\n' +
            '4. If primary fails, replica can be promoted to primary\n\n' +
            'Benefits:\n' +
            '- Low RPO (<1 second, only last second of data lost)\n' +
            '- Read scaling (queries can use replicas)\n' +
            '- Fast failover (replica already has data)\n\n' +
            'Check replication lag:\n' +
            '```sql\n' +
            'SELECT replay_lag FROM pg_stat_replication;\n' +
            '-- Result: 00:00:00.5 (500ms lag)\n' +
            '```\n\n' +
            'Alternative - Async replication: Cheaper but higher RPO (minutes of data loss).',
    },
    {
        id: 'disaster-recovery-failover-mc-4',
        question:
            'How often should you test your disaster recovery plan?',
        options: [
            'Once when first created',
            'Once per year',
            'Once per quarter',
            'Once per month',
        ],
        correctAnswer: 3,
        explanation:
            'Answer: Once per month.\n\n' +
            'Why monthly DR testing:\n\n' +
            '1. **Configuration drift**: Systems change constantly\n' +
            '   - New servers added\n' +
            '   - Network changes\n' +
            '   - Software updates\n' +
            '   - DR plan may be outdated\n\n' +
            '2. **Team turnover**: New engineers need DR training\n' +
            '   - Monthly tests ensure everyone knows the process\n' +
            '   - "Muscle memory" for incident response\n\n' +
            '3. **Catch issues early**: Monthly testing finds problems before real disaster\n' +
            '   - Example: Replica database not replicating\n' +
            '   - Better to find in test than during real outage\n\n' +
            'Best practice:\n' +
            '- **Weekly**: Automated failover test (scripts)\n' +
            '- **Monthly**: Manual DR drill (team participates)\n' +
            '- **Quarterly**: Full disaster simulation (entire company)\n\n' +
            'Real-world: Amazon requires all services to test DR monthly. Chaos engineering (intentionally breaking systems) is done continuously at Netflix.',
    },
    {
        id: 'disaster-recovery-failover-mc-5',
        question:
            'Your primary datacenter (New York) is destroyed by a natural disaster. Your secondary datacenter (Chicago) takes over. What is this called?',
        options: [
            'Backup',
            'Replication',
            'Failover',
            'Load balancing',
        ],
        correctAnswer: 2,
        explanation:
            'Answer: Failover.\n\n' +
            '**Failover**: Automatic or manual switch from failed primary to secondary system.\n\n' +
            'Disaster scenario:\n' +
            '1. NY datacenter destroyed (fire, earthquake, power outage)\n' +
            '2. Health checks detect NY is down\n' +
            '3. DNS/load balancer routes traffic to Chicago\n' +
            '4. Chicago datacenter now handles all orders\n\n' +
            'Requirements for geo-redundant DR:\n' +
            '- **Database replication**: Real-time NY → Chicago\n' +
            '- **Network**: Low-latency link between datacenters\n' +
            '- **Capacity**: Chicago sized to handle 100% of traffic\n' +
            '- **Testing**: Test geo-failover quarterly\n\n' +
            'Cost: Running two full datacenters = 2x infrastructure cost. But critical for 99.99% uptime.\n\n' +
            'Real-world: NYSE has backup datacenter in New Jersey (10 miles from Manhattan). Can failover in <15 minutes.',
    },
];


export const completeTradingSystemMC = [
    {
        id: 'complete-trading-system-mc-1',
        question:
            'What is the MOST important metric for a production trading system?',
        options: [
            'CPU usage',
            'Order success rate',
            'Disk space',
            'Number of servers',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Order success rate.\n\n' +
            'Order success rate = (Successful orders / Total orders) × 100%\n\n' +
            'Why most important:\n' +
            '- Directly measures business value (are orders executing?)\n' +
            '- Customer impact (failed orders = unhappy customers)\n' +
            '- Revenue impact (failed orders = missed opportunities)\n\n' +
            'Industry standards:\n' +
            '- **Excellent**: >99.9% (1 failure per 1,000 orders)\n' +
            '- **Good**: 99-99.9%\n' +
            '- **Poor**: <99%\n\n' +
            'Why other metrics are less important:\n' +
            '- **CPU**: High CPU doesn\'t mean orders are failing\n' +
            '- **Disk**: Can be 90% full but orders still work\n' +
            '- **Servers**: Number doesn\'t matter if orders succeed\n\n' +
            'Production alert: Page on-call if success rate <95%.',
    },
    {
        id: 'complete-trading-system-mc-2',
        question:
            'Your trading system needs to scale from 1,000 orders/day to 100,000 orders/day. What is the FIRST bottleneck you should address?',
        options: [
            'Buy more servers',
            'Profile the system to find the slowest component',
            'Rewrite everything in C++',
            'Hire more engineers',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Profile the system to find the slowest component.\n\n' +
            'Scaling strategy:\n\n' +
            '1. **Profile first** (don\'t guess):\n' +
            '   - Use profiling tools (cProfile for Python, perf for C++)\n' +
            '   - Identify bottleneck: Database? Network? CPU?\n' +
            '   - Optimize bottleneck\n\n' +
            '2. **Common bottlenecks**:\n' +
            '   - **Database queries**: Add indexes, use caching (Redis)\n' +
            '   - **Network**: Use connection pooling, reduce round trips\n' +
            '   - **CPU**: Optimize hot paths, use compiled languages\n\n' +
            '3. **Example profiling**:\n' +
            '   ```\n' +
            '   Component          Time    % Total\n' +
            '   Database query     80ms    80%      ← BOTTLENECK\n' +
            '   Order validation   10ms    10%\n' +
            '   FIX message        10ms    10%\n' +
            '   Total             100ms   100%\n' +
            '   ```\n' +
            '   → Fix database queries first (biggest impact)\n\n' +
            'Why other options are premature:\n' +
            '- **More servers**: May not help if bottleneck is single database\n' +
            '- **Rewrite in C++**: Expensive, may not address bottleneck\n' +
            '- **More engineers**: Can\'t scale faster without fixing bottleneck\n\n' +
            'Rule: Measure first, optimize second.',
    },
    {
        id: 'complete-trading-system-mc-3',
        question:
            'Which component should you implement FIRST when building a trading system from scratch?',
        options: [
            'Order Management System (OMS)',
            'Machine learning strategy',
            'Fancy dashboard',
            'Mobile app',
        ],
        correctAnswer: 0,
        explanation:
            'Answer: Order Management System (OMS).\n\n' +
            'Build order (critical to nice-to-have):\n\n' +
            '1. **OMS (Week 1)**: Core system\n' +
            '   - Order validation\n' +
            '   - Order persistence\n' +
            '   - Order routing\n' +
            '   - Can\'t trade without OMS\n\n' +
            '2. **EMS (Week 2)**: Execution\n' +
            '   - Broker connectivity\n' +
            '   - Fill processing\n\n' +
            '3. **Position Tracker (Week 3)**: Risk\n' +
            '   - Know what you own\n' +
            '   - Calculate P&L\n\n' +
            '4. **Risk Manager (Week 4)**: Safety\n' +
            '   - Position limits\n' +
            '   - P&L limits\n\n' +
            '5. **Dashboard (Week 5)**: Visibility\n' +
            '   - Nice to have, not critical\n\n' +
            '6. **ML Strategy (Week 10+)**: Alpha generation\n' +
            '   - Build after infrastructure is solid\n\n' +
            'Why OMS first:\n' +
            '- Foundation of trading system\n' +
            '- Can manually trade once OMS works\n' +
            '- Everything else depends on OMS\n\n' +
            'Common mistake: Building fancy strategies before infrastructure is solid.',
    },
    {
        id: 'complete-trading-system-mc-4',
        question:
            'Your trading system has been running for 1 year. What is the BEST way to ensure it continues to work reliably?',
        options: [
            'Never change the code',
            'Continuous monitoring, testing, and gradual improvements',
            'Rewrite from scratch every year',
            'Hire QA team to manually test everything',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Continuous monitoring, testing, and gradual improvements.\n\n' +
            'Long-term reliability strategy:\n\n' +
            '1. **Monitoring** (24/7):\n' +
            '   - Track key metrics (success rate, latency, P&L)\n' +
            '   - Alert on degradation\n' +
            '   - Dashboards for visibility\n\n' +
            '2. **Testing** (Continuous):\n' +
            '   - Unit tests (run on every commit)\n' +
            '   - Integration tests (daily)\n' +
            '   - Load tests (weekly)\n' +
            '   - DR drills (monthly)\n\n' +
            '3. **Gradual Improvements** (Ongoing):\n' +
            '   - Fix bugs as discovered\n' +
            '   - Optimize bottlenecks\n' +
            '   - Refactor technical debt\n' +
            '   - Add features incrementally\n\n' +
            '4. **Production Hygiene**:\n' +
            '   - Code reviews (2+ reviewers)\n' +
            '   - Deployment checklist\n' +
            '   - Post-mortems for incidents\n' +
            '   - Documentation updates\n\n' +
            'Why other options fail:\n' +
            '- **Never change**: Markets change, system must evolve\n' +
            '- **Rewrite yearly**: Risky, expensive, not necessary\n' +
            '- **Manual QA**: Too slow, can\'t catch everything\n\n' +
            'Real-world: Top trading firms invest 20-30% of engineering time in reliability and testing.',
    },
    {
        id: 'complete-trading-system-mc-5',
        question:
            'You are building a trading system for a client. They ask: "When will it be done?" What is the BEST answer?',
        options: [
            '"It will never be done - trading systems are continuously improved"',
            '"2 weeks"',
            '"MVP in 6 weeks, production-ready in 12 weeks, continuous improvements thereafter"',
            '"Impossible to estimate"',
        ],
        correctAnswer: 2,
        explanation:
            'Answer: MVP in 6 weeks, production-ready in 12 weeks, continuous improvements thereafter.\n\n' +
            'Trading system timeline:\n\n' +
            '**MVP (6 weeks):**\n' +
            '- Core functionality: Place orders, track positions, calculate P&L\n' +
            '- Manual testing only\n' +
            '- Demo-able but not production-ready\n\n' +
            '**Production-ready (12 weeks):**\n' +
            '- All Phase 1 features complete\n' +
            '- Automated tests (unit, integration, load)\n' +
            '- Monitoring and alerting\n' +
            '- Disaster recovery\n' +
            '- Documentation\n' +
            '- Regulatory compliance\n\n' +
            '**Continuous improvements (ongoing):**\n' +
            '- New features (smart routing, advanced strategies)\n' +
            '- Performance optimization\n' +
            '- Scale improvements\n' +
            '- Bug fixes\n\n' +
            'Why this answer works:\n' +
            '1. **Specific timeline**: Client knows when to expect MVP and production\n' +
            '2. **Realistic**: 12 weeks for production-ready is achievable\n' +
            '3. **Sets expectations**: Trading systems are never "done", always improving\n\n' +
            'Why other options fail:\n' +
            '- **"Never done"**: True but unhelpful for planning\n' +
            '- **"2 weeks"**: Unrealistic for production-ready system\n' +
            '- **"Impossible"**: Suggests poor planning\n\n' +
            'Real-world: Most trading systems take 3-6 months to go from concept to production.',
    },
];


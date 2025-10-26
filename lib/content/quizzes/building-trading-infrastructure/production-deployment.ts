export const productionDeploymentQuiz = [
    {
        id: 'production-deployment-q-1',
        question:
            'You need to deploy a critical order management system update during market hours (9:30 AM - 4:00 PM). Explain your deployment strategy to ensure zero downtime.',
        sampleAnswer:
            'Zero-Downtime Deployment During Market Hours:\n\n' +
            '**Strategy: Blue-Green Deployment**\n\n' +
            '**Timeline (Deploy at 12:00 PM - Low Volume Period):**\n\n' +
            '**11:30 AM** - Pre-deployment:\n' +
            '- Notify traders of upcoming deployment\n' +
            '- Verify green environment is ready\n' +
            '- Confirm rollback plan\n\n' +
            '**12:00 PM** - Deploy to green:\n' +
            '- Deploy new version to green environment\n' +
            '- Blue environment continues handling orders (no impact)\n\n' +
            '**12:15 PM** - Testing:\n' +
            '- Run automated smoke tests on green\n' +
            '- Send 10 test orders through green\n' +
            '- Verify: Orders execute correctly, latency <100ms, no errors\n\n' +
            '**12:30 PM** - Switch traffic:\n' +
            '- Load balancer switches 100% traffic to green\n' +
            '- Blue goes offline\n' +
            '- Total downtime: 0 seconds ✓\n\n' +
            '**12:30-1:30 PM** - Monitor:\n' +
            '- Order success rate: 99%+ ✓\n' +
            '- P99 latency: <100ms ✓\n' +
            '- Error rate: <0.1% ✓\n\n' +
            'If issues: Instant rollback (switch back to blue)',
        keyPoints: [
            'Blue-Green: Deploy to green while blue handles orders (zero downtime), switch traffic instantly',
            'Timing: 12 PM deployment (low market volume), avoid market open (9:30 AM) and close (4 PM)',
            'Testing: Automated smoke tests + 10 test orders on green before switching traffic',
            'Monitoring: Track success rate (99%+), latency (<100ms), error rate (<0.1%) for 1 hour post-deployment',
            'Rollback: If issues detected, switch back to blue instantly (zero downtime rollback)',
        ],
    },
    {
        id: 'production-deployment-q-2',
        question:
            'Compare Blue-Green vs Canary deployment for a trading system. What are the trade-offs?',
        sampleAnswer:
            'Blue-Green vs Canary:\n\n' +
            '**Blue-Green Deployment:**\n\n' +
            'Pros:\n' +
            '- Instant switchover (1 second)\n' +
            '- Easy rollback (switch back to blue)\n' +
            '- Simple to implement\n\n' +
            'Cons:\n' +
            '- All-or-nothing (100% of users on new version immediately)\n' +
            '- Requires 2x infrastructure (blue + green running simultaneously)\n' +
            '- If bug exists, impacts all users\n\n' +
            '**Canary Deployment:**\n\n' +
            'Pros:\n' +
            '- Gradual rollout (5% → 25% → 50% → 100%)\n' +
            '- Limits blast radius (bug only affects 5% initially)\n' +
            '- Can catch issues before full rollout\n\n' +
            'Cons:\n' +
            '- Slower (4 hours vs 1 minute for blue-green)\n' +
            '- More complex (traffic splitting)\n' +
            '- Two versions running simultaneously (data consistency issues)\n\n' +
            '**Recommendation:**\n' +
            '- **Low-risk changes**: Blue-Green (faster, simpler)\n' +
            '- **High-risk changes**: Canary (safer, gradual)',
        keyPoints: [
            'Blue-Green: Instant switchover (1 sec), easy rollback, but all-or-nothing (100% users immediately), requires 2x infrastructure',
            'Canary: Gradual rollout (5%→25%→50%→100%), limits blast radius, catches issues early, but slower (4 hours)',
            'Blue-Green cons: Bug impacts all users immediately, requires 2x infrastructure cost',
            'Canary cons: Slower deployment, complex traffic splitting, two versions running (data consistency issues)',
            'Recommendation: Blue-Green for low-risk changes (config updates), Canary for high-risk (algorithm changes)',
        ],
    },
    {
        id: 'production-deployment-q-3',
        question:
            'After deploying, you notice order success rate dropped from 99% to 95%. What should you do?',
        sampleAnswer:
            'Production Incident Response:\n\n' +
            '**Immediate Action (First 2 Minutes):**\n' +
            '1. **Rollback immediately**\n' +
            '   - Success rate <95% = critical issue\n' +
            '   - Switch traffic back to old version (blue-green rollback)\n' +
            '   - Takes 30 seconds\n\n' +
            '2. **Verify rollback successful**\n' +
            '   - Check success rate returns to 99%\n' +
            '   - If yes: Crisis averted\n' +
            '   - If no: Deeper issue (not deployment related)\n\n' +
            '**Investigation (Next 30 Minutes):**\n' +
            '1. **Check logs**\n' +
            '   - What errors are causing 4% failure?\n' +
            '   - Common: Timeout, validation error, broker rejection\n\n' +
            '2. **Identify root cause**\n' +
            '   - Code bug?\n' +
            '   - Configuration issue?\n' +
            '   - Broker API change?\n\n' +
            '3. **Fix and re-test**\n' +
            '   - Fix bug in green environment\n' +
            '   - Test thoroughly in staging\n' +
            '   - Deploy again (after approval)\n\n' +
            '**Production Rule:**\n' +
            'Success rate <95% = ALWAYS ROLLBACK (no exceptions)',
        keyPoints: [
            'Immediate action: Rollback within 2 minutes, 95% success rate is critical failure threshold',
            'Rollback: Switch traffic back to old version (blue-green), takes 30 seconds, verify success rate returns to 99%',
            'Investigation: Check logs for error patterns (timeout, validation, broker rejection), identify root cause',
            'Fix: Fix bug in green environment, test thoroughly in staging, re-deploy after approval',
            'Production rule: Success rate <95% = ALWAYS ROLLBACK, no exceptions, customer trust and regulatory compliance',
        ],
    },
];


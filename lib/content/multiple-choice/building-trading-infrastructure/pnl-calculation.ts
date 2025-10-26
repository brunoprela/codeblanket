export const pnlCalculationMC = [
    {
        id: 'pnl-calculation-mc-1',
        question:
            'You buy 100 shares @ $100, then sell 150 shares @ $110. What is your realized P&L? (Assume you can short sell)',
        options: [
            '$1,000 profit',
            '$1,500 profit',
            '$500 profit',
            'Cannot calculate without knowing current price',
        ],
        correctAnswer: 0,
        explanation:
            'Answer: $1,000 profit.\n\n' +
            'You sold 150 shares but only owned 100, so:\n' +
            '1. First 100 shares sold: These are LONG shares you owned\n' +
            '   - Realized P&L = 100 × ($110 - $100) = $1,000 profit\n' +
            '2. Next 50 shares sold: These are SHORT (you don\'t own them yet)\n' +
            '   - No realized P&L yet (short position is open)\n' +
            '   - Will realize P&L when you cover the short\n\n' +
            'Final position: Short 50 shares @ $110 avg cost\n' +
            'Realized P&L: $1,000 (from closing the long)\n' +
            'Unrealized P&L: Depends on current price (need to cover short)\n\n' +
            'Real-world: This is correct. Realized P&L only occurs when you close a position.',
    },
    {
        id: 'pnl-calculation-mc-2',
        question:
            'Your real-time P&L system must update within 100μs of each fill. Which storage solution is BEST?',
        options: [
            'PostgreSQL with indexed queries',
            'Redis (in-memory store)',
            'MySQL with replication',
            'MongoDB with sharding',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Redis (in-memory store).\n\n' +
            '**Latency comparison:**\n' +
            '- Redis: 0.1-10μs (in-memory, no disk I/O)\n' +
            '- PostgreSQL: 1-10ms (disk-based, index lookups)\n' +
            '- MySQL: 1-10ms (similar to PostgreSQL)\n' +
            '- MongoDB: 1-5ms (document store, still disk-based)\n\n' +
            'For 100μs P&L updates:\n' +
            '- Read position: 10μs\n' +
            '- Calculate P&L: 30μs\n' +
            '- Write position: 10μs\n' +
            '- Publish event: 20μs\n' +
            '- Aggregation: 30μs\n' +
            '- Total: ~100μs ✓\n\n' +
            '**Redis Advantages:**\n' +
            '1. All data in RAM (no disk I/O)\n' +
            '2. Single-threaded (no lock contention)\n' +
            '3. Atomic operations (INCR, GET/SET)\n' +
            '4. Pub/Sub for events\n\n' +
            '**Production Pattern:**\n' +
            '- Hot path: Redis for real-time P&L\n' +
            '- Cold path: Async write to PostgreSQL every 100ms for durability\n' +
            '- Recovery: Restore from PostgreSQL + replay Kafka log\n\n' +
            'Real-world: Citadel, Jane Street, and Two Sigma all use in-memory stores (Redis, Hazelcast, or custom) for real-time P&L.',
    },
    {
        id: 'pnl-calculation-mc-3',
        question:
            'At market close, your internal system shows $1,050,000 total P&L, but your broker reports $1,045,000. What should you do FIRST?',
        options: [
            'Immediately adjust internal P&L to match broker',
            'Investigate the $5,000 discrepancy to find the root cause',
            'Ignore it - $5K difference is immaterial for a $1M P&L',
            'Report the discrepancy to the SEC',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Investigate the $5,000 discrepancy.\n\n' +
            '**Investigation Steps:**\n\n' +
            '1. **Check Position Differences**:\n' +
            '   - Do internal positions match broker positions?\n' +
            '   - If positions differ, that explains the P&L difference\n' +
            '   - Example: Missing 100 shares of AAPL @ $150 = $15K position difference\n\n' +
            '2. **Check Price Differences**:\n' +
            '   - Are you using the same closing prices as the broker?\n' +
            '   - Brokers use official closing prices (typically 4:00 PM last trade)\n' +
            '   - If you marked at 3:59 PM, prices may differ slightly\n\n' +
            '3. **Check Fee Differences**:\n' +
            '   - Did you account for all fees? (commission, exchange, SEC, TAF)\n' +
            '   - Broker may include fees in P&L, you may track separately\n\n' +
            '4. **Check Funding/Interest**:\n' +
            '   - Did broker charge overnight interest or stock borrow fees?\n' +
            '   - These reduce P&L but are easy to miss\n\n' +
            '**Common Causes:**\n' +
            '- Position discrepancy (most common): Missing fill or duplicate fill\n' +
            '- Pricing difference: Different closing prices\n' +
            '- Fee accounting: Fees tracked differently\n' +
            '- Corporate action: Stock split, dividend not yet processed\n\n' +
            '**Why NOT the Other Options:**\n' +
            '- Option A (adjust immediately): Hides root cause, could mask a bug\n' +
            '- Option C (ignore): $5K is 0.5% of $1M - significant enough to investigate\n' +
            '- Option D (report to SEC): Not required unless you suspect fraud\n\n' +
            '**Production Practice:**\n' +
            '- Reconcile P&L daily against broker\n' +
            '- Tolerance: <0.1% difference is acceptable (due to timing/rounding)\n' +
            '- >0.5% difference: Requires investigation\n' +
            '- >1% difference: Urgent escalation to CTO\n\n' +
            'Real-world: Renaissance Technologies reconciles P&L to the penny. Any discrepancy triggers immediate investigation.',
    },
    {
        id: 'pnl-calculation-mc-4',
        question:
            'Your P&L system calculates unrealized P&L using real-time market prices. During a flash crash, prices drop 10% in 1 second. What is the RISK?',
        options: [
            'No risk - P&L accurately reflects market value',
            'Risk of false risk limits triggering automatic position liquidation',
            'Risk of overestimating profits',
            'Risk of SEC violations',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Risk of false risk limits triggering automatic liquidation.\n\n' +
            '**The Problem:**\n' +
            '1. Flash crash: Prices drop 10% in 1 second\n' +
            '2. Unrealized P&L drops 10% instantly\n' +
            '3. Risk system sees huge loss, triggers stop-loss\n' +
            '4. System automatically sells positions at crash prices\n' +
            '5. Prices recover in 10 seconds\n' +
            '6. You locked in losses that would have recovered\n\n' +
            '**Real-World Example: Knight Capital (2012)**\n' +
            '- Trading glitch caused $440M loss in 45 minutes\n' +
            '- Positions liquidated at worst prices\n' +
            '- If they had waited, losses would have been much smaller\n\n' +
            '**Solutions:**\n\n' +
            '1. **Price Validation**:\n' +
            '```python\n' +
            'def validate_price(symbol, new_price, old_price):\n' +
            '    # Reject price moves >5% in <1 second\n' +
            '    price_change_pct = abs(new_price - old_price) / old_price * 100\n' +
            '    time_diff = new_price.timestamp - old_price.timestamp\n' +
            '    \n' +
            '    if price_change_pct > 5 and time_diff < 1:\n' +
            '        # Flash crash detected - use last valid price\n' +
            '        return old_price\n' +
            '    \n' +
            '    return new_price\n' +
            '```\n\n' +
            '2. **Time-Weighted Average**:\n' +
            '   - Don\'t use instantaneous price\n' +
            '   - Use 5-minute VWAP for risk calculations\n' +
            '   - Flash crashes smooth out over 5 minutes\n\n' +
            '3. **Human Intervention**:\n' +
            '   - Alert traders on large P&L swings\n' +
            '   - Require manual approval for stop-loss >$100K\n' +
            '   - Don\'t auto-liquidate during circuit breakers\n\n' +
            '4. **Circuit Breakers**:\n' +
            '   - Pause trading if portfolio P&L drops >10% in <1 minute\n' +
            '   - Wait for human review before liquidating\n\n' +
            '**Production Practice:**\n' +
            '- Use NBBO (National Best Bid/Offer) prices, not last trade\n' +
            '- Reject prices that differ >5% from previous tick\n' +
            '- Use time-weighted prices for risk calculations\n' +
            '- Alert on large P&L swings, don\'t auto-liquidate\n\n' +
            'Real-world: After the 2010 Flash Crash, SEC implemented circuit breakers that pause trading on individual stocks that move >10% in 5 minutes.',
    },
    {
        id: 'pnl-calculation-mc-5',
        question:
            'End-of-day, your system shows $500K realized P&L and $200K unrealized P&L. For regulatory reporting, which figure matters?',
        options: [
            'Only realized P&L ($500K)',
            'Only unrealized P&L ($200K)',
            'Both realized and unrealized ($700K total P&L)',
            'Neither - only report net asset value (NAV)',
        ],
        correctAnswer: 2,
        explanation:
            'Answer: Both realized and unrealized ($700K total P&L).\n\n' +
            '**Regulatory Reporting Requirements:**\n\n' +
            '1. **Form PF (Private Fund)**:\n' +
            '   - Hedge funds >$150M AUM must report quarterly\n' +
            '   - Report TOTAL P&L (realized + unrealized)\n' +
            '   - Separate realized vs unrealized in footnotes\n\n' +
            '2. **Form 13F (Institutional Investment Manager)**:\n' +
            '   - Report long positions (market value)\n' +
            '   - Market value includes unrealized gains/losses\n\n' +
            '3. **Form ADV (Investment Adviser)**:\n' +
            '   - Report total AUM (includes unrealized)\n\n' +
            '**Why Both Matter:**\n\n' +
            '**Realized P&L ($500K):**\n' +
            '- Cash in the bank (actually earned)\n' +
            '- Taxable income (for tax reporting)\n' +
            '- Used for: Tax filing, cash flow analysis\n\n' +
            '**Unrealized P&L ($200K):**\n' +
            '- Paper gains (not yet cash)\n' +
            '- Still represents current value\n' +
            '- Used for: NAV calculation, investor reporting\n\n' +
            '**Total P&L ($700K):**\n' +
            '- Economic profit (what investors care about)\n' +
            '- Used for: Performance fees, regulatory reporting, risk management\n\n' +
            '**Example: Investor Statement**\n' +
            '```\n' +
            'Starting NAV (Jan 1):   $10,000,000\n' +
            'Realized P&L:              +$500,000\n' +
            'Unrealized P&L:            +$200,000\n' +
            'Total P&L:                 +$700,000\n' +
            'Ending NAV (Dec 31):    $10,700,000\n' +
            'Return: 7.0%\n' +
            '```\n\n' +
            '**Tax Reporting (Different):**\n' +
            '- Only realized P&L is taxable\n' +
            '- Unrealized P&L is NOT taxable until realized\n' +
            '- But for regulatory reporting, both count\n\n' +
            '**Production Practice:**\n' +
            '- Calculate both realized and unrealized daily\n' +
            '- Reconcile total P&L against broker\n' +
            '- Generate separate reports for: Investors (total P&L), Tax (realized P&L), Regulators (both)\n\n' +
            'Real-world: Bridgewater Associates reports daily P&L to investors including both realized and unrealized. Tax forms (K-1) only report realized P&L.',
    },
];


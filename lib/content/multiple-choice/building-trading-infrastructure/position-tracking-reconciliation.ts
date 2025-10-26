export const positionTrackingReconciliationMC = [
    {
        id: 'position-tracking-reconciliation-mc-1',
        question:
            'Your position tracking system shows 1,000 shares of AAPL, but your broker reports 1,100 shares. What is the FIRST step you should take?',
        options: [
            'Immediately adjust your internal position to match the broker',
            'Pause trading in AAPL and verify the break is real by checking both systems',
            'Ignore it - 100 shares is within tolerance',
            'File a complaint with the broker for incorrect reporting',
        ],
        correctAnswer: 1,
        explanation:
            'The FIRST step is to pause trading and verify the break is real. Before making any adjustments, you need to:\n\n' +
            '1. **Confirm internal position**: Query your positions table to verify 1,000 shares\n' +
            '2. **Confirm broker position**: Check broker portal AND FIX position report (not just one source)\n' +
            '3. **Check timing**: Broker positions may be delayed 5-10 minutes\n' +
            '4. **Pause trading**: Prevent the discrepancy from growing\n\n' +
            'Option A (adjust immediately) is WRONG - you need to understand the root cause first. Blindly adjusting could hide a serious bug (like duplicate fills). Option C (ignore) is WRONG - 100 shares × $150 = $15,000 exposure is significant. Option D (complain) is premature - the error might be on your side.\n\n' +
            'Real-world: At Citadel, any reconciliation break >$5,000 triggers an immediate investigation. You have 1 hour to identify the root cause before escalating to the CTO.',
    },
    {
        id: 'position-tracking-reconciliation-mc-2',
        question:
            'You are building a position tracking system for a multi-strategy hedge fund with 50 traders. How should you model positions to support both real-time trading AND daily reconciliation?',
        options: [
            'Single table: positions(symbol, quantity, avg_cost)',
            'Multi-dimensional: positions(symbol, account, strategy, trader, broker, quantity, avg_cost)',
            'Separate tables for each strategy',
            'Store only aggregate positions, reconstruct details from fills',
        ],
        correctAnswer: 1,
        explanation:
            'Option B is correct: Multi-dimensional position tracking.\n\n' +
            '**Why Multi-Dimensional:**\n' +
            '1. **Real-time trading**: Traders need to see their strategy-level positions instantly\n' +
            '2. **Risk management**: Risk team needs positions by trader, by strategy, by account\n' +
            '3. **Reconciliation**: Must match broker positions at account level\n' +
            '4. **Attribution**: P&L attribution requires strategy-level positions\n\n' +
            '**Schema:**\n' +
            '```sql\n' +
            'CREATE TABLE positions (\n' +
            '  id SERIAL PRIMARY KEY,\n' +
            '  symbol VARCHAR(10),\n' +
            '  account VARCHAR(50),\n' +
            '  strategy VARCHAR(50),\n' +
            '  trader VARCHAR(50),\n' +
            '  broker VARCHAR(50),\n' +
            '  quantity DECIMAL(18,4),\n' +
            '  avg_cost DECIMAL(18,4),\n' +
            '  realized_pnl DECIMAL(18,2),\n' +
            '  unrealized_pnl DECIMAL(18,2),\n' +
            '  last_update TIMESTAMP,\n' +
            '  UNIQUE(symbol, account, strategy, trader, broker)\n' +
            ');\n' +
            '```\n\n' +
            '**Aggregation Views:**\n' +
            '- By trader: `SELECT trader, symbol, SUM(quantity) FROM positions GROUP BY trader, symbol`\n' +
            '- By strategy: `SELECT strategy, symbol, SUM(quantity) FROM positions GROUP BY strategy, symbol`\n' +
            '- By broker (for reconciliation): `SELECT broker, symbol, SUM(quantity) FROM positions GROUP BY broker, symbol`\n\n' +
            'Option A (single table) is TOO SIMPLE - cannot support multi-strategy fund. Option C (separate tables per strategy) is WRONG - makes aggregation queries impossible. Option D (reconstruct from fills) is TOO SLOW - would take seconds to calculate positions, not microseconds.\n\n' +
            'Real-world: Jane Street tracks positions with 6 dimensions (symbol, account, strategy, trader, venue, book) to support 200+ strategies.',
    },
    {
        id: 'position-tracking-reconciliation-mc-3',
        question:
            'A stock split occurs: AAPL 2:1 on 2025-10-26. Your position is 100 shares @ $150 avg cost. After the split, what should your position be?',
        options: [
            '100 shares @ $150 (no change)',
            '200 shares @ $150 ($30,000 total value)',
            '200 shares @ $75 ($15,000 total value)',
            '100 shares @ $75 ($7,500 total value)',
        ],
        correctAnswer: 2,
        explanation:
            'Option C is correct: 200 shares @ $75 (total value unchanged at $15,000).\n\n' +
            '**Stock Split Math:**\n' +
            '- Before split: 100 shares × $150 = $15,000\n' +
            '- Split ratio: 2:1 (each share becomes 2 shares)\n' +
            '- After split: 200 shares × $75 = $15,000 ✓\n\n' +
            '**Key Rules:**\n' +
            '1. **Quantity adjustment**: New quantity = old quantity × split ratio\n' +
            '   - 100 shares × 2 = 200 shares\n' +
            '2. **Cost basis adjustment**: New cost = old cost / split ratio\n' +
            '   - $150 / 2 = $75\n' +
            '3. **Total value UNCHANGED**: Stock splits do not create value\n' +
            '   - Before: $15,000\n' +
            '   - After: $15,000\n\n' +
            '**Implementation:**\n' +
            '```python\n' +
            'def handle_stock_split(position, ratio):\n' +
            '    position.quantity *= ratio  # 100 → 200\n' +
            '    position.avg_cost /= ratio  # 150 → 75\n' +
            '    # Value: 200 × $75 = $15,000 ✓\n' +
            '```\n\n' +
            'Option A (no change) is WRONG - ignores the split. Option B ($30,000 value) is WRONG - stock splits don\'t create value. Option D (100 shares @ $75) is WRONG - quantity must increase.\n\n' +
            'Real-world: Apple (AAPL) has split 5 times since 1980. Most recently 4:1 in August 2020 ($400 → $100 per share).',
    },
    {
        id: 'position-tracking-reconciliation-mc-4',
        question:
            'You are tracking a multi-currency position: 1,000 shares of SONY (Japan) at ¥15,000 per share. The current FX rate is 150 JPY/USD (i.e., $1 = ¥150). What is the position value in USD?',
        options: [
            '$100',
            '$1,000',
            '$10,000',
            '$100,000',
        ],
        correctAnswer: 3,
        explanation:
            'Option D is correct: $100,000.\n\n' +
            '**Multi-Currency Position Math:**\n' +
            '1. **Position value in local currency (JPY)**:\n' +
            '   - 1,000 shares × ¥15,000/share = ¥15,000,000\n\n' +
            '2. **Convert to USD**:\n' +
            '   - FX rate: 150 JPY/USD means ¥150 = $1\n' +
            '   - FX rate as decimal: 1 USD = 150 JPY → 1 JPY = 1/150 USD = 0.00667 USD\n' +
            '   - Position value in USD: ¥15,000,000 × (1/150) = $100,000 ✓\n\n' +
            '**Alternative calculation:**\n' +
            '- ¥15,000,000 ÷ 150 JPY/USD = $100,000\n\n' +
            '**Implementation:**\n' +
            '```python\n' +
            'def calculate_usd_value(local_value_jpy, fx_rate_jpy_per_usd):\n' +
            '    """\n' +
            '    local_value_jpy: ¥15,000,000\n' +
            '    fx_rate_jpy_per_usd: 150 (means ¥150 = $1)\n' +
            '    """\n' +
            '    return local_value_jpy / fx_rate_jpy_per_usd\n' +
            '\n' +
            'usd_value = calculate_usd_value(15_000_000, 150)\n' +
            '# Result: $100,000\n' +
            '```\n\n' +
            '**FX P&L Impact:**\n' +
            'If JPY strengthens to 145 JPY/USD (¥145 = $1):\n' +
            '- New USD value: ¥15,000,000 ÷ 145 = $103,448\n' +
            '- FX P&L: $103,448 - $100,000 = +$3,448 (gain from JPY appreciation)\n\n' +
            'Real-world: Global hedge funds must track FX exposure. A $1B AUM fund with 20% in Japan has $200M JPY exposure. A 5% FX move = $10M P&L impact.',
    },
    {
        id: 'position-tracking-reconciliation-mc-5',
        question:
            'Your reconciliation system identifies 50 breaks at 6 PM daily. 40 are <10 shares, 8 are 10-100 shares, and 2 are >100 shares. What is the BEST reconciliation strategy?',
        options: [
            'Manually review all 50 breaks (takes 2 hours)',
            'Auto-resolve <10 shares, manually review 10-100 shares, urgent escalation for >100 shares',
            'Ignore breaks <100 shares, only fix large breaks',
            'Adjust all internal positions to match broker (no investigation)',
        ],
        correctAnswer: 1,
        explanation:
            'Option B is correct: Severity-based reconciliation strategy.\n\n' +
            '**Tiered Reconciliation Approach:**\n\n' +
            '1. **Auto-Resolve (<10 shares):** 40 breaks\n' +
            '   - Likely causes: Rounding errors, odd lots, minor timing differences\n' +
            '   - Action: Automatically adjust internal position to match broker\n' +
            '   - Time: <1 second per break (automated)\n' +
            '   - Total time: <1 minute\n\n' +
            '2. **Manual Review (10-100 shares):** 8 breaks\n' +
            '   - Likely causes: Missing fill, duplicate fill, incorrect quantity\n' +
            '   - Action: Ops team investigates, identifies root cause, fixes\n' +
            '   - Time: ~15 minutes per break\n' +
            '   - Total time: ~2 hours\n\n' +
            '3. **Urgent Escalation (>100 shares):** 2 breaks\n' +
            '   - Likely causes: System bug, major fill processing error\n' +
            '   - Action: Page on-call engineer, immediate investigation\n' +
            '   - Time: Must resolve within 1 hour\n' +
            '   - Financial impact: Likely >$10K per break\n\n' +
            '**Implementation:**\n' +
            '```python\n' +
            'async def reconcile_positions(breaks):\n' +
            '    auto_resolved = 0\n' +
            '    manual_review = []\n' +
            '    urgent = []\n' +
            '    \n' +
            '    for break_item in breaks:\n' +
            '        diff = abs(break_item.difference)\n' +
            '        \n' +
            '        if diff < 10:\n' +
            '            # Auto-resolve\n' +
            '            await auto_resolve_break(break_item)\n' +
            '            auto_resolved += 1\n' +
            '        elif diff < 100:\n' +
            '            # Manual review\n' +
            '            manual_review.append(break_item)\n' +
            '        else:\n' +
            '            # Urgent\n' +
            '            urgent.append(break_item)\n' +
            '            await alert_oncall(break_item)\n' +
            '    \n' +
            '    print(f"Auto-resolved: {auto_resolved}")\n' +
            '    print(f"Manual review: {len(manual_review)}")\n' +
            '    print(f"Urgent: {len(urgent)}")\n' +
            '```\n\n' +
            '**Total reconciliation time:**\n' +
            '- Auto: <1 minute (40 breaks)\n' +
            '- Manual: ~2 hours (8 breaks × 15 min)\n' +
            '- Urgent: Within 1 hour (2 breaks, high priority)\n' +
            '- **Total: ~3 hours** (vs 2 hours if manually reviewing all 50)\n\n' +
            'Option A (manual review all) is INEFFICIENT - wasting time on trivial breaks. Option C (ignore small breaks) is WRONG - small breaks can accumulate to large errors. Option D (adjust all) is DANGEROUS - hides root causes, could mask serious bugs.\n\n' +
            'Real-world: Two Sigma auto-resolves 95% of breaks (<$100 impact), manually reviews 4%, and escalates 1% to engineering. Average reconciliation time: 30 minutes for 1,000+ daily positions.',
    },
];


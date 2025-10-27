export const tradeReconciliationMC = [
  {
    id: 'trade-reconciliation-mc-1',
    question: 'What is the PRIMARY purpose of trade reconciliation?',
    options: [
      'To calculate trading fees',
      'To ensure internal trades match broker/exchange confirmations before settlement',
      'To optimize execution algorithms',
      'To generate tax reports',
    ],
    correctAnswer: 1,
    explanation:
      'Answer: To ensure internal trades match broker/exchange confirmations before settlement.\n\n' +
      'Trade reconciliation verifies that:\n' +
      '1. Every internal trade has a broker confirmation\n' +
      '2. Every broker confirmation matches an internal trade\n' +
      '3. Details match: Symbol, side, quantity, price, time\n\n' +
      'Why this matters:\n' +
      '- Trades must reconcile before T+2 settlement\n' +
      '- Unreconciled trades may not settle correctly\n' +
      '- Could result in fails-to-deliver (regulatory violations)\n\n' +
      'Other options are secondary: Fees (option A) are part of reconciliation but not the primary purpose. Execution optimization (option C) uses fill data but is separate from reconciliation. Tax reporting (option D) uses reconciled trades but is downstream.',
  },
  {
    id: 'trade-reconciliation-mc-2',
    question:
      'Your internal system shows you BOUGHT 100 shares, but the broker has no record of this trade. What is the MOST likely cause?',
    options: [
      'Broker lost the trade (broker error)',
      'Order was rejected and never executed',
      'Internal system duplicated the trade record',
      'Trade is still pending broker confirmation',
    ],
    correctAnswer: 1,
    explanation:
      'Answer: Order was rejected and never executed.\n\n' +
      'Most common scenario:\n' +
      '1. You sent BUY order to broker\n' +
      '2. Broker rejected order (insufficient buying power, halt, etc.)\n' +
      '3. Rejection message was missed or not processed\n' +
      '4. Internal system recorded "trade" but no actual execution\n\n' +
      'Investigation steps:\n' +
      '- Check FIX logs for OrderCancelReject or ExecutionReport with status=REJECTED\n' +
      '- Verify order was actually sent (check OMS logs)\n' +
      '- Contact broker to confirm no execution\n\n' +
      'Why other options are less likely:\n' +
      '- Broker lost trade (A): Very rare, brokers have robust systems\n' +
      '- Duplicate internal (C): Would show broker confirmation for original\n' +
      '- Pending confirmation (D): Broker confirms within seconds, not hours',
  },
  {
    id: 'trade-reconciliation-mc-3',
    question: 'When should trade reconciliation occur?',
    options: [
      'Once per month',
      'Once per week',
      'Once per day (end of day)',
      'Real-time (continuous)',
    ],
    correctAnswer: 3,
    explanation:
      'Answer: Real-time (continuous).\n\n' +
      'Modern trade reconciliation is CONTINUOUS:\n' +
      '- Match trades as soon as broker confirms (within seconds/minutes)\n' +
      '- Identify breaks immediately (within 1 hour)\n' +
      '- Resolve breaks same day\n\n' +
      'Why real-time:\n' +
      '1. **Faster issue detection**: Catch missing fills immediately\n' +
      '2. **Easier investigation**: Trader remembers what happened\n' +
      '3. **Settlement requirements**: Must reconcile by T+1 for T+2 settlement\n' +
      '4. **Risk management**: Know true positions real-time\n\n' +
      'Traditional approach (EOD reconciliation) is obsolete:\n' +
      '- Too slow (discover breaks 6+ hours after trade)\n' +
      '- Hard to investigate (trader forgot details)\n' +
      '- Risk of missed settlement deadline\n\n' +
      'Production practice:\n' +
      '- Continuous matching during market hours\n' +
      '- EOD report at 4 PM confirming 100% reconciliation\n' +
      '- T+1 final reconciliation before settlement',
  },
  {
    id: 'trade-reconciliation-mc-4',
    question:
      'A break is identified: Internal trade shows $150.00, broker shows $150.01. Should this auto-resolve?',
    options: [
      'No - must manually review all price discrepancies',
      'Yes - $0.01 difference is within rounding tolerance',
      'No - broker is always wrong, use internal price',
      'Yes - but only if quantity also matches',
    ],
    correctAnswer: 1,
    explanation:
      'Answer: Yes - $0.01 difference is within rounding tolerance.\n\n' +
      'Why $0.01 tolerance:\n' +
      '1. **Rounding**: Prices may be rounded differently (half-cent rounding)\n' +
      '2. **Partial fills**: Average price may differ slightly\n' +
      '3. **Timestamp**: Filled at different milliseconds, slightly different prices\n\n' +
      'Auto-resolution policy:\n' +
      '- ≤$0.01 difference: Auto-accept broker price (99% of cases)\n' +
      '- $0.01-$0.10: Auto-accept with alert to trader\n' +
      '- >$0.10: Manual review required\n\n' +
      'Implementation:\n' +
      '```python\n' +
      'def should_auto_resolve(internal_price, broker_price):\n' +
      '    diff = abs(internal_price - broker_price)\n' +
      '    if diff <= 0.01:\n' +
      '        return True  # Auto-resolve\n' +
      '    elif diff <= 0.10:\n' +
      '        alert_trader()\n' +
      '        return True  # Auto-resolve with alert\n' +
      '    else:\n' +
      '        return False  # Manual review\n' +
      '```\n\n' +
      'Why accept broker price:\n' +
      '- Broker price is "official" for settlement\n' +
      '- Internal price is an estimate\n' +
      '- SEC requires using broker confirmations',
  },
  {
    id: 'trade-reconciliation-mc-5',
    question:
      'Your firm executes 10,000 trades/day with 50 unreconciled breaks at EOD. What is your reconciliation rate?',
    options: ['50%', '95%', '99.5%', '99.95%'],
    correctAnswer: 2,
    explanation:
      'Answer: 99.5%.\n\n' +
      'Reconciliation rate = (Matched trades / Total trades) × 100%\n' +
      '= (10,000 - 50) / 10,000 × 100%\n' +
      '= 9,950 / 10,000 × 100%\n' +
      '= **99.5%**\n\n' +
      'Industry benchmarks:\n' +
      '- **Excellent**: >99.9% (1 break per 1,000 trades)\n' +
      '- **Good**: 99.5-99.9% (5-10 breaks per 1,000 trades)\n' +
      '- **Acceptable**: 99.0-99.5% (50 breaks per 10,000 trades) ← You are here\n' +
      '- **Poor**: <99.0% (>100 breaks per 10,000 trades)\n\n' +
      '99.5% is ACCEPTABLE but not excellent. Improvement targets:\n' +
      '1. Improve FIX connectivity (reduce missed confirmations)\n' +
      '2. Better order validation (reduce rejections)\n' +
      '3. Real-time matching (catch breaks faster)\n' +
      '4. Auto-resolution for small price differences\n\n' +
      'Real-world:\n' +
      '- High-frequency firms: >99.99% (1 break per 10,000 trades)\n' +
      '- Retail brokers: 99.5-99.8%\n' +
      '- Manual trading: 98-99%',
  },
];

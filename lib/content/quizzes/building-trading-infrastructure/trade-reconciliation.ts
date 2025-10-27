export const tradeReconciliationQuiz = [
  {
    id: 'trade-reconciliation-q-1',
    question:
      'Design a "trade reconciliation system" for a firm executing 50,000 trades per day across 5 brokers. How would you automate matching, identify breaks within 1 hour, and ensure all trades reconcile before T+2 settlement?',
    sampleAnswer:
      'Automated Trade Reconciliation System:\n\n' +
      '**Architecture:**\n' +
      '1. **Real-Time Trade Capture**:\n' +
      '   - Internal trades: Captured from EMS immediately on fill\n' +
      '   - Broker trades: Received via FIX ExecutionReport or API\n' +
      '   - Store both in PostgreSQL with indexed columns (symbol, side, timestamp, account)\n\n' +
      '2. **Automated Matching Engine**:\n' +
      '   - Match trades based on: Symbol, Side, Quantity, Price (±$0.01), Time (±60s), Account\n' +
      '   - Run matching every 5 minutes during market hours\n' +
      '   - Use fuzzy matching for price (allow $0.01 tolerance for rounding)\n\n' +
      '3. **Break Identification** (within 1 hour):\n' +
      '   - **Missing Broker**: Internal trade >1 hour old, no broker confirmation\n' +
      '   - **Missing Internal**: Broker trade with no internal match\n' +
      '   - **Quantity Mismatch**: Internal 100 shares, broker 90 shares\n' +
      '   - **Price Mismatch**: Internal $150.00, broker $150.50\n\n' +
      '4. **Break Resolution Workflow**:\n' +
      '   - **Auto-Resolve**: Price difference <$0.01 → accept broker price\n' +
      '   - **Manual Review**: Quantity mismatch or price >$0.10 difference\n' +
      '   - **Escalation**: >10 breaks for same broker → alert operations manager\n\n' +
      '5. **Pre-Settlement Reconciliation** (T+1 by 6 PM):\n' +
      '   - All trades from T must reconcile by T+1 6 PM (before T+2 settlement)\n' +
      '   - Automated report at 5 PM showing unreconciled trades\n' +
      '   - Alert if >1% of trades unreconciled\n\n' +
      'Implementation: Daily reconciliation completed by 6 PM, 99%+ auto-match rate',
    keyPoints: [
      'Real-time capture: Internal trades from EMS, broker trades via FIX/API, store in indexed PostgreSQL',
      'Automated matching: Match on symbol, side, quantity, price (±$0.01), time (±60s), account every 5 minutes',
      'Break identification: Missing broker (>1 hour), missing internal, quantity/price mismatch within 1 hour',
      'Break resolution: Auto-resolve <$0.01 price diff, manual review for quantity/price >$0.10, escalate >10 breaks',
      'Pre-settlement: All T trades reconcile by T+1 6 PM (before T+2 settlement), alert if >1% unreconciled',
    ],
  },
  {
    id: 'trade-reconciliation-q-2',
    question:
      'Your system shows an internal trade: BUY 100 AAPL @ $150.00 at 10:00:00. Broker confirms: BUY 100 AAPL @ $150.50 at 10:00:05. Should this match? What is the $0.50 difference?',
    sampleAnswer:
      'Trade Matching Analysis:\n\n' +
      '**Should it match? YES** (with tolerance)\n\n' +
      'Matching criteria:\n' +
      '1. ✓ Symbol: AAPL = AAPL\n' +
      '2. ✓ Side: BUY = BUY\n' +
      '3. ✓ Quantity: 100 = 100\n' +
      '4. ✓ Price: $150.00 vs $150.50 → $0.50 difference\n' +
      '5. ✓ Time: 10:00:00 vs 10:00:05 → 5 seconds (within 60s tolerance)\n\n' +
      '**Price Difference: $0.50**\n' +
      'This is larger than typical rounding ($0.01), so investigate:\n\n' +
      '**Possible Causes:**\n' +
      '1. **Slippage**: Market order filled at worse price than expected\n' +
      '   - You estimated $150.00, actually filled at $150.50\n' +
      '   - This is NORMAL for market orders\n\n' +
      '2. **Partial fills at different prices**:\n' +
      '   - 50 shares @ $150.00, 50 shares @ $151.00\n' +
      '   - Your system recorded $150.00 (first fill), broker reports $150.50 (average)\n\n' +
      '3. **Fee inclusion**:\n' +
      '   - Broker includes $0.50 commission in price\n' +
      '   - Your system tracks commission separately\n\n' +
      '**Resolution:**\n' +
      '- Accept broker price ($150.50) as official\n' +
      '- Update internal trade to match broker\n' +
      '- Adjust P&L: 100 × ($150.50 - $150.00) = -$50 cost increase\n\n' +
      'Production rule: Price tolerance ±$0.10 for auto-match, >$0.10 requires manual review',
    keyPoints: [
      'Should match: All criteria met (symbol, side, quantity, time within 60s)',
      'Price difference $0.50: Larger than typical rounding, requires investigation',
      'Possible causes: Slippage (market order), partial fills at different prices, fee inclusion',
      'Resolution: Accept broker price as official, update internal to $150.50, adjust P&L by -$50',
      'Tolerance: ±$0.10 for auto-match, >$0.10 requires manual review',
    ],
  },
  {
    id: 'trade-reconciliation-q-3',
    question:
      'At 6 PM (end of trading day), you have 50 unreconciled trades out of 10,000 total trades. What are the risks and how should you handle this?',
    sampleAnswer:
      '50 Unreconciled Trades (0.5% unreconciled rate):\n\n' +
      '**Risks:**\n\n' +
      '1. **Settlement Risk (T+2)**:\n' +
      '   - Trades must settle T+2 (trade date + 2 business days)\n' +
      '   - Unreconciled trades may not settle correctly\n' +
      '   - Could result in "fails-to-deliver" (FTD) → SEC fines\n\n' +
      '2. **Position Risk**:\n' +
      "   - If broker didn't confirm, trade may not have executed\n" +
      '   - Your position may be wrong → incorrect risk calculations\n\n' +
      '3. **P&L Risk**:\n' +
      '   - Unreconciled trades may have wrong price/quantity\n' +
      '   - P&L could be overstated or understated\n\n' +
      '4. **Regulatory Risk**:\n' +
      '   - SEC Rule 15c3-3: Must reconcile trades daily\n' +
      '   - Systematic failures → fines up to $10M\n\n' +
      '**Handling Procedure:**\n\n' +
      '**Step 1: Classify breaks (6:00-6:30 PM)**\n' +
      '- Missing broker confirmation: 30 trades (60%)\n' +
      '- Missing internal trade: 15 trades (30%)\n' +
      '- Quantity/price mismatch: 5 trades (10%)\n\n' +
      '**Step 2: Immediate actions (6:30-7:00 PM)**\n' +
      'For missing broker confirmations:\n' +
      '- Contact broker via phone/email\n' +
      '- Request immediate trade confirmation\n' +
      '- Check if broker API was down (delayed confirmations)\n\n' +
      'For missing internal trades:\n' +
      '- Check if fill was rejected by OMS\n' +
      '- Verify with trader if order was actually sent\n' +
      '- May need to manually enter internal trade\n\n' +
      '**Step 3: Resolution timeline**\n' +
      '- By 8:00 PM (T): Resolve 80% of breaks (40/50)\n' +
      '- By 10:00 AM (T+1): Resolve remaining 20% (10/50)\n' +
      '- By 4:00 PM (T+1): 100% reconciliation required\n\n' +
      '**Step 4: Escalation**\n' +
      'If >1% unreconciled by end of T+1:\n' +
      '- Escalate to COO/CFO\n' +
      '- Consider delaying settlement (notify broker)\n' +
      '- Prepare regulatory explanation (Form 15c3-3)\n\n' +
      '**Production Metrics:**\n' +
      '- Target: >99.5% same-day reconciliation\n' +
      '- Acceptable: >99.9% by T+1\n' +
      '- Unacceptable: <99.9% by T+1 (settlement risk too high)',
    keyPoints: [
      'Risks: Settlement risk (fails-to-deliver), position risk (wrong positions), P&L risk (incorrect pricing), regulatory risk (SEC fines)',
      'Classify breaks: 60% missing broker, 30% missing internal, 10% quantity/price mismatch',
      'Immediate actions: Contact broker for missing confirmations, check OMS for rejected fills, verify with traders',
      'Resolution timeline: 80% by 8 PM (T), 100% by 4 PM (T+1), escalate if >1% unreconciled',
      'Target metrics: >99.5% same-day reconciliation, >99.9% by T+1, <99.9% requires escalation',
    ],
  },
];

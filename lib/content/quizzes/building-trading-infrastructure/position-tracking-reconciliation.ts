export const positionTrackingReconciliationQuiz = [
  {
    id: 'position-tracking-reconciliation-q-1',
    question:
      'Design a "position tracking system" for a multi-strategy hedge fund with 50 traders, 100+ strategies, and 10 prime brokers. How would you handle real-time position updates, multi-dimensional position views, and daily reconciliation? What are the key architectural decisions?',
    sampleAnswer:
      'Multi-Strategy Position Tracking System:\n\n' +
      '**Architecture:**\n' +
      '1. Event-Driven Position Updates:\n' +
      '   - Fill events flow from EMS → Position Service\n' +
      '   - Process fills in microseconds (<100μs)\n' +
      '   - Atomic position updates (no partial states)\n' +
      '   - Publish position change events to downstream systems\n\n' +
      '2. Multi-Dimensional Position Model:\n' +
      '   - Primary key: (Symbol, Account, Strategy, Trader, Broker)\n' +
      '   - Aggregation views: By symbol, by strategy, by trader, by broker\n' +
      '   - Cost basis tracking: FIFO, LIFO, or weighted average\n' +
      '   - Support long/short, cash/margin, multiple currencies\n\n' +
      '3. Storage Strategy:\n' +
      '   - In-memory: Redis for real-time positions (100K positions ~ 1GB)\n' +
      '   - Database: PostgreSQL for historical positions and audit trail\n' +
      '   - Time-series: TimescaleDB for position snapshots (every 1 min)\n' +
      '   - Write-ahead log for disaster recovery\n\n' +
      '4. Daily Reconciliation Process:\n' +
      '   - 6:00 PM: Collect broker SOD positions via FIX or API\n' +
      '   - 6:15 PM: Compare internal positions vs broker positions\n' +
      '   - 6:30 PM: Identify breaks, classify by severity\n' +
      '   - 7:00 PM: Ops team resolves breaks before next trading day\n' +
      '   - 8:00 PM: Final reconciliation report to risk/compliance\n\n' +
      '5. Break Resolution Workflow:\n' +
      '   - AUTO-RESOLVE: <10 shares difference (rounding)\n' +
      '   - MANUAL-REVIEW: 10-100 shares (investigate recent fills)\n' +
      '   - URGENT: >100 shares or >$10K (page on-call engineer)\n' +
      '   - Track resolution time, escalate if >2 hours unresolved\n\n' +
      '**Implementation:**\n' +
      '```python\n' +
      'class MultiStrategyPositionTracker:\n' +
      '    def __init__(self):\n' +
      '        self.redis = Redis()  # Real-time positions\n' +
      '        self.postgres = PostgreSQL()  # Historical positions\n' +
      '        self.timescale = TimescaleDB()  # Position snapshots\n' +
      '        \n' +
      '    async def process_fill(self, fill):\n' +
      '        # 1. Update in-memory position\n' +
      '        position = await self.get_position(\n' +
      '            symbol=fill.symbol,\n' +
      '            account=fill.account,\n' +
      '            strategy=fill.strategy,\n' +
      '            trader=fill.trader,\n' +
      '            broker=fill.broker\n' +
      '        )\n' +
      '        \n' +
      '        # 2. Calculate new position\n' +
      '        new_qty, new_cost, realized_pnl = position.apply_fill(\n' +
      '            side=fill.side,\n' +
      '            quantity=fill.quantity,\n' +
      '            price=fill.price\n' +
      '        )\n' +
      '        \n' +
      '        # 3. Atomic update\n' +
      '        await self.redis.set(position.key, position.serialize())\n' +
      '        \n' +
      '        # 4. Persist to DB (async)\n' +
      '        await self.postgres.insert_position_update(position)\n' +
      '        \n' +
      '        # 5. Publish event\n' +
      '        await self.event_bus.publish("position.updated", position)\n' +
      '```\n\n' +
      '**Reconciliation Process:**\n' +
      '```python\n' +
      'async def daily_reconciliation(self, date):\n' +
      '    # 1. Get internal EOD positions\n' +
      '    internal_positions = await self.get_eod_positions(date)\n' +
      '    \n' +
      '    # 2. Get broker positions (from all 10 brokers)\n' +
      '    broker_positions = await asyncio.gather(*[\n' +
      '        self.get_broker_positions(broker, date)\n' +
      '        for broker in self.brokers\n' +
      '    ])\n' +
      '    \n' +
      '    # 3. Compare and identify breaks\n' +
      '    breaks = []\n' +
      '    for broker_pos in broker_positions:\n' +
      '        internal_pos = internal_positions.get(broker_pos.key)\n' +
      '        if internal_pos.quantity != broker_pos.quantity:\n' +
      '            breaks.append(ReconciliationBreak(\n' +
      '                symbol=broker_pos.symbol,\n' +
      '                internal_qty=internal_pos.quantity,\n' +
      '                broker_qty=broker_pos.quantity,\n' +
      '                difference=internal_pos.quantity - broker_pos.quantity,\n' +
      '                severity=self.classify_severity(difference)\n' +
      '            ))\n' +
      '    \n' +
      '    # 4. Auto-resolve small breaks\n' +
      '    for break_item in breaks:\n' +
      '        if break_item.severity == "LOW":\n' +
      '            await self.auto_resolve(break_item)\n' +
      '    \n' +
      '    # 5. Alert on unresolved breaks\n' +
      '    unresolved = [b for b in breaks if not b.resolved]\n' +
      '    if unresolved:\n' +
      '        await self.alert_ops_team(unresolved)\n' +
      '```\n\n' +
      '**Performance Considerations:**\n' +
      '- Real-time position updates: <100μs per fill\n' +
      '- Reconciliation time: <15 minutes for 100K positions across 10 brokers\n' +
      '- Storage: ~10GB per day for position history (365 days = 3.6TB)\n' +
      '- Break resolution: 95% auto-resolved, 5% manual review\n\n' +
      '**Production Checklist:**\n' +
      '- Position snapshot every 1 minute for audit trail\n' +
      '- Alert if position update latency >1 second\n' +
      '- Daily reconciliation must complete before market open\n' +
      '- Support corporate actions (splits, mergers, dividends)\n' +
      '- Multi-currency position tracking with FX rates',
    keyPoints: [
      'Event-driven architecture: Process fills in <100μs, atomic updates, publish position change events',
      'Multi-dimensional positions: Track by symbol, account, strategy, trader, broker with aggregation views',
      'Storage strategy: Redis for real-time (1GB), PostgreSQL for historical, TimescaleDB for snapshots',
      'Daily reconciliation: Compare internal vs broker positions at 6 PM, classify breaks by severity, auto-resolve <10 shares',
      'Break resolution: Auto-resolve small breaks, manual review for 10-100 shares, urgent escalation for >100 shares or >$10K',
    ],
  },
  {
    id: 'position-tracking-reconciliation-q-2',
    question:
      'You discover a reconciliation break: internal system shows 1,000 shares of AAPL, but broker shows 950 shares. The 50-share discrepancy represents $7,500. How do you investigate and resolve this? What are the possible causes and resolution steps?',
    sampleAnswer:
      'Reconciliation Break Investigation:\n\n' +
      '**Step 1: Immediate Actions (First 5 Minutes)**\n' +
      '1. Check if position is actively trading:\n' +
      '   - If yes, pause trading immediately to prevent further discrepancy\n' +
      '   - Alert trader/risk manager\n\n' +
      '2. Verify the break is real:\n' +
      '   - Confirm internal position: SELECT * FROM positions WHERE symbol="AAPL"\n' +
      '   - Confirm broker position: Check broker portal + FIX position report\n' +
      '   - Check if broker position is delayed (some brokers lag 5-10 minutes)\n\n' +
      '3. Calculate financial impact:\n' +
      '   - 50 shares × $150 = $7,500 notional exposure\n' +
      '   - Check if this affects margin/buying power\n' +
      '   - Determine if regulatory threshold exceeded (e.g., 13F filing)\n\n' +
      '**Step 2: Root Cause Analysis (Minutes 5-30)**\n\n' +
      'Possible Causes & Investigation:\n\n' +
      '1. **Missing Fill** (most common ~40% of breaks):\n' +
      '   - Check FIX execution reports: Did broker send ExecutionReport but we missed it?\n' +
      '   - Check network logs: Was there a network blip during market hours?\n' +
      '   - Check fill processing logs: Did our fill handler crash/reject a fill?\n' +
      '   - Resolution: Request broker to resend ExecutionReport, manually enter fill\n\n' +
      '2. **Duplicate Fill** (common ~30% of breaks):\n' +
      '   - Check if we processed same fill twice (duplicate ExecID)\n' +
      '   - Check fill timestamps: Two fills within milliseconds?\n' +
      '   - Resolution: Reverse duplicate fill, add ExecID deduplication\n\n' +
      '3. **Incorrect Fill Quantity** (common ~20% of breaks):\n' +
      '   - Broker filled 950 shares, reported 1,000 shares in ExecutionReport\n' +
      "   - Check broker's trade report vs our fill: Quantities match?\n" +
      '   - Resolution: Adjust internal position to match broker confirmation\n\n' +
      '4. **Corporate Action** (less common ~5%):\n' +
      '   - Stock split, dividend, merger - did we account for it?\n' +
      '   - Check corporate action calendar for AAPL\n' +
      '   - Resolution: Apply corporate action adjustment\n\n' +
      '5. **Broker Error** (rare ~5%):\n' +
      "   - Broker's position is wrong, not ours\n" +
      "   - Cross-check with broker's trade confirmations\n" +
      '   - Resolution: Escalate to broker, request position correction\n\n' +
      '**Step 3: Investigation Commands**\n' +
      '```sql\n' +
      '-- Check all AAPL fills today\n' +
      'SELECT fill_time, side, quantity, price, exec_id, source\n' +
      'FROM fills\n' +
      'WHERE symbol = "AAPL" AND date = CURRENT_DATE\n' +
      'ORDER BY fill_time;\n\n' +
      '-- Check position history (should show 1,000 shares)\n' +
      'SELECT timestamp, quantity, avg_cost, change_reason\n' +
      'FROM position_history\n' +
      'WHERE symbol = "AAPL" AND account = "MAIN"\n' +
      'ORDER BY timestamp DESC\n' +
      'LIMIT 20;\n\n' +
      '-- Check for duplicate ExecIDs\n' +
      'SELECT exec_id, COUNT(*)\n' +
      'FROM fills\n' +
      'WHERE symbol = "AAPL" AND date = CURRENT_DATE\n' +
      'GROUP BY exec_id\n' +
      'HAVING COUNT(*) > 1;\n\n' +
      '-- Sum up all AAPL fills (should = 1,000 if no missing/duplicate)\n' +
      'SELECT \n' +
      '  SUM(CASE WHEN side = "BUY" THEN quantity ELSE -quantity END) as net_position\n' +
      'FROM fills\n' +
      'WHERE symbol = "AAPL";\n' +
      '```\n\n' +
      '**Step 4: Resolution**\n\n' +
      'Scenario A: Missing Fill Detected\n' +
      '```python\n' +
      '# Found missing fill: SELL 50 shares at 14:32:15\n' +
      'await position_tracker.process_fill(\n' +
      '    symbol="AAPL",\n' +
      '    side="SELL",\n' +
      '    quantity=50,\n' +
      '    price=150.00,\n' +
      '    fill_time=datetime(2025, 10, 26, 14, 32, 15),\n' +
      '    exec_id="BROKER-12345",\n' +
      '    is_manual_adjustment=True,\n' +
      '    reason="Reconciliation - Missing fill from broker"\n' +
      ')\n' +
      '# Internal position now: 950 shares ✓\n' +
      '```\n\n' +
      'Scenario B: Duplicate Fill Detected\n' +
      '```python\n' +
      '# Found duplicate fill: BUY 50 shares processed twice\n' +
      'await position_tracker.reverse_fill(\n' +
      '    fill_id="FILL-98765",\n' +
      '    reason="Reconciliation - Duplicate fill detected"\n' +
      ')\n' +
      '# Internal position now: 950 shares ✓\n' +
      '```\n\n' +
      '**Step 5: Prevention**\n' +
      '1. Add idempotency check: Reject fills with duplicate ExecID\n' +
      '2. Real-time reconciliation: Check position every 15 minutes vs broker API\n' +
      '3. Alert on position divergence: If internal vs broker differs by >1%, alert immediately\n' +
      '4. Audit trail: Log every position change with reason and timestamp\n' +
      '5. Automated tests: Simulate missing fills, duplicate fills in staging\n\n' +
      '**Timeline:**\n' +
      '- 0-5 min: Detect and verify break\n' +
      '- 5-30 min: Root cause analysis\n' +
      '- 30-45 min: Implement fix and verify\n' +
      '- 45-60 min: Document incident, update monitoring\n\n' +
      'Total resolution time: <1 hour for $7,500 break',
    keyPoints: [
      'Immediate actions: Pause trading, verify break is real, calculate financial impact ($7,500 notional)',
      'Root cause analysis: Check for missing fill (40%), duplicate fill (30%), incorrect quantity (20%), corporate action (5%), or broker error (5%)',
      'Investigation: Query fills table, position history, check for duplicate ExecIDs, verify with broker confirmations',
      'Resolution: Process missing fill manually or reverse duplicate fill, update position to match broker',
      'Prevention: Add ExecID deduplication, real-time reconciliation every 15 min, alert on >1% divergence, comprehensive audit trail',
    ],
  },
  {
    id: 'position-tracking-reconciliation-q-3',
    question:
      'Design a "position tracking system" that supports corporate actions (stock splits, dividends, mergers), multi-currency positions, and options positions. What are the key challenges and how would you handle them?',
    sampleAnswer:
      'Advanced Position Tracking System:\n\n' +
      '**1. Corporate Actions**\n\n' +
      'Challenges:\n' +
      '- Stock splits change quantity but not value (2:1 split doubles shares)\n' +
      "- Dividends create cash but don't change shares\n" +
      '- Mergers convert shares to new ticker (AAPL → NEWCO)\n' +
      '- Must adjust historical cost basis for tax reporting\n\n' +
      'Implementation:\n' +
      '```python\n' +
      'class CorporateActionHandler:\n' +
      '    async def handle_stock_split(self, symbol, ratio, ex_date):\n' +
      '        """\n' +
      '        Stock Split: AAPL 2:1 on 2025-10-26\n' +
      '        Before: 100 shares @ $150 = $15,000\n' +
      '        After: 200 shares @ $75 = $15,000\n' +
      '        """\n' +
      '        positions = self.get_positions_by_symbol(symbol)\n' +
      '        \n' +
      '        for position in positions:\n' +
      '            old_qty = position.quantity\n' +
      '            old_cost = position.avg_cost\n' +
      '            \n' +
      '            # Adjust quantity and cost\n' +
      '            position.quantity = old_qty * ratio\n' +
      '            position.avg_cost = old_cost / ratio\n' +
      '            \n' +
      '            # Log for audit\n' +
      '            await self.log_corporate_action(\n' +
      '                symbol=symbol,\n' +
      '                action_type="SPLIT",\n' +
      '                ratio=ratio,\n' +
      '                old_qty=old_qty,\n' +
      '                new_qty=position.quantity,\n' +
      '                ex_date=ex_date\n' +
      '            )\n' +
      '    \n' +
      '    async def handle_dividend(self, symbol, amount_per_share, ex_date):\n' +
      '        """\n' +
      '        Dividend: AAPL pays $0.50/share on 2025-10-26\n' +
      '        Position: 100 shares → Receive $50 cash\n' +
      '        """\n' +
      '        positions = self.get_positions_by_symbol(symbol)\n' +
      '        \n' +
      '        for position in positions:\n' +
      '            dividend_amount = position.quantity * amount_per_share\n' +
      '            \n' +
      '            # Credit cash account\n' +
      '            await self.credit_cash(\n' +
      '                account=position.account,\n' +
      '                amount=dividend_amount,\n' +
      '                reason=f"Dividend: {symbol} {amount_per_share}/share"\n' +
      '            )\n' +
      '            \n' +
      '            # For tax reporting, dividends reduce cost basis\n' +
      '            # (qualified dividends are tax-advantaged)\n' +
      '            position.avg_cost -= amount_per_share\n' +
      '    \n' +
      '    async def handle_merger(self, old_symbol, new_symbol, conversion_ratio, cash_component, merger_date):\n' +
      '        """\n' +
      '        Merger: AAPL acquired by NEWCO\n' +
      '        Each AAPL share → 1.5 NEWCO shares + $10 cash\n' +
      '        """\n' +
      '        positions = self.get_positions_by_symbol(old_symbol)\n' +
      '        \n' +
      '        for position in positions:\n' +
      '            old_qty = position.quantity\n' +
      '            \n' +
      '            # Create new position in NEWCO\n' +
      '            new_qty = old_qty * conversion_ratio\n' +
      '            new_cost = position.avg_cost / conversion_ratio\n' +
      '            \n' +
      '            await self.create_position(\n' +
      '                symbol=new_symbol,\n' +
      '                quantity=new_qty,\n' +
      '                avg_cost=new_cost,\n' +
      '                account=position.account\n' +
      '            )\n' +
      '            \n' +
      '            # Cash component\n' +
      '            cash_received = old_qty * cash_component\n' +
      '            await self.credit_cash(\n' +
      '                account=position.account,\n' +
      '                amount=cash_received,\n' +
      '                reason=f"Merger: {old_symbol} → {new_symbol}"\n' +
      '            )\n' +
      '            \n' +
      '            # Close old position\n' +
      '            await self.close_position(position, reason="Merger")\n' +
      '```\n\n' +
      '**2. Multi-Currency Positions**\n\n' +
      'Challenges:\n' +
      '- Position in SONY (JPY) but report P&L in USD\n' +
      '- FX rates change continuously → unrealized P&L changes\n' +
      '- Need to hedge FX risk (or accept it)\n\n' +
      'Implementation:\n' +
      '```python\n' +
      'class MultiCurrencyPosition:\n' +
      '    def __init__(self, symbol, quantity, local_price, currency, fx_rate_to_usd):\n' +
      '        self.symbol = symbol  # "SONY"\n' +
      '        self.quantity = quantity  # 1000 shares\n' +
      '        self.local_price = local_price  # ¥15,000 per share\n' +
      '        self.currency = currency  # "JPY"\n' +
      '        self.fx_rate_to_usd = fx_rate_to_usd  # 0.0067 (150 JPY = 1 USD)\n' +
      '        \n' +
      '        # Calculate USD values\n' +
      '        self.local_value = quantity * local_price  # ¥15,000,000\n' +
      '        self.usd_value = self.local_value * fx_rate_to_usd  # $100,000\n' +
      '    \n' +
      '    def update_fx_rate(self, new_fx_rate):\n' +
      '        """\n' +
      '        FX rate changed: 150 JPY/USD → 145 JPY/USD (JPY strengthened)\n' +
      '        Same ¥15M position now worth more USD\n' +
      '        """\n' +
      '        old_usd_value = self.usd_value\n' +
      '        self.fx_rate_to_usd = new_fx_rate\n' +
      '        self.usd_value = self.local_value * new_fx_rate\n' +
      '        \n' +
      '        # FX P&L\n' +
      '        fx_pnl = self.usd_value - old_usd_value\n' +
      '        return fx_pnl\n' +
      '    \n' +
      '    def calculate_pnl(self, current_local_price, current_fx_rate):\n' +
      '        """\n' +
      '        Total P&L = Local P&L (stock price change) + FX P&L (rate change)\n' +
      '        """\n' +
      '        # Local P&L (in local currency)\n' +
      '        local_pnl = self.quantity * (current_local_price - self.local_price)\n' +
      '        \n' +
      '        # Convert to USD\n' +
      '        local_pnl_usd = local_pnl * current_fx_rate\n' +
      '        \n' +
      '        # FX P&L (position value at new FX rate)\n' +
      '        fx_pnl = self.local_value * (current_fx_rate - self.fx_rate_to_usd)\n' +
      '        \n' +
      '        return {\n' +
      '            "local_pnl": local_pnl,\n' +
      '            "local_pnl_usd": local_pnl_usd,\n' +
      '            "fx_pnl": fx_pnl,\n' +
      '            "total_pnl_usd": local_pnl_usd + fx_pnl\n' +
      '        }\n' +
      '```\n\n' +
      '**3. Options Positions**\n\n' +
      'Challenges:\n' +
      '- Options have expiration (expire worthless or get exercised)\n' +
      '- Multiple contracts: AAPL 150C, AAPL 155C, AAPL 160C all different positions\n' +
      '- Exercise/assignment changes stock position\n' +
      '- Greeks change continuously (delta, gamma, vega, theta)\n\n' +
      'Implementation:\n' +
      '```python\n' +
      'class OptionsPosition:\n' +
      '    def __init__(self, symbol, strike, expiration, option_type, quantity, premium):\n' +
      '        self.symbol = symbol  # "AAPL"\n' +
      '        self.strike = strike  # 150\n' +
      '        self.expiration = expiration  # 2025-11-15\n' +
      '        self.option_type = option_type  # "CALL" or "PUT"\n' +
      '        self.quantity = quantity  # 10 contracts = 1000 shares\n' +
      '        self.premium = premium  # $5.00 per share\n' +
      '        \n' +
      '        # Greeks (updated real-time)\n' +
      '        self.delta = 0.60\n' +
      '        self.gamma = 0.05\n' +
      '        self.vega = 0.20\n' +
      '        self.theta = -0.10  # Losing $0.10/day per share\n' +
      '    \n' +
      '    async def handle_expiration(self, stock_price):\n' +
      '        """\n' +
      '        Option expiration: Either worthless or exercised\n' +
      '        """\n' +
      '        if self.option_type == "CALL":\n' +
      '            if stock_price > self.strike:\n' +
      '                # ITM: Exercised, receive stock\n' +
      '                await self.exercise_call(stock_price)\n' +
      '            else:\n' +
      '                # OTM: Expire worthless\n' +
      '                await self.expire_worthless()\n' +
      '        else:  # PUT\n' +
      '            if stock_price < self.strike:\n' +
      '                # ITM: Exercised, sell stock\n' +
      '                await self.exercise_put(stock_price)\n' +
      '            else:\n' +
      '                # OTM: Expire worthless\n' +
      '                await self.expire_worthless()\n' +
      '    \n' +
      '    async def exercise_call(self, stock_price):\n' +
      '        """\n' +
      '        Long Call Exercise:\n' +
      '        - Close option position\n' +
      '        - Open stock position (buy at strike)\n' +
      '        """\n' +
      '        shares = self.quantity * 100\n' +
      '        \n' +
      '        # Close option\n' +
      '        await self.close_position()\n' +
      '        \n' +
      '        # Buy stock at strike\n' +
      '        await self.open_stock_position(\n' +
      '            symbol=self.symbol,\n' +
      '            quantity=shares,\n' +
      '            price=self.strike,\n' +
      '            reason="Call Exercise"\n' +
      '        )\n' +
      '```\n\n' +
      '**Production Checklist:**\n' +
      '- Corporate action calendar: Check daily for upcoming events\n' +
      '- FX rates: Update every minute during market hours\n' +
      '- Options expiration: Auto-handle expiration on ex-date\n' +
      '- Multi-leg options: Track spreads as single position\n' +
      '- Tax lots: Track for FIFO/LIFO cost basis',
    keyPoints: [
      'Corporate actions: Handle stock splits (adjust quantity/cost), dividends (credit cash), mergers (convert to new symbol + cash component)',
      'Multi-currency: Track positions in local currency, convert to USD for P&L, separate local P&L from FX P&L, update FX rates continuously',
      'Options positions: Track strike/expiration/type, handle expiration (ITM exercised, OTM expire worthless), exercise changes stock position',
      'Options Greeks: Update delta/gamma/vega/theta real-time, calculate option-equivalent stock exposure (delta × quantity × 100)',
      'Production considerations: Daily corporate action check, FX rate updates every minute, auto-handle options expiration, track tax lots for FIFO/LIFO',
    ],
  },
];

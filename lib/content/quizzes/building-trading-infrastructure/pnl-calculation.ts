export const pnlCalculationQuiz = [
    {
        id: 'pnl-calculation-q-1',
        question:
            'Design a "real-time P&L calculation system" for a high-frequency trading firm executing 10,000 trades per second. The system must calculate P&L within 100μs of each fill and support multi-dimensional attribution (strategy, trader, symbol). What are the key architectural decisions?',
        sampleAnswer:
            'High-Frequency P&L System:\n\n' +
            '**Architecture:**\n' +
            '1. In-Memory P&L Store (Redis/Hazelcast):\n' +
            '   - Store all positions and P&L in memory\n' +
            '   - Key schema: `pnl:{symbol}:{account}:{strategy}`\n' +
            '   - Value: Serialized PnLSnapshot (quantity, avg_cost, realized_pnl, unrealized_pnl)\n' +
            '   - Latency: <10μs read/write\n\n' +
            '2. Event-Driven Updates:\n' +
            '   - Fill event arrives → Update P&L immediately\n' +
            '   - No database queries in hot path\n' +
            '   - Atomic updates using Redis transactions\n\n' +
            '3. P&L Calculation Logic:\n' +
            '   - Realized P&L: Calculate on position close (FIFO, LIFO, or weighted avg)\n' +
            '   - Unrealized P&L: (current_price - avg_cost) × quantity\n' +
            '   - Update on every fill AND every price tick\n\n' +
            '4. Multi-Dimensional Attribution:\n' +
            '   - Primary positions: By (symbol, account, strategy, trader)\n' +
            '   - Aggregation views: Real-time materialized views in Redis\n' +
            '   - Example: Strategy P&L = SUM of all positions in that strategy\n\n' +
            '5. Persistence:\n' +
            '   - Async write to PostgreSQL every 100ms\n' +
            '   - Write-ahead log for crash recovery\n' +
            '   - Kafka topic for P&L events (consumed by risk systems)\n\n' +
            '**Performance Target: 100μs per Fill**\n' +
            '```python\n' +
            'async def process_fill_ultra_fast(fill):\n' +
            '    # Step 1: Read position from Redis (10μs)\n' +
            '    position_key = f"pos:{fill.symbol}:{fill.account}:{fill.strategy}"\n' +
            '    position = await redis.get(position_key)\n' +
            '    \n' +
            '    # Step 2: Calculate new P&L (30μs)\n' +
            '    old_qty = position.quantity\n' +
            '    new_qty, new_cost, realized_pnl = calculate_pnl(\n' +
            '        old_qty=old_qty,\n' +
            '        old_cost=position.avg_cost,\n' +
            '        fill_side=fill.side,\n' +
            '        fill_qty=fill.quantity,\n' +
            '        fill_price=fill.price\n' +
            '    )\n' +
            '    \n' +
            '    # Step 3: Update Redis (10μs)\n' +
            '    position.quantity = new_qty\n' +
            '    position.avg_cost = new_cost\n' +
            '    position.realized_pnl += realized_pnl\n' +
            '    await redis.set(position_key, position)\n' +
            '    \n' +
            '    # Step 4: Publish event to Kafka (20μs async)\n' +
            '    await kafka.publish_async("pnl.updates", position)\n' +
            '    \n' +
            '    # Step 5: Update aggregations (30μs)\n' +
            '    await update_strategy_pnl(fill.strategy, realized_pnl)\n' +
            '    \n' +
            '    # Total: ~100μs\n' +
            '```\n\n' +
            '**Scalability:**\n' +
            '- 10,000 trades/sec = 1 trade per 100μs\n' +
            '- Single-threaded event loop handles this easily\n' +
            '- For 100K trades/sec, shard by symbol (10 shards = 10K/shard)\n\n' +
            '**Disaster Recovery:**\n' +
            '- Snapshot P&L to disk every 1 minute\n' +
            '- On crash: Restore from snapshot + replay Kafka log\n' +
            '- Recovery time: <30 seconds\n\n' +
            '**Production Checklist:**\n' +
            '- Monitor P&L update latency (alert if >200μs)\n' +
            '- Cross-check P&L vs broker every 15 minutes\n' +
            '- Store P&L snapshots for regulatory audit\n' +
            '- Support P&L restatement (e.g., broker corrects fill price)',
        keyPoints: [
            'In-memory store: Redis for <10μs P&L reads/writes, key schema pnl:{symbol}:{account}:{strategy}',
            'Event-driven: Process fills immediately, no DB queries in hot path, atomic Redis transactions',
            'P&L calculation: Realized P&L on position close (FIFO/LIFO), unrealized P&L = (current_price - avg_cost) × quantity',
            'Multi-dimensional: Track by (symbol, account, strategy, trader), real-time aggregation views for strategy/trader P&L',
            'Performance: 100μs per fill (10μs read + 30μs calc + 10μs write + 20μs Kafka + 30μs aggregation), handles 10K trades/sec',
        ],
    },
    {
        id: 'pnl-calculation-q-2',
        question:
            'You buy 100 shares of AAPL at $150, then sell 50 shares at $153, then buy 30 shares at $151. The current market price is $152. Calculate the realized P&L, unrealized P&L, and total P&L. Show your work using FIFO cost basis.',
        sampleAnswer:
            'AAPL P&L Calculation (FIFO Cost Basis):\n\n' +
            '**Step 1: Initial Position**\n' +
            '- BUY 100 shares @ $150\n' +
            '- Position: 100 shares @ $150 avg cost\n' +
            '- Total cost: 100 × $150 = $15,000\n' +
            '- Realized P&L: $0\n' +
            '- Unrealized P&L: $0\n\n' +
            '**Step 2: Sell 50 Shares @ $153 (FIFO: Sell oldest shares first)**\n' +
            '- Selling 50 of the original 100 shares bought @ $150\n' +
            '- Realized P&L = 50 × ($153 - $150) = 50 × $3 = **$150**\n' +
            '- Remaining position: 50 shares @ $150 avg cost (from original purchase)\n' +
            '- Unrealized P&L: 50 × ($152 current - $150 cost) = **$100**\n\n' +
            '**Step 3: Buy 30 Shares @ $151**\n' +
            '- Now have: 50 shares @ $150 + 30 shares @ $151\n' +
            '- Total quantity: 80 shares\n' +
            '- Weighted average cost: (50 × $150 + 30 × $151) / 80\n' +
            '  = ($7,500 + $4,530) / 80\n' +
            '  = $12,030 / 80\n' +
            '  = **$150.375** per share\n\n' +
            '**Step 4: Mark-to-Market @ $152**\n' +
            '- Position: 80 shares @ $150.375 avg cost\n' +
            '- Current value: 80 × $152 = $12,160\n' +
            '- Cost basis: 80 × $150.375 = $12,030\n' +
            '- Unrealized P&L: $12,160 - $12,030 = **$130**\n\n' +
            '**Final P&L:**\n' +
            '- **Realized P&L: $150** (from selling 50 shares @ $153)\n' +
            '- **Unrealized P&L: $130** (from holding 80 shares marked @ $152)\n' +
            '- **Total P&L: $150 + $130 = $280**\n\n' +
            '**Verification:**\n' +
            '- Cash out: Sold 50 @ $153 = $7,650\n' +
            '- Cash in: Bought 100 @ $150 = -$15,000, Bought 30 @ $151 = -$4,530\n' +
            '- Net cash: $7,650 - $15,000 - $4,530 = -$11,880 (cash out)\n' +
            '- Current position value: 80 × $152 = $12,160\n' +
            '- Total P&L: $12,160 - $11,880 = $280 ✓\n\n' +
            '**Implementation:**\n' +
            '```python\n' +
            'position = Position(symbol="AAPL", quantity=0, avg_cost=0, realized_pnl=0)\n' +
            '\n' +
            '# Step 1: Buy 100 @ $150\n' +
            'position.apply_fill("BUY", 100, 150.00)\n' +
            '# Position: 100 @ $150, Realized: $0\n' +
            '\n' +
            '# Step 2: Sell 50 @ $153\n' +
            'position.apply_fill("SELL", 50, 153.00)\n' +
            '# Realized P&L: 50 × ($153 - $150) = $150\n' +
            '# Position: 50 @ $150, Realized: $150\n' +
            '\n' +
            '# Step 3: Buy 30 @ $151\n' +
            'position.apply_fill("BUY", 30, 151.00)\n' +
            '# Position: 80 @ $150.375, Realized: $150\n' +
            '\n' +
            '# Step 4: Mark-to-market @ $152\n' +
            'position.mark_to_market(152.00)\n' +
            '# Unrealized: 80 × ($152 - $150.375) = $130\n' +
            '\n' +
            'print(f"Realized P&L: ${position.realized_pnl:.2f}")  # $150.00\n' +
            'print(f"Unrealized P&L: ${position.unrealized_pnl:.2f}")  # $130.00\n' +
            'print(f"Total P&L: ${position.total_pnl():.2f}")  # $280.00\n' +
            '```',
        keyPoints: [
            'Step 1: Buy 100 @ $150, position = 100 shares @ $150 cost, realized P&L = $0',
            'Step 2: Sell 50 @ $153 (FIFO), realized P&L = 50 × ($153 - $150) = $150, position = 50 @ $150',
            'Step 3: Buy 30 @ $151, position = 80 shares @ $150.375 weighted avg cost [(50×$150 + 30×$151)/80]',
            'Step 4: Mark-to-market @ $152, unrealized P&L = 80 × ($152 - $150.375) = $130',
            'Final: Realized P&L = $150, Unrealized P&L = $130, Total P&L = $280',
        ],
    },
    {
        id: 'pnl-calculation-q-3',
        question:
            'Design a "P&L attribution system" that breaks down daily P&L into components: price change, position change, fees, and funding costs (for overnight positions). Why is attribution important and how would you implement it?',
        sampleAnswer:
            'P&L Attribution System:\n\n' +
            '**Why Attribution Matters:**\n' +
            '1. **Strategy Optimization**: Identify which strategies are profitable\n' +
            '2. **Risk Management**: Separate P&L from price vs position changes\n' +
            '3. **Cost Analysis**: Track fees and funding eating into profits\n' +
            '4. **Performance Review**: Trader compensation based on attribution\n\n' +
            '**P&L Components:**\n\n' +
            '1. **Price Change P&L** (market moved):\n' +
            '   - P&L from holding constant position while price changed\n' +
            '   - Formula: position_qty × (new_price - old_price)\n' +
            '   - Example: Hold 100 shares, price $150 → $152 = 100 × $2 = $200\n\n' +
            '2. **Position Change P&L** (position changed):\n' +
            '   - P&L from changing position size (realized gains/losses)\n' +
            '   - Formula: trade_qty × (trade_price - avg_cost)\n' +
            '   - Example: Sell 50 @ $153, avg cost $150 = 50 × $3 = $150\n\n' +
            '3. **Fees** (trading costs):\n' +
            '   - Commission, exchange fees, SEC fees, TAF fees\n' +
            '   - Example: $0.01/share × 100 shares = -$1.00\n\n' +
            '4. **Funding Costs** (overnight interest):\n' +
            '   - For long positions: Pay interest on margin\n' +
            '   - For short positions: Pay stock borrow cost\n' +
            '   - Example: Borrow $100K overnight @ 5% APY = -$13.70/day\n\n' +
            '**Implementation:**\n' +
            '```python\n' +
            '@dataclass\n' +
            'class PnLAttribution:\n' +
            '    """P&L attribution breakdown"""\n' +
            '    date: date\n' +
            '    symbol: str\n' +
            '    \n' +
            '    # Components\n' +
            '    pnl_from_price: Decimal = Decimal(\'0\')  # Price moved\n' +
            '    pnl_from_position: Decimal = Decimal(\'0\')  # Position changed (realized)\n' +
            '    pnl_from_fees: Decimal = Decimal(\'0\')  # Trading fees\n' +
            '    pnl_from_funding: Decimal = Decimal(\'0\')  # Overnight interest/borrow\n' +
            '    \n' +
            '    def total_pnl(self) -> Decimal:\n' +
            '        return (\n' +
            '            self.pnl_from_price +\n' +
            '            self.pnl_from_position +\n' +
            '            self.pnl_from_fees +\n' +
            '            self.pnl_from_funding\n' +
            '        )\n' +
            '\n' +
            'class AttributionCalculator:\n' +
            '    def calculate_attribution(self, position, old_price, new_price, fill=None):\n' +
            '        attr = PnLAttribution(date=date.today(), symbol=position.symbol)\n' +
            '        \n' +
            '        # 1. Price change P&L (before fill)\n' +
            '        attr.pnl_from_price = position.quantity * (new_price - old_price)\n' +
            '        \n' +
            '        # 2. Position change P&L (from fill)\n' +
            '        if fill:\n' +
            '            if fill.side == "SELL" and position.quantity > 0:\n' +
            '                # Selling long: Realized gain\n' +
            '                attr.pnl_from_position = fill.quantity * (fill.price - position.avg_cost)\n' +
            '            elif fill.side == "BUY" and position.quantity < 0:\n' +
            '                # Covering short: Realized gain\n' +
            '                attr.pnl_from_position = fill.quantity * (position.avg_cost - fill.price)\n' +
            '            \n' +
            '            # 3. Fees\n' +
            '            attr.pnl_from_fees = -fill.fees\n' +
            '        \n' +
            '        # 4. Funding costs (calculated EOD)\n' +
            '        attr.pnl_from_funding = self.calculate_funding(position)\n' +
            '        \n' +
            '        return attr\n' +
            '    \n' +
            '    def calculate_funding(self, position):\n' +
            '        """\n' +
            '        Funding cost for overnight position\n' +
            '        """\n' +
            '        position_value = abs(position.quantity * position.current_price)\n' +
            '        \n' +
            '        if position.quantity > 0:\n' +
            '            # Long: Pay interest on margin (if using margin)\n' +
            '            margin_rate = Decimal(\'0.05\')  # 5% APY\n' +
            '            daily_cost = position_value * margin_rate / 365\n' +
            '            return -daily_cost  # Cost\n' +
            '        else:\n' +
            '            # Short: Pay stock borrow cost\n' +
            '            borrow_rate = Decimal(\'0.03\')  # 3% APY (easy to borrow)\n' +
            '            daily_cost = position_value * borrow_rate / 365\n' +
            '            return -daily_cost  # Cost\n' +
            '```\n\n' +
            '**Example: Full Day Attribution**\n' +
            '```python\n' +
            '# Start of day: 100 shares @ $150, price = $150\n' +
            'position = Position(symbol="AAPL", quantity=100, avg_cost=150.00)\n' +
            '\n' +
            '# 10:00 AM: Price moves to $152 (no trades yet)\n' +
            'attr_1 = calc.calculate_attribution(position, old_price=150, new_price=152)\n' +
            '# pnl_from_price: 100 × ($152 - $150) = $200\n' +
            '# pnl_from_position: $0 (no trade)\n' +
            '# pnl_from_fees: $0\n' +
            '# Total: $200\n' +
            '\n' +
            '# 11:00 AM: Sell 50 @ $153\n' +
            'fill = Fill(side="SELL", quantity=50, price=153.00, fees=1.00)\n' +
            'attr_2 = calc.calculate_attribution(position, old_price=152, new_price=153, fill=fill)\n' +
            '# pnl_from_price: 100 × ($153 - $152) = $100 (before fill)\n' +
            '# pnl_from_position: 50 × ($153 - $150) = $150 (realized gain)\n' +
            '# pnl_from_fees: -$1.00\n' +
            '# Total: $249\n' +
            '\n' +
            '# End of day: Calculate funding\n' +
            'position.quantity = 50  # After sell\n' +
            'position.current_price = 153.00\n' +
            'attr_3 = calc.calculate_attribution(position, old_price=153, new_price=153)\n' +
            '# pnl_from_funding: -(50 × $153 × 0.05 / 365) = -$1.05\n' +
            '\n' +
            '# Daily P&L Attribution:\n' +
            '# Price change: $200 + $100 = $300\n' +
            '# Position change: $150\n' +
            '# Fees: -$1.00\n' +
            '# Funding: -$1.05\n' +
            '# Total: $447.95\n' +
            '```\n\n' +
            '**Production Checklist:**\n' +
            '- Calculate attribution on every fill and price update\n' +
            '- Store daily attribution in database for analysis\n' +
            '- Generate attribution reports by strategy, trader, symbol\n' +
            '- Alert if fees >2% of P&L (excessive trading costs)\n' +
            '- Track funding costs separately (can be significant for large positions)',
        keyPoints: [
            'P&L Attribution: Break down P&L into price change (market moved), position change (realized gains), fees (trading costs), funding (overnight interest/borrow)',
            'Price change P&L: position_qty × (new_price - old_price), represents P&L from holding constant position',
            'Position change P&L: trade_qty × (trade_price - avg_cost), represents realized gains from closing positions',
            'Fees: Commission + exchange fees, typically -$0.001-$0.01 per share',
            'Funding costs: For long positions pay margin interest (~5% APY), for short positions pay stock borrow cost (~3% APY), calculated daily',
        ],
    },
];


export const marketParticipantsQuiz = [
  {
    id: 'fm-1-9-q-1',
    question:
      'Institutional investors have predictable rebalancing patterns that create exploitable price impacts. Design a quantitative strategy to capitalize on quarter-end rebalancing. What are the ethical implications? When does this become illegal manipulation?',
    sampleAnswer: `**Quarter-End Rebalancing Exploitation:**

**Pattern:** Institutions rebalance to target allocations (typically 60/40 stocks/bonds) at quarter-end.

**Strategy:**
If stocks outperform bonds in Q1:
- Actual allocation shifts to 64% stocks / 36% bonds
- Institutions must sell 4% stocks, buy 4% bonds
- $11T in passive assets × 4% = $440B selling pressure on stocks

**Backtest Implementation:**
\`\`\`python
def quarter_end_strategy():
    # If Q1 stocks up >10%:
    # Day before quarter-end: Short S&P
    # Day after: Cover short
    # Avg reversal: 0.5-1.0%
    
    results = {
        'frequency': '25/40 quarters (62.5%)',
        'avg_return_per_trade': '0.8%',
        'annual_return': '2.0%',
        'sharpe': 1.5
    }
    return results
\`\`\`

**Ethical/Legal Analysis:**

**LEGAL:**
- ✅ Analyzing public information (rebalancing dates known)
- ✅ Predicting behavior based on market returns
- ✅ Trading ahead of anticipated flow
- ✅ Providing liquidity when needed

**ILLEGAL:**
- ❌ Front-running confidential client orders (broker knows client will sell)
- ❌ Price manipulation to trigger rebalancing
- ❌ Coordinating with others to move prices

**The Line:** PUBLIC information (known rebalancing dates) = legal. PRIVATE information (specific client orders) = illegal front-running.

**Bottom Line:** Exploiting known institutional patterns is legal and common HFT strategy. Using confidential order information is securities fraud.`,
    keyPoints: [
      'Institutional rebalancing: $11T assets rebalance quarterly to target allocations',
      'Strategy: Short stocks on last day of strong quarters, cover next day (0.5-1% reversal)',
      'Legal: Trading on public patterns (known dates). Illegal: Front-running private client orders',
      'Difference: Information source (public vs confidential) determines legality',
      'Annual return: ~2% with Sharpe 1.5, but getting crowded as more know about it',
    ],
  },
  {
    id: 'fm-1-9-q-2',
    question:
      "HFT firms claim they improve markets (tighter spreads, liquidity) but critics argue they're predatory (latency arbitrage, quote stuffing). Evaluate both sides. Design regulations to keep HFT benefits while preventing predatory behavior.",
    sampleAnswer: `**HFT Benefits vs Harms:**

**Benefits:**
- Tighter spreads: $0.05 → $0.01 (saves retail $4B+/year)
- Deep liquidity: Trade $10M with <0.5% impact
- Price efficiency: Keeps ETF = NAV, futures = spot

**Harms:**
- Latency arbitrage: Fast traders exploit slow traders' stale quotes ($1-2B/year transfer)
- Flash crashes: 2010 Dow -1000pts in minutes (HFT withdrew liquidity)
- Arms race: $10B+ spent on speed (zero societal benefit)
- Quote stuffing: Fake orders to create noise

**Net Assessment:** Benefits > costs, but need regulation to prevent abuse.

**Regulatory Proposal:**

**1. Speed Bumps (IEX Model):**
- 350 microsecond delay on all orders
- Eliminates latency arbitrage
- Still allows market making

**2. Minimum Quote Life:**
- Quotes must stay active 500ms
- Prevents quote stuffing
- Ensures liquidity is real

**3. Maker-Taker Reform:**
- Remove rebates for posting liquidity
- Eliminates incentive for quote spam

**4. Circuit Breakers:**
- Stock-level pause after 5% move in 5min
- Prevents flash crash cascades

**5. Transparency:**
- HFT firms must register
- Annual strategy reporting
- Allows regulators to monitor manipulation

**6. Ban Predatory Tactics:**
- No "pinging" dark pools
- No layering/spoofing
- Enforce with $100M+ fines

**Implementation:**
\`\`\`python
class HFTRegulation:
    def enforce_rules (self, trade):
        # Speed bump
        if trade.latency < 350:  # microseconds
            delay(350 - trade.latency)
        
        # Minimum quote life
        if trade.is_quote and trade.duration < 500_000:
            reject("Quote must stay 500ms")
        
        # Detect spoofing
        if trade.cancel_rate > 0.90:
            flag_for_review("High cancel rate - possible spoofing")
        
        # Circuit breaker
        if stock.move_5min > 0.05:
            pause_trading (stock, duration=300)  # 5min pause
\`\`\`

**Bottom Line:** Keep HFT benefits (spreads, liquidity), prevent harms (speed bumps, quote life, ban manipulation). Net effect: Better markets for all.`,
    keyPoints: [
      'HFT benefits: Tighter spreads (\$4B+ savings), liquidity, efficiency',
      'HFT harms: Latency arbitrage ($1-2B transfer), flash crashes, wasteful arms race',
      'Regulation: Speed bumps (350µs), minimum quote life (500ms), circuit breakers',
      'Ban: Quote stuffing, pinging dark pools, layering/spoofing',
      'Goal: Keep benefits (lower costs) while preventing predatory tactics',
    ],
  },
  {
    id: 'fm-1-9-q-3',
    question:
      'Market makers profit from bid-ask spread while providing liquidity. Design a market making strategy with inventory risk management, adverse selection mitigation, and dynamic spread adjustment. How would you handle a flash crash scenario?',
    sampleAnswer: `**Market Making Strategy:**

**Core Concept:** Quote bid/ask, profit from spread, manage inventory risk.

**Implementation:**
\`\`\`python
class MarketMaker:
    def __init__(self):
        self.inventory = 0  # Target: 0 (neutral)
        self.max_inventory = 10000  # Risk limit
        self.base_spread = 0.0005  # 5 bps
    
    def quote (self, fair_value):
        # Adjust spread based on inventory
        inventory_skew = self.inventory / self.max_inventory
        
        # If long (positive inventory), widen ask, tighten bid
        # (want to sell more, buy less)
        bid = fair_value - self.base_spread / 2 - inventory_skew * 0.0002
        ask = fair_value + self.base_spread / 2 + inventory_skew * 0.0002
        
        return bid, ask
    
    def adjust_spread_for_risk (self, volatility, order_flow_toxicity):
        """
        Wider spreads when:
        - High volatility (more price risk)
        - Toxic order flow (informed traders)
        """
        vol_multiplier = volatility / 0.01  # Scale to 1% base vol
        toxicity_multiplier = 1 + order_flow_toxicity
        
        adjusted_spread = self.base_spread * vol_multiplier * toxicity_multiplier
        
        return adjusted_spread
    
    def detect_adverse_selection (self, recent_fills):
        """
        If we keep buying at bid and price drops,
        or selling at ask and price rises,
        we're being adversely selected (losing to informed traders)
        """
        buys_followed_by_drop = sum(1 for trade in recent_fills 
                                    if trade.side == 'buy' and trade.pnl < 0)
        
        toxicity = buys_followed_by_drop / len (recent_fills)
        
        if toxicity > 0.6:  # Losing on 60%+ of trades
            return 'HIGH_TOXICITY'  # Widen spreads or stop quoting
    
    def handle_flash_crash (self):
        """
        Flash crash (2010): Dow -1000pts in minutes
        
        Market makers faced:
        - Prices moving 5% in seconds
        - Unable to hedge fast enough
        - Risk of bankruptcy
        
        Response: PULL QUOTES (stop providing liquidity)
        """
        if volatility > 5 * normal_volatility:
            # Emergency: Cancel all orders
            self.cancel_all_quotes()
            
            # Hedge existing inventory aggressively
            self.hedge_inventory_at_market()
            
            # Don't resume until volatility normalizes
            while volatility > 2 * normal_volatility:
                wait()
            
            # Resume with MUCH wider spreads
            self.base_spread *= 3
\`\`\`

**Flash Crash Scenario:**

**What Happened (May 6, 2010):**
1. Large sell order (\$4.1B E-mini futures)
2. HFT market makers bought, then tried to sell
3. No buyers → HFTs sold to each other in hot potato
4. Prices cascaded down
5. Market makers hit risk limits → PULLED ALL QUOTES
6. Liquidity evaporated → Dow -1000pts in 5min
7. Circuit breakers didn't exist yet

**Market Maker Response:**
- Detect extreme volatility → stop quoting
- Flatten inventory (sell everything now, worry about loss later)
- Survival > profit in tail events

**Post-Crisis Improvements:**
- Circuit breakers: Pause trading after 5% move
- Position limits: Can't build huge inventory
- Kill switches: Auto-stop if losses exceed threshold

**Bottom Line:** Market making = sell liquidity, earn spread. But in crashes, liquidity becomes infinitely expensive (no one wants to provide it). Pull quotes to survive.`,
    keyPoints: [
      'Market making: Quote bid/ask, profit from spread, manage inventory risk',
      'Inventory management: Skew quotes to reduce position (long → widen ask, tighten bid)',
      'Adverse selection: Detect toxic flow (losing on 60%+ trades), widen spreads or stop',
      'Flash crash response: Pull all quotes, hedge inventory, wait for volatility to normalize',
      '2010 lesson: Liquidity disappears when needed most. Circuit breakers now prevent cascades.',
    ],
  },
];

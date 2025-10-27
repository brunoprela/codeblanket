export const tradingVenuesExchangesQuiz = [
  {
    id: 'fm-1-10-q-1',
    question:
      'Compare NYSE (specialist model) vs NASDAQ (dealer network) vs dark pools (hidden liquidity). For a $10M institutional order, which venue minimizes market impact and why? Design a smart order routing algorithm.',
    sampleAnswer: `**Venue Comparison:**

**NYSE (Specialist/DMM Model):**
- Single designated market maker per stock
- Centralized order book
- Best for: Price discovery, auction mechanisms
- Market impact: Moderate (visible liquidity)

**NASDAQ (Competing Dealers):**
- Multiple market makers compete
- Decentralized quotes
- Best for: Tight spreads (competition)
- Market impact: Moderate (also visible)

**Dark Pools:**
- Hidden orders (no pre-trade transparency)
- Institutional block trading
- Best for: Large orders (no front-running)
- Market impact: Low (orders don't move market before execution)

**$10M Order Strategy:**

For large institutional order, use **combination**:

\`\`\`python
class SmartOrderRouter:
    def route_large_order (self, symbol, size=10_000_000):
        """
        $10M order needs careful routing to minimize market impact
        """
        # 1. Try dark pools first (40% of order)
        dark_pool_fill = self.try_dark_pools (symbol, size * 0.40)
        remaining = size - dark_pool_fill
        
        # 2. Lit markets with TWAP (30%)
        lit_fill = self.twap_execution (symbol, size * 0.30, duration_hours=4)
        remaining -= lit_fill
        
        # 3. Iceberg orders on exchanges (20%)
        iceberg_fill = self.iceberg_order (symbol, size * 0.20, display_size=1000)
        remaining -= iceberg_fill
        
        # 4. Remaining via VWAP (10%)
        vwap_fill = self.vwap_execution (symbol, remaining)
        
        return {
            'dark_pools': dark_pool_fill,
            'twap': lit_fill,
            'iceberg': iceberg_fill,
            'vwap': vwap_fill,
            'total_filled': size,
            'estimated_impact': '10-15 bps'  # 0.10-0.15%
        }
    
    def try_dark_pools (self, symbol, size):
        """
        Route to multiple dark pools
        Priority: Largest pools with best fill rates
        """
        dark_pools = ['BlockCross', 'Sigma X', 'Level ATS']
        filled = 0
        
        for pool in dark_pools:
            # Check available liquidity (hidden)
            fill = pool.try_match (symbol, size - filled)
            filled += fill
            
            if filled >= size:
                break
        
        return filled
    
    def calculate_market_impact (self, order_size, adv):
        """
        Market impact model
        
        Impact = α × (Order Size / ADV)^β
        α = 0.1 (constant)
        β = 0.6 (concavity)
        ADV = Average Daily Volume
        """
        impact_bps = 10 * (order_size / adv) **0.6
        
        return impact_bps
\`\`\`

**Why Dark Pools for Large Orders:**
- No pre-trade transparency → can't front-run
- Match with other institutions (natural liquidity)
- Minimal market impact (orders hidden)
- Example: Block of 100K shares at midpoint

**Risks of Dark Pools:**
- Lower fill rates (less liquidity than lit markets)
- Information leakage (some pools leak)
- Adverse selection (trading against informed flow)

**Optimal Strategy:**1. **Start with dark pools** (40% of order, 0 market impact)
2. **TWAP on lit markets** (30%, spread over 4 hours)
3. **Iceberg orders** (20%, show 1K shares, hide 9K)
4. **VWAP for remainder** (10%, match market rhythm)

**Bottom Line:** Large orders need multi-venue strategy. Dark pools for blocks, lit markets with algos for rest. Total impact: 10-15 bps vs 50+ bps if dumping on one exchange.`,
    keyPoints: [
      'NYSE: Specialist model, centralized, good price discovery. NASDAQ: Competing dealers, tight spreads',
      'Dark pools: Hidden liquidity, no front-running, best for large institutional orders',
      'Smart routing: 40% dark pools, 30% TWAP, 20% iceberg, 10% VWAP',
      'Market impact model: Impact = α × (Size / ADV)^0.6',
      'Dark pool advantage: Orders invisible → no market impact before execution',
    ],
  },
  {
    id: 'fm-1-10-q-2',
    question:
      'MiFID II (Europe) and Reg NMS (US) mandate best execution. Define best execution beyond just price. Design a transaction cost analysis (TCA) system that measures execution quality across multiple dimensions.',
    sampleAnswer: `**Best Execution Dimensions:**

**Not just price!** Must consider:
1. **Price:** Execution price vs benchmark
2. **Speed:** How fast was order filled?
3. **Certainty:** Fill rate (100%? or partial?)
4. **Market impact:** Did order move market?
5. **Opportunity cost:** Unfilled portion cost
6. **Information leakage:** Did others detect order?

**Transaction Cost Analysis (TCA) System:**

\`\`\`python
class TCASystem:
    def analyze_execution (self, order, executions):
        """
        Comprehensive execution quality analysis
        """
        # 1. Implementation Shortfall
        IS = self.implementation_shortfall (order, executions)
        
        # 2. Market impact
        impact = self.market_impact (order, executions)
        
        # 3. Timing cost
        timing = self.timing_cost (order, executions)
        
        # 4. Opportunity cost (unfilled portion)
        opportunity = self.opportunity_cost (order, executions)
        
        # 5. Spread cost
        spread_cost = self.spread_captured (order, executions)
        
        total_cost_bps = IS + impact + timing + opportunity + spread_cost
        
        return {
            'implementation_shortfall_bps': IS,
            'market_impact_bps': impact,
            'timing_cost_bps': timing,
            'opportunity_cost_bps': opportunity,
            'spread_cost_bps': spread_cost,
            'total_cost_bps': total_cost_bps,
            'grade': self.grade_execution (total_cost_bps)
        }
    
    def implementation_shortfall (self, order, executions):
        """
        IS = (Execution Price - Decision Price) / Decision Price
        
        Decision price: When you decided to trade
        Execution price: Weighted average fill price
        """
        decision_price = order.price_at_decision
        
        # Weighted average execution price
        total_filled_value = sum (e.shares * e.price for e in executions)
        total_filled_shares = sum (e.shares for e in executions)
        avg_exec_price = total_filled_value / total_filled_shares
        
        # Shortfall (in bps)
        if order.side == 'BUY':
            shortfall = (avg_exec_price - decision_price) / decision_price
        else:  # SELL
            shortfall = (decision_price - avg_exec_price) / decision_price
        
        return shortfall * 10000  # Convert to bps
    
    def market_impact (self, order, executions):
        """
        How much did our order move the market?
        
        Compare price before/after our trades
        """
        price_before = order.price_at_arrival
        price_after = self.get_price_after_completion (order.symbol)
        
        if order.side == 'BUY':
            impact = (price_after - price_before) / price_before
        else:
            impact = (price_before - price_after) / price_before
        
        return impact * 10000
    
    def timing_cost (self, order, executions):
        """
        Cost of NOT trading immediately
        
        If price moved against us while we slowly traded
        """
        arrival_price = order.price_at_arrival
        vwap_during_period = self.calculate_vwap (order.symbol, order.start_time, order.end_time)
        
        if order.side == 'BUY':
            timing = (vwap_during_period - arrival_price) / arrival_price
        else:
            timing = (arrival_price - vwap_during_period) / arrival_price
        
        return timing * 10000
    
    def opportunity_cost (self, order, executions):
        """
        Cost of unfilled portion
        
        If we wanted 10K shares but only got 8K, 
        what was cost of missing 2K?
        """
        target_shares = order.target_shares
        filled_shares = sum (e.shares for e in executions)
        unfilled = target_shares - filled_shares
        
        if unfilled <= 0:
            return 0  # Fully filled
        
        # Price movement on unfilled portion
        exec_price = order.final_execution_price
        closing_price = self.get_closing_price (order.symbol, order.date)
        
        if order.side == 'BUY':
            opportunity = (closing_price - exec_price) / exec_price
        else:
            opportunity = (exec_price - closing_price) / exec_price
        
        # Weight by unfilled portion
        return opportunity * (unfilled / target_shares) * 10000
    
    def grade_execution (self, total_cost_bps):
        """Grade execution quality"""
        if total_cost_bps < 5:
            return 'EXCELLENT (A)'
        elif total_cost_bps < 15:
            return 'GOOD (B)'
        elif total_cost_bps < 30:
            return 'FAIR (C)'
        else:
            return 'POOR (D/F)'

# Example
tca = TCASystem()
analysis = tca.analyze_execution(
    order={'side': 'BUY', 'target_shares': 10000, 'price_at_decision': 100.00},
    executions=[
        {'shares': 5000, 'price': 100.05},
        {'shares': 3000, 'price': 100.10},
        {'shares': 2000, 'price': 100.15}
    ]
)

print(f"Implementation Shortfall: {analysis['implementation_shortfall_bps']:.1f} bps")
print(f"Market Impact: {analysis['market_impact_bps']:.1f} bps")
print(f"Total Cost: {analysis['total_cost_bps']:.1f} bps")
print(f"Grade: {analysis['grade']}")
\`\`\`

**Regulatory Requirements:**
- **Reg NMS (US):** Best execution = best available price considering all factors
- **MiFID II (EU):** Must publish TCA reports, demonstrate best execution
- **Brokers must:** Compare execution quality across venues, route smartly

**Bottom Line:** Best execution ≠ lowest price. Must minimize total cost: price + impact + timing + opportunity. Good execution: <15 bps total cost.`,
    keyPoints: [
      'Best execution: Price + speed + certainty + impact + opportunity cost + information leakage',
      'Implementation shortfall: (Execution price - Decision price), key metric',
      'TCA components: Shortfall + market impact + timing + opportunity + spread',
      'Grading: <5 bps excellent, <15 good, <30 fair, >30 poor',
      'Regulation: Reg NMS (US) and MiFID II (EU) mandate best execution with TCA proof',
    ],
  },
  {
    id: 'fm-1-10-q-3',
    question:
      "Exchange co-location allows HFT firms to place servers next to exchange servers, reducing latency from milliseconds to microseconds. Analyze the arms race dynamics. Should regulators allow/ban co-location? Design a 'fairness' alternative.",
    sampleAnswer: `**Co-Location Arms Race:**

**Current State:**
- Exchange rack space: $10K-50K/month
- Latency advantage: 5ms → 50µs (100x faster)
- Benefit: See orders milliseconds before others
- Cost: $100M+ infrastructure per firm

**Arms Race Dynamics:**
\`\`\`python
class CoLocationArmsRace:
    def model_equilibrium (self):
        """
        Game theory: Prisoner\'s dilemma
        
        If no one co-locates: Everyone saves $100M
        If one firm co-locates: They dominate, others lose
        Equilibrium: EVERYONE co-locates (wasteful)
        """
        scenarios = {
            'No one co-locates': {
                'cost': 0,
                'advantage': 0,
                'social_welfare': 'HIGH'
            },
            'Only Firm A co-locates': {
                'firm_a_profit': '+$50M/year',
                'others_loss': '-$200M combined',
                'social_welfare': 'ZERO-SUM'
            },
            'Everyone co-locates': {
                'cost_per_firm': '$100M',
                'advantage': 0,  # Relative advantage = 0
                'social_welfare': 'NEGATIVE (wasteful)'
            }
        }
        
        return 'Nash Equilibrium: Everyone co-locates (worst outcome)'
\`\`\`

**Arguments For Co-Location:**
- ✅ Voluntary: Exchange offers, firms choose
- ✅ Equal access: Anyone can pay for rack space
- ✅ Speed = efficiency (faster price discovery)
- ✅ Liquidity: HFT provides tight spreads

**Arguments Against:**
- ❌ Wasteful arms race (\$100M+ per firm, zero-sum)
- ❌ Unfair: Retail can't compete with microseconds
- ❌ Fragile: 2010 flash crash (speed → instability)
- ❌ Socially useless: No economic value from being 10µs faster

**Regulatory Options:**

**Option 1: Ban Co-Location**
- Pros: Level playing field, end arms race
- Cons: Liquidity might suffer, exchanges lose revenue

**Option 2: Mandate Equal Latency (IEX Model)**
- All orders delayed equally (350µs)
- Pros: Eliminates latency arbitrage, keeps liquidity
- Cons: Exchanges resist (lose co-location revenue)

**Option 3: Frequent Batch Auctions**
- Orders batched every 1 millisecond
- Execute all at once (like mini-auctions)
- Pros: Speed irrelevant within 1ms window
- Cons: Different market structure, untested at scale

**Recommended: IEX Speed Bump Model**

\`\`\`python
class FairnessAlternative:
    """
    IEX Exchange approach: 350µs speed bump for ALL orders
    """
    
    def implement_speed_bump (self, order):
        # Coil fiber optic cable to create 350µs delay
        delay_microseconds = 350
        
        # EVERY order delayed equally
        # Co-located: 50µs + 350µs = 400µs
        # Remote: 5000µs + 350µs = 5350µs
        
        # Relative difference: 4950µs vs previous 4950µs
        # But within exchange: everyone equal
        
        time.sleep (delay_microseconds / 1_000_000)
        
        return {
            'effect': 'Latency arbitrage eliminated',
            'liquidity': 'Maintained (350µs negligible for humans)',
            'fairness': 'All market participants see same prices'
        }
    
    def auction_alternative (self):
        """
        Frequent batch auctions (1ms)
        """
        # Collect all orders in 1ms window
        orders = self.collect_orders (duration_ms=1)
        
        # Execute all simultaneously at market-clearing price
        clearing_price = self.find_equilibrium (orders)
        
        # Allocate: Pro-rata or price-time
        fills = self.allocate_fills (orders, clearing_price)
        
        return {
            'effect': 'Speed irrelevant (all orders in 1ms treated equal)',
            'fairness': 'HIGH',
            'adoption': 'LOW (exchanges resist change)'
        }

\`\`\`

**Recommendation:**
**Mandate 350µs speed bumps** (IEX model) at all exchanges.

**Why:**
- Eliminates latency arbitrage (350µs >> any co-location advantage)
- Maintains liquidity (350µs is instant for humans)
- No harm to legitimate market making
- Ends wasteful arms race

**Bottom Line:** Co-location arms race is socially wasteful prisoner's dilemma. Speed bumps eliminate advantage while preserving liquidity. Regulators should mandate IEX-style delays.`,
    keyPoints: [
      'Co-location: Pay $50K/month for rack space, reduce latency from 5ms to 50µs',
      "Arms race: Prisoner's dilemma → everyone spends $100M +, zero relative advantage (wasteful)",
      'IEX solution: 350µs speed bump for ALL orders → eliminates latency arbitrage',
      'Batch auctions: Execute all orders in 1ms window simultaneously → speed irrelevant',
      'Recommendation: Mandate speed bumps (preserves liquidity, ends wasteful race, levels field)',
    ],
  },
];

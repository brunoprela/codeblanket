export const spreadsStrategiesQuiz = [
  {
    id: 'spreads-strategies-q-1',
    question:
      'Design a "spread trading system" that identifies optimal vertical spread opportunities. The system should: (1) Screen stocks by IV rank and liquidity, (2) Determine bull call spread vs bull put spread based on IV regime, (3) Calculate optimal strikes (probability of profit, risk/reward ratio), (4) Generate trade recommendations with expected value, (5) Implement adjustment rules if spread goes against you, (6) Backtest on historical data to measure win rate and returns. Explain decision logic and provide detailed metrics.',
    sampleAnswer:
      'Vertical Spread Trading System: **Strategy Selection Logic**: If bullish + high IV (>60%) → Bull Put Spread (sell premium), If bullish + low IV (<40%) → Bull Call Spread (buy cheap), If bearish + high IV → Bear Call Spread, If bearish + low IV → Bear Put Spread. **Strike Selection**: For credit spreads (bull put, bear call): Sell at 1 standard deviation OTM (30-35 delta), Buy at 2 SD OTM (10-15 delta) for protection, Target 20-30% credit of spread width. For debit spreads (bull call, bear put): Buy ATM or slightly OTM (50-60 delta), Sell further OTM (30-40 delta), Target 30-40% debit of spread width. **Expected Value Calculation**: EV = (Prob_Profit × Max_Profit) - (Prob_Loss × Max_Loss). Example Bull Put Spread: Sell 95 put @ 30 delta, buy 90 put, credit = $1.50, max profit = $150, max loss = $350, Prob_profit ≈ 70% (based on delta), EV = 0.70×$150 - 0.30×$350 = $105 - $105 = $0. Need edge from IV mean reversion! **Backtest Results**: Bull put spreads in high IV (rank > 70%): Win rate: 72%, avg profit $120, avg loss -$280, EV = $39 per trade. Bull call spreads in low IV (rank < 30%): Win rate: 58%, avg profit $180, avg loss -$140, EV = $45 per trade.',
    keyPoints: [
      'Strategy selection: Bullish + high IV → bull put spread (sell premium), bullish + low IV → bull call spread (buy cheap)',
      'Strike selection: Credit spreads sell at 1 SD (30 delta), buy at 2 SD (10 delta); target 20-30% credit of width',
      'Expected value: EV = prob_profit × max_profit - prob_loss × max_loss; need IV edge to be profitable',
      'Backtest shows: Bull put spreads in high IV win 72% with $39 EV; bull call spreads in low IV win 58% with $45 EV',
      'Adjustments: Exit at 50% max loss or 75% max profit; roll tested side if breached with > 21 DTE',
    ],
  },
  {
    id: 'spreads-strategies-q-2',
    question:
      'Compare "Iron Butterfly" vs "Iron Condor" for neutral income strategies. For each: (1) Calculate P&L at various stock prices, (2) Analyze probability of profit, (3) Determine max profit/loss and capital requirements, (4) Evaluate risk/reward ratios, (5) Assess which is better in different volatility regimes. Provide specific trade examples with numerical analysis.',
    sampleAnswer:
      'Iron Butterfly vs Iron Condor: **IRON BUTTERFLY**: Structure: Sell ATM put + call at 100, buy 90 put + 110 call (wings), Credit: $5.00 (higher premium), Max profit: $500 (at $100 exactly), Max loss: $500 ($10 wing - $5 credit), Breakevens: $95, $105 (narrow range!), Prob profit: ~40% (must stay in $95-$105). **IRON CONDOR**: Structure: Sell 95 put + 105 call, buy 90 put + 110 call, Credit: $2.00 (lower premium), Max profit: $200 (between $95-$105), Max loss: $300 ($5 wing - $2 credit), Breakevens: $93, $107 (wider range), Prob profit: ~65% (wider profit zone). **Comparison**: Iron Butterfly: Higher credit, tighter range, LOWER prob profit (40%), Best when: Stock pinned, very low vol expected. Iron Condor: Lower credit, wider range, HIGHER prob profit (65%), Best when: Neutral but want buffer, high IV (sell premium). **Risk/Reward**: Butterfly: $500 profit / $500 risk = 1:1, but only 40% win rate, EV = 0.40×$500 - 0.60×$500 = -$100 (negative!). Condor: $200 profit / $300 risk = 0.67:1, but 65% win rate, EV = 0.65×$200 - 0.35×$300 = $130 - $105 = $25 (positive). **Conclusion**: Iron Condors generally superior due to higher win probability.',
    keyPoints: [
      'Iron Butterfly: Sell ATM straddle + wings; $5 credit, $5 max loss, 40% prob profit (tight $95-$105 range)',
      'Iron Condor: Sell OTM strangle + wings; $2 credit, $3 max loss, 65% prob profit (wider $93-$107 range)',
      'Risk/reward: Butterfly 1:1 with 40% win rate (EV = -$100), Condor 0.67:1 with 65% win rate (EV = +$25)',
      'Iron Condor superior: Wider profit zone and higher probability outweigh lower credit',
      'Use butterfly only when: Stock pinned (post-earnings, specific catalyst), very high confidence in narrow range',
    ],
  },
  {
    id: 'spreads-strategies-q-3',
    question:
      'Design a "calendar spread trading strategy" that profits from time decay and volatility changes. The system should: (1) Select optimal underlyings (low volatility expected), (2) Determine strike selection (ATM vs OTM), (3) Calculate profit zones and ideal holding period, (4) Implement vega exposure management (IV changes), (5) Define exit rules and adjustment triggers, (6) Compare single calendar vs double calendar spreads. Provide detailed implementation.',
    sampleAnswer:
      'Calendar Spread System: **Setup**: Sell near-term 30-day option, Buy back-month 90-day option, Same strike (usually ATM for max theta). **Underlying Selection**: Low expected movement: Utilities, consumer staples, dividend aristocrats (KO, PG, JNJ), IV Rank 30-60% (moderate, not extremes), Liquid options: OI > 500, tight bid-ask. **Strike Selection**: ATM Calendar: Strike = current stock price, Max theta capture, Best if stock stays flat, Higher cost ($3-4 per share). OTM Calendar (bullish lean): Strike 2-3% above stock, Cheaper ($2-3), Profit if stock drifts up slowly. **Profit Zones**: Maximum profit at front-month expiration if stock = strike, Losses if stock moves significantly away from strike, Example: Stock $100, 100-strike calendar, cost $3: Profit zone: $97-$103 at front expiration, Max profit: ~$1.50 (50% return) at $100, Losses: if stock < $95 or > $105. **Vega Exposure**: Calendar = positive vega (benefit from IV expansion), Risk: If IV crashes, spread loses value even if stock at strike, Hedge: Mix with negative vega trades (short iron condors). **Exit Rules**: Close at front-month expiration (or 3-5 days before), Exit early if: stock moves > 5% away from strike (take loss), IV collapses > 20% (vega loss), 50% of max profit achieved. **Double Calendar**: Instead of 1 strike, use 2 strikes (e.g., 95 and 105), Creates wider profit zone, Lower max profit per spread but higher prob profit.',
    keyPoints: [
      'Calendar spread: Sell 30-day, buy 90-day, same ATM strike; profit from front-month theta decay',
      'Strike selection: ATM = max theta ($3-4 cost), 2-3% OTM = cheaper ($2-3), directional bias',
      'Profit zone: $97-$103 for $100 strike; max profit ~50% return at strike at front expiration',
      'Vega risk: Calendar = long vega; loses value if IV crashes even with stock at strike',
      'Exit rules: Close at front expiration, exit early if stock > 5% away or IV drops > 20%',
    ],
  },
];


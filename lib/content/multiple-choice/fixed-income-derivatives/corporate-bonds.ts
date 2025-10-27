import { MultipleChoiceQuestion } from '@/lib/types';

export const corporateBondsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'cb-mc-1',
    question:
      'A callable bond has a straight bond value of $1,050 and an embedded call option value of $30. What is the callable bond value?',
    options: ['$1,020', '$1,050', '$1,080', '$30'],
    correctAnswer: 0,
    explanation:
      "Callable Bond Value = Straight Bond Value - Call Option Value. Calculation: $1,050 - $30 = $1,020. Logic: The call option belongs to the ISSUER (company can call bond early). This is a DISadvantage to investors (upside capped). Therefore, option value SUBTRACTS from bond value. Why subtract? When rates fall, straight bond would rise to $1,100+, but callable capped near call price ($1,020 typical). Investor loses potential gains → worth less. Call option value $30 represents this lost upside. Real-world example: 10-year bond, 6% coupon, callable at 102 after 5 years. Straight bond trading at 105 (rates fell to 5%). Callable only trading at 102 (issuer will call soon). Difference $30 = call option value (issuer's benefit, investor's loss). Why not $1,080? That would be ADDING option value. Wrong because call option benefits issuer, not investor. Putable bonds (investor has put option) DO add: Putable = Straight + Put. Why not $1,050? That's straight bond value ignoring the call feature. Market price for callable must be lower. Important: Callable bonds offer HIGHER coupons to compensate (typically +20-50bp vs non-callable). This compensates investors for giving up call option.",
  },
  {
    id: 'cb-mc-2',
    question:
      'A convertible bond has face value $1,000, conversion ratio 25, and stock currently trades at $35. What is the conversion value?',
    options: ['$875', '$1,000', '$40', '$25'],
    correctAnswer: 0,
    explanation:
      "Conversion Value = Conversion Ratio × Current Stock Price. Calculation: 25 shares × $35/share = $875. Interpretation: If you convert today, you receive 25 shares worth $35 each = $875 total. This is the equity value of the convertible bond. Conversion Price (strike): $1,000 / 25 = $40 per share. Current stock $35 < Conversion price $40 = OUT of the money. Converting now would be foolish (lose $125: give up $1,000 bond for $875 stock). Bond should trade ABOVE $875 (has bond floor + time value). Analysis: Bond Floor: As straight debt, might be worth $950 (corporate bond). Conversion Value: $875 (equity value if converted). Bond should trade at: max($950, $875) + Time Value = $950+ (above both). Conversion Premium: (Bond_price - Conversion_value) / Conversion_value. If bond trades at $1,000: Premium = ($1,000 - $875) / $875 = 14.3%. Why this premium? Time value of conversion option (stock could rise above $40). Scenario analysis: Stock rises to $50: Conversion value = 25 × $50 = $1,250. Bond should trade ≈ $1,250+ (now in-the-money). Stock falls to $25: Conversion value = $625. Bond should trade ≈ $950 (bond floor protects downside). Why not $1,000? That's face value, not conversion value. Not relevant unless stock at exactly $40. Why not $40? That's conversion PRICE (strike), not conversion VALUE. Real-world: Convertibles act like bonds when stock low (downside protection), act like stock when stock high (participate in upside). Sweet spot for investors: Get bond safety + equity upside.",
  },
  {
    id: 'cb-mc-3',
    question:
      'Why do callable bonds typically have negative convexity when interest rates fall below the call price?',
    options: [
      'Price appreciation is capped near the call price (issuer will call)',
      'Duration increases as rates fall',
      'Credit risk increases',
      'Liquidity decreases',
    ],
    correctAnswer: 0,
    explanation:
      'Negative convexity: Price gains are limited (asymmetric), losses are not. Mechanism: Rates fall significantly → straight bond would rally to $1,100+. But callable bond capped near call price (e.g., $1,020). Issuer will call bond and refinance at lower rates. Rational issuer calls when bond price > call price + costs. Price-yield curve: Rates high (6%+): Bond trades well below call price, behaves like straight bond, positive convexity. Rates fall to 5%: Bond approaches call price ($1,020), curve starts flattening. Rates fall to 4%: Bond stuck at ~$1,020 (maybe $1,025 with call hesitation), straight bond would be $1,100, negative convexity region. Mathematical: Convexity = ∂²P/∂y². For straight bond: Always positive (curved upward). For callable near call price: Negative (curved downward / flat top). Impact on returns: Rates fall 1%: Straight bond +10%, callable bond +2% (capped). Rates rise 1%: Straight bond -10%, callable bond -10% (same downside). Asymmetric: Limited upside, full downside = BAD. Why investors accept this: Higher coupon (compensation for negative convexity). Callable bond: 6.0% coupon, straight bond: 5.5% coupon. 50bp extra yield offsets call risk (sometimes). Effective duration: Also shortens in call region (price less sensitive to rates). Duration = 7 when rates high, drops to 2-3 when rates low (near call). Portfolio management: Avoid callable bonds when expecting rates to fall (miss rally). Prefer when rates stable or rising (collect higher coupon, call unlikely). Important: Negative convexity is the main risk of callable bonds. Must be compensated with higher yield to be attractive.',
  },
  {
    id: 'cb-mc-4',
    question:
      'A floating rate note resets quarterly based on 3-month SOFR + 150bp. If SOFR is currently 4.5%, what is the approximate duration?',
    options: [
      '0.25 years (time to next reset)',
      '10 years (maturity)',
      '5 years (half of maturity)',
      '0 years (no duration)',
    ],
    correctAnswer: 0,
    explanation:
      "FRN duration ≈ time to next reset (for RATE duration). Calculation: Quarterly reset = 3 months = 0.25 years. Modified duration ≈ 0.25 years. Why so low? FRN coupon adjusts with rates, protecting investors. Rates rise from 4.5% to 5.5%: Fixed bond: Price falls significantly (duration × yield change). FRN: At next reset (3 months), coupon increases from 6.0% (4.5%+1.5%) to 7.0% (5.5%+1.5%). Higher future coupons offset rate rise. Price stable near par. Only time to next reset matters. Mechanics: Day 0: SOFR = 4.5%, coupon = 6.0% for next quarter. Day 89: SOFR jumps to 5.5%, but current coupon still 6.0% (locked in). Day 90 (reset): New coupon = 7.0%, bond reprices to ~par. Risk: 1 day before reset: Exposed to 90 days of old rate, duration ≈ 0.25. 1 day after reset: Exposed to new 90-day period, duration ≈ 0.25. SPREAD duration vs RATE duration: RATE duration ≈ 0.25 (low, hedge interest rate risk). SPREAD duration ≈ maturity-dependent (5-7 years typical). Exposed to credit spread changes (quoted margin fixed). If credit deteriorates, spread widens, price falls. Why not 10 years? That would be if coupon were FIXED (like standard bond). FRN coupon floats → not exposed to long-term rate changes. Why not 0? Would be true if continuous reset (instantaneous adjustment). With quarterly lag, small duration = time_to_reset. Important: FRNs protect from interest rate risk but NOT credit risk. If company's credit worsens, discount margin increases, price falls. Use cases: Rising rate environment: Investors prefer FRNs (coupon increases with rates). Alternative to cash: Earn spread over reference rate with minimal rate risk. Why not 5 years? There's no averaging or halfway point. Duration is strictly time to next reset for FRNs.",
  },
  {
    id: 'cb-mc-5',
    question:
      'An investor purchases a convertible bond for $1,000 and immediately shorts 20 shares of the underlying stock at $40/share. This strategy is known as:',
    options: [
      'Convertible arbitrage (delta-neutral hedge)',
      'Credit default swap',
      'Covered call strategy',
      'Naked short selling',
    ],
    correctAnswer: 0,
    explanation:
      "Convertible arbitrage: Classic hedge fund strategy combining long convertible + short stock. Setup: Long convertible bond: $1,000 position. Short stock: 20 shares × $40 = $800 short. Net delta: If convertible delta ≈ 0.80 (80% equity-like), delta = 0.80 × 25 shares (conversion ratio) = 20 shares. Shorting 20 shares makes position delta-neutral. Strategy objectives: Delta-neutral: Insensitive to small stock price moves. Long volatility (positive gamma): Benefit from large moves either direction. Credit exposure: Long credit (benefit if spreads tighten). Theta management: Time decay can be positive or negative. Profit sources: Volatility: Stock moves $10 up or down, adjustments capture profit. Stock up $10 to $50: Convertible gains $250 (delta 0.80 × $10 × 25 shares × 0.80). Short stock loses $200 (20 shares × $10). Net: +$50 (gamma profit). Rebalance: Cover some shorts (delta increased). Stock down $10 to $30: Convertible loses $200, short stock gains $200, roughly flat (bond floor provides support). Credit spread tightening: If company credit improves, bond floor rises, profit. Rebalancing: Must adjust hedge continuously (dynamic hedging). Stock rises → delta increases → short more stock. Stock falls → delta decreases → cover shorts. Frequency: Daily or intra-day for large funds. Transaction costs: Key consideration (impacts profitability). Real-world: Hedge funds specialize in convert arb. Typical: Long 100-200 different convertibles, each hedged. Target: 8-12% annual returns with low volatility (Sharpe > 1.5). Risks: Credit deterioration: If company defaults, bond floor collapses, large loss. Volatility decrease: If realized vol < implied vol paid, lose money. Forced conversion: Issuer calls bond, forced to unwind at suboptimal time. Gamma flipping: Deep in-the-money converts act like stock (lose optionality). Why not CDS? That's credit insurance, different instrument. Would buy CDS protection if bearish on credit. Why not covered call? That's selling calls on stock you own. Convertible arb is long bond + short stock (different payoff). Why not naked short? Naked = short stock with no hedge. Convert arb is hedged (long bond offsets short stock). Important: Convertible arbitrage is market-neutral strategy, seeks to profit from relative value, not directional moves.",
  },
];

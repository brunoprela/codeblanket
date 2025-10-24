import { MultipleChoiceQuestion } from '@/lib/types';

export const derivativesPricingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'dp-mc-1',
    question:
      'Stock=$100, r=5%, q=2%, T=1 year. Fair forward price F_0 = S×e^((r-q)T). If forward trades at $106, what is the arbitrage?',
    options: [
      'Buy forward at $106, short stock at $100, invest $100 at 5%',
      'Short forward at $106, buy stock at $100, borrow $100 at 5%',
      'Buy forward at $106, buy stock at $100, no arbitrage opportunity',
      'No arbitrage; $106 is within bid-ask spread of fair value',
    ],
    correctAnswer: 1,
    explanation:
      'Fair forward: F_0 = $100×e^(0.05-0.02) = $100×e^0.03 = $103.05. Market forward $106 is OVERPRICED by $2.95. Arbitrage (exploit expensive forward): Short forward at $106 (agree to sell at $106 in 1 year), Buy stock at $100 (borrow $100 at 5%), Hold 1 year, collect dividends. At maturity: Deliver stock via forward (receive $106), Received $2 dividends, Repay loan $105.13. Net profit: $106 + $2 - $105.13 = $2.87 per share risk-free. This is pure arbitrage: zero initial capital, zero risk, positive profit.',
  },
  {
    id: 'dp-mc-2',
    question:
      'Interest rate swap: 3-year, notional $10M, fixed rate 5%, current SOFR 4.5%. What is value to fixed-rate payer?',
    options: [
      'Zero (swaps always have zero initial value)',
      'Positive (paying below-market fixed rate)',
      'Negative (paying above-market fixed rate)',
      'Depends on credit spreads (not enough info)',
    ],
    correctAnswer: 2,
    explanation:
      'Value to fixed payer = V_floating - V_fixed. V_floating = $10M (floating-rate bond always worth par at reset). V_fixed = PV of 5% fixed payments discounted at 4.5% SOFR. Since paying 5% fixed when market is 4.5% (par rate), paying ABOVE market → negative value. V_fixed > $10M → V_swap = $10M - V_fixed < 0. Fixed payer LOSES money (would need to pay counterparty to exit). If rates rise to 6%: Then paying 5% becomes favorable → swap value becomes positive. Par swap rate (zero initial value) = current SOFR = 4.5%.',
  },
  {
    id: 'dp-mc-3',
    question:
      'Crude oil futures are in contango (F > S). 1-month future=$82, spot=$80. What does this indicate?',
    options: [
      'Market expects oil price to rise $2 next month',
      'Storage cost + interest exceeds convenience yield',
      'Oil shortage; immediate demand exceeds future demand',
      'Arbitrage opportunity: buy spot, sell future, profit $2',
    ],
    correctAnswer: 1,
    explanation:
      'Contango (F > S) means future price above spot. General formula: F = S×e^((r+c-q-y)T). Where r=interest, c=storage, q=income yield, y=convenience yield. Contango when: Storage cost + interest > convenience yield. r+c > q+y. Example: S=$80, r=5%, c=3%, y=2% (convenience), T=1 month. F = $80×e^((0.05+0.03-0.02)/12) = $80.40. $82 future vs $80 spot = $2 contango reflects carry costs. This is NOT price expectation (risk-neutral pricing). Storage arbitrage: Buy spot $80, store (pay $3), sell future $82. Profit = $82 - $80 - $3 = $-1 loss! No arbitrage (contango reflects storage).',
  },
  {
    id: 'dp-mc-4',
    question:
      'Up-and-out call (S=$100, K=$100, barrier=$120) typically costs 30% less than vanilla call. Why?',
    options: [
      'Barrier options are less liquid (liquidity discount)',
      'Option disappears if stock hits $120 (gives up upside gains)',
      'Black-Scholes systematically misprices barrier options',
      'Market makers charge lower premiums for exotic options',
    ],
    correctAnswer: 1,
    explanation:
      'Up-and-out call knocks out (becomes worthless) if stock ever touches barrier $120. Loses value two ways: (1) Expires below strike K=$100 (like vanilla), (2) Ever hits barrier during life (unique to barrier). Gives up gains beyond $120. If stock goes $100→$125: Vanilla call payoff: $25, Up-and-out call payoff: $0 (knocked out at $120). Cheaper because: Less optionality (conditional payoff), Higher knockout probability → lower expected value. Typically 20-40% discount vs vanilla. Use case: Bull investor expects $100→$115 (not $120+), saves premium by selling upside beyond $120.',
  },
  {
    id: 'dp-mc-5',
    question:
      'Asian call option (payoff based on average price over 3 months) vs vanilla call. Which has higher value?',
    options: [
      'Asian call (additional averaging benefit)',
      'Vanilla call (more optionality, higher volatility exposure)',
      'Equal value (same underlying, same maturity)',
      'Depends on current stock price relative to strike',
    ],
    correctAnswer: 1,
    explanation:
      'Vanilla call > Asian call. Reason: Averaging reduces volatility. S_avg has lower volatility than S_T (averages smooth out fluctuations). Lower volatility → lower option value (less chance of extreme profitable moves). Example: Stock σ=30% daily. S_T (terminal price) volatility: 30%. S_avg (3-month average) volatility: ~17% (σ_avg ≈ σ/√n for n observations). Vanilla call (benefits from S_T=150 spike on last day): Payoff = $50. Asian call (S_avg=120 even if S_T=150 spike): Payoff = $20. Asian options: Cheaper (lower vol), Reduce manipulation (avg price harder to manipulate), Used for hedging steady exposures (FX, commodity averaging over period).',
  },
];

import { MultipleChoiceQuestion } from '@/lib/types';

export const optionsPricingMentalMathMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'opmm-mc-1',
      question:
        'Stock is $100, ATM 1-year call with 25% vol is worth approximately:',
      options: ['$5', '$10', '$15', '$25'],
      correctAnswer: 1,
      explanation:
        'Using ATM approximation: C ≈ 0.4 × S × σ × √T = 0.4 × 100 × 0.25 × 1 = $10. This is the mental math shortcut for ATM options. The exact Black-Scholes value would be very close (~$9.98 with zero interest rate).',
    },
    {
      id: 'opmm-mc-2',
      question:
        'You observe: Stock at $50, 3-month call (K=$50) at $3, put (K=$50) at $4. What is the arbitrage opportunity?',
      options: [
        'Buy call, sell put, short stock - profit $1',
        'Sell call, buy put, buy stock - profit $1',
        'Buy call, buy put, short stock - profit $1',
        'No arbitrage exists',
      ],
      correctAnswer: 0,
      explanation:
        'Put-call parity: C - P should equal S - K. Here: $3 - $4 = -$1, but S - K = $50 - $50 = $0. The put is overpriced by $1. Arbitrage: Buy call (+position), Sell put (+cash), Short stock (+cash). Net: Buy call for $3, sell put for $4, short stock for $50. Net cash: -$3 + $4 + $50 = $51. At expiration, position is worth $50 (strike). Profit: $51 - $50 = $1.',
    },
    {
      id: 'opmm-mc-3',
      question:
        'For an ATM option with S=$200, σ=20%, T=3 months, what is the approximate daily theta?',
      options: [
        '-$0.05 per day',
        '-$0.11 per day',
        '-$0.22 per day',
        '-$0.44 per day',
      ],
      correctAnswer: 2,
      explanation:
        'Annual theta: Θ ≈ -0.5 × S × σ / √T = -0.5 × 200 × 0.20 / √0.25 = -20 / 0.5 = -$40 per year. Daily: -40 / 365 ≈ -$0.11 per day. Wait, let me recalculate more carefully: Θ (per year) = -0.5 × 200 × 0.20 / 0.5 = -40. Daily = -40/365 = -0.1096. Actually, a better estimate: for 3 months, the option is worth approximately C ≈ 0.4 × 200 × 0.20 × 0.5 = $8. Over 90 days, it decays to $0, so roughly $8/90 ≈ $0.089 per day for ATM, but accelerates near expiry. At the midpoint (1.5 months out), daily decay ≈ $0.11-0.15. The closest answer is -$0.11.',
    },
    {
      id: 'opmm-mc-4',
      question: 'A 10% OTM call (S=$100, K=$110) has approximately what delta?',
      options: ['0.25', '0.35', '0.50', '0.65'],
      correctAnswer: 1,
      explanation:
        'For 10% OTM, delta ≈ 0.35. Mental math rule: ATM delta = 0.5, 10% OTM ≈ 0.35, 20% OTM ≈ 0.20. The exact value depends on volatility and time, but 0.35 is a good approximation for typical parameters (20-30% vol, 3-12 months to expiry).',
    },
    {
      id: 'opmm-mc-5',
      question:
        'If an ATM call price is $5 with S=$50, T=3 months, what is the approximate implied volatility?',
      options: ['20%', '30%', '40%', '50%'],
      correctAnswer: 3,
      explanation:
        "Using ATM inversion: σ = C / (0.4 × S × √T). Here: σ = 5 / (0.4 × 50 × √0.25) = 5 / (0.4 × 50 × 0.5) = 5 / 10 = 0.5 = 50%. Verification: at 50% vol, C ≈ 0.4 × 50 × 0.5 × 0.5 = 5 ✓. At 40% vol, C ≈ 0.4 × 50 × 0.4 × 0.5 = 4. So to get $5, we need 50% implied volatility. While 50% is high for most stocks, it's reasonable for volatile stocks or during periods of high uncertainty (e.g., biotech stocks, earnings announcements, market crises).",
    },
  ];

import { MultipleChoiceQuestion } from '@/lib/types';

export const fixedIncomeMarketsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fm-1-2-mc-1',
    question:
      'A 10-year Treasury bond with a 3% coupon is trading at $100 (par). If the yield rises from 3% to 4%, and the bond has a duration of 8.5 years, what is the approximate new price?',
    options: ['$91.50', '$95.00', '$98.00', '$100.00'],
    correctAnswer: 0,
    explanation:
      'Price change ≈ -Duration × ΔYield × Price = -8.5 × 0.01 × $100 = -$8.50. New price ≈ $100 - $8.50 = $91.50. This is an approximation; exact calculation using present value gives $91.89. Duration provides quick estimate of price sensitivity to yield changes.',
  },
  {
    id: 'fm-1-2-mc-2',
    question:
      'The yield curve shows: 2-year at 4.0%, 5-year at 3.5%, 10-year at 3.0%. This curve shape is called:',
    options: [
      'Normal (upward sloping)',
      'Inverted (downward sloping)',
      'Flat',
      'Humped',
    ],
    correctAnswer: 1,
    explanation:
      'Inverted yield curve: short-term rates > long-term rates. This signals market expects Fed to cut rates (recession coming). Historically, inversions precede recessions with 12-18 month lag. Normal curve: long rates > short rates (typical). Flat: all rates similar. Humped: medium-term rates highest.',
  },
  {
    id: 'fm-1-2-mc-3',
    question:
      'A corporate bond trades at $950 with a 5% coupon and 10-year maturity. The 10-year Treasury yields 3%. What is the credit spread?',
    options: ['1.5%', '2.0%', '2.5%', '3.0%'],
    correctAnswer: 1,
    explanation:
      'First, calculate bond YTM: discount rate that makes PV of cash flows = $950. Using financial calculator: PV=-950, FV=1000, PMT=50, N=10 → YTM = 5.59%. Credit spread = Corporate YTM - Treasury = 5.59% - 3.0% = 2.59% ≈ 2.0%. Credit spread compensates for default risk.',
  },
  {
    id: 'fm-1-2-mc-4',
    question:
      'A bond portfolio manager has $100M in 10-year bonds (duration = 8 years) and wants to hedge against rising rates using Treasury futures (duration = 10 years). How much futures notional should they short?',
    options: ['$50M', '$80M', '$100M', '$125M'],
    correctAnswer: 1,
    explanation:
      'Duration hedge: Portfolio dollar duration = $100M × 8 = $800M. Futures dollar duration per $1M = 10. Hedge ratio = $800M / 10 = $80M notional. Short $80M of futures to offset rate risk. This makes portfolio duration-neutral to parallel yield curve shifts.',
  },
  {
    id: 'fm-1-2-mc-5',
    question:
      'During a credit crisis, corporate bond spreads widen from 200 bps to 400 bps. A bond with 5-year maturity and $1000 face value approximately loses how much value?',
    options: ['$50', '$100', '$150', '$200'],
    correctAnswer: 1,
    explanation:
      'Spread widening = 200 bps = 2%. Approximate loss = Duration × ΔSpread × Price. For 5-year bond, duration ≈ 4.5 years. Loss = 4.5 × 0.02 × $1000 = $90 ≈ $100. Actual loss depends on exact duration and convexity, but order of magnitude is ~10% for 200 bps widening.',
  },
];

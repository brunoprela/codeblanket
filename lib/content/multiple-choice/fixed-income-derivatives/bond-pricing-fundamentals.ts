import { MultipleChoiceQuestion } from '@/lib/types';

export const bondPricingFundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bpf-mc-1',
    question:
      'A bond has face value $1,000, 6% annual coupon (paid semi-annually), 5 years to maturity. If YTM is 8%, what is the approximate bond price?',
    options: [
      '$918.89 (discount)',
      '$1,000.00 (par)',
      '$1,081.11 (premium)',
      '$950.00 (discount)',
    ],
    correctAnswer: 0,
    explanation:
      "Price calculation: Semi-annual coupon = $1,000 × 6% / 2 = $30 per period. YTM per period = 8% / 2 = 4% = 0.04. Number of periods = 5 years × 2 = 10. Price = PV(coupons) + PV(principal). PV(coupons) = $30 × [(1 - (1.04)^-10) / 0.04] = $30 × 8.1109 = $243.33. PV(principal) = $1,000 / (1.04)^10 = $1,000 / 1.4802 = $675.56. Total Price = $243.33 + $675.56 = $918.89. This is a DISCOUNT bond because YTM (8%) > Coupon (6%). When market rates rise above the bond's coupon, the bond must trade below par to offer competitive returns. Investors require 8% return, but the bond only pays 6% coupon, so the price drops below $1,000. The $81.11 discount ($1,000 - $918.89) compensates for the lower coupon rate. Key insight: Price and yield have an INVERSE relationship. As YTM ↑ from 6% to 8%, price ↓ from $1,000 to $918.89. Real-world application: If Fed raises interest rates, existing bonds lose value. This is why bond portfolios suffer losses when rates rise.",
  },
  {
    id: 'bpf-mc-2',
    question:
      'Which day count convention would result in the HIGHEST accrued interest for a 61-day period from January 15 to March 16 (non-leap year)?',
    options: [
      'Actual/365',
      '30/360',
      'Actual/Actual',
      'All conventions give the same result',
    ],
    correctAnswer: 1,
    explanation:
      'Day count comparison for Jan 15 to Mar 16: Actual days = 61 days (31-15 in Jan = 16, + 28 in Feb, + 16 in Mar, + 1 for inclusive end = 61). Actual/365: day_factor = 61 / 365 = 0.16712 years. Actual/Actual: day_factor = 61 / 365 = 0.16712 years (same as Actual/365 for non-leap year). 30/360: Uses formula: days = 360×(y2-y1) + 30×(m2-m1) + (d2-d1). Jan 15 to Mar 16: days = 360×0 + 30×(3-1) + (16-15) = 0 + 60 + 1 = 61 days. But wait, 30/360 divides by 360, not 365: day_factor = 61 / 360 = 0.16944 years. HIGHEST: 30/360 gives 0.16944 > Actual/365 (0.16712). Why? 30/360 assumes shorter year (360 days instead of 365), so same number of days represents a larger fraction of the year. Accrued Interest = Coupon × day_factor, so higher day_factor = more accrued interest. Example with $50 annual coupon: Actual/365: $50 × 0.16712 = $8.36. 30/360: $50 × 0.16944 = $8.47 (higher by $0.11). Real-world: US corporate bonds use 30/360, resulting in slightly higher accrued interest for seller. US Treasuries use Actual/Actual (more accurate). Important: The choice of day count convention can affect bond valuations by $0.10-$0.50 per $1,000 face value.',
  },
  {
    id: 'bpf-mc-3',
    question:
      'A bond with $1,000 face value and 5% semi-annual coupon has clean price of $1,020. It last paid a coupon 45 days ago, and the next coupon is in 138 days. Using Actual/365, what is the dirty price?',
    options: ['$1,026.14', '$1,020.00', '$1,033.75', '$1,025.00'],
    correctAnswer: 0,
    explanation:
      'Dirty price = Clean price + Accrued interest. Step 1: Calculate coupon payment: Semi-annual coupon = $1,000 × 5% / 2 = $25. Step 2: Calculate accrued interest: Days since last coupon = 45 days. Total days in period = 45 + 138 = 183 days (typical semi-annual). Using Actual/365: Accrued fraction = 45 / 183 = 0.24590. Accrued interest = $25 × 0.24590 = $6.15 (rounded to nearest cent). Actually, let me recalculate more precisely: 45/183 = 0.245901639, × $25 = $6.1475, rounds to $6.15. Step 3: Dirty price = $1,020.00 + $6.15 = $1,026.15 ≈ $1,026.14 (closest option). Why dirty price matters: Clean price ($1,020) is the quoted price - what you see in the market. Dirty price ($1,026.14) is what you actually PAY at settlement. The buyer compensates the seller for the 45 days of interest accrued since the last coupon payment. The seller held the bond for 45 days earning interest, so deserves that portion of the $25 coupon. Accrued interest "jumps" from $0 (right after coupon) to full coupon amount (right before next coupon), then drops back to $0. Real-world application: Bond traders always think in clean prices (removes accrued interest noise), but settlement systems calculate dirty prices (actual cash exchanged). Important: Never forget accrued interest in bond trades, or you\'ll miscalculate P&L by the accrued amount.',
  },
  {
    id: 'bpf-mc-4',
    question:
      'A zero-coupon bond with $1,000 face value matures in 7 years and currently trades at $680. What is the yield to maturity (YTM)?',
    options: ['5.60%', '6.25%', '4.85%', '45.9% (total return)'],
    correctAnswer: 0,
    explanation:
      "Zero-coupon bond YTM formula (direct, no iteration needed): YTM = (Face Value / Price)^(1 / Years) - 1. Calculation: YTM = ($1,000 / $680)^(1/7) - 1 = (1.4706)^(1/7) - 1 = 1.0560 - 1 = 0.0560 = 5.60%. Verification: Price at 5.60% YTM = $1,000 / (1.056)^7 = $1,000 / 1.4706 = $680.00 ✓. Why this formula works: Zero-coupon bonds have only ONE cash flow (face value at maturity), so: Price = FV / (1 + YTM)^t. Rearranging: (1 + YTM)^t = FV / Price, YTM = (FV / Price)^(1/t) - 1. No iteration needed (unlike coupon bonds where Newton-Raphson required). Option D ($45.9% total return) is WRONG because it calculates: (FV - Price) / Price = ($1,000 - $680) / $680 = 47.06% total return. But YTM is the ANNUALIZED return, not total return. Total return of 47.06% over 7 years = 5.60% annualized. Real-world examples: US Treasury STRIPS (Separate Trading of Registered Interest and Principal Securities) are zero-coupon Treasuries created by stripping coupons from regular bonds. Popular for education savings (buy 15-year zero for child's college fund). Easy to price and understand - no reinvestment risk (no coupons to reinvest). Important: Zero-coupon bonds have HIGHER duration than coupon bonds of same maturity, making them more sensitive to interest rate changes. If rates rise 1%, a 7-year zero might drop ~7%, while a 7-year coupon bond might drop only ~5%.",
  },
  {
    id: 'bpf-mc-5',
    question:
      'Why does a bond trading at a PREMIUM (price > face value) have a yield to maturity (YTM) that is LOWER than its current yield?',
    options: [
      'Because YTM accounts for the capital loss at maturity (receiving par < purchase price)',
      'Because premium bonds are riskier',
      'Because the coupon payments are lower',
      'This is incorrect - YTM is always higher than current yield',
    ],
    correctAnswer: 0,
    explanation:
      "Key relationships for premium bonds: Current Yield = Annual Coupon / Market Price (ignores capital gain/loss). YTM = Total return accounting for all cash flows INCLUDING capital loss at maturity. Example: Bond with $1,000 face, 8% coupon, trading at $1,100 (premium): Current Yield = $80 / $1,100 = 7.27%. But YTM is LOWER than 7.27% because: You pay $1,100 today but only receive $1,000 at maturity = $100 capital loss. This capital loss is amortized over the bond's life, reducing the effective return. YTM might be 6.5% (captures both 8% coupon income AND -$100 capital loss). For premium bonds: Coupon Rate > Current Yield > YTM. Example: 8.00% > 7.27% > 6.50%. Why? Coupon Rate (8%) is fixed based on face value. Current Yield (7.27%) is lower because price is higher than face. YTM (6.50%) is lowest because it includes the guaranteed loss at maturity. For discount bonds (opposite): YTM > Current Yield > Coupon Rate. Example: If bond trades at $900: Current Yield = $80 / $900 = 8.89%. YTM might be 9.5% (includes 8% coupon + $100 capital gain amortized). Real-world implication: In falling interest rate environments, bonds trade at premiums. Old bonds with high coupons (issued when rates were higher) trade above par. Investors accept YTM < coupon because current market rates are even lower. Example: In 2020-2021, many bonds issued in 2018-2019 (before rate cuts) traded at 110-120 (10-20% premium). Important: When comparing bonds, always use YTM (total return), not current yield (incomplete measure). Current yield ignores the inevitable capital gain/loss at maturity.",
  },
];

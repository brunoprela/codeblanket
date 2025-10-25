export const fixedIncomeMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'A 10-year Treasury bond with a 4% coupon is currently priced at $950 (below par value of $1,000). What can you conclude about the current market yield?',
    options: [
      'The market yield is less than 4% (yields have fallen since issuance)',
      'The market yield is exactly 4% (same as coupon)',
      'The market yield is greater than 4% (yields have risen since issuance)',
      "Cannot determine without knowing the bond's duration",
      'The bond must be in danger of default',
    ],
    correctAnswer: 2,
    explanation:
      "When a bond trades at a **discount** (below par), it means the market yield is **higher** than the coupon rate. Think about it: If you can buy NEW bonds yielding 5%, why would you pay full price for an OLD bond paying only 4%? You'd pay less (discount) so your total return (coupon + price appreciation to par) equals 5%. **Key rule**: Price and yield move INVERSELY. When yields rise → prices fall (discount). When yields fall → prices rise (premium).",
    difficulty: 'intermediate',
  },
  {
    id: 2,
    question:
      'An investor in the 37% federal tax bracket is choosing between a taxable corporate bond yielding 6% and a municipal bond yielding 4%. Which is better from a tax perspective?',
    options: [
      'Corporate bond is clearly better (6% > 4%)',
      "Municipal bond is equivalent to 6.35% taxable yield, so it's better",
      'They provide the same after-tax return',
      'Corporate bond is better if held in IRA (tax-deferred)',
      'Cannot determine without knowing state taxes',
    ],
    correctAnswer: 1,
    explanation:
      "Municipal bond interest is federal tax-exempt. To compare fairly, calculate the **tax-equivalent yield (TEY)**: TEY = Muni Yield / (1 - Tax Rate) = 4% / (1 - 0.37) = 4% / 0.63 = **6.35%**. Since 6.35% > 6%, the muni bond is better. **Key insight**: Munis are attractive for high-income investors in high tax brackets. For someone in 22% bracket, TEY = 4% / 0.78 = 5.13%, making the 6% corporate bond better. This is why munis are called 'tax-advantaged' securities.",
    difficulty: 'intermediate',
  },
  {
    id: 3,
    question:
      'The yield curve inverts (short-term rates > long-term rates). What does this typically signal, and why does it happen?',
    options: [
      'It signals economic expansion because investors demand more return for longer investments',
      'It signals recession because investors expect the Fed to cut rates in the future (flight to safety in long bonds)',
      "It\'s meaningless - just statistical noise in bond markets",
      "It only happens when there's a credit crisis in corporate bonds",
      'It signals high inflation is expected long-term',
    ],
    correctAnswer: 1,
    explanation:
      'An **inverted yield curve** (2-year yield > 10-year yield) has predicted **every U.S. recession** in the past 50 years, typically 12-18 months in advance. **Why it happens**: Investors expect a recession → Fed will cut rates → future short-term rates will be lower → long-term bonds become attractive (locking in current rates) → demand for long bonds pushes their prices up and yields down. **Example**: In 2022-2023, the curve inverted (2Y at 5%, 10Y at 4%) as markets anticipated Fed rate cuts due to slowing economy. **Key**: The yield curve is one of the most reliable recession predictors.',
    difficulty: 'intermediate',
  },
  {
    id: 4,
    question:
      "A bond has a duration of 7 years. If interest rates rise by 1%, approximately how much will the bond's price change?",
    options: [
      'The bond will gain approximately 7% in value',
      'The bond will lose approximately 7% in value',
      "The bond's price will not change significantly (duration doesn't affect price)",
      'The bond will lose 7% of its coupon payments',
      "Cannot determine without knowing the bond's convexity",
    ],
    correctAnswer: 1,
    explanation:
      '**Duration** measures interest rate sensitivity: **Price Change ≈ -Duration × Yield Change**. If duration = 7 years and rates rise 1%, price change ≈ -7 × 1% = **-7%**. The negative sign shows the **inverse relationship** between yields and prices. **Intuition**: Longer duration = more sensitive to rate changes. A 2-year duration bond would only lose 2%, while a 15-year duration bond would lose 15%. **Why this matters**: In 2022 when Fed raised rates rapidly, long-duration bond funds (10+ years) lost 20-30% while short-duration funds lost only 2-5%. Duration is THE key risk metric for bond portfolios.',
    difficulty: 'beginner',
  },
  {
    id: 5,
    question:
      "A corporate bond is rated BBB by S&P. The company's financial situation deteriorates and the rating is downgraded to BB. What happens immediately?",
    options: [
      "Nothing changes - ratings are just opinions and don't affect markets",
      "The bond's yield will likely decrease as investors perceive it as safer",
      "The bond becomes 'junk' and many institutional investors must sell it by policy, causing the price to drop sharply",
      'The company must immediately repay the bondholders',
      "The bond's coupon rate increases to compensate for higher risk",
    ],
    correctAnswer: 2,
    explanation:
      'BBB- is the **lowest investment-grade rating**. Below that (BB and below) is **junk/high-yield**. This is a critical threshold because: 1) Many institutional investors (pension funds, insurance companies) are **prohibited by policy** from holding junk bonds, 2) They must SELL immediately ("forced selling"), 3) This creates sudden selling pressure → price drops sharply → yields spike. This is called a **"Fallen Angel"**. **Real example**: When Ford was downgraded to junk in 2020, billions of dollars of forced selling hit the market. **Key insight**: The BBB to BB downgrade is far more impactful than any other single-notch downgrade due to institutional restrictions.',
    difficulty: 'advanced',
  },
];

export const derivativesMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'A trader has a long position in oil futures contracts (1,000 barrels per contract) at $80/barrel. The current price is $85/barrel. With initial margin of $8,000 per contract and maintenance margin of $6,000, what happens if oil drops to $75/barrel the next day?',
    options: [
      'Nothing happens - futures are only settled at expiration',
      'The trader loses $5,000 but no action is required since account balance ($8,000 - $5,000 = $3,000) is positive',
      'MARGIN CALL - The trader must deposit $5,000 to bring account back to initial margin ($8,000)',
      'The position is automatically liquidated immediately when price hits $74',
      "The trader's maximum loss is the initial margin ($8,000), so they're protected",
    ],
    correctAnswer: 2,
    explanation:
      'With futures, **daily mark-to-market** means P&L is settled daily. Oil drops from $85 to $75 = -$10/barrel loss. With 1,000 barrel contract = -$10,000 loss. Account balance = $8,000 - $10,000 = **-$2,000** (NEGATIVE!). This is below maintenance margin ($6,000), triggering a **MARGIN CALL**. Trader must deposit enough to restore account to **initial margin** ($8,000), so needs to add $10,000. **Key difference from stocks**: Futures have DAILY settlement and can result in owing money beyond initial investment. This is why futures are so risky - losses can exceed your deposit.',
    difficulty: 'advanced',
  },
  {
    id: 2,
    question:
      'The main difference between a forward contract and a futures contract is:',
    options: [
      'Forwards are for commodities, futures are for financial instruments',
      'Forwards settle at maturity only, futures are marked-to-market daily on exchanges',
      'Forwards are always cash-settled, futures involve physical delivery',
      'Forwards have no counterparty risk, futures have high counterparty risk',
      'There is no meaningful difference - they are essentially the same',
    ],
    correctAnswer: 1,
    explanation:
      "The critical difference is **settlement**: Forwards are OTC (over-the-counter) bilateral agreements that settle ONLY at maturity, while futures are exchange-traded and **marked-to-market daily** with a clearinghouse. This means: (1) Futures have daily cash flows (winners receive money daily, losers pay daily), (2) Futures have much lower counterparty risk because the clearinghouse guarantees all trades, (3) Futures require daily margin maintenance, (4) Futures are standardized and liquid, forwards are customized. **Example**: If you're long a 3-month oil forward and oil rises, you don't get any money until maturity. With a 3-month oil future, you receive cash daily as prices rise. This daily settlement reduces counterparty risk but requires active margin management.",
    difficulty: 'intermediate',
  },
  {
    id: 3,
    question: "An option has 'limited downside, unlimited upside.' This means:",
    options: [
      'The option can never lose value, and gains are unlimited',
      'As a buyer, your maximum loss is the premium paid, but potential gains are theoretically infinite',
      'Options are always profitable investments',
      'Selling options is always safer than buying options',
      'Options have no risk if you hold to expiration',
    ],
    correctAnswer: 1,
    explanation:
      "For option **BUYERS**: Maximum loss = premium paid (if option expires worthless), but gains can be huge if the underlying moves favorably. **Example**: Buy a call for $5. Stock crashes → you lose $5 (max loss). Stock moons to $200 above strike → you make $195 profit (huge gain). **CRITICAL**: This asymmetry only applies to BUYERS. For option **SELLERS**, it's reversed: Limited upside (premium collected) but unlimited downside risk. This is why selling naked call options is extremely dangerous - if stock goes to infinity, your losses are infinite. **Why options exist**: This asymmetry makes them perfect for hedging (buying protection) and speculation (leveraged bets with known max loss).",
    difficulty: 'beginner',
  },
  {
    id: 4,
    question:
      'An interest rate swap where you pay fixed 4% and receive floating SOFR is MOST beneficial when:',
    options: [
      'Interest rates stay constant at 4% throughout the swap',
      'Interest rates fall below 4% (you pay more than you receive)',
      'Interest rates rise above 4% (you receive more than you pay)',
      'Interest rates are volatile but average 4% over the period',
      'The swap is always neutral regardless of rate movements',
    ],
    correctAnswer: 2,
    explanation:
      "You **pay fixed 4%** and **receive floating** (SOFR). You profit when the floating rate > 4%. **Scenario**: SOFR rises to 6%. You pay 4% (fixed) but receive 6% (floating) = net receive 2%. You profit! **Why use this**: If you have floating-rate debt (paying SOFR + spread), this swap converts it to fixed-rate debt. Or, speculatively, you're betting rates will rise. **Example**: You pay a bank 4% on $10M notional, they pay you SOFR on $10M. If SOFR = 3%, you pay net 1% (\$100K loss). If SOFR = 6%, you receive net 2% (\$200K gain). **Real-world use**: Companies with floating-rate debt use swaps to 'lock in' fixed rates when they expect rates to rise.",
    difficulty: 'intermediate',
  },
  {
    id: 5,
    question:
      'During the 2008 financial crisis, derivatives like CDOs (Collateralized Debt Obligations) and CDS (Credit Default Swaps) played a destructive role primarily because:',
    options: [
      "They were illegal instruments that regulators didn't know about",
      'They created massive hidden leverage and interconnected counterparty risk that amplified a housing crisis into a systemic crisis',
      'They caused the housing market to collapse in the first place',
      'Banks were forced by the government to trade them',
      'They were poorly designed products that lost money by mistake',
    ],
    correctAnswer: 1,
    explanation:
      "Derivatives **amplified** the crisis through: (1) **Leverage**: Banks held 30-40x leveraged positions in CDOs, turning small losses into catastrophic ones, (2) **Complexity**: No one understood the risk - AAA-rated CDOs contained junk mortgages, (3) **Interconnection**: AIG sold $500B+ in CDS 'insurance' without capital to pay. When AIG nearly failed, it threatened to bring down everyone who bought protection from them, (4) **Lack of transparency**: OTC markets hid systemic risk. **Result**: A $1T mortgage problem became a $10T+ global crisis. **Warren Buffett\'s warning**: 'Financial weapons of mass destruction' - he was right. **Key lesson**: Derivatives don't cause crises, but they can turn small problems into catastrophic ones through leverage and counterparty chains.",
    difficulty: 'advanced',
  },
];

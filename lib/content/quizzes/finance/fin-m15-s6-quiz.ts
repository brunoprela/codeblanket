export default {
  id: 'fin-m15-s6-quiz',
  title: 'Credit Risk Management - Quiz',
  questions: [
    {
      id: 1,
      question:
        'A corporate loan has: PD = 2%, LGD = 40%, EAD = $10M. What is the Expected Loss?',
      options: ['$80,000', '$200,000', '$400,000', '$800,000'],
      correctAnswer: 0,
      explanation:
        'Expected Loss = PD × LGD × EAD = 0.02 × 0.40 × $10M = $80,000. This is the average annual loss from this loan. Option B ($200K) would be if LGD = 100% (no recovery). Option C ($400K) would be if PD = 4%. Option D ($800K) would be if both PD = 4% and LGD = 80%. Expected Loss should be covered by: (1) Pricing—loan spread should include $80K annual charge, (2) Provisions—set aside reserves for expected losses. Capital is for Unexpected Loss (volatility around the expected). For example, if loan is priced at 300bp spread on $10M = $300K revenue, and expected loss is $80K, bank keeps $220K after expected loss. The $80K expected loss is a cost of doing business, not a surprise.',
    },
    {
      id: 2,
      question:
        'Why did the 2008 crisis cause massive losses for buyers of AAA-rated CDO tranches despite low historical default rates for AAA?',
      options: [
        'Rating agencies were bribed',
        'AAA ratings assumed diversification, but all mortgages defaulted together (correlation = 1)',
        'The tranches were actually BBB, mislabeled as AAA',
        'AAA ratings only apply to corporate bonds, not structured products',
      ],
      correctAnswer: 1,
      explanation:
        'The crisis exposed the flaw in CDO models: they assumed mortgage defaults were independent (low correlation). If 1% of mortgages default independently, a senior AAA tranche with 30% credit enhancement is virtually risk-free. But in 2008, mortgages defaulted together (correlation approached 1) because all were tied to housing prices. When housing fell 30% nationally, defaults spiked to 15-20% even in "diversified" pools, overwhelming the 30% credit enhancement. Option A is conspiratorial—ratings were wrong due to model failure, not bribes. Option C is factually incorrect—tranches were legally AAA. Option D is wrong—AAA applies to any debt. The lesson: Correlation risk in tail events. Models assumed 30% default correlation; reality was 80%+. This "correlation breakdown" is now central to credit risk modeling.',
    },
    {
      id: 3,
      question:
        'A bank has 99.9% Credit VaR of $500M. What does this represent?',
      options: [
        'Expected credit losses over one year',
        'Maximum possible credit loss',
        'Credit loss exceeded 0.1% of the time (1 year in 1000)',
        'Required credit loss provisions',
      ],
      correctAnswer: 2,
      explanation:
        "Credit VaR at 99.9% confidence (Basel standard) means there is a 0.1% probability (1 year in 1000) of credit losses exceeding $500M. This is Unexpected Loss—volatility around expected loss. Option A (expected losses) is wrong—that's Expected Loss (EL), not VaR. Option B (maximum) is wrong—losses could exceed $500M in extreme tail. Option D (provisions) is wrong—provisions cover expected losses, not unexpected. The 99.9% threshold corresponds to a AA credit rating for the bank (0.1% default probability). Banks must hold capital = Unexpected Loss = Credit VaR - Expected Loss. If EL = $100M and VaR = $500M, Unexpected Loss = $400M → need $400M capital. The 99.9% confidence was chosen to ensure banks can survive a 1-in-1000 year credit loss event without failing.",
    },
    {
      id: 4,
      question:
        'Post-Dodd-Frank, CDS markets shifted to central clearing. What is the main benefit?',
      options: [
        'Lower transaction costs',
        'Reduces bilateral counterparty risk—CCP becomes counterparty to all trades',
        'Eliminates the need for collateral',
        'Makes CDS prices more favorable to buyers',
      ],
      correctAnswer: 1,
      explanation:
        "Central clearing through CCPs (Central Counterparties) means the CCP becomes the counterparty to both sides of every trade, eliminating bilateral exposure. Pre-crisis, if you bought CDS protection from Lehman and Lehman failed, you lost protection exactly when you needed it most (wrong-way risk). With CCP clearing, you have exposure to the CCP (which is well-capitalized and mutualized), not to individual dealer. Option A is wrong—clearing actually increases costs (initial margin, clearing fees). Option C is backwards—clearing REQUIRES more collateral (initial margin + variation margin). Option D is wrong—clearing doesn't affect pricing favorably. The 2008 lesson: CDS web with no central clearing created systemic risk. If one major dealer failed, cascade of exposures. CCPs prevent this by netting and centralizing, though they create new risk: what if CCP fails? (regulators monitor closely).",
    },
    {
      id: 5,
      question: 'CVA (Credit Valuation Adjustment) represents:',
      options: [
        'The current market value of a derivative',
        'The expected loss from counterparty default, deducted from derivative value',
        'The gain from the possibility of our own default',
        'The regulatory capital charge for credit risk',
      ],
      correctAnswer: 1,
      explanation:
        "CVA is the present value of expected loss from counterparty defaulting on a derivative. It reduces the derivative's value because there's a chance the counterparty won't pay. If you have a $10M in-the-money swap with a counterparty who has 2% default probability and 60% LGD, CVA ≈ $10M × 2% × 60% = $120K. You deduct this from the swap value. Option A describes gross value (before CVA). Option C describes DVA (Debit Valuation Adjustment—our default benefit), not CVA. Option D describes capital, not valuation. CVA became mandatory after 2008 when Lehman defaulted and counterparties realized their derivatives were worth less than marked. Banks now have CVA desks that calculate and hedge this risk. Basel III requires capital for CVA volatility because CVA can change rapidly as counterparty credit deteriorates.",
    },
  ],
} as const;

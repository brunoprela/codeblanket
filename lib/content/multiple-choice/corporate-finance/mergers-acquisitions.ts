import { MultipleChoiceQuestion } from '@/lib/types';

export const mergersAcquisitionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mergers-acquisitions-mc-1',
    question:
      'Acquirer standalone value $8B, Target standalone value $2B, Synergies $500M, Price paid $2.6B. What is the NPV to the acquirer?',
    options: ['-$100M', '+$100M', '+$500M', '-$500M'],
    correctAnswer: 0,
    explanation:
      'NPV to acquirer = (Combined value) - (Acquirer standalone) - (Price paid). Combined value = $8B + $2B + $0.5B = $10.5B. NPV = $10.5B - $8B - $2.6B = -$0.1B = -$100M. The acquirer DESTROYS $100M value because they paid a $600M premium ($2.6B - $2.0B) but only got $500M in synergies. Premium > Synergies → value destruction. The acquirer overpaid by $100M.',
  },
  {
    id: 'mergers-acquisitions-mc-2',
    question:
      'What does a 30% control premium in an M&A transaction represent?',
    options: [
      'The percentage of synergies the acquirer expects',
      "The amount above the target's standalone market value the acquirer pays",
      'The tax benefit from the acquisition',
      'The cost of integrating the target',
    ],
    correctAnswer: 1,
    explanation:
      "Control premium = (Offer Price - Pre-announcement market price) / Pre-announcement price. It represents how much above the target's standalone trading value the acquirer pays to gain control. A 30% premium means if target traded at $100/share, acquirer offers $130/share. Typical premiums: 20-40%. Premium justified by: Synergies, Strategic value, Control benefits. If premium > synergies, acquirer overpays.",
  },
  {
    id: 'mergers-acquisitions-mc-3',
    question:
      'In an EPS accretion/dilution analysis, an all-stock acquisition is most likely to be accretive when:',
    options: [
      "The acquirer's P/E ratio is lower than the target's P/E ratio",
      "The acquirer's P/E ratio is higher than the target's P/E ratio",
      'Both companies have the same P/E ratio',
      'The target has negative earnings',
    ],
    correctAnswer: 1,
    explanation:
      'Stock deals are accretive when acquirer P/E > target P/E. Example: Acquirer at 20× P/E buys target at 15× P/E. Acquirer uses "expensive" stock (20×) to buy "cheap" earnings (15×). This arbitrages the P/E multiple difference. If acquirer P/E < target P/E: Dilutive (buying expensive earnings with cheap stock). If P/E equal: Neutral. However: EPS accretion ≠ value creation! Can be accretive but destroy value if overpay.',
  },
  {
    id: 'mergers-acquisitions-mc-4',
    question:
      "Which M&A defense mechanism dilutes a hostile acquirer's ownership by allowing other shareholders to buy stock at a discount?",
    options: [
      'Golden parachute',
      'Staggered board',
      'Poison pill',
      'White knight',
    ],
    correctAnswer: 2,
    explanation:
      'Poison pill (shareholder rights plan): If hostile acquirer buys >15% (trigger threshold), other shareholders can buy stock at 50% discount. This massively dilutes the hostile acquirer, making takeover prohibitively expensive. Golden parachute: Large severance for executives (increases cost). Staggered board: Only 1/3 elected each year (delays control). White knight: Friendly acquirer outbids hostile bidder. Poison pill is most powerful defense—forces negotiation with board.',
  },
  {
    id: 'mergers-acquisitions-mc-5',
    question:
      'In M&A valuation, precedent transaction analysis typically yields higher valuations than comparable company analysis because:',
    options: [
      'Transaction multiples include control premiums and synergies',
      'Public companies are generally overvalued',
      'Transaction multiples use lower discount rates',
      'Precedent transactions are more liquid',
    ],
    correctAnswer: 0,
    explanation:
      'Precedent transactions (M&A comps) include control premium (20-40%) and often reflect strategic value/synergies. Comparable company analysis (trading comps) reflects minority stake (no control). Example: Company trades at 10× EV/EBITDA (trading comps). M&A transaction might be 12-13× (precedent transaction). Difference: 20-30% control premium. When valuing acquisition target: Use trading comps for floor (minority value). Use precedent transactions for ceiling (control value). Fair price: Somewhere in between, depending on synergies.',
  },
];

import { MultipleChoiceQuestion } from '@/lib/types';

export const probabilityPuzzlesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pp-mc-1',
    question:
      'In the Monty Hall problem, you choose door 1. The host opens door 3 (revealing a goat). Before you decide whether to switch, the host offers you the option to open door 2 and look behind it without choosing it. If you see a goat, you must keep door 1. If you see a car, you can switch to door 2. Should you pay $100 for this option if you win $1000 for getting the car?',
    options: [
      'Yes, because it gives you additional information worth more than $100',
      'No, because you should always switch anyway, so the peek is worthless',
      'Yes, but only if the $100 is less than 10% of your net worth',
      "No, because the peek doesn't change the fundamental 2/3 probability",
    ],
    correctAnswer: 1,
    explanation:
      "The peek is worthless because you should ALWAYS switch in Monty Hall (2/3 vs 1/3). The peek adds no value: if you see a goat behind door 2, you're forced to stay with door 1 (1/3 chance of car). If you see the car, you switch (certain win). But without the peek, just switching gives you 2/3 chance. The peek option gives: (2/3)×0 + (1/3)×1 = 1/3 success rate—WORSE than just switching! Never pay for information that hurts your strategy.",
  },
  {
    id: 'pp-mc-2',
    question:
      'You draw cards from a standard deck without replacement. What is the probability the first ace appears on the 5th card?',
    options: [
      '(4/52) × (48/51) × (47/50) × (46/49) × (48/48)',
      '(48/52) × (47/51) × (46/50) × (45/49) × (4/48)',
      '4/52',
      '(4/52) × (3/51) × (2/50) × (1/49)',
    ],
    correctAnswer: 1,
    explanation:
      'First 4 cards must be non-aces (48 non-aces in 52 cards), then 5th card is an ace. P(1st not ace) = 48/52. P(2nd not ace | 1st not ace) = 47/51. P(3rd not ace) = 46/50. P(4th not ace) = 45/49. P(5th is ace) = 4/48 (4 aces remain out of 48 cards). Multiply: (48×47×46×45×4) / (52×51×50×49×48) ≈ 0.0299.',
  },
  {
    id: 'pp-mc-3',
    question:
      'In the medical test problem with 1% disease prevalence and 95% test accuracy, you test positive. What additional information would most increase the posterior probability you have the disease?',
    options: [
      'Taking the same test again and getting another positive result',
      'Learning that you have risk factors that make prevalence 10% in your group',
      "Learning the test's sensitivity is 98% (higher than 95%)",
      'Testing everyone in your family',
    ],
    correctAnswer: 1,
    explanation:
      'The prior prevalence (1%) is the dominant factor keeping posterior probability low (16.1%). Increasing prevalence to 10% changes P(D|T) dramatically: P(D|T) = (0.95×0.10) / (0.95×0.10 + 0.05×0.90) = 0.095 / 0.14 = 67.9%! Taking the test again helps but only updates to ~75% after second positive. Increasing sensitivity from 95% to 98% only moves posterior from 16.1% to 18.1%. Prior prevalence is the strongest lever.',
  },
  {
    id: 'pp-mc-4',
    question:
      'You flip a fair coin until you get two heads in a row (HH). What is the probability the sequence ends with exactly two heads (HH) versus more than two heads (e.g., HHH)?',
    options: [
      'Exactly 2 heads: 100%, because we stop at HH',
      'Exactly 2 heads: 75%, More than 2: 25%',
      'Exactly 2 heads: 50%, More than 2: 50%',
      'Cannot be determined without counting all possible sequences',
    ],
    correctAnswer: 0,
    explanation:
      'We stop IMMEDIATELY upon seeing HH for the first time. By definition, the sequence ends with exactly HH, not HHH or longer. Once we see HH, we don\'t flip again, so probability of ending with exactly HH is 100%. This is a definitional question testing careful reading. Tricky version: "What\'s the probability HH appears starting from the end?" Still 100% by the stopping rule.',
  },
  {
    id: 'pp-mc-5',
    question:
      'You have two envelopes, one containing twice as much money as the other. You open one and find $100. Your friend offers to let you switch to the other envelope. Expected value analysis says: other envelope has $50 or $200 with equal probability, so E[other] = 0.5×$50 + 0.5×$200 = $125 > $100. Should you always switch?',
    options: [
      'Yes, the expected value is higher so always switch',
      'No, this is the Two Envelope Paradox—the logic is flawed',
      "Yes, but only if you're risk-neutral",
      'It depends on the prior distribution of envelope amounts',
    ],
    correctAnswer: 1,
    explanation:
      "This is the Two Envelope Paradox. The flaw: you can't assign equal probability to $50 and $200 without knowing the prior. If you see $100, and if the original amounts were ($50, $100), other envelope has $50. If original amounts were ($100, $200), other has $200. Equal probability requires symmetric prior, but that leads to infinite expected values (paradox). Resolution: switching gives no advantage on average—E[other | you see $100] = $100 assuming any reasonable prior. The paradox highlights the danger of using symmetry without checking if a proper prior exists.",
  },
];

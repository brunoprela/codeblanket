import { MultipleChoiceQuestion } from '@/lib/types';

export const monteCarloValuationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mcv-mc-1',
    question: 'A Monte Carlo simulation with 10,000 iterations shows: P10 = $8B, P50 = $12B, P90 = $18B. What does "P10 = $8B" mean?',
    options: [
      'The company is worth $8B in the worst 10% of scenarios',
      'There is a 10% probability the company is worth less than $8B',
      'The company is worth $8B with 10% confidence',
      'The company lost $8B in 10% of simulations'],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: "10% probability worth LESS than $8B." P10 is the 10th percentile—10% of simulations resulted in valuations below $8B, and 90% resulted in valuations above $8B. It marks the boundary: worse than P10 is the bottom 10% (tail risk), better than P10 is the top 90%. Option 1 is imprecise—P10 is not "the value IN worst 10%," it\'s the threshold separating worst 10% from rest. Option 3 confuses percentile with confidence level. Option 4 is nonsense—P10 is a valuation level ($8B), not a loss. Usage: P10 quantifies downside risk. If acquiring at $12B (P50), there\'s a 40% chance company worth less than paid (between P10 and P50). Risk metric: Distance from purchase price to P10 = potential loss in stress scenario.',
  },
  {
    id: 'mcv-mc-2',
    question: 'You model revenue growth as Normal(10%, 3%) in Monte Carlo. What percentage of simulations will have revenue growth between 7% and 13%?',
    options: ['50%', '68%', '95%', '99.7%'],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: 68%. Normal distribution rule: ±1 standard deviation from mean captures 68% of outcomes. Here: Mean = 10%, Std dev = 3%. Mean - 1 std = 10% - 3% = 7%. Mean + 1 std = 10% + 3% = 13%. Therefore, 68% of simulations fall between 7% and 13%. Remaining 32% split: 16% below 7% (left tail), 16% above 13% (right tail). Other ranges: ±2 std dev (4% to 16%) = 95% of outcomes, ±3 std dev (1% to 19%) = 99.7% of outcomes. Application: If you want to capture "most likely" scenarios, use mean ± 1 std dev (68%). For comprehensive risk analysis, use mean ± 2 std dev (95%).',
  },
  {
    id: 'mcv-mc-3',
    question: 'Which distribution type is MOST appropriate for modeling stock prices in Monte Carlo simulation?',
    options: ['Normal distribution', 'Lognormal distribution', 'Uniform distribution', 'Triangular distribution'],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: Lognormal distribution. Reasons: (1) Ensures positive values—stock prices can\'t be negative. Normal distribution allows negative values (problematic if mean - 2×std < 0). (2) Right-skewed—stock prices have limited downside (worst case is $0) but unlimited upside (can go to $infinity). Lognormal captures this asymmetry. (3) Multiplicative growth—stock returns compound multiplicatively, not additively. Lognormal is natural distribution for multiplicative processes. Example: Stock at $100 with 20% volatility (annual). Normal(100, 20) implies: P5 = $67, P95 = $133 (symmetric). But also allows negative prices if volatility high. Lognormal(log(100), 0.2) implies: P5 = $70, P95 = $143 (right-skewed, always positive). Option 1 (normal) is theoretically wrong for prices (allows negatives). Option 3 (uniform) implies all prices equally likely (wrong—prices cluster near current level). Option 4 (triangular) is for expert judgment scenarios, not continuous price modeling.',
  },
  {
    id: 'mcv-mc-4',
    question: 'Two variables have correlation of +0.6. In Monte Carlo simulation, this means:',
    options: [
      'When one variable increases, the other increases 60% of the time',
      'The variables move together with 60% strength (positive relationship)',
      'One variable explains 60% of the variance in the other',
      '60% of simulations have both variables above their means'],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: Variables move together with 60% strength. Correlation of +0.6 means: (1) Positive relationship—when X increases, Y tends to increase (not always, but on average). (2) Moderate-strong relationship—correlation ranges from -1 (perfect negative) to +1 (perfect positive). 0.6 is moderately strong. (3) Linear association—measures linear relationship between variables. Option 1 is wrong—correlation is not about frequency of co-movement (60% of time). It\'s about strength of average co-movement. Option 3 confuses correlation with R-squared (R² = 0.6² = 0.36 = 36% variance explained). Option 4 is probability statement unrelated to correlation. Interpretation: Correlation +0.6 means: If X is 1 std dev above its mean, Y is on average 0.6 std dev above its mean (not exactly, but expected value). Application: Positive correlation between revenue growth and EBITDA margin means: In simulations with high growth (top 10%), expect above-average margins (operating leverage). In simulations with low growth (bottom 10%), expect below-average margins.',
  },
  {
    id: 'mcv-mc-5',
    question: 'You run Monte Carlo DCF and get: Mean = $11B, Median = $10B, Std Dev = $3B. Which statement is correct?',
    options: [
      'The distribution is left-skewed (negative skew)',
      'The distribution is right-skewed (positive skew)',
      'The distribution is symmetric (normal)',
      'Cannot determine skewness from mean and median alone'],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: Right-skewed (positive skew). Rule: Mean > Median → Right skew (long tail to the right). Mean < Median → Left skew (long tail to the left). Mean = Median → Symmetric (normal-like). Here: Mean $11B > Median $10B → Right skew. Interpretation: Distribution has longer tail on high side (more extreme upside scenarios than downside). Causes: Valuations have bounded downside (can\'t be negative) but unbounded upside (breakthrough scenarios). Few very high outcomes (P95 = $20B+) pull mean higher than median. Implications: Expected value (mean $11B) exceeds "typical" outcome (median $10B) by 10%. For portfolio decisions, use mean (accounts for rare but large wins). For single deal, use median (more representative of central tendency). Visual: Histogram would show peak at $10B (median), but long right tail extending to $20B+. Option 1 (left skew) is backwards. Option 3 (symmetric) would require mean = median. Option 4 is wrong—mean vs median definitively indicates skew direction.',
  },
];

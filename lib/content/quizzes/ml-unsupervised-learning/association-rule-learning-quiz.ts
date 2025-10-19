/**
 * Quiz: Association Rule Learning
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { QuizQuestion } from '../../../types';

export const association_rule_learningQuiz: QuizQuestion[] = [
  {
    id: 'association-rule-learning-q1',
    question: `Explain the three key metrics in association rule mining: support, confidence, and lift. Why is lift more informative than confidence alone, and how do you interpret lift values?`,
    hint: 'Cover the formulas and provide examples showing why confidence can be misleading.',
    sampleAnswer: `SUPPORT: P(X) = how often itemset appears. support({milk, bread}) = 0.3 means 30% of transactions contain both. Indicates frequency/importance. CONFIDENCE: P(Y|X) = how often rule is true. confidence({milk}→{bread}) = P({milk,bread})/P({milk}). If 0.8, then 80% of customers who buy milk also buy bread. Problem: misleading if bread is very popular. LIFT: P(Y|X)/P(Y) = how much more likely Y is purchased with X vs without. lift({milk}→{bread}) = confidence/P({bread}). INTERPRETATION: Lift > 1: positive correlation (buying X increases chance of Y), Lift = 1: independent (X doesn't affect Y), Lift < 1: negative correlation (buying X decreases chance of Y). EXAMPLE: bread has P(bread)=0.9 (very popular). Rule {milk}→{bread} has confidence=0.9. Sounds strong! But lift=0.9/0.9=1.0. Actually independent - milk doesn't increase bread purchases. WHY LIFT MATTERS: Controls for base rate. Confidence alone misleading for popular items. Lift reveals true association. PRACTICAL: Focus on rules with lift > 1.2 (20% increase). Very high lift (>2) indicates strong association worth investigating.`,
    keyPoints: [
      'Support: frequency of itemset in data',
      'Confidence: conditional probability P(Y|X)',
      'Lift: confidence adjusted for base rate of Y',
      'Lift > 1: positive association; = 1: independent; < 1: negative',
      'Lift more informative than confidence alone',
    ],
  },
  {
    id: 'association-rule-learning-q2',
    question: `Explain the Apriori algorithm and the Apriori property. How does this property dramatically reduce the number of candidate itemsets that need to be checked, and why is this important for computational efficiency?`,
    hint: 'Cover the pruning strategy and how it prevents combinatorial explosion.',
    sampleAnswer: `APRIORI PROPERTY (Monotonicity): If an itemset is frequent, all its subsets must also be frequent. Contrapositive: If an itemset is infrequent, all its supersets must also be infrequent. IMPORTANCE: Massive computational savings. EXAMPLE: If {milk, bread} has support < threshold, then {milk, bread, eggs}, {milk, bread, butter}, etc., must also be infrequent. No need to check! ALGORITHM: (1) Find frequent 1-itemsets (scan database), (2) Generate candidate k-itemsets from frequent (k-1)-itemsets, (3) Prune candidates: if any (k-1)-subset is infrequent, remove candidate, (4) Count support of remaining candidates (scan database), (5) Keep frequent k-itemsets, (6) Repeat until no new frequent itemsets. WITHOUT PRUNING: With n items, must check 2^n possible itemsets. For n=100: 2^100 ≈ 10^30 itemsets - impossible! WITH PRUNING: If 95 items are infrequent individually, 2^95 supersets eliminated immediately. Practical: check thousands instead of billions. TRADE-OFF: Multiple database scans (slow) but dramatically fewer candidates. Alternative: FP-Growth avoids repeated scans using tree structure (faster for large databases).`,
    keyPoints: [
      'Apriori property: frequent itemset → all subsets frequent',
      'Allows pruning of candidate itemsets',
      'Prevents combinatorial explosion (2^n possibilities)',
      'Dramatically reduces computational cost',
      'Multiple database scans but far fewer candidates',
    ],
  },
  {
    id: 'association-rule-learning-q3',
    question: `Discuss practical applications and limitations of association rule mining. Why don't association rules imply causation, and what are common pitfalls in interpreting rules?`,
    hint: 'Cover use cases, the causation issue, spurious correlations, and business implementation.',
    sampleAnswer: `APPLICATIONS: (1) RETAIL: 'Customers who bought X also bought Y' recommendations, product placement (separate associated items to increase store traversal), bundling/promotions. (2) E-COMMERCE: product recommendations, frequently bought together. (3) HEALTHCARE: comorbidity analysis (diseases that co-occur), treatment protocols. (4) WEB: page navigation patterns, click analysis. CAUSATION WARNING: Association ≠ Causation! {diapers}→{beer} doesn't mean diapers cause beer purchases. Possible explanations: (1) common cause (new parents), (2) coincidence, (3) temporal patterns (both bought on weekends). Cannot determine causality from association rules. Need experiments (A/B tests) or causal inference methods. COMMON PITFALLS: (1) SPURIOUS CORRELATIONS: random chance in large datasets. (2) POPULAR ITEMS: dominate rules but aren't interesting. (3) TRIVIAL RULES: {bread, butter}→{milk} and {bread, milk}→{butter} are redundant. (4) STATIC: rules change over time (seasonality, trends). (5) ACTIONABILITY: interesting ≠ actionable. Need business context. BEST PRACTICES: Use lift > 1.2, validate with domain experts, remove trivial/redundant rules, update periodically, A/B test before implementing, consider confounding factors.`,
    keyPoints: [
      'Applications: recommendations, product placement, comorbidity analysis',
      'Association does not imply causation',
      'Watch for spurious correlations and trivial rules',
      'Rules are static, need periodic updates',
      'Validate with domain knowledge and A/B testing',
    ],
  },
];

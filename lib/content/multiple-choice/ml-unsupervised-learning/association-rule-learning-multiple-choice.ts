/**
 * Multiple Choice Questions: Association Rule Learning
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { MultipleChoiceQuestion } from '../../../types';

export const association_rule_learningMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'association-rule-learning-mc1',
      question:
        'In association rule mining, what does the support metric measure?',
      options: [
        'How reliable the rule is',
        'How frequently an itemset appears in the dataset',
        'How much better than random the rule is',
        'The strength of the relationship between items',
      ],
      correctAnswer: 1,
      explanation:
        'Support measures how frequently an itemset appears in the dataset. For example, support({milk, bread}) = 0.3 means 30% of all transactions contain both milk and bread.',
    },
    {
      id: 'association-rule-learning-mc2',
      question: 'Why is lift more informative than confidence alone?',
      options: [
        'Lift is easier to calculate',
        'Lift adjusts for the base rate of the consequent item',
        'Lift can only be positive',
        'Lift works with continuous data',
      ],
      correctAnswer: 1,
      explanation:
        'Lift = confidence / P(Y) adjusts for how common Y is. A rule might have high confidence simply because Y is popular. Lift > 1 indicates X truly increases the likelihood of Y beyond its base rate.',
    },
    {
      id: 'association-rule-learning-mc3',
      question: 'The Apriori property states that:',
      options: [
        'If an itemset is frequent, all its subsets must be frequent',
        'Frequent items always appear together',
        'The algorithm must start with 1-itemsets',
        'Support decreases as itemset size increases',
      ],
      correctAnswer: 0,
      explanation:
        'The Apriori property (monotonicity) states that if an itemset is frequent, all its subsets must also be frequent. This allows pruning of candidates: if {A,B} is infrequent, no need to check {A,B,C}.',
    },
    {
      id: 'association-rule-learning-mc4',
      question: 'What does a lift value of 0.8 for a rule {A} → {B} indicate?',
      options: [
        'B is 80% likely when A is purchased',
        '80% of transactions contain both A and B',
        'Purchasing A decreases the likelihood of purchasing B',
        'The rule has 80% accuracy',
      ],
      correctAnswer: 2,
      explanation:
        'Lift < 1 indicates negative correlation: purchasing A actually makes B less likely than its base rate. Lift = 1 is independence, lift > 1 is positive correlation.',
    },
    {
      id: 'association-rule-learning-mc5',
      question:
        'You find a rule {diapers} → {beer} with support=0.05, confidence=0.6, lift=1.8. What should you conclude?',
      options: [
        'The rule is useless because support is too low',
        'The rule suggests a real association: 60% of diaper buyers also buy beer, which is 1.8× the baseline',
        'The rule indicates diapers and beer are independent',
        'You should reject this rule because lift should be > 2',
      ],
      correctAnswer: 1,
      explanation:
        'Support=0.05 means 5% of transactions have both (low but might still be valuable for targeted marketing). Confidence=0.6 means 60% of diaper buyers buy beer. Lift=1.8 means buying diapers makes beer 1.8× more likely than baseline, indicating a real association. Low support is not necessarily bad - it depends on business context (rare but valuable patterns). There is no universal threshold for lift.',
    },
  ];

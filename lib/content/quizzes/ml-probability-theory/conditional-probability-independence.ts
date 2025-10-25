/**
 * Quiz questions for Conditional Probability & Independence section
 */

export const conditionalprobabilityindependenceQuiz = [
  {
    id: 'q1',
    question:
      "Explain conditional probability and why it's central to machine learning. Give at least three examples of how conditional probability appears in ML systems.",
    hint: 'Think about predictions, feature relationships, and model outputs.',
    sampleAnswer:
      "Conditional probability P(A|B) is the probability of A occurring given that B has occurred. It\'s calculated as P(A|B) = P(A∩B)/P(B). This is central to ML because: (1) All classification models output P(class|features) - the probability of a class given observed features. (2) Feature importance is measured by how much knowing a feature changes the prediction: P(Y|X₁) vs P(Y). (3) Bayesian methods update beliefs: P(hypothesis|data) combines prior beliefs with evidence. (4) Causal inference asks: does P(Y|X) differ from P(Y)? (5) Model confidence: P(correct|high_confidence) should be higher than P(correct) overall. Without conditional probability, we couldn't make informed predictions, measure feature importance, or reason about uncertainty. Every time an ML model makes a prediction, it's computing a conditional probability.",
    keyPoints: [
      'P(A|B) = P(A∩B)/P(B) - probability of A given B',
      'All predictions are conditional: P(class|features)',
      'Feature importance: how much does knowing feature change probability',
      'Bayesian inference: P(hypothesis|data)',
      'Foundation of all probabilistic ML',
    ],
  },
  {
    id: 'q2',
    question:
      'Two events can be dependent overall but conditionally independent given a third event. Explain this concept and why it matters for Naive Bayes classifiers.',
    hint: 'Think about a common cause creating correlation.',
    sampleAnswer:
      'Conditional independence means P(A∩B|C) = P(A|C)×P(B|C) - A and B are independent once we condition on C. Classic example: exam scores. Test1 and Test2 are correlated overall (students who score high on one tend to score high on the other). But this correlation is because both depend on study time. Given study time (low or high), Test1 and Test2 become independent - knowing one doesn\'t help predict the other. Naive Bayes exploits this: it assumes features are conditionally independent given the class, even though features may be correlated overall. Formula: P(x₁,x₂,...|y) = P(x₁|y)×P(x₂|y)×... This "naive" assumption simplifies computation (factorial to linear) and often works well in practice. Even when violated, Naive Bayes can still give good predictions because we only need relative probabilities, not exact ones.',
    keyPoints: [
      'Conditional independence: A ⊥ B | C means A,B independent given C',
      'Events can be dependent overall but independent given C',
      'Common cause creates overall correlation',
      'Naive Bayes assumes features conditionally independent given class',
      'Simplifies computation and often works despite violation',
    ],
  },
  {
    id: 'q3',
    question:
      'When testing for independence between events A and B, what are the three equivalent conditions you can check? Why are they all equivalent?',
    sampleAnswer:
      "Three equivalent definitions of independence: (1) P(A|B) = P(A) - knowing B doesn't change probability of A. (2) P(B|A) = P(B) - knowing A doesn't change probability of B. (3) P(A∩B) = P(A)×P(B) - joint probability factors. They're equivalent by the multiplication rule. Starting from (1): P(A|B) = P(A). Multiply both sides by P(B): P(B)×P(A|B) = P(B)×P(A). Left side is P(A∩B) by multiplication rule, giving (3). From (3), divide by P(B): P(A∩B)/P(B) = P(A), which is P(A|B) = P(A), giving (1). Similarly for (2). Practical testing: (3) is easiest to check numerically - just compute three probabilities and verify multiplication. If violated, events are dependent. Common in ML: check if P(feature₁, feature₂) = P(feature₁)×P(feature₂) to test feature independence.",
    keyPoints: [
      'Three equivalent definitions: P(A|B)=P(A), P(B|A)=P(B), P(A∩B)=P(A)×P(B)',
      'Multiplication rule connects them',
      'Test (3) easiest: check if joint factors',
      'Used to test feature independence in ML',
      'Violation indicates dependence/correlation',
    ],
  },
];

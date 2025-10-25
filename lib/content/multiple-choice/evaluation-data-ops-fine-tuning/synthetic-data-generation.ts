/**
 * Multiple choice questions for Synthetic Data Generation section
 */

export const syntheticDataGenerationMultipleChoice = [
  {
    id: 'synthetic-data-mc-1',
    question:
      'You generate 10,000 synthetic examples using GPT-4. Training on these increases accuracy from 72% (real data only) to 79%. However, when you manually review 100 synthetic examples, you find 15% contain factual errors. What should you do?',
    options: [
      'Use them anyway—the 7% accuracy gain outweighs the 15% error rate',
      'Filter out the errors using an automatic fact-checking system before training',
      'Mix synthetic and real data with real data weighted higher to mitigate errors',
      'Regenerate with lower temperature to reduce hallucinations',
    ],
    correctAnswer: 2,
    explanation:
      "Option C (mix with higher real data weight) is most practical. Here\'s why: Option A (use anyway) is risky—model will learn incorrect facts. Option B (auto fact-check) sounds ideal but fact-checking systems have ~70-80% accuracy themselves, so you'll still miss errors. Option D (lower temperature) reduces hallucinations but also reduces diversity, diminishing value of synthetic data. Best approach: Weighted training where real data has 2-3x the weight of synthetic. This means model \"trusts\" real data more when learning. Example: Train with loss = 3.0 × loss_real + 1.0 × loss_synthetic. Model still benefits from synthetic diversity (79% vs 72%) but grounds itself in real facts. Additionally: Filter obvious errors (check named entities against knowledge base), Add data uncertainty flags (label synthetic examples for model to know they're less reliable), Monitor model on factual accuracy benchmarks.",
  },
  {
    id: 'synthetic-data-mc-2',
    question:
      'You measure diversity of synthetic data using average cosine distance between embeddings. Real data has 0.68 average distance, your synthetic data has 0.34. What does this indicate?',
    options: [
      'Synthetic data is high quality (more focused/coherent)',
      'Synthetic data lacks diversity (too repetitive/similar)',
      'Real data has errors or outliers that should be cleaned',
      'The embedding model is not suitable for this task',
    ],
    correctAnswer: 1,
    explanation:
      'Option B (lacks diversity) is correct. Lower average distance = examples are more similar to each other. Real data at 0.68 has high diversity (examples cover different topics, styles, contexts). Synthetic at 0.34 is much more similar—likely due to: Generator using repetitive patterns, Limited few-shot examples causing mode collapse, Low temperature causing conservative generation, Same prompting strategy for all examples. Consequences: Model trains on repetitive data, learns narrow distribution, poor generalization. Solutions: Increase temperature (0.7 → 1.2), Use cluster-based prompting (generate from different contexts), Add diversity rejection sampling (reject examples too similar to existing), Use multiple generation methods. Target: Get synthetic diversity to 0.55-0.65 (close to real). Option A is wrong—too low diversity is bad. Option C is wrong—real data diversity is expected. Option D is possible but unlikely if embedding model works for real data.',
  },
  {
    id: 'synthetic-data-mc-3',
    question:
      'You want to detect if your training data contains synthetic examples from GPT-4 (data contamination). Which method is MOST effective?',
    options: [
      'Search for common GPT-4 phrases like "as an AI language model"',
      'Check response length distribution (GPT-4 tends to be verbose)',
      'Train a classifier on known GPT-4 vs human text, apply to your data',
      'All of the above—use multiple detection methods',
    ],
    correctAnswer: 3,
    explanation:
      'Option D (all methods) is correct. Different methods catch different types of contamination: Method A (phrase search): Catches obvious cases, but sophisticated users remove these phrases. Still useful for quick first pass. Find: "It\'s important to note", "arguably", "various factors", numbered lists. Method B (statistical features): GPT-4 has signatures—avg length, vocabulary diversity, sentence structure. Compare distributions of suspect data vs known human/GPT-4. Method C (classifier): Most robust. Train on known human vs GPT-4 examples, apply to suspect data. Use features: embeddings, perplexity, writing style. Achieves ~85-90% accuracy. In practice: Run all three, flag examples that trigger 2+ methods for manual review. Why contamination matters: If training data contains GPT-4 text, your model learns GPT-4\'s biases and errors instead of real-world patterns. False confidence—accuracy looks high on contaminated data but drops in production. Best practice: Keep provenance metadata (track data source), regularly audit for contamination, especially after acquiring new datasets.',
  },
  {
    id: 'synthetic-data-mc-4',
    question:
      'You generate 1,000 synthetic examples and measure their "utility" (impact on model performance). What is the BEST way to measure utility?',
    options: [
      'Human reviewers rate quality of synthetic examples (1-5 scale)',
      'Measure accuracy improvement: with vs without synthetic data',
      'Check if synthetic examples are similar to real examples (embedding distance)',
      'Count how many synthetic examples the model gets correct',
    ],
    correctAnswer: 1,
    explanation:
      "Option B (accuracy improvement) is the gold standard for utility. Utility means \"does it help the model?\" The only way to know is to measure model performance. Methodology: Baseline: Train on real data only (e.g., 500 examples) → 75% test accuracy. With synthetic: Train on real + synthetic (500 + 1000) → 82% test accuracy. Utility = +7% accuracy improvement. This directly measures value of synthetic data for your task. Option A (human quality ratings) is a proxy but doesn't guarantee utility—high-quality examples might not help if they're too easy/hard or redundant with real data. Option C (similarity to real) is wrong—you WANT diversity, not similarity. If synthetic is identical to real, it adds no value. Option D (model correctness on synthetic) measures if synthetic is learnable, but doesn't tell if it helps on real test data. Best practice: Always measure utility on a held-out test set of REAL data. If synthetic doesn't improve real-world performance by at least 3-5%, it's not worth the generation cost.",
  },
  {
    id: 'synthetic-data-mc-5',
    question:
      'You use back-translation (English → French → English) to create synthetic data. Original: "The product is great." Back-translated: "The product is excellent." Is this a good augmentation?',
    options: [
      'Yes—it creates a synonym variation while preserving meaning',
      'No—the meaning changed ("great" ≠ "excellent"), so it\'s noisy',
      "It depends—check semantic similarity; if >0.9, it's acceptable",
      'No—back-translation should produce identical text',
    ],
    correctAnswer: 2,
    explanation:
      'Option C (depends on similarity) is correct. "Great" and "excellent" are close synonyms, so this is probably acceptable. But you need a systematic way to decide. Method: Use semantic similarity (e.g., sentence embeddings). Compute cosine similarity between original and back-translated. If similarity > threshold (e.g., 0.90), accept; else reject. In this case: embedding("The product is great") vs embedding("The product is excellent") → likely ~0.95 similarity → Accept. Why not Option A: Some back-translations change meaning significantly, e.g., "The product isn\'t bad" → "The product is good" (lost negation). Can\'t blindly accept all back-translations. Why not Option D: Perfect identity is rare and not the goal. Back-translation intentionally creates paraphrases for data augmentation. The goal is semantic preservation, not word-for-word identity. Best practice: Back-translate, measure similarity, keep examples with 0.85-0.98 similarity. Too low (<0.85): meaning changed. Too high (>0.98): no augmentation value, nearly identical. Sweet spot: 0.90-0.95 where you get meaningful paraphrases.',
  },
];

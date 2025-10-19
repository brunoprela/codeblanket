/**
 * Quiz questions for Data Visualization section
 */

export const datavisualizationQuiz = [
  {
    id: 'q1',
    question:
      'You notice that your scatter plot of two features is completely overcrowded with 100,000 points, making it impossible to see the relationship. What visualization techniques can you use to better understand the data, and when would you use each?',
    hint: 'Think about density-based visualizations and aggregation methods.',
    sampleAnswer:
      'For overcrowded scatter plots with dense data, several techniques work better than standard scatter: (1) **Hexbin plots**: Aggregate points into hexagonal bins and color by density. Best for: Showing overall density patterns and relationships. Implementation: plt.hexbin() with gridsize parameter. (2) **2D histograms/heatmaps**: Divide space into grid and count points per cell. Best for: Clear density visualization, works well with millions of points. (3) **Contour plots**: Show density contours like a topographic map. Best for: Highlighting regions of high concentration. (4) **Alpha transparency with small points**: Use alpha=0.01 with s=1 in scatter. Works for: Moderate density (thousands of points), reveals overlapping regions. (5) **Sample plotting**: Plot random sample of 5-10k points. Quick check but loses some information. (6) **KDE plots**: Kernel density estimation shows smooth density surface. Best for: Understanding distribution shape. Choice depends on: dataset size, whether you need exact patterns or just general trends, and if you need to show regression lines. For ML: hexbin/2D histograms preserve all information while being readable.',
    keyPoints: [
      'Hexbin plots aggregate into hexagons, color by count',
      '2D histograms divide space into grid cells',
      'Alpha transparency reveals overlapping',
      'Sampling provides quick approximation',
      'KDE shows smooth density surface',
      'Choice depends on data size and analysis goals',
    ],
  },
  {
    id: 'q2',
    question:
      "You create a learning curve for your model and observe that training accuracy is 95% but validation accuracy plateaus at 70%, with a large gap that doesn't close even with more data. What does this indicate, and what should you try next?",
    hint: 'Think about the bias-variance tradeoff and what the gap indicates.',
    sampleAnswer:
      "This learning curve shows **high variance (overfitting)**. Key observations: (1) Large gap (25%) between training and validation indicates the model memorizes training data but doesn't generalize. (2) Adding more data doesn't close the gap, suggesting the model has too much capacity for the problem. (3) Training accuracy of 95% shows the model can fit the training data easily. Solutions to try: **Regularization**: (a) Increase L1/L2 regularization strength (higher alpha), (b) Add dropout (neural networks), (c) Early stopping based on validation loss. **Reduce model complexity**: (a) Fewer layers in neural network, (b) Lower max_depth in trees, (c) Fewer features (feature selection). **More training data**: Sometimes helps if gap is moderate. **Data augmentation**: Artificially increase training data variety. **Ensemble methods**: Can reduce variance. **Cross-validation**: Ensure it's not just a bad train/val split. What NOT to do: Making the model more complex would worsen overfitting. The goal is to increase bias slightly (accept lower training accuracy) to reduce variance (improve validation accuracy). Target: Close the gap, even if training accuracy drops to 80-85%.",
    keyPoints: [
      'Large train-val gap indicates high variance/overfitting',
      "Model memorizes training data, doesn't generalize",
      'Solutions: regularization, reduce complexity, more data',
      'Increase bias to reduce variance',
      'Goal: close the gap, not maximize training accuracy',
      "Cross-validate to confirm it's systematic",
    ],
  },
  {
    id: 'q3',
    question:
      "You're presenting ML results to non-technical stakeholders. What are the most important principles for creating effective visualizations that communicate insights without overwhelming or misleading your audience?",
    hint: 'Consider clarity, honesty, and storytelling.',
    sampleAnswer:
      'Effective stakeholder visualizations require: **Clarity**: (1) One clear message per chart - avoid cramming multiple insights. (2) Descriptive titles that state the finding (not just "Results"). (3) Annotate key insights directly on the plot (arrows, text boxes). (4) Remove unnecessary elements (reduce chart junk). **Honesty**: (1) Always start axes at zero (or clearly indicate if not). (2) Use appropriate scales - don\'t distort to exaggerate effects. (3) Show confidence intervals/error bars to indicate uncertainty. (4) Don\'t cherry-pick favorable views. **Storytelling**: (1) Order charts logically (build narrative). (2) Use color purposefully (highlight important elements). (3) Compare to baselines or benchmarks. (4) Show before/after or with/without model. **Accessibility**: (1) Large enough fonts (12pt minimum). (2) Colorblind-friendly palettes. (3) High resolution for projection. (4) Include key numbers as text (don\'t rely on visual only). **Specific for ML**: Show business metrics (revenue, cost savings) not just accuracy. Show model limitations and failure cases. Use familiar chart types (bar, line) over exotic ones. Example: Instead of "Model Accuracy: 94%", show "Model would have prevented 94 out of 100 fraudulent transactions, saving $470K annually" with comparison to current system.',
    keyPoints: [
      'One clear message per visualization',
      'Honest representation (no distorted scales)',
      'Annotate insights directly on charts',
      'Use business metrics, not just technical metrics',
      'Show uncertainty and limitations',
      'Large fonts, colorblind-friendly, high resolution',
    ],
  },
];

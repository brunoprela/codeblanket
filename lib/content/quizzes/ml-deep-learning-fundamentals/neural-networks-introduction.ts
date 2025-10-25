import { QuizQuestion } from '../../../types';

export const neuralNetworksIntroductionQuiz: QuizQuestion[] = [
  {
    id: 'neural-networks-introduction-dq-1',
    question:
      'Explain why the XOR problem was historically significant in the development of neural networks. Why could the perceptron not solve it, and how did multi-layer perceptrons (MLPs) overcome this limitation?',
    sampleAnswer: `The XOR problem was a pivotal moment in AI history that led to the "AI winter" of the 1970s-80s and ultimately drove the development of modern neural networks:

**Why Perceptrons Cannot Solve XOR:**

The XOR (exclusive OR) function has the following truth table:
- (0,0) → 0
- (0,1) → 1
- (1,0) → 1
- (1,1) → 0

The fundamental limitation is **linear separability**. A perceptron learns a linear decision boundary defined by w₁x₁ + w₂x₂ + b = 0, which is a straight line in 2D space. However, XOR requires a non-linear boundary—you cannot draw a single straight line that separates the two classes.

Geometrically, the points (0,1) and (1,0) belong to class 1, while (0,0) and (1,1) belong to class 0. These points are diagonally arranged, making linear separation impossible.

**Historical Significance:**

In 1969, Marvin Minsky and Seymour Papert published "Perceptrons," mathematically proving this limitation. This led to:
1. Severe funding cuts for neural network research
2. A shift toward symbolic AI and expert systems
3. Nearly 20 years of stagnation in neural network development
4. The first "AI winter"

**How MLPs Solve XOR:**

Multi-layer perceptrons with hidden layers and non-linear activations overcome this by:

1. **Hierarchical Transformations**: The hidden layer transforms the input space into a new representation where the problem becomes linearly separable.

2. **Non-linear Decision Boundaries**: With activation functions like sigmoid or ReLU, MLPs can learn complex, non-linear decision boundaries—circles, curves, or any arbitrary shape.

3. **Feature Learning**: The hidden layer essentially learns new features (combinations of inputs) that make the classification task easier.

For XOR, a 2-3-1 MLP works as follows:
- Hidden neurons learn to detect patterns: "Is exactly one input on?"
- The output neuron combines these patterns
- The network effectively learns: output = (x₁ AND NOT x₂) OR (NOT x₁ AND x₂)

**Modern Implications:**

This breakthrough demonstrated that depth (multiple layers) provides fundamental computational advantages. Modern deep learning (100+ layers) extends this principle—each layer learns increasingly abstract representations, enabling networks to solve problems far beyond what any shallow model could achieve.

The XOR problem teaches us that representation matters. Given the right representation (learned by hidden layers), complex problems become simple.`,
    keyPoints: [
      'XOR is not linearly separable - no single line can separate the classes',
      'Perceptrons can only learn linear decision boundaries',
      "Minsky & Papert\'s proof led to the first AI winter (1970s-80s)",
      'MLPs with non-linear activations transform the input space to make problems linearly separable',
      'Hidden layers learn new representations where complex problems become simple',
      'Modern deep learning extends this principle to learn hierarchical abstractions',
    ],
  },
  {
    id: 'neural-networks-introduction-dq-2',
    question:
      'The Universal Approximation Theorem states that a single hidden layer network can approximate any continuous function. If this is true, why do we need deep networks with many layers? Discuss the practical trade-offs between shallow wide networks and deep narrow networks.',
    sampleAnswer: `While the Universal Approximation Theorem guarantees that shallow networks CAN approximate any function, it doesn't address how EFFICIENTLY they do so. Deep networks offer practical advantages that make them superior in real-world applications:

**Why Depth Matters Despite Universal Approximation:**

1. **Exponential Efficiency**: Deep networks require exponentially fewer parameters than shallow networks for the same representational power.
   - Example: Approximating a function with n variables
   - Shallow network: Might need O(2^n) neurons
   - Deep network: Might need only O(n²) neurons
   - For n=100 variables, this is the difference between 2^100 neurons (impossible) vs 10,000 neurons (feasible)

2. **Hierarchical Feature Learning**: Deep networks naturally learn hierarchical abstractions:
   - Layer 1: Low-level features (edges, textures in images)
   - Layer 2: Mid-level features (shapes, patterns)
   - Layer 3: High-level features (object parts)
   - Layer 4: Abstract concepts (complete objects)
   
   This mirrors how humans understand the world—we build complex concepts from simpler ones.

3. **Better Generalization**: Deep architectures encode useful inductive biases. The hierarchical structure acts as a form of regularization, leading to models that generalize better to unseen data.

**Practical Trade-offs:**

**Shallow Wide Networks:**
Advantages:
- Easier to train (no vanishing gradient problems)
- Faster to evaluate (single matrix multiply for hidden layer)
- Better theoretical guarantees (universal approximation)
- Fewer hyperparameters to tune

Disadvantages:
- Require enormous width (millions of neurons) for complex tasks
- Poor parameter efficiency (billions of parameters)
- Learn "flat" representations without hierarchical structure
- Don't transfer well to related tasks
- Overfit more easily on limited data

**Deep Narrow Networks:**
Advantages:
- Extremely parameter efficient
- Learn reusable hierarchical features
- Better transfer learning (pre-trained models)
- Better generalization on complex tasks
- More aligned with natural information processing

Disadvantages:
- Harder to train (vanishing/exploding gradients)
- Require careful initialization and normalization
- More hyperparameters (depth, layer sizes, skip connections)
- Longer training time (more sequential operations)

**Real-World Example - Image Classification:**

Shallow approach (1 hidden layer):
- Input: 224×224×3 = 150,528 features
- Hidden: 10,000 neurons needed
- Parameters: 150,528 × 10,000 = 1.5 billion

Deep approach (ResNet-50):
- 50 layers with skip connections
- Parameters: ~25 million
- 60x fewer parameters, far better accuracy

**Financial Trading Context:**

For predicting stock prices:
- Shallow: Might learn that "high RSI + low price" indicates buy signal
- Deep: Learns market regimes in lower layers, specific patterns in middle layers, and optimal timing in upper layers
- The deep network can adapt its strategy based on detected market conditions

**Conclusion:**

The Universal Approximation Theorem is a theoretical guarantee, not a practical guide. Deep networks are the standard in modern ML because they're more efficient, learn better representations, and generalize better—even though shallow networks could theoretically do the same job with astronomical width.`,
    keyPoints: [
      'Universal Approximation guarantees capability, not efficiency',
      'Deep networks need exponentially fewer parameters than shallow networks (O(n²) vs O(2^n))',
      'Hierarchical feature learning provides better representations and generalization',
      'Shallow wide networks overfit more and learn flat representations',
      'Deep networks transfer better to related tasks via learned hierarchies',
      'Modern success (ImageNet, GPT) relies on depth, not width',
    ],
  },
  {
    id: 'neural-networks-introduction-dq-3',
    question:
      'In quantitative trading, neural networks can be used for price prediction. Discuss the challenges unique to applying neural networks to financial time series data, and what precautions should be taken to avoid common pitfalls like overfitting to noise.',
    sampleAnswer: `Applying neural networks to financial prediction presents unique challenges that differ significantly from standard ML tasks like image classification. Financial markets are non-stationary, noisy, and adversarial:

**Unique Challenges in Financial Time Series:**

1. **Non-Stationarity**: Financial data distributions change over time
   - Market regimes shift (bull/bear markets, crises)
   - Relationships between features break down
   - Past patterns don't guarantee future performance
   - Neural networks trained on historical data may fail on future data

2. **Low Signal-to-Noise Ratio**: Financial data is extremely noisy
   - Random walk hypothesis: Prices may be mostly unpredictable
   - Many "patterns" are spurious correlations
   - True alpha signals are weak and fleeting
   - Easy to overfit to noise rather than signal

3. **Look-Ahead Bias**: Accidentally using future information
   - Feature engineering must be causally valid (only past data)
   - Train/test splits must respect temporal ordering
   - Must account for data revisions (economic indicators get revised)

4. **Survivorship Bias**: Historical data excludes failed companies
   - Training on survivors creates upward bias
   - Real-time trading includes companies that will fail
   - Backtests are overly optimistic

5. **Limited Data**: Despite seeming large, financial data is limited
   - Only ~20 years of high-quality data for most assets
   - Rare events (crashes) have few examples
   - Many regime changes haven't occurred in training period

6. **Transaction Costs**: Prediction accuracy alone insufficient
   - Trading costs (commissions, slippage, spread) can eliminate profits
   - High-frequency predictions need very high accuracy
   - Model must account for execution realities

7. **Adversarial Environment**: Markets adapt to strategies
   - If many traders use same strategy, it stops working (alpha decay)
   - Markets are adversarial—other traders exploit your patterns
   - Not like image classification where cats don't change to avoid detection

**Precautions to Avoid Overfitting:**

**1. Rigorous Validation:**
\`\`\`python
# BAD: Random split
X_train, X_test = train_test_split (data, test_size=0.2, random_state=42)

# GOOD: Time-based split (respects temporal order)
split_date = '2020-01-01'
train_data = data[data.index < split_date]
test_data = data[data.index >= split_date]

# BEST: Walk-forward validation (multiple train/test periods)
for train_end, test_end in rolling_windows:
    train = data[:train_end]
    test = data[train_end:test_end]
    model.fit (train)
    evaluate (model, test)
\`\`\`

**2. Regularization Techniques:**
- Use dropout (20-50%) to prevent co-adaptation
- L2 regularization on weights
- Early stopping based on validation loss
- Keep networks small (don't use ResNet-50 for price prediction!)
- Ensemble multiple models to reduce variance

**3. Feature Engineering Discipline:**
- Only use features that will be available in real-time
- Avoid using future information (e.g., don't use today's close in morning predictions)
- Use robust features less prone to overfitting (moving averages vs raw prices)
- Domain knowledge crucial (technical indicators, fundamentals)

**4. Cross-Validation for Time Series:**
\`\`\`python
from sklearn.model_selection import TimeSeriesSplit

# Time series cross-validation (respects temporal order)
tscv = TimeSeriesSplit (n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    # Train and validate
\`\`\`

**5. Out-of-Sample Testing:**
- Reserve completely untouched holdout set (e.g., last 6 months)
- Never use for hyperparameter tuning
- Only evaluate once to get realistic performance estimate

**6. Paper Trading:**
- Deploy model with paper trading (simulated) first
- Verify performance matches backtests
- Check for implementation bugs, data issues
- Gradually increase capital allocation

**7. Performance Metrics:**
- Don't just use accuracy—use risk-adjusted metrics
- Sharpe ratio: return per unit risk
- Maximum drawdown: worst peak-to-trough decline
- Win rate and profit factor
- Transaction cost analysis

**8. Ensemble and Diversification:**
- Combine multiple models (neural network + tree models + classical)
- Use different time horizons and features
- Diversification reduces dependence on any single model

**9. Regular Retraining:**
- Financial data is non-stationary
- Retrain model regularly (weekly/monthly)
- Monitor performance degradation
- Be prepared to stop trading if performance deteriorates

**10. Realistic Expectations:**
- 55% directional accuracy can be profitable with good risk management
- Focus on risk-adjusted returns, not raw predictions
- Many "working" strategies have Sharpe ratios < 1.5

**Example: Responsible Neural Network Trading:**

\`\`\`python
# Good practices in one pipeline
class TradingNeuralNetwork:
    def __init__(self):
        self.model = build_small_mlp()  # Keep it simple
        self.scaler = RobustScaler()  # Robust to outliers
        self.retrain_frequency = '30D'
        
    def create_features (self, data, lookback=20):
        # Only use past information
        features = {
            'returns': data['close',].pct_change (lookback),
            'volatility': data['returns',].rolling (lookback).std(),
            'rsi': calculate_rsi (data['close',]),
            # etc.
        }
        return features
    
    def train_with_validation (self, data):
        # Time-series split
        tscv = TimeSeriesSplit (n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split (data):
            train, val = data.iloc[train_idx], data.iloc[val_idx]
            self.model.fit (train)
            score = self.evaluate_sharpe (self.model, val)
            scores.append (score)
        
        return np.mean (scores)
    
    def evaluate_sharpe (self, model, data):
        predictions = model.predict (data)
        returns = predictions * data['forward_returns',]
        return np.mean (returns) / np.std (returns) * np.sqrt(252)
\`\`\`

**Conclusion:**

Financial ML is harder than standard ML because markets are non-stationary, noisy, and adversarial. Success requires rigorous validation, strong regularization, domain knowledge, and realistic expectations. Most importantly, remember that even small improvements over random (52-55% accuracy) can be profitable with proper risk management.`,
    keyPoints: [
      'Financial data is non-stationary—distributions change over time',
      'Signal-to-noise ratio is very low—easy to overfit to noise',
      'Must use time-based train/test splits, never random splits',
      'Walk-forward validation provides realistic performance estimates',
      'Keep models simple with strong regularization (dropout, L2, early stopping)',
      'Ensemble multiple models for robustness',
      'Account for transaction costs in backtests',
      'Paper trade before live deployment to verify performance',
      'Regular retraining essential as markets evolve',
      'Success is possible with 52-55% accuracy and good risk management',
    ],
  },
];

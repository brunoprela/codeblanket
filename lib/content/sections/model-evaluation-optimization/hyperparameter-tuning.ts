export const hyperparameterTuning = {
  title: 'Hyperparameter Tuning',
  content: `
# Hyperparameter Tuning

## Introduction

Hyperparameters are the configuration settings of machine learning algorithms that cannot be learned from the data. Unlike model parameters (which are learned during training, like weights in neural networks), hyperparameters must be set before training begins.

**Examples:**
- Learning rate in gradient descent
- Number of trees in random forest
- Regularization strength (α in Ridge/Lasso)
- Number of layers/neurons in neural networks
- Kernel type in SVM

**Why Tuning Matters**: Poor hyperparameter choices can lead to underfitting, overfitting, or slow convergence. Proper tuning is often the difference between a mediocre and excellent model.

## Manual Tuning vs Automated Search

\`\`\`python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import pandas as pd
import time
import matplotlib.pyplot as plt

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Hyperparameter Tuning")
print("="*70)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Task: Binary classification")

# Manual tuning example
print("\\n" + "="*70)
print("Manual Hyperparameter Tuning")
print("="*70)

manual_results = []

for n_estimators in [50, 100, 200]:
    for max_depth in [5, 10, None]:
        for min_samples_split in [2, 5]:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            val_score = model.score(X_train, y_train)
            
            manual_results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'accuracy': val_score
            })

df_manual = pd.DataFrame(manual_results).sort_values('accuracy', ascending=False)
print("\\nTop 5 configurations:")
print(df_manual.head().to_string(index=False))

print("\\n⚠️  Manual tuning limitations:")
print("  • Time-consuming and tedious")
print("  • Easy to miss optimal combinations")
print("  • Hard to explore large parameter spaces")
print("  • Requires domain knowledge")
\`\`\`

## Grid Search

Grid search exhaustively tries all combinations of specified hyperparameter values.

\`\`\`python
print("\\n" + "="*70)
print("Grid Search")
print("="*70)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Calculate total combinations
total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\\nParameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

print(f"\\nTotal combinations: {total_combinations}")
print(f"With 5-fold CV: {total_combinations * 5} model trainings")

# Perform grid search
print("\\nRunning Grid Search...")
start_time = time.time()

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print(f"\\n✓ Grid search completed in {grid_time:.2f} seconds")

# Best parameters
print(f"\\nBest parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\\nBest cross-validation score: {grid_search.best_score_:.4f}")

# Test on held-out set
test_score = grid_search.score(X_test, y_test)
print(f"Test set score: {test_score:.4f}")

# Analyze all results
cv_results = pd.DataFrame(grid_search.cv_results_)

print("\\nTop 10 configurations:")
top_configs = cv_results.nsmallest(10, 'rank_test_score')[
    ['param_n_estimators', 'param_max_depth', 'param_min_samples_split',
     'param_min_samples_leaf', 'mean_test_score', 'std_test_score']
]
print(top_configs.to_string(index=False))
\`\`\`

### Visualizing Grid Search Results

\`\`\`python
print("\\n" + "="*70)
print("Visualizing Grid Search Results")
print("="*70)

# Heatmap of 2D parameter interaction
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# n_estimators vs max_depth
pivot_data = cv_results.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_n_estimators',
    aggfunc='mean'
)

import seaborn as sns
sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0])
axes[0].set_title('Grid Search: n_estimators vs max_depth', fontsize=12, fontweight='bold')
axes[0].set_xlabel('n_estimators')
axes[0].set_ylabel('max_depth')

# min_samples_split vs min_samples_leaf
pivot_data2 = cv_results.pivot_table(
    values='mean_test_score',
    index='param_min_samples_split',
    columns='param_min_samples_leaf',
    aggfunc='mean'
)

sns.heatmap(pivot_data2, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1])
axes[1].set_title('Grid Search: min_samples_split vs min_samples_leaf', 
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('min_samples_leaf')
axes[1].set_ylabel('min_samples_split')

plt.tight_layout()
plt.savefig('grid_search_heatmap.png', dpi=150, bbox_inches='tight')
print("\\nHeatmap saved to 'grid_search_heatmap.png'")

# Score distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(cv_results['mean_test_score'], bins=30, edgecolor='black', alpha=0.7)
plt.axvline(grid_search.best_score_, color='red', linestyle='--', 
           linewidth=2, label=f'Best: {grid_search.best_score_:.4f}')
plt.xlabel('Mean CV Score')
plt.ylabel('Frequency')
plt.title('Distribution of CV Scores', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(cv_results['mean_fit_time'], cv_results['mean_test_score'], 
           alpha=0.6, edgecolors='black', linewidth=0.5)
plt.xlabel('Mean Fit Time (seconds)')
plt.ylabel('Mean CV Score')
plt.title('Score vs Training Time', fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('grid_search_analysis.png', dpi=150, bbox_inches='tight')
print("Analysis plots saved to 'grid_search_analysis.png'")
\`\`\`

## Random Search

Random search samples random combinations from the parameter space, often more efficient than grid search.

\`\`\`python
from scipy.stats import randint, uniform

print("\\n" + "="*70)
print("Random Search")
print("="*70)

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

print("Parameter distributions:")
for param, dist in param_distributions.items():
    print(f"  {param}: {dist}")

# Perform random search
n_iter = 100
print(f"\\nRunning Random Search with {n_iter} iterations...")
start_time = time.time()

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=n_iter,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
random_time = time.time() - start_time

print(f"\\n✓ Random search completed in {random_time:.2f} seconds")

# Compare with grid search
print("\\n" + "="*70)
print("Grid Search vs Random Search Comparison")
print("="*70)

comparison = pd.DataFrame({
    'Method': ['Grid Search', 'Random Search'],
    'Best Score': [grid_search.best_score_, random_search.best_score_],
    'Time (s)': [grid_time, random_time],
    'Iterations': [total_combinations, n_iter]
})

print("\\n" + comparison.to_string(index=False))

print(f"\\nRandom search:")
print(f"  • Explored {n_iter} configurations")
print(f"  • Found score of {random_search.best_score_:.4f}")
print(f"  • {grid_time/random_time:.1f}x faster than grid search")

# Best parameters from random search
print(f"\\nBest parameters (Random Search):")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

# Convergence plot
cv_results_random = pd.DataFrame(random_search.cv_results_)
scores_sorted = cv_results_random.sort_values('mean_test_score', ascending=False)
cumulative_best = scores_sorted['mean_test_score'].cummax()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(cumulative_best) + 1), cumulative_best.values, linewidth=2)
plt.xlabel('Number of Iterations')
plt.ylabel('Best Score Found')
plt.title('Random Search Convergence', fontweight='bold')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(range(len(cv_results_random)), cv_results_random['mean_test_score'],
           alpha=0.6, edgecolors='black', linewidth=0.5)
plt.axhline(y=random_search.best_score_, color='red', linestyle='--', 
           linewidth=2, label='Best score')
plt.xlabel('Iteration')
plt.ylabel('CV Score')
plt.title('Score Distribution Across Iterations', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('random_search_convergence.png', dpi=150, bbox_inches='tight')
print("\\nConvergence plot saved to 'random_search_convergence.png'")
\`\`\`

### Why Random Search Often Outperforms Grid Search

\`\`\`python
print("\\n" + "="*70)
print("Why Random Search Can Be Better")
print("="*70)

print("""
Key Advantages of Random Search:

1. **Better coverage with fewer iterations**
   Grid search: Tries every combination in discrete grid
   Random search: Samples from continuous/wider distributions

2. **More efficient for unimportant parameters**
   If parameter A is important and B isn't:
   - Grid search: Wastes time on all B values for each A
   - Random search: Explores more A values naturally

3. **Easier to parallelize and resume**
   Can add more iterations anytime without recomputing

4. **Works with continuous distributions**
   Can sample from continuous ranges (e.g., learning_rate ∈ [0.0001, 0.1])

5. **Diminishing returns**
   After exploring ~60-100 combinations, additional grid points add little value
""")

# Demonstrate with toy example
print("\\nToy Example: One important, one unimportant parameter")
print("-"*70)

# Simulate scenario
important_param_values = np.linspace(0, 1, 100)
unimportant_param_values = np.linspace(0, 1, 100)

# Performance only depends on important parameter
def performance(important, unimportant):
    return 1 - (important - 0.7)**2 - 0.01 * np.random.randn()

# Grid search (10x10 = 100 evaluations)
grid_important = np.linspace(0, 1, 10)
grid_unimportant = np.linspace(0, 1, 10)
grid_best = -np.inf
for imp in grid_important:
    for unimp in grid_unimportant:
        score = performance(imp, unimp)
        grid_best = max(grid_best, score)

# Random search (100 evaluations)
np.random.seed(42)
random_important = np.random.uniform(0, 1, 100)
random_unimportant = np.random.uniform(0, 1, 100)
random_best = max([performance(imp, unimp) 
                   for imp, unimp in zip(random_important, random_unimportant)])

print(f"Grid search best score: {grid_best:.4f}")
print(f"  Explored 10 values of important parameter")
print(f"Random search best score: {random_best:.4f}")
print(f"  Explored 100 values of important parameter")
print(f"\\nRandom search explored {100/10}x more values of important parameter!")
\`\`\`

## Bayesian Optimization

Bayesian optimization uses probabilistic models to guide the search, focusing on promising regions.

\`\`\`python
print("\\n" + "="*70)
print("Bayesian Optimization with Optuna")
print("="*70)

try:
    import optuna
    
    # Suppress optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Define objective function
    def objective(trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        
        # Create and evaluate model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation score
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        return scores.mean()
    
    # Run optimization
    print("\\nRunning Bayesian Optimization...")
    start_time = time.time()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    
    bayesian_time = time.time() - start_time
    
    print(f"\\n✓ Bayesian optimization completed in {bayesian_time:.2f} seconds")
    print(f"\\nBest value: {study.best_value:.4f}")
    print(f"\\nBest parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Optimization history
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    trials_df = study.trials_dataframe()
    plt.plot(trials_df['number'], trials_df['value'], 'o-', alpha=0.6)
    plt.plot(trials_df['number'], trials_df['value'].cummax(), 
            'r-', linewidth=2, label='Best so far')
    plt.xlabel('Trial')
    plt.ylabel('Objective Value (Accuracy)')
    plt.title('Bayesian Optimization History', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())
        values = list(importance.values())
        
        plt.barh(params, values)
        plt.xlabel('Importance')
        plt.title('Hyperparameter Importance', fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
    except:
        plt.text(0.5, 0.5, 'Importance analysis\\nnot available', 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('bayesian_optimization.png', dpi=150, bbox_inches='tight')
    print("\\nBayesian optimization plots saved to 'bayesian_optimization.png'")
    
    # Compare all methods
    print("\\n" + "="*70)
    print("Method Comparison Summary")
    print("="*70)
    
    comparison_full = pd.DataFrame({
        'Method': ['Grid Search', 'Random Search', 'Bayesian Optimization'],
        'Best Score': [grid_search.best_score_, random_search.best_score_, study.best_value],
        'Time (s)': [grid_time, random_time, bayesian_time],
        'Trials': [total_combinations * 5, n_iter * 5, 50 * 5]
    })
    
    print("\\n" + comparison_full.to_string(index=False))
    
except ImportError:
    print("\\n⚠️  Optuna not installed. Install with: pip install optuna")
    print("   Bayesian optimization provides:")
    print("   • Intelligent search guided by previous results")
    print("   • Often finds better parameters with fewer evaluations")
    print("   • Automatic parameter importance analysis")
\`\`\`

## Halving Search (Successive Halving)

Successive halving allocates more resources to promising configurations.

\`\`\`python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

print("\\n" + "="*70)
print("Successive Halving")
print("="*70)

print("""
Successive Halving Strategy:
1. Start with many candidates on small data samples
2. Eliminate worst performers
3. Give remaining candidates more data
4. Repeat until one winner emerges

Advantages:
• Much faster than standard grid/random search
• Quickly eliminates poor configurations
• Focuses resources on promising candidates
""")

# Halving Grid Search
print("\\nRunning Halving Grid Search...")
start_time = time.time()

halving_grid = HalvingGridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    factor=3,  # Eliminate 2/3 of candidates each iteration
    resource='n_samples',
    max_resources='auto',
    random_state=42,
    n_jobs=-1
)

halving_grid.fit(X_train, y_train)
halving_time = time.time() - start_time

print(f"\\n✓ Halving grid search completed in {halving_time:.2f} seconds")
print(f"Best score: {halving_grid.best_score_:.4f}")
print(f"\\nSpeedup vs standard grid search: {grid_time/halving_time:.1f}x")

# Show how candidates were eliminated
cv_results_halving = pd.DataFrame(halving_grid.cv_results_)

print(f"\\nSuccessive Halving Rounds:")
for iteration in sorted(cv_results_halving['iter'].unique()):
    iter_results = cv_results_halving[cv_results_halving['iter'] == iteration]
    n_candidates = len(iter_results)
    n_resources = iter_results['n_resources'].iloc[0]
    
    print(f"  Iteration {iteration}: {n_candidates} candidates, "
          f"{n_resources} samples each")
\`\`\`

## Learning Rate Scheduling (for Neural Networks)

\`\`\`python
print("\\n" + "="*70)
print("Learning Rate Scheduling")
print("="*70)

print("""
Learning Rate Schedules:

1. **Constant**: Same LR throughout training
   • Simple but may not converge well

2. **Step Decay**: Reduce LR by factor every N epochs
   • lr = lr0 * (drop_rate)^(epoch // drop_every)

3. **Exponential Decay**: Gradual exponential decrease
   • lr = lr0 * exp(-decay_rate * epoch)

4. **Cosine Annealing**: Smooth decrease following cosine curve
   • lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * epoch / max_epochs))

5. **Warm Restarts**: Periodic resets to high LR
   • Helps escape local minima

6. **Adaptive** (Adam, AdaGrad, RMSprop): Per-parameter adaptation
""")

# Demonstrate learning rate schedules
epochs = 100
lr_initial = 0.1

schedules = {
    'Constant': [lr_initial] * epochs,
    'Step Decay': [lr_initial * (0.5 ** (e // 30)) for e in range(epochs)],
    'Exponential': [lr_initial * np.exp(-0.05 * e) for e in range(epochs)],
    'Cosine Annealing': [0.001 + 0.5 * (lr_initial - 0.001) * 
                         (1 + np.cos(np.pi * e / epochs)) for e in range(epochs)],
}

plt.figure(figsize=(14, 6))

for name, schedule in schedules.items():
    plt.plot(schedule, label=name, linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedules', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('learning_rate_schedules.png', dpi=150, bbox_inches='tight')
print("\\nLearning rate schedules saved to 'learning_rate_schedules.png'")
\`\`\`

## Best Practices and Recommendations

\`\`\`python
print("\\n" + "="*70)
print("Hyperparameter Tuning Best Practices")
print("="*70)

practices = {
    'General': [
        '1. Always use separate validation set or cross-validation',
        '2. Start with coarse search, then refine around best values',
        '3. Use logarithmic scale for parameters spanning orders of magnitude',
        '4. Consider computational budget (time/resources)',
        '5. Document all experiments and results'
    ],
    'Search Strategy': [
        '• Small datasets (<10K): Grid search on narrow ranges',
        '• Medium datasets: Random search (100-200 iterations)',
        '• Large datasets: Bayesian optimization or halving search',
        '• Very large: Successive halving with early stopping'
    ],
    'Parameter Selection': [
        '• Focus on most impactful parameters first',
        '• Learning rate: Most important for neural networks',
        '• Regularization: Critical for preventing overfitting',
        '• Model capacity: Balance complexity with data size'
    ],
    'Validation': [
        '• Use same metric as final evaluation',
        '• Ensure validation set is representative',
        '• Watch for overfitting to validation set',
        '• Final evaluation MUST be on held-out test set'
    ]
}

for category, tips in practices.items():
    print(f"\\n{category}:")
    for tip in tips:
        print(f"  {tip}")
\`\`\`

## Common Hyperparameters by Model Type

\`\`\`python
print("\\n" + "="*70)
print("Common Hyperparameters by Model")
print("="*70)

hyperparams_guide = {
    'Random Forest': {
        'Critical': ['n_estimators', 'max_depth', 'min_samples_split'],
        'Important': ['min_samples_leaf', 'max_features'],
        'Fine-tuning': ['max_leaf_nodes', 'min_impurity_decrease'],
        'Typical ranges': {
            'n_estimators': '50-500',
            'max_depth': '5-50 or None',
            'min_samples_split': '2-20',
            'max_features': 'sqrt, log2, or None'
        }
    },
    'Gradient Boosting (XGBoost/LightGBM)': {
        'Critical': ['learning_rate', 'n_estimators', 'max_depth'],
        'Important': ['min_child_weight', 'subsample', 'colsample_bytree'],
        'Fine-tuning': ['gamma', 'reg_alpha', 'reg_lambda'],
        'Typical ranges': {
            'learning_rate': '0.001-0.3',
            'n_estimators': '100-1000',
            'max_depth': '3-10'
        }
    },
    'SVM': {
        'Critical': ['C', 'kernel', 'gamma (for RBF)'],
        'Important': ['degree (for poly)', 'coef0'],
        'Fine-tuning': ['class_weight', 'tol'],
        'Typical ranges': {
            'C': '0.1-100 (log scale)',
            'gamma': '0.0001-1 (log scale)'
        }
    },
    'Neural Networks': {
        'Critical': ['learning_rate', 'batch_size', 'architecture'],
        'Important': ['optimizer', 'activation', 'dropout'],
        'Fine-tuning': ['weight_decay', 'momentum', 'lr_schedule'],
        'Typical ranges': {
            'learning_rate': '0.0001-0.1 (log scale)',
            'batch_size': '16, 32, 64, 128, 256'
        }
    }
}

for model, params in hyperparams_guide.items():
    print(f"\\n{model}:")
    print(f"  Critical: {', '.join(params['Critical'])}")
    print(f"  Important: {', '.join(params['Important'])}")
    if 'Typical ranges' in params:
        print("  Typical ranges:")
        for param, range_val in params['Typical ranges'].items():
            print(f"    {param}: {range_val}")
\`\`\`

## Trading Application: Strategy Optimization

\`\`\`python
print("\\n" + "="*70)
print("Trading Application: Strategy Hyperparameter Tuning")
print("="*70)

# Simulate trading strategy with hyperparameters
np.random.seed(42)
n_days = 1000

# Generate price data
returns = np.random.randn(n_days) * 0.015
prices = 100 * np.exp(np.cumsum(returns))

def moving_average_strategy(prices, short_window, long_window, stop_loss_pct):
    """
    Simple MA crossover strategy with stop loss.
    Returns: total return percentage.
    """
    short_ma = pd.Series(prices).rolling(short_window).mean()
    long_ma = pd.Series(prices).rolling(long_window).mean()
    
    position = 0  # 0=no position, 1=long
    entry_price = 0
    returns = []
    
    for i in range(max(short_window, long_window), len(prices)):
        # Entry signal: short MA crosses above long MA
        if position == 0 and short_ma.iloc[i] > long_ma.iloc[i]:
            position = 1
            entry_price = prices[i]
        
        # Exit signals
        elif position == 1:
            # Stop loss
            if prices[i] < entry_price * (1 - stop_loss_pct / 100):
                returns.append((prices[i] - entry_price) / entry_price)
                position = 0
            # Exit signal: short MA crosses below long MA
            elif short_ma.iloc[i] < long_ma.iloc[i]:
                returns.append((prices[i] - entry_price) / entry_price)
                position = 0
    
    # Close any open position
    if position == 1:
        returns.append((prices[-1] - entry_price) / entry_price)
    
    total_return = np.sum(returns) * 100 if returns else 0
    return total_return

# Define parameter space
param_space = {
    'short_window': [5, 10, 15, 20],
    'long_window': [20, 30, 50, 100],
    'stop_loss_pct': [1, 2, 5, 10]
}

print("\\nTuning trading strategy parameters...")
print(f"Parameter space:")
for param, values in param_space.items():
    print(f"  {param}: {values}")

# Grid search for trading strategy
best_return = -np.inf
best_params = None
all_results = []

for short in param_space['short_window']:
    for long in param_space['long_window']:
        if short >= long:  # Invalid: short window must be < long window
            continue
        for stop in param_space['stop_loss_pct']:
            total_return = moving_average_strategy(prices, short, long, stop)
            
            all_results.append({
                'short_window': short,
                'long_window': long,
                'stop_loss_pct': stop,
                'return_%': total_return
            })
            
            if total_return > best_return:
                best_return = total_return
                best_params = {
                    'short_window': short,
                    'long_window': long,
                    'stop_loss_pct': stop
                }

print(f"\\nBest strategy parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"\\nExpected return: {best_return:.2f}%")

# Show top 5
df_trading = pd.DataFrame(all_results).sort_values('return_%', ascending=False)
print("\\nTop 5 configurations:")
print(df_trading.head().to_string(index=False))

print("\\n⚠️  Trading Strategy Tuning Warnings:")
print("  • Risk of overfitting to historical data")
print("  • Must validate on out-of-sample period")
print("  • Consider transaction costs")
print("  • Market conditions change")
print("  • Use walk-forward optimization")
\`\`\`

## Key Takeaways

1. **Hyperparameters vs Parameters**: Set before training vs learned during training
2. **Grid Search**: Exhaustive but expensive
3. **Random Search**: Often better with same budget
4. **Bayesian Optimization**: Intelligent, guided search
5. **Successive Halving**: Fast elimination of poor candidates
6. **Always validate properly**: Use CV or separate validation set
7. **Start coarse, then refine**: Don't jump to fine-tuning
8. **Watch for overfitting**: To both training and validation data
9. **Document everything**: Track all experiments
10. **Test set is sacred**: Never tune on test set

**Practical Workflow:**
1. Start with default parameters
2. Identify most important parameters
3. Coarse search with random/Bayesian
4. Fine-tune around best values
5. Validate on held-out test set

## Further Reading

- Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization"
- Snoek, J., et al. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms"
- Li, L., et al. (2017). "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
`,
  exercises: [
    {
      prompt:
        'Create a comprehensive hyperparameter tuning framework that compares grid search, random search, and Bayesian optimization on the same problem, tracks all experiments, and provides detailed analysis and recommendations.',
      solution: `# Comprehensive solution in full evaluation framework`,
    },
  ],
  quizId: 'model-evaluation-optimization-hyperparameter-tuning',
  multipleChoiceId: 'model-evaluation-optimization-hyperparameter-tuning-mc',
};

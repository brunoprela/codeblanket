import { Content } from '@/lib/types';

const strategyParameterOptimization: Content = {
  title: 'Strategy Parameter Optimization',
  description:
    'Master parameter optimization techniques including grid search, Bayesian optimization, genetic algorithms, and preventing overfitting during the optimization process',
  sections: [
    {
      title: 'The Parameter Optimization Challenge',
      content: `
# Strategy Parameter Optimization

Every trading strategy has parameters: moving average periods, position sizes, stop-loss levels, rebalancing frequencies. Finding optimal parameters is critical—but dangerous if done incorrectly.

## The Overfitting Trap

**Case Study - The Optimized Failure**: A researcher optimized a momentum strategy across 5 parameters. After testing 10,000 parameter combinations, they found the "perfect" set: Sharpe 3.2 in backtest. In paper trading: Sharpe 0.3.

**What happened?** The optimization process curve-fit to historical noise. The parameters that worked best in-sample were highly specific to that particular data sample and didn't generalize.

## Parameter Optimization Methods

### 1. Grid Search (Exhaustive)

\`\`\`python
from typing import Dict, List, Iterator, Tuple
from itertools import product
import numpy as np
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import time

@dataclass
class ParameterSet:
    """A set of strategy parameters"""
    params: Dict[str, any]
    sharpe: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0

class GridSearchOptimizer:
    """
    Exhaustive grid search over parameter space
    
    Pros: Guaranteed to find best combination in grid
    Cons: Computationally expensive, doesn't scale beyond ~4-5 parameters
    """
    
    def __init__(
        self,
        backtest_func: callable,
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        metric: str = 'sharpe',
        n_jobs: int = 4
    ):
        """
        Initialize grid search
        
        Args:
            backtest_func: Function that takes data and params, returns metrics
            data: Historical data
            param_grid: Dictionary of parameter_name -> list of values
            metric: Metric to optimize ('sharpe', 'total_return', etc.)
            n_jobs: Number of parallel workers
        """
        self.backtest_func = backtest_func
        self.data = data
        self.param_grid = param_grid
        self.metric = metric
        self.n_jobs = n_jobs
        
        # Calculate search space size
        self.search_space_size = np.prod([len(v) for v in param_grid.values()])
        
        print(f"\\nGrid Search Optimizer initialized")
        print(f"Parameters: {list(param_grid.keys())}")
        print(f"Search space size: {self.search_space_size:,} combinations")
        print(f"Estimated time: ~{self.search_space_size / (n_jobs * 60):.1f} minutes\\n")
    
    def generate_parameter_combinations(self) -> Iterator[Dict]:
        """Generate all parameter combinations"""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        for combination in product(*values):
            yield dict(zip(keys, combination))
    
    def optimize(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Run grid search optimization
        
        Returns:
            (best_params, results_dataframe)
        """
        print(f"Starting grid search over {self.search_space_size:,} combinations...")
        start_time = time.time()
        
        # Generate all combinations
        param_combinations = list(self.generate_parameter_combinations())
        
        # Run backtests in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(
                self._run_single_backtest,
                param_combinations
            ))
        
        elapsed = time.time() - start_time
        
        # Create results DataFrame
        results_df = pd.DataFrame([
            {**r.params, self.metric: getattr(r, self.metric)}
            for r in results
        ])
        
        # Find best parameters
        best_idx = results_df[self.metric].idxmax()
        best_params = {
            k: results_df.loc[best_idx, k]
            for k in self.param_grid.keys()
        }
        best_score = results_df.loc[best_idx, self.metric]
        
        print(f"\\nGrid search complete!")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Best {self.metric}: {best_score:.3f}")
        print(f"Best parameters: {best_params}")
        
        return best_params, results_df
    
    def _run_single_backtest(self, params: Dict) -> ParameterSet:
        """Run single backtest with given parameters"""
        try:
            metrics = self.backtest_func(self.data, params)
            return ParameterSet(
                params=params,
                sharpe=metrics.get('sharpe', 0),
                total_return=metrics.get('total_return', 0),
                max_drawdown=metrics.get('max_drawdown', 0),
                num_trades=metrics.get('num_trades', 0)
            )
        except Exception as e:
            # Return zero metrics on failure
            return ParameterSet(params=params)
    
    def visualize_results(self, results_df: pd.DataFrame):
        """Visualize optimization results"""
        import matplotlib.pyplot as plt
        
        # For 2D parameter space, create heatmap
        if len(self.param_grid) == 2:
            param_names = list(self.param_grid.keys())
            pivot = results_df.pivot(
                index=param_names[0],
                columns=param_names[1],
                values=self.metric
            )
            
            plt.figure(figsize=(10, 8))
            plt.imshow(pivot.values, aspect='auto', cmap='RdYlGn')
            plt.colorbar(label=self.metric)
            plt.xlabel(param_names[1])
            plt.ylabel(param_names[0])
            plt.title(f'Parameter Optimization: {self.metric}')
            plt.tight_layout()
            plt.savefig('grid_search_results.png', dpi=300)
            plt.close()


### 2. Bayesian Optimization (Smart Search)

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class BayesianOptimizer:
    """
    Bayesian optimization using Gaussian Processes
    
    Pros: Much more efficient than grid search, explores intelligently
    Cons: More complex, can get stuck in local optima
    """
    
    def __init__(
        self,
        backtest_func: callable,
        data: pd.DataFrame,
        param_bounds: Dict[str, Tuple[float, float]],
        n_iterations: int = 50,
        n_initial_points: int = 10
    ):
        """
        Initialize Bayesian optimizer
        
        Args:
            backtest_func: Function that takes data and params, returns metrics
            data: Historical data
            param_bounds: Dict of param_name -> (min, max)
            n_iterations: Number of optimization iterations
            n_initial_points: Random points to sample initially
        """
        self.backtest_func = backtest_func
        self.data = data
        self.param_bounds = param_bounds
        self.n_iterations = n_iterations
        self.n_initial_points = n_initial_points
        
        # History
        self.X_observed = []  # Parameter sets tried
        self.y_observed = []  # Corresponding scores
        
        # Gaussian Process
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
    
    def optimize(self) -> Dict:
        """
        Run Bayesian optimization
        
        Returns:
            Best parameters found
        """
        print(f"\\nBayesian Optimization")
        print(f"Iterations: {self.n_iterations}")
        print(f"Initial random samples: {self.n_initial_points}\\n")
        
        # Phase 1: Random exploration
        print("Phase 1: Random exploration...")
        for i in range(self.n_initial_points):
            params = self._random_sample()
            score = self._evaluate(params)
            self.X_observed.append(params)
            self.y_observed.append(score)
            print(f"  {i+1}/{self.n_initial_points}: {score:.3f}")
        
        # Phase 2: Guided optimization
        print("\\nPhase 2: Bayesian optimization...")
        for i in range(self.n_iterations - self.n_initial_points):
            # Fit GP to observed data
            self.gp.fit(self.X_observed, self.y_observed)
            
            # Find next point to sample using acquisition function
            next_params = self._acquisition_function()
            score = self._evaluate(next_params)
            
            self.X_observed.append(next_params)
            self.y_observed.append(score)
            
            current_best = max(self.y_observed)
            print(f"  {i+1}/{self.n_iterations - self.n_initial_points}: "
                  f"{score:.3f} (best: {current_best:.3f})")
        
        # Return best parameters
        best_idx = np.argmax(self.y_observed)
        best_params_array = self.X_observed[best_idx]
        
        best_params = {
            name: best_params_array[i]
            for i, name in enumerate(self.param_bounds.keys())
        }
        
        print(f"\\nOptimization complete!")
        print(f"Best score: {self.y_observed[best_idx]:.3f}")
        print(f"Best parameters: {best_params}")
        
        return best_params
    
    def _random_sample(self) -> np.ndarray:
        """Sample random point from parameter space"""
        return np.array([
            np.random.uniform(bounds[0], bounds[1])
            for bounds in self.param_bounds.values()
        ])
    
    def _evaluate(self, params_array: np.ndarray) -> float:
        """Evaluate parameters"""
        params = {
            name: params_array[i]
            for i, name in enumerate(self.param_bounds.keys())
        }
        
        metrics = self.backtest_func(self.data, params)
        return metrics.get('sharpe', 0)
    
    def _acquisition_function(self) -> np.ndarray:
        """
        Expected Improvement acquisition function
        
        Balances exploration (high uncertainty) vs exploitation (high predicted value)
        """
        # Sample many random points
        n_samples = 1000
        X_random = np.array([self._random_sample() for _ in range(n_samples)])
        
        # Get GP predictions
        mu, sigma = self.gp.predict(X_random, return_std=True)
        
        # Calculate Expected Improvement
        current_best = max(self.y_observed)
        
        with np.errstate(divide='ignore'):
            Z = (mu - current_best) / sigma
            ei = (mu - current_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        # Return point with highest EI
        best_idx = np.argmax(ei)
        return X_random[best_idx]


### 3. Genetic Algorithm (Evolutionary Optimization)

class GeneticAlgorithmOptimizer:
    """
    Genetic algorithm for parameter optimization
    
    Mimics biological evolution: selection, crossover, mutation
    """
    
    def __init__(
        self,
        backtest_func: callable,
        data: pd.DataFrame,
        param_bounds: Dict[str, Tuple[float, float]],
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ):
        self.backtest_func = backtest_func
        self.data = data
        self.param_bounds = param_bounds
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.param_names = list(param_bounds.keys())
        self.n_params = len(self.param_names)
    
    def optimize(self) -> Dict:
        """Run genetic algorithm"""
        print(f"\\nGenetic Algorithm Optimization")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.n_generations}\\n")
        
        # Initialize population
        population = self._initialize_population()
        fitness_scores = self._evaluate_population(population)
        
        best_fitness_history = []
        
        for generation in range(self.n_generations):
            # Selection
            parents = self._selection(population, fitness_scores)
            
            # Crossover
            offspring = self._crossover(parents)
            
            # Mutation
            offspring = self._mutation(offspring)
            
            # Evaluate offspring
            offspring_fitness = self._evaluate_population(offspring)
            
            # Combine and select next generation
            combined_pop = np.vstack([population, offspring])
            combined_fitness = np.concatenate([fitness_scores, offspring_fitness])
            
            # Keep best individuals
            best_indices = np.argsort(combined_fitness)[-self.population_size:]
            population = combined_pop[best_indices]
            fitness_scores = combined_fitness[best_indices]
            
            best_fitness = fitness_scores[-1]
            best_fitness_history.append(best_fitness)
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.3f}")
        
        # Return best individual
        best_individual = population[-1]
        best_params = {
            name: best_individual[i]
            for i, name in enumerate(self.param_names)
        }
        
        print(f"\\nOptimization complete!")
        print(f"Best fitness: {fitness_scores[-1]:.3f}")
        print(f"Best parameters: {best_params}")
        
        return best_params
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = [
                np.random.uniform(bounds[0], bounds[1])
                for bounds in self.param_bounds.values()
            ]
            population.append(individual)
        return np.array(population)
    
    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness of entire population"""
        fitness = []
        for individual in population:
            params = {
                name: individual[i]
                for i, name in enumerate(self.param_names)
            }
            metrics = self.backtest_func(self.data, params)
            fitness.append(metrics.get('sharpe', 0))
        return np.array(fitness)
    
    def _selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Tournament selection"""
        n_parents = self.population_size // 2
        parents = []
        
        for _ in range(n_parents):
            # Tournament: pick 3 random, select best
            tournament_idx = np.random.choice(len(population), 3, replace=False)
            winner_idx = tournament_idx[np.argmax(fitness[tournament_idx])]
            parents.append(population[winner_idx])
        
        return np.array(parents)
    
    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """Single-point crossover"""
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            
            if np.random.random() < self.crossover_rate:
                # Crossover
                crossover_point = np.random.randint(1, self.n_params)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            else:
                # No crossover
                child1, child2 = parent1.copy(), parent2.copy()
            
            offspring.extend([child1, child2])
        
        return np.array(offspring)
    
    def _mutation(self, offspring: np.ndarray) -> np.ndarray:
        """Gaussian mutation"""
        for i in range(len(offspring)):
            for j in range(self.n_params):
                if np.random.random() < self.mutation_rate:
                    # Mutate with Gaussian noise
                    bounds = list(self.param_bounds.values())[j]
                    mutation = np.random.normal(0, (bounds[1] - bounds[0]) * 0.1)
                    offspring[i][j] += mutation
                    # Clip to bounds
                    offspring[i][j] = np.clip(offspring[i][j], bounds[0], bounds[1])
        
        return offspring


## Preventing Overfitting During Optimization

# Critical: Use walk-forward analysis during optimization

class WalkForwardOptimizer:
    """
    Walk-forward optimization to prevent overfitting
    
    Optimize on training window, test on OOS window, roll forward
    """
    
    def __init__(
        self,
        optimizer: 'Optimizer',
        data: pd.DataFrame,
        train_size: int = 504,  # 2 years
        test_size: int = 126,   # 6 months
        step_size: int = 63     # 3 months
    ):
        self.optimizer = optimizer
        self.data = data
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
    
    def optimize(self) -> Dict:
        """Run walk-forward optimization"""
        results = []
        
        n = len(self.data)
        start_idx = 0
        
        while start_idx + self.train_size + self.test_size <= n:
            # Split data
            train_end = start_idx + self.train_size
            test_end = train_end + self.test_size
            
            train_data = self.data.iloc[start_idx:train_end]
            test_data = self.data.iloc[train_end:test_end]
            
            # Optimize on training data
            self.optimizer.data = train_data
            best_params = self.optimizer.optimize()
            
            # Test on OOS data
            oos_metrics = self.optimizer.backtest_func(test_data, best_params)
            
            results.append({
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'best_params': best_params,
                'oos_sharpe': oos_metrics['sharpe']
            })
            
            # Roll forward
            start_idx += self.step_size
        
        # Average best parameters across periods
        return self._aggregate_parameters(results)
    
    def _aggregate_parameters(self, results: List[Dict]) -> Dict:
        """Aggregate parameters across walk-forward periods"""
        # Simple: take median of each parameter
        all_params = [r['best_params'] for r in results]
        param_names = all_params[0].keys()
        
        aggregated = {}
        for param_name in param_names:
            values = [p[param_name] for p in all_params]
            aggregated[param_name] = np.median(values)
        
        return aggregated


# Example usage
if __name__ == "__main__":
    # Sample backtest function
    def simple_backtest(data: pd.DataFrame, params: Dict) -> Dict:
        fast_period = int(params['fast_period'])
        slow_period = int(params['slow_period'])
        
        # Simple MA crossover
        fast_ma = data['close'].rolling(fast_period).mean()
        slow_ma = data['close'].rolling(slow_period).mean()
        
        signal = (fast_ma > slow_ma).astype(int)
        returns = data['close'].pct_change() * signal.shift(1)
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'sharpe': sharpe,
            'total_return': returns.sum(),
            'max_drawdown': (returns.cumsum() - returns.cumsum().cummax()).min()
        }
    
    # Generate sample data
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    }, index=dates)
    
    # Grid search
    grid_optimizer = GridSearchOptimizer(
        backtest_func=simple_backtest,
        data=data,
        param_grid={
            'fast_period': [10, 20, 30, 50],
            'slow_period': [100, 150, 200]
        },
        n_jobs=4
    )
    
    best_params, results = grid_optimizer.optimize()
\`\`\`

## Key Principles

1. **Use Walk-Forward**: Optimize on training, validate on OOS
2. **Penalize Complexity**: Prefer simpler parameter sets
3. **Average Parameters**: Use median across multiple periods
4. **Test Stability**: Small parameter changes shouldn't drastically change performance
5. **Multiple Metrics**: Don't optimize on Sharpe alone

## Production Checklist

- [ ] Walk-forward optimization implemented
- [ ] Out-of-sample testing after optimization
- [ ] Parameter stability analysis
- [ ] Multiple metric evaluation
- [ ] Overfitting safeguards in place
- [ ] Results documented and reviewed
`,
    },
  ],
};

export default strategyParameterOptimization;

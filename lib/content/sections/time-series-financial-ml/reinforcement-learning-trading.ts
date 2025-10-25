export const reinforcementLearningTrading = {
  title: 'Reinforcement Learning for Trading',
  id: 'reinforcement-learning-trading',
  content: `
# Reinforcement Learning for Trading

## Introduction

Reinforcement Learning (RL) offers a fundamentally different approach to trading: instead of predicting prices, agents learn optimal trading policies through trial-and-error interaction with markets.

**Why RL for Trading?**
- **Sequential Decision-Making**: Trading is inherently sequential (today's action affects tomorrow's state)
- **Delayed Rewards**: Profit/loss realized over time, not immediately
- **Complex Dynamics**: Non-linear relationships, regime changes
- **Exploration**: Discovers non-obvious strategies
- **Adaptive**: Learns from market feedback

**Challenges**:
- **Sample Inefficiency**: Needs lots of data/experience
- **Reward Engineering**: Hard to define "good" trading
- **Non-Stationarity**: Markets change, policies decay
- **Risk**: Exploration can be expensive
- **Overfitting**: Easy to overfit to historical data

**RL vs Supervised Learning**:
- **Supervised**: "What will price be?" → Predict
- **RL**: "What should I do?" → Act optimally

---

## Trading as MDP (Markov Decision Process)

\`\`\`python
"""
Formulate trading as RL problem
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict

class TradingEnvironment (gym.Env):
    """
    OpenAI Gym-compatible trading environment
    
    Components:
    - State: Market observations + portfolio state
    - Action: Buy / Sell / Hold (or continuous position sizing)
    - Reward: Return, Sharpe, risk-adjusted metrics
    - Transition: Market dynamics
    """
    
    def __init__(self, data: pd.DataFrame, 
                 initial_balance: float = 100000,
                 commission: float = 0.001,
                 max_position: float = 1.0):
        """
        Args:
            data: OHLCV + features DataFrame
            initial_balance: Starting capital
            commission: Transaction cost (0.001 = 0.1%)
            max_position: Max position as fraction of capital
        """
        super().__init__()
        
        self.data = data.reset_index (drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_position = max_position
        
        # State space: [price features, position, cash, unrealized P&L, ...]
        # Observation: normalized market data + portfolio state
        n_features = len (data.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_features + 4,),  # +4 for portfolio state
            dtype=np.float32
        )
        
        # Action space: 
        # Discrete: 0=sell, 1=hold, 2=buy
        # Continuous: -1 to 1 (position sizing)
        self.discrete_actions = True
        if self.discrete_actions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        
        self.reset()
    
    def reset (self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # Number of shares
        self.entry_price = 0
        self.total_profit = 0
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        
        return self._get_observation()
    
    def _get_observation (self) -> np.ndarray:
        """
        Construct state observation
        
        State includes:
        - Market features (price, volume, indicators)
        - Portfolio state (position, cash, P&L)
        """
        if self.current_step >= len (self.data):
            return np.zeros (self.observation_space.shape)
        
        # Market features (normalized)
        market_obs = self.data.iloc[self.current_step].values
        
        # Portfolio state
        current_price = self.data.iloc[self.current_step]['Close']
        position_value = self.position * current_price
        total_value = self.balance + position_value
        
        portfolio_obs = np.array([
            self.position / self.max_position if self.max_position > 0 else 0,  # Normalized position
            self.balance / self.initial_balance,  # Normalized cash
            position_value / self.initial_balance if self.initial_balance > 0 else 0,  # Position value
            (total_value - self.initial_balance) / self.initial_balance  # Total P&L %
        ])
        
        # Combine
        obs = np.concatenate([market_obs, portfolio_obs]).astype (np.float32)
        
        return obs
    
    def step (self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (observation, reward, done, info)
        
        Args:
            action: 0=sell, 1=hold, 2=buy (discrete)
                   or -1 to 1 (continuous position target)
        
        Returns:
            observation: Next state
            reward: Immediate reward
            done: Episode finished?
            info: Additional information
        """
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Execute action
        reward = 0
        
        if self.discrete_actions:
            if action == 2:  # Buy
                self._execute_buy (current_price)
            elif action == 0:  # Sell
                reward = self._execute_sell (current_price)
            # action == 1: Hold (do nothing)
        else:
            # Continuous action: target position
            target_position = action * self.max_position * self.initial_balance / current_price
            self._adjust_position (target_position, current_price)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len (self.data) - 1
        
        # Calculate portfolio value
        if self.current_step < len (self.data):
            current_price = self.data.iloc[self.current_step]['Close']
        position_value = self.position * current_price
        portfolio_value = self.balance + position_value
        self.portfolio_values.append (portfolio_value)
        
        # Reward: Change in portfolio value (can customize)
        if len (self.portfolio_values) > 1:
            reward = (self.portfolio_values[-1] - self.portfolio_values[-2]) / self.portfolio_values[-2]
        
        # Get next observation
        obs = self._get_observation()
        
        # Info
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'total_profit': portfolio_value - self.initial_balance,
            'num_trades': len (self.trades)
        }
        
        return obs, reward, done, info
    
    def _execute_buy (self, price: float):
        """Execute buy order"""
        if self.balance <= 0 or self.position > 0:
            return  # Already in position or no cash
        
        # Buy max shares with available balance
        max_shares = (self.balance / price) * (1 - self.commission)
        
        if max_shares > 0:
            self.position = max_shares
            self.entry_price = price
            self.balance = 0
            
            self.trades.append({
                'step': self.current_step,
                'action': 'buy',
                'price': price,
                'shares': max_shares
            })
    
    def _execute_sell (self, price: float) -> float:
        """Execute sell order and return profit"""
        if self.position <= 0:
            return 0  # No position to sell
        
        # Sell all shares
        proceeds = self.position * price * (1 - self.commission)
        profit = proceeds - (self.position * self.entry_price)
        
        self.balance = proceeds
        self.total_profit += profit
        
        self.trades.append({
            'step': self.current_step,
            'action': 'sell',
            'price': price,
            'shares': self.position,
            'profit': profit
        })
        
        self.position = 0
        self.entry_price = 0
        
        # Return normalized profit as reward
        return profit / self.initial_balance
    
    def _adjust_position (self, target_shares: float, price: float):
        """Adjust position to target (continuous action)"""
        delta = target_shares - self.position
        
        if delta > 0:  # Buy
            cost = delta * price * (1 + self.commission)
            if cost <= self.balance:
                self.position += delta
                self.balance -= cost
        elif delta < 0:  # Sell
            proceeds = abs (delta) * price * (1 - self.commission)
            self.position += delta  # delta is negative
            self.balance += proceeds
    
    def render (self, mode='human'):
        """Render current state"""
        current_price = self.data.iloc[self.current_step]['Close']
        portfolio_value = self.balance + self.position * current_price
        
        print(f"Step: {self.current_step}")
        print(f"  Price: \${current_price:.2f}")
        print(f"  Position: {self.position:.2f} shares")
        print(f"  Balance: \${self.balance:.2f}")
        print(f"  Portfolio Value: \${portfolio_value:.2f}")
        print(f"  P&L: \${portfolio_value - self.initial_balance:.2f}")


# ============================================================================
# EXAMPLE: CREATE TRADING ENVIRONMENT
# ============================================================================

import yfinance as yf

# Download data
data = yf.download('SPY', start='2020-01-01', end='2024-01-01')

# Add features
data['Returns'] = data['Close'].pct_change()
data['SMA_20'] = data['Close'].rolling(20).mean()
data['SMA_50'] = data['Close'].rolling(50).mean()
data['RSI'] = calculate_rsi (data['Close'], 14)
data = data.dropna()

# Create environment
env = TradingEnvironment(
    data=data[['Close', 'Volume', 'Returns', 'SMA_20', 'SMA_50', 'RSI']],
    initial_balance=100000,
    commission=0.001
)

# Test environment
obs = env.reset()
print("\\nInitial observation shape:", obs.shape)

for _ in range(5):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step (action)
    print(f"Action: {action}, Reward: {reward:.4f}, Portfolio: \${info['portfolio_value']:,.2f}")
    if done:
        break
\`\`\`

---

## Deep Q-Network (DQN) Agent

\`\`\`python
"""
Complete DQN implementation for trading
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    """
    Deep Q-Network architecture
    
    Maps state → Q-values for each action
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear (input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append (nn.Linear (input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward (self, x):
        return self.network (x)


class DQNAgent:
    """
    DQN trading agent with experience replay and target network
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            memory_size: Experience replay buffer size
            batch_size: Training batch size
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Experience replay memory
        self.memory = deque (maxlen=memory_size)
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict (self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam (self.policy_net.parameters(), lr=learning_rate)
        
        # Training stats
        self.losses = []
    
    def act (self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state observation
            training: If True, use epsilon-greedy; else greedy
        
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return random.randrange (self.action_dim)
        
        # Exploit: best action according to Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor (state).unsqueeze(0)
            q_values = self.policy_net (state_tensor)
            return q_values.argmax().item()
    
    def remember (self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay (self):
        """
        Train on batch from experience replay
        
        Uses Double DQN to reduce overestimation bias
        """
        if len (self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample (self.memory, self.batch_size)
        
        states = torch.FloatTensor (np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor (np.array([e[3] for e in batch]))
        dones = torch.FloatTensor([e[4] for e in batch])
        
        # Current Q-values
        current_q_values = self.policy_net (states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net (next_states).argmax(1)
            next_q_values = self.target_net (next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append (loss.item())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network (self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict (self.policy_net.state_dict())
    
    def save (self, path: str):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load (self, path: str):
        """Load model"""
        checkpoint = torch.load (path)
        self.policy_net.load_state_dict (checkpoint['policy_net'])
        self.target_net.load_state_dict (checkpoint['target_net'])
        self.optimizer.load_state_dict (checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_dqn_agent (env, agent, num_episodes=500, update_target_every=10):
    """
    Train DQN agent on trading environment
    
    Args:
        env: Trading environment
        agent: DQN agent
        num_episodes: Number of training episodes
        update_target_every: Update target network every N episodes
    
    Returns:
        Training history
    """
    episode_rewards = []
    episode_profits = []
    
    for episode in range (num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select and perform action
            action = agent.act (state, training=True)
            next_state, reward, done, info = env.step (action)
            
            # Store experience
            agent.remember (state, action, reward, next_state, done)
            
            # Train
            agent.replay()
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append (total_reward)
        episode_profits.append (info['total_profit'])
        
        # Update target network
        if (episode + 1) % update_target_every == 0:
            agent.update_target_network()
        
        # Log progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean (episode_rewards[-50:])
            avg_profit = np.mean (episode_profits[-50:])
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (50 ep): {avg_reward:.4f}")
            print(f"  Avg Profit (50 ep): \${avg_profit:,.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Replay Memory: {len (agent.memory)}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_profits': episode_profits,
        'losses': agent.losses
    }


# Train agent
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_decay=0.995
)

print("\\n" + "="*70)
print("TRAINING DQN AGENT")
print("="*70)

history = train_dqn_agent (env, agent, num_episodes=200, update_target_every=10)

print("\\nTraining complete!")
print(f"Final epsilon: {agent.epsilon:.3f}")
print(f"Total experiences: {len (agent.memory)}")

# Evaluate trained agent
state = env.reset()
done = False
portfolio_values = [env.initial_balance]

while not done:
    action = agent.act (state, training=False)  # Greedy policy
    state, reward, done, info = env.step (action)
    portfolio_values.append (info['portfolio_value'])

final_return = (portfolio_values[-1] - env.initial_balance) / env.initial_balance

print(f"\\nEvaluation Results:")
print(f"  Final Portfolio Value: \${portfolio_values[-1]:,.2f}")
print(f"  Total Return: {final_return:.2%}")
print(f"  Number of Trades: {len (env.trades)}")
\`\`\`

---

## Policy Gradient Methods (PPO)

\`\`\`python
"""
Proximal Policy Optimization for continuous action spaces
"""

# Using stable-baselines3 (production-ready RL library)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Wrap environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Create PPO agent
    ppo_agent = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # Train
    print("\\nTraining PPO Agent...")
    ppo_agent.learn (total_timesteps=100000)
    
    # Evaluate
    obs = vec_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = ppo_agent.predict (obs, deterministic=True)
        obs, reward, done, info = vec_env.step (action)
        total_reward += reward[0]
    
    print(f"\\nPPO Evaluation:")
    print(f"  Total Reward: {total_reward:.4f}")
    
except ImportError:
    print("Install stable-baselines3: pip install stable-baselines3")
\`\`\`

---

## Key Takeaways

**RL Advantages for Trading**:
- **Sequential**: Natural fit for trading (states → actions → rewards)
- **Complex Policies**: Can learn non-linear, adaptive strategies
- **No Labels**: Doesn't need price predictions
- **Risk-Aware**: Can optimize for Sharpe, not just returns

**Challenges**:
- **Sample Efficiency**: Needs lots of data
- **Exploration**: Can lose money learning
- **Non-Stationary**: Markets change, policies decay
- **Reward Engineering**: Hard to define "good"
- **Overfitting**: Easy to overfit to training environment

**Algorithms**:
- **DQN**: Discrete actions, off-policy, experience replay
- **PPO**: Continuous actions, on-policy, stable
- **SAC**: Continuous, off-policy, entropy regularization
- **A3C**: Parallel actors, advantage estimation

**Best Practices**:
1. **Start Simple**: Basic DQN before complex algorithms
2. **Reward Shaping**: Carefully design rewards (Sharpe, not just P&L)
3. **Risk Constraints**: Incorporate stop-losses in environment
4. **Combine Approaches**: RL + supervised learning
5. **Paper Trade First**: Validate in simulation
6. **Monitor Distribution Shift**: Market regime changes

**Practical Tips**:
- Use pre-trained models (transfer learning)
- Ensemble RL with rule-based strategies
- Add domain knowledge as constraints
- Use RL for position sizing, not entry/exit
- Continuous retraining essential

**Remember**: RL is powerful but requires careful engineering. Most successful applications combine RL with traditional methods rather than pure RL.
`,
};

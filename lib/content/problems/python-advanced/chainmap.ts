/**
 * ChainMap for Layered Dictionaries
 * Problem ID: advanced-chainmap
 * Order: 35
 */

import { Problem } from '../../../types';

export const chainmapProblem: Problem = {
  id: 'advanced-chainmap',
  title: 'ChainMap for Layered Dictionaries',
  difficulty: 'Medium',
  description: `Use collections.ChainMap to manage multiple dictionaries as a single view.

Use ChainMap for:
- Configuration layers (defaults, user, environment)
- Scope chains (local, global)
- Template contexts
- Fallback lookups

**Benefit:** Efficient layered lookups without copying dictionaries.`,
  examples: [
    {
      input: 'ChainMap(user_config, default_config)',
      output: 'User config with defaults as fallback',
    },
  ],
  constraints: [
    'Use collections.ChainMap',
    'Understand lookup order',
    'First dict has priority',
  ],
  hints: [
    'ChainMap(*dicts) creates layered view',
    'Lookups search from first to last',
    'Updates only affect first dict',
  ],
  starterCode: `from collections import ChainMap

def create_config_system(defaults, user_config, env_config):
    """Create layered configuration system.
    
    Args:
        defaults: Default config dict
        user_config: User overrides dict
        env_config: Environment overrides dict
        
    Returns:
        ChainMap with proper priority
    """
    # Priority: env > user > defaults
    pass


def simulate_scope_chain():
    """Simulate variable scope chain.
    
    Returns:
        ChainMap representing local -> global scope
    """
    global_scope = {'x': 1, 'y': 2, 'z': 3}
    local_scope = {'x': 10, 'y': 20}
    
    # Create ChainMap for scope lookup
    pass


# Test
defaults = {'host': 'localhost', 'port': 8080, 'debug': False}
user = {'port': 3000, 'debug': True}
env = {'host': '0.0.0.0'}

config = create_config_system(defaults, user, env)
print(dict(config))  # Should show env > user > defaults priority
`,
  testCases: [
    {
      input: [{ a: 1 }, { a: 2, b: 3 }],
      expected: '{"a": 1, "b": 3}',
    },
  ],
  solution: `from collections import ChainMap

def create_config_system(defaults, user_config, env_config):
    # First dict in ChainMap has highest priority
    return ChainMap(env_config, user_config, defaults)


def simulate_scope_chain():
    global_scope = {'x': 1, 'y': 2, 'z': 3}
    local_scope = {'x': 10, 'y': 20}
    return ChainMap(local_scope, global_scope)`,
  timeComplexity: 'O(k) where k is number of maps (usually small)',
  spaceComplexity: 'O(1) - no copying, just references',
  order: 35,
  topic: 'Python Advanced',
};

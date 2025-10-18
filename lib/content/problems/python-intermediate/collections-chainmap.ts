/**
 * ChainMap for Nested Contexts
 * Problem ID: intermediate-collections-chainmap
 * Order: 26
 */

import { Problem } from '../../../types';

export const intermediate_collections_chainmapProblem: Problem = {
  id: 'intermediate-collections-chainmap',
  title: 'ChainMap for Nested Contexts',
  difficulty: 'Medium',
  description: `Use ChainMap to combine multiple dictionaries.

ChainMap features:
- Search multiple dicts
- Prioritizes first dict
- Fast lookups
- Used for nested scopes

**Use Case:** Configuration layers, template contexts

This tests:
- collections.ChainMap
- Dictionary chaining
- Scope resolution`,
  examples: [
    {
      input: 'ChainMap(local, global, default)',
      output: 'Searches in order',
    },
  ],
  constraints: ['Use ChainMap', 'Multiple dicts'],
  hints: [
    'from collections import ChainMap',
    'ChainMap(dict1, dict2, ...)',
    'Searches left to right',
  ],
  starterCode: `from collections import ChainMap

def get_config_value(key, local, global_config, defaults):
    """
    Get config value from layered dicts.
    
    Args:
        key: Config key
        local: Local overrides
        global_config: Global settings
        defaults: Default values
        
    Returns:
        Value from first dict containing key
        
    Examples:
        >>> local = {'color': 'red'}
        >>> global_config = {'color': 'blue', 'size': 10}
        >>> defaults = {'color': 'black', 'size': 8, 'style': 'solid'}
        >>> get_config_value('color', local, global_config, defaults)
        'red'
    """
    pass


# Test
print(get_config_value('size', {}, {'size': 10}, {'size': 8}))
`,
  testCases: [
    {
      input: ['size', {}, { size: 10 }, { size: 8 }],
      expected: 10,
    },
    {
      input: ['color', { color: 'red' }, { color: 'blue' }, {}],
      expected: 'red',
    },
  ],
  solution: `from collections import ChainMap

def get_config_value(key, local, global_config, defaults):
    config = ChainMap(local, global_config, defaults)
    return config.get(key)`,
  timeComplexity: 'O(n) where n is number of dicts',
  spaceComplexity: 'O(1)',
  order: 26,
  topic: 'Python Intermediate',
};

/**
 * JSON Configuration Manager
 * Problem ID: intermediate-json-config
 * Order: 3
 */

import { Problem } from '../../../types';

export const intermediate_json_configProblem: Problem = {
  id: 'intermediate-json-config',
  title: 'JSON Configuration Manager',
  difficulty: 'Medium',
  description: `Create a configuration manager that reads, writes, and updates JSON configuration files.

**Features:**
- Load configuration from JSON file
- Get configuration value by key (support nested keys with dot notation)
- Set configuration value
- Save configuration back to file
- Handle missing files by creating defaults

**Example:**
\`\`\`python
config = ConfigManager("config.json")
db_host = config.get("database.host")  # Nested key
config.set("database.port", 5432)
config.save()
\`\`\``,
  examples: [
    {
      input: 'config.get("database.host")',
      output: '"localhost"',
    },
  ],
  constraints: [
    'Support nested keys with dot notation',
    'Create file if not exists',
    'Validate JSON format',
  ],
  hints: [
    'Split dot notation into nested keys',
    'Use dict.get() for safe access',
    'Handle FileNotFoundError for new files',
  ],
  starterCode: `import json

class ConfigManager:
    """Manage JSON configuration files."""
    
    def __init__(self, filename):
        """
        Initialize configuration manager.
        
        Args:
            filename: Path to JSON config file
        """
        self.filename = filename
        self.config = self.load()
    
    def load(self):
        """Load configuration from file."""
        # TODO: Load JSON from file, handle FileNotFoundError
        return {}  # Return empty dict for now to prevent crashes
    
    def get(self, key, default=None):
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config.get("database.host")
            "localhost"
        """
        # TODO: Split key by '.' and navigate nested dicts
        pass
    
    def set(self, key, value):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        # TODO: Split key by '.' and set value in nested dict
        pass
    
    def save(self):
        """Save configuration to file."""
        # TODO: Write config dict to JSON file
        pass


# Create a virtual config file for testing
with open('config.json', 'w') as f:
    f.write('{"database": {"host": "localhost", "port": 3306}}')

# Test
config = ConfigManager("config.json")
print(config.get("database.host", "localhost"))
config.set("database.port", 5432)
config.save()


# Test helper function (for automated testing)
def test_config_manager(filename, key):
    """Test function for ConfigManager - implement the class methods above first!"""
    try:
        config = ConfigManager(filename)
        return config.get(key, 'localhost')
    except:
        return None  # Return None if methods not yet implemented
`,
  testCases: [
    {
      input: ['config.json', 'database.host'],
      expected: 'localhost',
      functionName: 'test_config_manager',
    },
  ],
  solution: `import json

class ConfigManager:
    def __init__(self, filename):
        self.filename = filename
        self.config = self.load()
    
    def load(self):
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return empty config if file doesn't exist
            return {}
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {self.filename}")
    
    def get(self, key, default=None):
        # Split dot notation
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key, value):
        keys = key.split('.')
        config = self.config
        
        # Navigate to nested dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
    
    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.config, f, indent=2)


# Create a virtual config file for testing
with open('config.json', 'w') as f:
    f.write('{"database": {"host": "localhost", "port": 3306}}')


# Test helper function (for automated testing)
def test_config_manager(filename, key):
    """Test function for ConfigManager."""
    config = ConfigManager(filename)
    return config.get(key, 'localhost')`,
  timeComplexity: 'O(d) where d is depth of nested keys',
  spaceComplexity: 'O(n) where n is config size',
  order: 3,
  topic: 'Python Intermediate',
};

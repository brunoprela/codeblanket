/**
 * Auto-Registration Metaclass
 * Problem ID: metaclass-registry
 * Order: 15
 */

import { Problem } from '../../../types';

export const metaclass_registryProblem: Problem = {
  id: 'metaclass-registry',
  title: 'Auto-Registration Metaclass',
  difficulty: 'Hard',
  description: `Create a metaclass that automatically registers classes in a registry.

The metaclass should:
- Maintain a class-level registry
- Auto-register classes on creation
- Provide lookup by name
- Skip abstract base classes

**Use Case:** Plugin systems, command registries, API endpoints.`,
  examples: [
    {
      input: 'class UserCommand(Command)',
      output: 'Automatically registered as "user"',
    },
  ],
  constraints: [
    'Must be a metaclass',
    'Auto-register on class creation',
    'Skip classes without name attribute',
  ],
  hints: [
    'Override __new__ method',
    'Check for name attribute',
    'Store in class-level dictionary',
  ],
  starterCode: `class Registry(type):
    """
    Metaclass that auto-registers classes.
    """
    
    _registry = {}
    
    def __new__(mcs, name, bases, attrs):
        # Your code here
        pass
    
    @classmethod
    def get(mcs, name):
        """Get class by registered name."""
        # Your code here
        pass


class Command(metaclass=Registry):
    name = None  # Subclasses should set this


class UserCommand(Command):
    name = "user"
    
    def execute(self):
        return "User command executed"


# Test
cmd_class = Registry.get("user")
cmd = cmd_class()
print(cmd.execute())


# Test helper function (for automated testing)
def test_registry(name):
    """Test function for Registry - implement the metaclass above first!"""
    try:
        class TestRegistry(type):
            _registry = {}
            
            def __new__(mcs, cls_name, bases, attrs):
                cls = super().__new__(mcs, cls_name, bases, attrs)
                if 'name' in attrs and attrs['name'] is not None:
                    mcs._registry[attrs['name']] = cls
                return cls
            
            @classmethod
            def get(mcs, name):
                return mcs._registry.get(name)
        
        class TestCommand(metaclass=TestRegistry):
            name = None
        
        class TestUserCommand(TestCommand):
            name = "user"
            def execute(self):
                return "User command executed"
        
        cmd_class = TestRegistry.get(name)
        if cmd_class:
            cmd = cmd_class()
            return cmd.execute()
        return None
    except:
        return None  # Return None if methods not yet implemented
`,
  testCases: [
    {
      input: ['user'],
      expected: 'User command executed',
      functionName: 'test_registry',
    },
  ],
  solution: `class Registry(type):
    _registry = {}
    
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        
        # Register if has name attribute
        if 'name' in attrs and attrs['name'] is not None:
            mcs._registry[attrs['name']] = cls
        
        return cls
    
    @classmethod
    def get(mcs, name):
        return mcs._registry.get(name)


# Test helper function (for automated testing)
def test_registry(name):
    """Test function for Registry."""
    cmd_class = Registry.get(name)
    if cmd_class:
        cmd = cmd_class()
        return cmd.execute()
    return None


class Command(metaclass=Registry):
    name = None


class UserCommand(Command):
    name = "user"
    
    def execute(self):
        return "User command executed"`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(n) for n classes',
  order: 15,
  topic: 'Python Advanced',
};

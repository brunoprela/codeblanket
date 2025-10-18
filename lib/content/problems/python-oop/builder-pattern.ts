/**
 * Builder Pattern for Complex Objects
 * Problem ID: oop-builder-pattern
 * Order: 6
 */

import { Problem } from '../../../types';

export const builder_patternProblem: Problem = {
  id: 'oop-builder-pattern',
  title: 'Builder Pattern for Complex Objects',
  difficulty: 'Medium',
  description: `Implement the Builder pattern for constructing complex \`House\` objects step by step.

Create:
- \`House\` class with multiple optional attributes (walls, roof, windows, doors, garage)
- \`HouseBuilder\` class with fluent interface (method chaining)
- Builder methods return self for chaining
- \`build()\` method returns the constructed House

**Pattern:** Builder separates object construction from representation.`,
  examples: [
    {
      input: 'HouseBuilder().add_walls().add_roof().add_garage().build()',
      output: 'House with walls, roof, and garage',
    },
  ],
  constraints: [
    'Use method chaining (fluent interface)',
    'All parts are optional',
    'build() returns House instance',
  ],
  hints: [
    'Return self from builder methods',
    'Store parts in builder, not House',
    'Create House in build() method',
  ],
  starterCode: `class House:
    """House with various components."""
    
    def __init__(self, walls=False, roof=False, windows=0, doors=0, garage=False):
        self.walls = walls
        self.roof = roof
        self.windows = windows
        self.doors = doors
        self.garage = garage
    
    def __str__(self):
        parts = []
        if self.walls:
            parts.append("walls")
        if self.roof:
            parts.append("roof")
        if self.windows:
            parts.append(f"{self.windows} windows")
        if self.doors:
            parts.append(f"{self.doors} doors")
        if self.garage:
            parts.append("garage")
        return f"House with: {', '.join(parts) if parts else 'nothing'}"


class HouseBuilder:
    """Builder for constructing houses."""
    
    def __init__(self):
        # Initialize builder state
        pass
    
    def add_walls(self):
        """Add walls to house."""
        # Return self for chaining
        pass
    
    def add_roof(self):
        """Add roof to house."""
        pass
    
    def add_windows(self, count):
        """Add windows to house."""
        pass
    
    def add_doors(self, count):
        """Add doors to house."""
        pass
    
    def add_garage(self):
        """Add garage to house."""
        pass
    
    def build(self):
        """Build and return the house."""
        pass


# Test fluent interface
house = (HouseBuilder()
         .add_walls()
         .add_roof()
         .add_windows(4)
         .add_doors(2)
         .add_garage()
         .build())
print(house)


def test_builder_pattern(*components):
    """Test function for Builder pattern."""
    builder = HouseBuilder()
    for component in components:
        if component == 'walls':
            builder.add_walls()
        elif component == 'roof':
            builder.add_roof()
        elif component == 'garage':
            builder.add_garage()
    house = builder.build()
    return house.__class__.__name__
`,
  testCases: [
    {
      input: ['walls', 'roof', 'garage'],
      expected: 'House',
      functionName: 'test_builder_pattern',
    },
  ],
  solution: `class House:
    def __init__(self, walls=False, roof=False, windows=0, doors=0, garage=False):
        self.walls = walls
        self.roof = roof
        self.windows = windows
        self.doors = doors
        self.garage = garage
    
    def __str__(self):
        parts = []
        if self.walls:
            parts.append("walls")
        if self.roof:
            parts.append("roof")
        if self.windows:
            parts.append(f"{self.windows} windows")
        if self.doors:
            parts.append(f"{self.doors} doors")
        if self.garage:
            parts.append("garage")
        return f"House with: {', '.join(parts) if parts else 'nothing'}"


class HouseBuilder:
    def __init__(self):
        self._walls = False
        self._roof = False
        self._windows = 0
        self._doors = 0
        self._garage = False
    
    def add_walls(self):
        self._walls = True
        return self
    
    def add_roof(self):
        self._roof = True
        return self
    
    def add_windows(self, count):
        self._windows = count
        return self
    
    def add_doors(self, count):
        self._doors = count
        return self
    
    def add_garage(self):
        self._garage = True
        return self
    
    def build(self):
        return House(
            walls=self._walls,
            roof=self._roof,
            windows=self._windows,
            doors=self._doors,
            garage=self._garage
        )


def test_builder_pattern(*components):
    """Test function for Builder pattern."""
    builder = HouseBuilder()
    for component in components:
        if component == 'walls':
            builder.add_walls()
        elif component == 'roof':
            builder.add_roof()
        elif component == 'garage':
            builder.add_garage()
    house = builder.build()
    return house.__class__.__name__`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 6,
  topic: 'Python Object-Oriented Programming',
};

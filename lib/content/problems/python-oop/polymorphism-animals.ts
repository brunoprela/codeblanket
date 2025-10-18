/**
 * Animal Polymorphism System
 * Problem ID: polymorphism-animals
 * Order: 3
 */

import { Problem } from '../../../types';

export const polymorphism_animalsProblem: Problem = {
  id: 'polymorphism-animals',
  title: 'Animal Polymorphism System',
  difficulty: 'Medium',
  description: `Create an animal class hierarchy that demonstrates polymorphism, inheritance, and composition.

Implement:
- Base \`Animal\` class with name, age, and speak() method
- \`Dog\` and \`Cat\` subclasses with specific speak() implementations
- \`Owner\` class that "has-a" relationship with animals (composition)
- Method to make all owned animals speak

**Key Concepts:**
- Inheritance: Dog and Cat inherit from Animal
- Polymorphism: Different speak() implementations
- Composition: Owner has a list of animals
- Encapsulation: Private attributes with properties`,
  examples: [
    {
      input: 'owner.add_animal(Dog("Buddy", 5)); owner.make_all_speak()',
      output: 'Buddy says Woof!',
    },
  ],
  constraints: [
    'Use inheritance for Dog and Cat',
    'Use composition for Owner',
    'Demonstrate polymorphism in make_all_speak()',
  ],
  hints: [
    'Animal is the base class',
    'Each subclass overrides speak()',
    'Owner stores animals in a list',
    'Loop through animals and call speak()',
  ],
  starterCode: `class Animal:
    """
    Base class for all animals.
    """
    
    def __init__(self, name, age):
        # Your code here
        pass
    
    def speak(self):
        """
        Make the animal speak.
        Should be overridden by subclasses.
        """
        # Your code here
        pass
    
    def __str__(self):
        # Your code here
        pass


class Dog(Animal):
    """Dog that barks."""
    
    def speak(self):
        # Your code here
        pass


class Cat(Animal):
    """Cat that meows."""
    
    def speak(self):
        # Your code here
        pass


class Owner:
    """
    Person who owns animals (composition).
    """
    
    def __init__(self, name):
        # Your code here
        pass
    
    def add_animal(self, animal):
        """Add an animal to owner's collection."""
        # Your code here
        pass
    
    def make_all_speak(self):
        """Make all owned animals speak (demonstrates polymorphism)."""
        # Your code here
        pass
    
    @property
    def animal_count(self):
        """Get number of animals owned."""
        # Your code here
        pass


# Test
owner = Owner("Alice")
owner.add_animal(Dog("Buddy", 5))
owner.add_animal(Cat("Whiskers", 3))
owner.add_animal(Dog("Max", 2))

print(f"{owner.name} has {owner.animal_count} animals")
owner.make_all_speak()


def test_animal(animal_type, name):
    """Test function for Animal classes."""
    if animal_type == 'Dog':
        animal = Dog(name, 5)
    elif animal_type == 'Cat':
        animal = Cat(name, 3)
    else:
        raise ValueError(f"Unknown animal type: {animal_type}")
    return animal.speak()
`,
  testCases: [
    {
      input: ['Dog', 'Buddy'],
      expected: 'Buddy says Woof!',
      functionName: 'test_animal',
    },
    {
      input: ['Cat', 'Whiskers'],
      expected: 'Whiskers says Meow!',
      functionName: 'test_animal',
    },
  ],
  solution: `class Animal:
    def __init__(self, name, age):
        self._name = name
        self._age = age
    
    @property
    def name(self):
        return self._name
    
    @property
    def age(self):
        return self._age
    
    def speak(self):
        return f"{self._name} makes a sound"
    
    def __str__(self):
        return f"{self.__class__.__name__}(name={self._name}, age={self._age})"


class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"


class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"


class Owner:
    def __init__(self, name):
        self.name = name
        self._animals = []
    
    def add_animal(self, animal):
        if not isinstance(animal, Animal):
            raise TypeError("Can only add Animal instances")
        self._animals.append(animal)
    
    def make_all_speak(self):
        for animal in self._animals:
            print(animal.speak())
    
    @property
    def animal_count(self):
        return len(self._animals)


def test_animal(animal_type, name):
    """Test function for Animal classes."""
    if animal_type == 'Dog':
        animal = Dog(name, 5)
    elif animal_type == 'Cat':
        animal = Cat(name, 3)
    else:
        raise ValueError(f"Unknown animal type: {animal_type}")
    return animal.speak()`,
  timeComplexity: 'O(n) for make_all_speak where n is number of animals',
  spaceComplexity: 'O(n) to store n animals',
  order: 3,
  topic: 'Python Object-Oriented Programming',
};

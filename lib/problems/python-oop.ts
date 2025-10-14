/**
 * Python Object-Oriented Programming problems
 */

import { Problem } from '../types';

export const pythonOOPProblems: Problem[] = [
  {
    id: 'class-bankaccount',
    title: 'Bank Account Class',
    difficulty: 'Easy',
    description: `Implement a \`BankAccount\` class with proper encapsulation and methods.

The class should:
- Store account holder name and balance (private)
- Provide deposit() and withdraw() methods
- Prevent negative balance with withdraw()
- Implement __str__ for readable output
- Use properties for controlled access

**Requirements:**
- Balance should be private (_balance)
- Withdraw should return True/False for success
- Deposit should only accept positive amounts`,
    examples: [
      {
        input: 'account = BankAccount("Alice", 1000); account.withdraw(200)',
        output: 'True, balance becomes 800',
      },
      {
        input: 'account.withdraw(2000)',
        output: 'False, insufficient funds',
      },
    ],
    constraints: [
      'Balance must be non-negative',
      'Deposit must be positive',
      'Use encapsulation (private attributes)',
    ],
    hints: [
      'Use _balance for private attribute',
      'Check balance before withdrawing',
      'Return True/False to indicate success',
    ],
    starterCode: `class BankAccount:
    """
    Bank account with deposit and withdraw operations.
    """
    
    def __init__(self, name, initial_balance=0):
        """
        Initialize account.
        
        Args:
            name: Account holder name
            initial_balance: Starting balance (default 0)
        """
        # Your code here
        pass
    
    def deposit(self, amount):
        """
        Deposit money into account.
        
        Args:
            amount: Amount to deposit
            
        Raises:
            ValueError: If amount is not positive
        """
        # Your code here
        pass
    
    def withdraw(self, amount):
        """
        Withdraw money from account.
        
        Args:
            amount: Amount to withdraw
            
        Returns:
            True if successful, False if insufficient funds
        """
        # Your code here
        pass
    
    @property
    def balance(self):
        """Get current balance."""
        # Your code here
        pass
    
    def __str__(self):
        """String representation."""
        # Your code here
        pass


# Test
account = BankAccount("Alice", 1000)
print(account)  # BankAccount(Alice, balance=1000)
account.deposit(500)
print(account.balance)  # 1500
print(account.withdraw(200))  # True
print(account.balance)  # 1300
print(account.withdraw(2000))  # False


def test_bank_account(name, initial_balance, deposit_amount, withdraw_amount):
    """Test function for BankAccount class."""
    account = BankAccount(name, initial_balance)
    account.deposit(deposit_amount)
    result = account.withdraw(withdraw_amount)
    if not result:
        return False
    return account.balance
`,
    testCases: [
      {
        input: ['Alice', 1000, 500, 200],
        expected: 1300,
        functionName: 'test_bank_account',
      },
      {
        input: ['Bob', 100, 0, 200],
        expected: false, // withdraw fails
        functionName: 'test_bank_account',
      },
    ],
    solution: `class BankAccount:
    def __init__(self, name, initial_balance=0):
        self.name = name
        self._balance = initial_balance
    
    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
    
    def withdraw(self, amount):
        if amount > self._balance:
            return False
        self._balance -= amount
        return True
    
    @property
    def balance(self):
        return self._balance
    
    def __str__(self):
        return f"BankAccount({self.name}, balance={self._balance})"


def test_bank_account(name, initial_balance, deposit_amount, withdraw_amount):
    """Test function for BankAccount class."""
    account = BankAccount(name, initial_balance)
    account.deposit(deposit_amount)
    result = account.withdraw(withdraw_amount)
    if not result:
        return False
    return account.balance`,
    timeComplexity: 'O(1) for all operations',
    spaceComplexity: 'O(1)',
    order: 1,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'inheritance-shapes',
    title: 'Shape Hierarchy with Inheritance',
    difficulty: 'Medium',
    description: `Create a shape class hierarchy using inheritance and polymorphism.

Implement:
- Abstract base class \`Shape\` with abstract methods area() and perimeter()
- \`Circle\` subclass with radius
- \`Rectangle\` subclass with width and height
- \`Square\` subclass inheriting from Rectangle

**Requirements:**
- Shape should be abstract (cannot instantiate)
- All shapes implement area() and perimeter()
- Square should reuse Rectangle logic
- Demonstrate polymorphism with a list of shapes`,
    examples: [
      {
        input: 'Circle(radius=5)',
        output: 'area() returns 78.54, perimeter() returns 31.42',
      },
      {
        input: 'Square(side=4)',
        output: 'area() returns 16, perimeter() returns 16',
      },
    ],
    constraints: [
      'Use ABC and abstractmethod',
      'All shapes must implement required methods',
      'Square should inherit from Rectangle',
    ],
    hints: [
      'Use abc.ABC and @abstractmethod',
      'Circle: area = π * r², perimeter = 2 * π * r',
      'Rectangle: area = w * h, perimeter = 2(w + h)',
      'Square: just set width = height = side',
    ],
    starterCode: `from abc import ABC, abstractmethod
import math

class Shape(ABC):
    """
    Abstract base class for shapes.
    """
    
    @abstractmethod
    def area(self):
        """Calculate area."""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate perimeter."""
        pass


class Circle(Shape):
    """Circle shape."""
    
    def __init__(self, radius):
        # Your code here
        pass
    
    def area(self):
        # Your code here
        pass
    
    def perimeter(self):
        # Your code here
        pass


class Rectangle(Shape):
    """Rectangle shape."""
    
    def __init__(self, width, height):
        # Your code here
        pass
    
    def area(self):
        # Your code here
        pass
    
    def perimeter(self):
        # Your code here
        pass


class Square(Rectangle):
    """Square shape (special case of rectangle)."""
    
    def __init__(self, side):
        # Your code here
        pass


# Test polymorphism
shapes = [
    Circle(5),
    Rectangle(4, 6),
    Square(4)
]

for shape in shapes:
    print(f"{shape.__class__.__name__}: area={shape.area():.2f}, perimeter={shape.perimeter():.2f}")


def test_shape(shape_type, *args):
    """Test function for Shape classes."""
    import math
    if shape_type == 'Circle':
        shape = Circle(args[0])
    elif shape_type == 'Rectangle':
        shape = Rectangle(args[0], args[1])
    elif shape_type == 'Square':
        shape = Square(args[0])
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    return round(shape.area(), 2)
`,
    testCases: [
      {
        input: ['Circle', 5],
        expected: 78.54, // area
        functionName: 'test_shape',
      },
      {
        input: ['Rectangle', 4, 6],
        expected: 24, // area
        functionName: 'test_shape',
      },
      {
        input: ['Square', 4],
        expected: 16, // area
        functionName: 'test_shape',
      },
    ],
    solution: `from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        return 2 * math.pi * self.radius


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)


class Square(Rectangle):
    def __init__(self, side):
        super().__init__(side, side)


def test_shape(shape_type, *args):
    """Test function for Shape classes."""
    import math
    if shape_type == 'Circle':
        shape = Circle(args[0])
    elif shape_type == 'Rectangle':
        shape = Rectangle(args[0], args[1])
    elif shape_type == 'Square':
        shape = Square(args[0])
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    return round(shape.area(), 2)`,
    timeComplexity: 'O(1) for all operations',
    spaceComplexity: 'O(1)',
    order: 2,
    topic: 'Python Object-Oriented Programming',
  },
  {
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
  },
  {
    id: 'oop-vehicle-factory',
    title: 'Vehicle Factory Pattern',
    difficulty: 'Medium',
    description: `Implement a factory pattern for creating different types of vehicles.

Create:
- Abstract \`Vehicle\` base class with speed and fuel_type properties
- \`Car\`, \`Motorcycle\`, and \`Truck\` subclasses
- \`VehicleFactory\` class with a static method \`create_vehicle(type)\`
- Factory should return the appropriate vehicle type based on string input

**Pattern:** Factory pattern centralizes object creation logic.`,
    examples: [
      {
        input: 'VehicleFactory.create_vehicle("car")',
        output: 'Returns Car instance',
      },
    ],
    constraints: [
      'Use abstract base class',
      'Factory method must be static or class method',
      'Handle invalid vehicle types',
    ],
    hints: [
      'Use @staticmethod or @classmethod for factory',
      'Use a dictionary to map types to classes',
      'Raise ValueError for unknown types',
    ],
    starterCode: `from abc import ABC, abstractmethod

class Vehicle(ABC):
    """Abstract vehicle base class."""
    
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def get_info(self):
        pass


class Car(Vehicle):
    """Car implementation."""
    pass


class Motorcycle(Vehicle):
    """Motorcycle implementation."""
    pass


class Truck(Vehicle):
    """Truck implementation."""
    pass


class VehicleFactory:
    """Factory for creating vehicles."""
    
    @staticmethod
    def create_vehicle(vehicle_type):
        """
        Create vehicle of specified type.
        
        Args:
            vehicle_type: Type of vehicle ('car', 'motorcycle', 'truck')
            
        Returns:
            Vehicle instance
            
        Raises:
            ValueError: If vehicle_type is unknown
        """
        pass


# Test
car = VehicleFactory.create_vehicle('car')
print(car.get_info())


def test_vehicle_factory(vehicle_type):
    """Test function for VehicleFactory."""
    vehicle = VehicleFactory.create_vehicle(vehicle_type)
    return vehicle.__class__.__name__
`,
    testCases: [
      {
        input: ['car'],
        expected: 'Car',
        functionName: 'test_vehicle_factory',
      },
      {
        input: ['motorcycle'],
        expected: 'Motorcycle',
        functionName: 'test_vehicle_factory',
      },
    ],
    solution: `from abc import ABC, abstractmethod

class Vehicle(ABC):
    def __init__(self, speed, fuel_type):
        self.speed = speed
        self.fuel_type = fuel_type
    
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def get_info(self):
        pass


class Car(Vehicle):
    def __init__(self):
        super().__init__(speed=120, fuel_type='gasoline')
    
    def start(self):
        return "Car engine starting"
    
    def get_info(self):
        return f"Car: {self.speed} km/h, {self.fuel_type}"


class Motorcycle(Vehicle):
    def __init__(self):
        super().__init__(speed=180, fuel_type='gasoline')
    
    def start(self):
        return "Motorcycle engine starting"
    
    def get_info(self):
        return f"Motorcycle: {self.speed} km/h, {self.fuel_type}"


class Truck(Vehicle):
    def __init__(self):
        super().__init__(speed=90, fuel_type='diesel')
    
    def start(self):
        return "Truck engine starting"
    
    def get_info(self):
        return f"Truck: {self.speed} km/h, {self.fuel_type}"


class VehicleFactory:
    _vehicles = {
        'car': Car,
        'motorcycle': Motorcycle,
        'truck': Truck
    }
    
    @staticmethod
    def create_vehicle(vehicle_type):
        vehicle_class = VehicleFactory._vehicles.get(vehicle_type.lower())
        if vehicle_class is None:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")
        return vehicle_class()


def test_vehicle_factory(vehicle_type):
    """Test function for VehicleFactory."""
    vehicle = VehicleFactory.create_vehicle(vehicle_type)
    return vehicle.__class__.__name__`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 4,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-observer-pattern',
    title: 'Observer Pattern Implementation',
    difficulty: 'Hard',
    description: `Implement the Observer pattern where subjects notify observers of state changes.

Create:
- \`Subject\` class that maintains list of observers
- Methods: \`attach(observer)\`, \`detach(observer)\`, \`notify()\`
- \`Observer\` interface with \`update(subject)\` method
- Concrete observer that reacts to subject changes

**Use Case:** Event systems, MVC patterns, pub-sub systems.`,
    examples: [
      {
        input: 'subject.state = 10; subject.notify()',
        output: 'All observers receive update',
      },
    ],
    constraints: [
      'Subject maintains observer list',
      'Observers implement update method',
      'Support multiple observers',
    ],
    hints: [
      'Store observers in a list',
      'Loop through observers in notify()',
      'Pass self to observer.update()',
    ],
    starterCode: `from abc import ABC, abstractmethod

class Observer(ABC):
    """Observer interface."""
    
    @abstractmethod
    def update(self, subject):
        """Called when subject changes."""
        pass


class Subject:
    """Subject that observers watch."""
    
    def __init__(self):
        self._observers = []
        self._state = None
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        self._state = value
        self.notify()
    
    def attach(self, observer):
        """Add an observer."""
        pass
    
    def detach(self, observer):
        """Remove an observer."""
        pass
    
    def notify(self):
        """Notify all observers of state change."""
        pass


class ConcreteObserver(Observer):
    """Concrete observer implementation."""
    
    def __init__(self, name):
        self.name = name
    
    def update(self, subject):
        """React to subject state change."""
        pass


# Test
subject = Subject()
observer1 = ConcreteObserver("Observer1")
observer2 = ConcreteObserver("Observer2")

subject.attach(observer1)
subject.attach(observer2)
subject.state = 10  # Should notify both observers


def test_observer_pattern(state, num_observers):
    """Test function for Observer pattern."""
    subject = Subject()
    for i in range(num_observers):
        observer = ConcreteObserver(f"Observer{i+1}")
        subject.attach(observer)
    subject.state = state
    return 'notified'
`,
    testCases: [
      {
        input: [10, 2], // state, num observers
        expected: 'notified',
        functionName: 'test_observer_pattern',
      },
    ],
    solution: `from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass


class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        self._state = value
        self.notify()
    
    def attach(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer):
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)


class ConcreteObserver(Observer):
    def __init__(self, name):
        self.name = name
    
    def update(self, subject):
        print(f"{self.name} received update: state = {subject.state}")


def test_observer_pattern(state, num_observers):
    """Test function for Observer pattern."""
    subject = Subject()
    for i in range(num_observers):
        observer = ConcreteObserver(f"Observer{i+1}")
        subject.attach(observer)
    subject.state = state
    return 'notified'`,
    timeComplexity: 'O(n) for notify where n is number of observers',
    spaceComplexity: 'O(n) to store observers',
    order: 5,
    topic: 'Python Object-Oriented Programming',
  },
  {
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
  },
  {
    id: 'oop-strategy-pattern',
    title: 'Strategy Pattern for Sorting',
    difficulty: 'Medium',
    description: `Implement the Strategy pattern to allow dynamic selection of sorting algorithms.

Create:
- \`SortStrategy\` interface with \`sort(data)\` method
- Concrete strategies: \`BubbleSort\`, \`QuickSort\`, \`MergeSort\`
- \`DataProcessor\` class that accepts and uses a strategy
- Ability to change strategy at runtime

**Pattern:** Strategy encapsulates interchangeable algorithms.`,
    examples: [
      {
        input: 'processor.set_strategy(QuickSort()); processor.sort([3,1,2])',
        output: '[1, 2, 3] using QuickSort',
      },
    ],
    constraints: [
      'Strategies implement common interface',
      'Strategy can be changed at runtime',
      'DataProcessor delegates to strategy',
    ],
    hints: [
      'Store strategy as instance variable',
      'Call strategy.sort() from processor',
      'Each strategy implements differently',
    ],
    starterCode: `from abc import ABC, abstractmethod

class SortStrategy(ABC):
    """Strategy interface for sorting."""
    
    @abstractmethod
    def sort(self, data):
        """Sort the data using this strategy."""
        pass


class BubbleSort(SortStrategy):
    """Bubble sort strategy."""
    
    def sort(self, data):
        # Implement bubble sort
        pass


class QuickSort(SortStrategy):
    """Quick sort strategy."""
    
    def sort(self, data):
        # Implement quick sort (simplified)
        pass


class MergeSort(SortStrategy):
    """Merge sort strategy."""
    
    def sort(self, data):
        # Implement merge sort (simplified)
        pass


class DataProcessor:
    """Processor that uses a sorting strategy."""
    
    def __init__(self, strategy=None):
        self._strategy = strategy
    
    def set_strategy(self, strategy):
        """Change the sorting strategy."""
        pass
    
    def sort(self, data):
        """Sort data using current strategy."""
        pass


# Test
processor = DataProcessor()
processor.set_strategy(QuickSort())
result = processor.sort([3, 1, 4, 1, 5, 9, 2, 6])
print(result)

processor.set_strategy(BubbleSort())
result = processor.sort([3, 1, 4, 1, 5, 9, 2, 6])
print(result)


def test_strategy_pattern(data, strategy_name):
    """Test function for Strategy pattern."""
    processor = DataProcessor()
    if strategy_name == 'QuickSort':
        processor.set_strategy(QuickSort())
    elif strategy_name == 'BubbleSort':
        processor.set_strategy(BubbleSort())
    elif strategy_name == 'MergeSort':
        processor.set_strategy(MergeSort())
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return processor.sort(data)
`,
    testCases: [
      {
        input: [[3, 1, 2], 'QuickSort'],
        expected: [1, 2, 3],
        functionName: 'test_strategy_pattern',
      },
    ],
    solution: `from abc import ABC, abstractmethod

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass


class BubbleSort(SortStrategy):
    def sort(self, data):
        arr = data.copy()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr


class QuickSort(SortStrategy):
    def sort(self, data):
        return sorted(data)  # Simplified using built-in


class MergeSort(SortStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return data.copy()
        
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result


class DataProcessor:
    def __init__(self, strategy=None):
        self._strategy = strategy
    
    def set_strategy(self, strategy):
        self._strategy = strategy
    
    def sort(self, data):
        if self._strategy is None:
            raise ValueError("No strategy set")
        return self._strategy.sort(data)


def test_strategy_pattern(data, strategy_name):
    """Test function for Strategy pattern."""
    processor = DataProcessor()
    if strategy_name == 'QuickSort':
        processor.set_strategy(QuickSort())
    elif strategy_name == 'BubbleSort':
        processor.set_strategy(BubbleSort())
    elif strategy_name == 'MergeSort':
        processor.set_strategy(MergeSort())
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return processor.sort(data)`,
    timeComplexity:
      'Depends on strategy (O(n²) for bubble, O(n log n) for others)',
    spaceComplexity: 'O(n)',
    order: 7,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-composite-pattern',
    title: 'Composite Pattern for File System',
    difficulty: 'Hard',
    description: `Implement the Composite pattern to represent a file system hierarchy.

Create:
- \`FileSystemItem\` abstract base class with \`get_size()\` method
- \`File\` class (leaf) with size attribute
- \`Directory\` class (composite) that can contain files and directories
- Methods: \`add(item)\`, \`remove(item)\`, \`get_size()\` (sum of all contents)

**Pattern:** Composite lets clients treat individual objects and compositions uniformly.`,
    examples: [
      {
        input: 'dir.add(File(100)); dir.add(File(200)); dir.get_size()',
        output: '300',
      },
    ],
    constraints: [
      'Both File and Directory inherit from FileSystemItem',
      'Directory can contain files and other directories',
      'get_size() recursively calculates total size',
    ],
    hints: [
      'Directory stores children in a list',
      'File returns its size directly',
      "Directory sums children's sizes",
    ],
    starterCode: `from abc import ABC, abstractmethod

class FileSystemItem(ABC):
    """Abstract base for files and directories."""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def get_size(self):
        """Get size in bytes."""
        pass


class File(FileSystemItem):
    """File with fixed size."""
    
    def __init__(self, name, size):
        super().__init__(name)
        self.size = size
    
    def get_size(self):
        # Return file size
        pass


class Directory(FileSystemItem):
    """Directory that can contain files and directories."""
    
    def __init__(self, name):
        super().__init__(name)
        self.children = []
    
    def add(self, item):
        """Add a file or directory."""
        pass
    
    def remove(self, item):
        """Remove a file or directory."""
        pass
    
    def get_size(self):
        """Get total size of all contents."""
        pass


# Test
root = Directory("root")
docs = Directory("documents")
pics = Directory("pictures")

docs.add(File("resume.pdf", 1024))
docs.add(File("cover_letter.pdf", 512))
pics.add(File("photo1.jpg", 2048))
pics.add(File("photo2.jpg", 1536))

root.add(docs)
root.add(pics)
root.add(File("readme.txt", 256))

print(f"Total size: {root.get_size()} bytes")


def test_composite_pattern(*file_sizes):
    """Test function for Composite pattern."""
    directory = Directory("test")
    for i, size in enumerate(file_sizes):
        directory.add(File(f"file{i}.txt", size))
    return directory.get_size()
`,
    testCases: [
      {
        input: [1024, 512, 256], // file sizes
        expected: 1792, // sum
        functionName: 'test_composite_pattern',
      },
    ],
    solution: `from abc import ABC, abstractmethod

class FileSystemItem(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def get_size(self):
        pass


class File(FileSystemItem):
    def __init__(self, name, size):
        super().__init__(name)
        self.size = size
    
    def get_size(self):
        return self.size


class Directory(FileSystemItem):
    def __init__(self, name):
        super().__init__(name)
        self.children = []
    
    def add(self, item):
        if not isinstance(item, FileSystemItem):
            raise TypeError("Can only add FileSystemItem")
        self.children.append(item)
    
    def remove(self, item):
        if item in self.children:
            self.children.remove(item)
    
    def get_size(self):
        return sum(child.get_size() for child in self.children)


def test_composite_pattern(*file_sizes):
    """Test function for Composite pattern."""
    directory = Directory("test")
    for i, size in enumerate(file_sizes):
        directory.add(File(f"file{i}.txt", size))
    return directory.get_size()`,
    timeComplexity: 'O(n) where n is total number of items',
    spaceComplexity: 'O(d) where d is depth of directory tree',
    order: 8,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-template-method',
    title: 'Template Method Pattern',
    difficulty: 'Medium',
    description: `Implement the Template Method pattern for a data processing pipeline.

Create:
- Abstract \`DataProcessor\` class with template method \`process()\`
- Template method calls: \`load_data()\`, \`parse_data()\`, \`analyze_data()\`, \`save_results()\`
- Subclasses: \`CSVProcessor\` and \`JSONProcessor\` that implement specific steps
- Template method defines the algorithm structure

**Pattern:** Template method defines skeleton, subclasses fill in steps.`,
    examples: [
      {
        input: 'CSVProcessor().process()',
        output: 'Executes CSV-specific steps in defined order',
      },
    ],
    constraints: [
      'Template method is concrete in base class',
      'Hook methods are abstract',
      "Subclasses don't override template method",
    ],
    hints: [
      'Template method calls other methods',
      'Make step methods abstract',
      'Each processor implements steps differently',
    ],
    starterCode: `from abc import ABC, abstractmethod

class DataProcessor(ABC):
    """Template for data processing pipeline."""
    
    def process(self):
        """Template method defining the algorithm structure."""
        # Define the steps here
        pass
    
    @abstractmethod
    def load_data(self):
        """Load data from source."""
        pass
    
    @abstractmethod
    def parse_data(self, raw_data):
        """Parse raw data."""
        pass
    
    @abstractmethod
    def analyze_data(self, parsed_data):
        """Analyze parsed data."""
        pass
    
    @abstractmethod
    def save_results(self, results):
        """Save analysis results."""
        pass


class CSVProcessor(DataProcessor):
    """Process CSV data."""
    
    def load_data(self):
        # Simulate loading CSV
        pass
    
    def parse_data(self, raw_data):
        # Parse CSV format
        pass
    
    def analyze_data(self, parsed_data):
        # Analyze data
        pass
    
    def save_results(self, results):
        # Save results
        pass


class JSONProcessor(DataProcessor):
    """Process JSON data."""
    
    def load_data(self):
        pass
    
    def parse_data(self, raw_data):
        pass
    
    def analyze_data(self, parsed_data):
        pass
    
    def save_results(self, results):
        pass


# Test
csv_processor = CSVProcessor()
csv_processor.process()


def test_template_method(processor_type):
    """Test function for Template Method pattern."""
    if processor_type == 'CSV':
        processor = CSVProcessor()
    elif processor_type == 'JSON':
        processor = JSONProcessor()
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")
    processor.process()
    return 'processed'
`,
    testCases: [
      {
        input: ['CSV'],
        expected: 'processed',
        functionName: 'test_template_method',
      },
    ],
    solution: `from abc import ABC, abstractmethod

class DataProcessor(ABC):
    def process(self):
        """Template method defining the algorithm structure."""
        print("Starting processing pipeline...")
        raw_data = self.load_data()
        parsed_data = self.parse_data(raw_data)
        results = self.analyze_data(parsed_data)
        self.save_results(results)
        print("Processing complete!")
        return results
    
    @abstractmethod
    def load_data(self):
        pass
    
    @abstractmethod
    def parse_data(self, raw_data):
        pass
    
    @abstractmethod
    def analyze_data(self, parsed_data):
        pass
    
    @abstractmethod
    def save_results(self, results):
        pass


class CSVProcessor(DataProcessor):
    def load_data(self):
        print("Loading CSV data...")
        return "name,age\\nAlice,30\\nBob,25"
    
    def parse_data(self, raw_data):
        print("Parsing CSV...")
        lines = raw_data.split('\\n')
        header = lines[0].split(',')
        data = [dict(zip(header, line.split(','))) for line in lines[1:]]
        return data
    
    def analyze_data(self, parsed_data):
        print("Analyzing CSV data...")
        avg_age = sum(int(row['age']) for row in parsed_data) / len(parsed_data)
        return {'average_age': avg_age}
    
    def save_results(self, results):
        print(f"Saving results: {results}")


class JSONProcessor(DataProcessor):
    def load_data(self):
        print("Loading JSON data...")
        return '{"users": [{"name": "Alice", "age": 30}]}'
    
    def parse_data(self, raw_data):
        print("Parsing JSON...")
        import json
        return json.loads(raw_data)
    
    def analyze_data(self, parsed_data):
        print("Analyzing JSON data...")
        users = parsed_data['users']
        return {'user_count': len(users)}
    
    def save_results(self, results):
        print(f"Saving results: {results}")


def test_template_method(processor_type):
    """Test function for Template Method pattern."""
    if processor_type == 'CSV':
        processor = CSVProcessor()
    elif processor_type == 'JSON':
        processor = JSONProcessor()
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")
    processor.process()
    return 'processed'`,
    timeComplexity: 'O(n) where n is data size',
    spaceComplexity: 'O(n)',
    order: 9,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-multiple-inheritance-mixin',
    title: 'Multiple Inheritance with Mixins',
    difficulty: 'Hard',
    description: `Create a flexible class system using mixins for shared functionality.

Implement:
- \`SerializableMixin\` with \`to_dict()\` and \`from_dict()\` methods
- \`TimestampMixin\` that adds created_at and updated_at timestamps
- \`User\` class that inherits from both mixins
- Proper MRO (Method Resolution Order) handling

**Pattern:** Mixins provide reusable functionality without deep inheritance hierarchies.`,
    examples: [
      {
        input: 'user = User("alice"); user.to_dict()',
        output: 'Dictionary with user data and timestamp',
      },
    ],
    constraints: [
      'Mixins should be independent',
      'User class combines both mixins',
      'Demonstrate proper MRO',
    ],
    hints: [
      "Mixins typically don't have __init__",
      'Use super() for cooperative inheritance',
      'Check class.mro() for resolution order',
    ],
    starterCode: `from datetime import datetime
import json

class SerializableMixin:
    """Mixin for JSON serialization."""
    
    def to_dict(self):
        """Convert object to dictionary."""
        pass
    
    @classmethod
    def from_dict(cls, data):
        """Create object from dictionary."""
        pass


class TimestampMixin:
    """Mixin for automatic timestamps."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def touch(self):
        """Update the updated_at timestamp."""
        pass


class User(SerializableMixin, TimestampMixin):
    """User class with serialization and timestamps."""
    
    def __init__(self, username, email=None):
        super().__init__()
        self.username = username
        self.email = email


# Test
user = User("alice", "alice@example.com")
print(user.to_dict())

user.touch()
print(user.updated_at)

# Test MRO
print(User.mro())


def test_mixin(username, email):
    """Test function for Mixin pattern."""
    user = User(username, email)
    result = user.to_dict()
    return type(result).__name__
`,
    testCases: [
      {
        input: ['alice', 'alice@example.com'],
        expected: 'dict',
        functionName: 'test_mixin',
      },
    ],
    solution: `from datetime import datetime
import json

class SerializableMixin:
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data):
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        for key, value in data.items():
            # Try to parse datetime strings
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value)
                except (ValueError, AttributeError):
                    pass
            setattr(instance, key, value)
        return instance
    
    def to_json(self):
        return json.dumps(self.to_dict())


class TimestampMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def touch(self):
        self.updated_at = datetime.now()


class User(SerializableMixin, TimestampMixin):
    def __init__(self, username, email=None):
        super().__init__()
        self.username = username
        self.email = email


def test_mixin(username, email):
    """Test function for Mixin pattern."""
    user = User(username, email)
    result = user.to_dict()
    return type(result).__name__`,
    timeComplexity: 'O(n) where n is number of attributes',
    spaceComplexity: 'O(n)',
    order: 10,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-dataclass-comparison',
    title: 'Custom Comparison with Dataclasses',
    difficulty: 'Medium',
    description: `Use Python's \`@dataclass\` decorator to create classes with automatic comparison methods.

Implement:
- \`Point\` dataclass with x and y coordinates
- Enable ordering (use \`order=True\`)
- Add custom \`distance_from_origin()\` method
- Create \`Rectangle\` dataclass with two Point corners
- Implement \`area()\` property

**Modern Pattern:** Dataclasses reduce boilerplate for data-holding classes.`,
    examples: [
      {
        input: 'Point(3, 4).distance_from_origin()',
        output: '5.0',
      },
    ],
    constraints: [
      'Use @dataclass decorator',
      'Points should be comparable',
      'Rectangle uses Points',
    ],
    hints: [
      'Import from dataclasses',
      'Use order=True for comparisons',
      'Add methods normally to dataclasses',
    ],
    starterCode: `from dataclasses import dataclass
import math

@dataclass(order=True)
class Point:
    """2D point with comparison support."""
    x: float
    y: float
    
    def distance_from_origin(self):
        """Calculate distance from origin."""
        pass
    
    def distance_to(self, other):
        """Calculate distance to another point."""
        pass


@dataclass
class Rectangle:
    """Rectangle defined by two corner points."""
    top_left: Point
    bottom_right: Point
    
    @property
    def width(self):
        """Calculate rectangle width."""
        pass
    
    @property
    def height(self):
        """Calculate rectangle height."""
        pass
    
    @property
    def area(self):
        """Calculate rectangle area."""
        pass


# Test
p1 = Point(0, 0)
p2 = Point(3, 4)
print(p2.distance_from_origin())
print(p1 < p2)  # Comparison works automatically

rect = Rectangle(Point(0, 10), Point(10, 0))
print(f"Area: {rect.area}")


def test_dataclass(x, y):
    """Test function for Dataclass."""
    point = Point(x, y)
    return point.distance_from_origin()
`,
    testCases: [
      {
        input: [3, 4],
        expected: 5.0,
        functionName: 'test_dataclass',
      },
    ],
    solution: `from dataclasses import dataclass
import math

@dataclass(order=True)
class Point:
    x: float
    y: float
    
    def distance_from_origin(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Rectangle:
    top_left: Point
    bottom_right: Point
    
    @property
    def width(self):
        return abs(self.bottom_right.x - self.top_left.x)
    
    @property
    def height(self):
        return abs(self.top_left.y - self.bottom_right.y)
    
    @property
    def area(self):
        return self.width * self.height
    
    def contains_point(self, point):
        return (self.top_left.x <= point.x <= self.bottom_right.x and
                self.bottom_right.y <= point.y <= self.top_left.y)


def test_dataclass(x, y):
    """Test function for Dataclass."""
    point = Point(x, y)
    return point.distance_from_origin()`,
    timeComplexity: 'O(1) for all operations',
    spaceComplexity: 'O(1)',
    order: 11,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-protocol-duck-typing',
    title: 'Protocol and Duck Typing',
    difficulty: 'Medium',
    description: `Implement Protocol (structural subtyping) for duck typing with type hints.

Create:
- \`Drawable\` Protocol with \`draw()\` method
- Multiple classes that implement draw() without inheriting
- \`Canvas\` class that accepts any Drawable
- Use typing.Protocol for static type checking

**Pattern:** "If it walks like a duck and quacks like a duck, it's a duck."`,
    examples: [
      {
        input: 'canvas.render(Circle()); canvas.render(Square())',
        output: 'Both work without common base class',
      },
    ],
    constraints: [
      'Use typing.Protocol',
      "Classes don't inherit from Drawable",
      'Canvas works with any object that has draw()',
    ],
    hints: [
      'Import Protocol from typing',
      'Protocol defines interface, not inheritance',
      'Classes implicitly satisfy protocol',
    ],
    starterCode: `from typing import Protocol

class Drawable(Protocol):
    """Protocol for drawable objects."""
    
    def draw(self) -> str:
        """Draw the object and return description."""
        ...


class Circle:
    """Circle that can be drawn (no inheritance!)."""
    
    def __init__(self, radius):
        self.radius = radius
    
    def draw(self) -> str:
        pass


class Square:
    """Square that can be drawn (no inheritance!)."""
    
    def __init__(self, side):
        self.side = side
    
    def draw(self) -> str:
        pass


class Triangle:
    """Triangle that can be drawn (no inheritance!)."""
    
    def __init__(self, base, height):
        self.base = base
        self.height = height
    
    def draw(self) -> str:
        pass


class Canvas:
    """Canvas that can render any Drawable."""
    
    def __init__(self):
        self.objects = []
    
    def add(self, obj: Drawable):
        """Add a drawable object."""
        pass
    
    def render(self) -> str:
        """Render all objects."""
        pass


# Test duck typing
canvas = Canvas()
canvas.add(Circle(5))
canvas.add(Square(4))
canvas.add(Triangle(3, 4))

print(canvas.render())


def test_protocol(shape_type, size):
    """Test function for Protocol pattern."""
    if shape_type == 'Circle':
        shape = Circle(size)
    elif shape_type == 'Square':
        shape = Square(size)
    elif shape_type == 'Triangle':
        shape = Triangle(size, size)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    result = shape.draw()
    # Check if result contains the shape type
    if shape_type in result:
        return f"{shape_type} drawn"
    return result
`,
    testCases: [
      {
        input: ['Circle', 5],
        expected: 'Circle drawn',
        functionName: 'test_protocol',
      },
    ],
    solution: `from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...


class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def draw(self) -> str:
        return f"Drawing circle with radius {self.radius}"


class Square:
    def __init__(self, side):
        self.side = side
    
    def draw(self) -> str:
        return f"Drawing square with side {self.side}"


class Triangle:
    def __init__(self, base, height):
        self.base = base
        self.height = height
    
    def draw(self) -> str:
        return f"Drawing triangle with base {self.base} and height {self.height}"


class Canvas:
    def __init__(self):
        self.objects = []
    
    def add(self, obj: Drawable):
        # Type checker ensures obj has draw() method
        self.objects.append(obj)
    
    def render(self) -> str:
        return '\\n'.join(obj.draw() for obj in self.objects)


def test_protocol(shape_type, size):
    """Test function for Protocol pattern."""
    if shape_type == 'Circle':
        shape = Circle(size)
    elif shape_type == 'Square':
        shape = Square(size)
    elif shape_type == 'Triangle':
        shape = Triangle(size, size)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    result = shape.draw()
    # Check if result contains the shape type
    if shape_type in result:
        return f"{shape_type} drawn"
    return result`,
    timeComplexity: 'O(n) for rendering n objects',
    spaceComplexity: 'O(n)',
    order: 12,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-enum-state-machine',
    title: 'State Machine with Enum',
    difficulty: 'Medium',
    description: `Implement a state machine using Python's \`Enum\` for a traffic light system.

Create:
- \`TrafficLightState\` Enum with RED, YELLOW, GREEN
- \`TrafficLight\` class that manages state transitions
- Transition rules: RED → GREEN → YELLOW → RED
- Time duration for each state
- \`next_state()\` method with validation

**Pattern:** Enums provide type-safe state management.`,
    examples: [
      {
        input: 'light.next_state() from RED',
        output: 'Changes to GREEN',
      },
    ],
    constraints: [
      'Use enum.Enum',
      'Enforce valid transitions only',
      'Each state has a duration',
    ],
    hints: [
      'Import Enum from enum',
      'Use dictionary for transition rules',
      'Store current_state in traffic light',
    ],
    starterCode: `from enum import Enum

class TrafficLightState(Enum):
    """Traffic light states."""
    RED = 1
    YELLOW = 2
    GREEN = 3


class TrafficLight:
    """Traffic light with state machine."""
    
    # State transitions
    TRANSITIONS = {
        TrafficLightState.RED: TrafficLightState.GREEN,
        TrafficLightState.GREEN: TrafficLightState.YELLOW,
        TrafficLightState.YELLOW: TrafficLightState.RED,
    }
    
    # Duration for each state (seconds)
    DURATIONS = {
        TrafficLightState.RED: 30,
        TrafficLightState.YELLOW: 5,
        TrafficLightState.GREEN: 25,
    }
    
    def __init__(self):
        self.current_state = TrafficLightState.RED
    
    def next_state(self):
        """Transition to next state."""
        pass
    
    def get_duration(self):
        """Get duration for current state."""
        pass
    
    def can_transition_to(self, state):
        """Check if transition to state is valid."""
        pass
    
    def __str__(self):
        return f"Traffic Light: {self.current_state.name}"


# Test
light = TrafficLight()
print(light)  # RED
print(f"Duration: {light.get_duration()}s")

light.next_state()
print(light)  # GREEN

light.next_state()
print(light)  # YELLOW

light.next_state()
print(light)  # RED again


def test_state_machine(initial_state):
    """Test function for State Machine with Enum."""
    from enum import Enum
    
    class TrafficLightState(Enum):
        RED = 1
        YELLOW = 2
        GREEN = 3
    
    light = TrafficLight()
    # Set initial state if specified
    if initial_state == 'RED':
        light.current_state = TrafficLightState.RED
    elif initial_state == 'YELLOW':
        light.current_state = TrafficLightState.YELLOW
    elif initial_state == 'GREEN':
        light.current_state = TrafficLightState.GREEN
    
    light.next_state()
    return light.current_state.name
`,
    testCases: [
      {
        input: ['RED'],
        expected: 'GREEN',
        functionName: 'test_state_machine',
      },
    ],
    solution: `from enum import Enum

class TrafficLightState(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3


class TrafficLight:
    TRANSITIONS = {
        TrafficLightState.RED: TrafficLightState.GREEN,
        TrafficLightState.GREEN: TrafficLightState.YELLOW,
        TrafficLightState.YELLOW: TrafficLightState.RED,
    }
    
    DURATIONS = {
        TrafficLightState.RED: 30,
        TrafficLightState.YELLOW: 5,
        TrafficLightState.GREEN: 25,
    }
    
    def __init__(self):
        self.current_state = TrafficLightState.RED
    
    def next_state(self):
        self.current_state = self.TRANSITIONS[self.current_state]
        return self.current_state
    
    def get_duration(self):
        return self.DURATIONS[self.current_state]
    
    def can_transition_to(self, state):
        return self.TRANSITIONS[self.current_state] == state
    
    def force_state(self, state):
        """Force transition to any state (e.g., for maintenance)."""
        if not isinstance(state, TrafficLightState):
            raise TypeError("State must be TrafficLightState")
        self.current_state = state
    
    def __str__(self):
        return f"Traffic Light: {self.current_state.name}"


def test_state_machine(initial_state):
    """Test function for State Machine with Enum."""
    from enum import Enum
    
    class TrafficLightState(Enum):
        RED = 1
        YELLOW = 2
        GREEN = 3
    
    light = TrafficLight()
    # Set initial state if specified
    if initial_state == 'RED':
        light.current_state = TrafficLightState.RED
    elif initial_state == 'YELLOW':
        light.current_state = TrafficLightState.YELLOW
    elif initial_state == 'GREEN':
        light.current_state = TrafficLightState.GREEN
    
    light.next_state()
    return light.current_state.name`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 13,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'complex-number-magic-methods',
    title: 'Complex Number with Magic Methods',
    difficulty: 'Medium',
    category: 'python-oop',
    description: `Create a \`ComplexNumber\` class that implements magic methods for arithmetic operations, comparison, and string representation.

Implement these magic methods:
- \`__init__(real, imag)\`: Initialize with real and imaginary parts
- \`__str__()\`: Return user-friendly string like "3 + 4i"
- \`__repr__()\`: Return developer string like "ComplexNumber(3, 4)"
- \`__add__(other)\`: Add two complex numbers
- \`__sub__(other)\`: Subtract complex numbers
- \`__mul__(other)\`: Multiply complex numbers
- \`__eq__(other)\`: Check equality
- \`__abs__()\`: Return magnitude (distance from origin)

**Examples:**
\`\`\`python
c1 = ComplexNumber(3, 4)
c2 = ComplexNumber(1, 2)

print(c1)           # "3 + 4i"
print(repr(c1))     # "ComplexNumber(3, 4)"
print(c1 + c2)      # "4 + 6i"
print(c1 * c2)      # "-5 + 10i"  (3+4i)*(1+2i) = 3 + 6i + 4i + 8i² = -5 + 10i
print(abs(c1))      # 5.0  (sqrt(3² + 4²))
print(c1 == ComplexNumber(3, 4))  # True
\`\`\`

**Constraints:**
- Handle negative imaginary parts correctly in \`__str__\`
- \`__abs__\` should return a float`,
    starterCode: `class ComplexNumber:
    def __init__(self, real, imag):
        """Initialize complex number with real and imaginary parts."""
        pass
    
    def __str__(self):
        """Return user-friendly string representation."""
        pass
    
    def __repr__(self):
        """Return developer-friendly representation."""
        pass
    
    def __add__(self, other):
        """Add two complex numbers."""
        pass
    
    def __sub__(self, other):
        """Subtract complex numbers."""
        pass
    
    def __mul__(self, other):
        """Multiply complex numbers."""
        pass
    
    def __eq__(self, other):
        """Check equality."""
        pass
    
    def __abs__(self):
        """Return magnitude."""
        pass`,
    testCases: [
      {
        input: [
          ['ComplexNumber', 3, 4],
          ['ComplexNumber', 1, 2],
          ['add'],
          ['str'],
        ],
        expected: '4 + 6i',
      },
      {
        input: [['ComplexNumber', 3, 4], ['abs']],
        expected: 5.0,
      },
      {
        input: [
          ['ComplexNumber', 3, 4],
          ['ComplexNumber', 1, 2],
          ['multiply'],
          ['str'],
        ],
        expected: '-5 + 10i',
      },
    ],
    hints: [
      'For __str__, handle negative imaginary with f"{real} - {abs(imag)}i"',
      'For __mul__, use (a+bi)*(c+di) = (ac-bd) + (ad+bc)i',
      'For __abs__, use sqrt(real² + imag²)',
      'Always check isinstance(other, ComplexNumber) in operations',
    ],
    solution: `import math

class ComplexNumber:
    def __init__(self, real, imag):
        """Initialize complex number with real and imaginary parts."""
        self.real = real
        self.imag = imag
    
    def __str__(self):
        """Return user-friendly string representation."""
        if self.imag >= 0:
            return f"{self.real} + {self.imag}i"
        else:
            return f"{self.real} - {abs(self.imag)}i"
    
    def __repr__(self):
        """Return developer-friendly representation."""
        return f"ComplexNumber({self.real}, {self.imag})"
    
    def __add__(self, other):
        """Add two complex numbers."""
        if not isinstance(other, ComplexNumber):
            return NotImplemented
        return ComplexNumber(self.real + other.real, self.imag + other.imag)
    
    def __sub__(self, other):
        """Subtract complex numbers."""
        if not isinstance(other, ComplexNumber):
            return NotImplemented
        return ComplexNumber(self.real - other.real, self.imag - other.imag)
    
    def __mul__(self, other):
        """Multiply complex numbers: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i"""
        if not isinstance(other, ComplexNumber):
            return NotImplemented
        real_part = self.real * other.real - self.imag * other.imag
        imag_part = self.real * other.imag + self.imag * other.real
        return ComplexNumber(real_part, imag_part)
    
    def __eq__(self, other):
        """Check equality."""
        if not isinstance(other, ComplexNumber):
            return False
        return self.real == other.real and self.imag == other.imag
    
    def __abs__(self):
        """Return magnitude: sqrt(real² + imag²)"""
        return math.sqrt(self.real ** 2 + self.imag ** 2)


# Test
c1 = ComplexNumber(3, 4)
c2 = ComplexNumber(1, 2)
print(c1 + c2)      # 4 + 6i
print(c1 * c2)      # -5 + 10i
print(abs(c1))      # 5.0`,
    timeComplexity: 'O(1) for all operations',
    spaceComplexity: 'O(1)',
    order: 14,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'custom-list-magic-methods',
    title: 'Custom List with Magic Methods',
    difficulty: 'Medium',
    category: 'python-oop',
    description: `Create a \`MyList\` class that behaves like Python's built-in list using magic methods.

Implement:
- \`__init__(items=[])\`: Initialize with items
- \`__len__()\`: Support \`len()\`
- \`__getitem__(index)\`: Support indexing \`mylist[i]\`
- \`__setitem__(index, value)\`: Support assignment \`mylist[i] = val\`
- \`__contains__(item)\`: Support \`in\` operator
- \`__iter__()\`: Support iteration
- \`__str__()\`: Return string like "[1, 2, 3]"
- \`append(item)\`: Add item to end

**Examples:**
\`\`\`python
mylist = MyList([1, 2, 3])
print(len(mylist))      # 3
print(mylist[0])        # 1
print(2 in mylist)      # True
mylist.append(4)
print(str(mylist))      # "[1, 2, 3, 4]"

for item in mylist:
    print(item)         # 1, 2, 3, 4
\`\`\``,
    starterCode: `class MyList:
    def __init__(self, items=None):
        """Initialize with items."""
        pass
    
    def __len__(self):
        """Return length."""
        pass
    
    def __getitem__(self, index):
        """Get item by index."""
        pass
    
    def __setitem__(self, index, value):
        """Set item by index."""
        pass
    
    def __contains__(self, item):
        """Check if item exists."""
        pass
    
    def __iter__(self):
        """Make iterable."""
        pass
    
    def __str__(self):
        """String representation."""
        pass
    
    def append(self, item):
        """Add item to end."""
        pass`,
    testCases: [
      {
        input: [['MyList', [1, 2, 3]], ['len']],
        expected: 3,
      },
      {
        input: [['MyList', [1, 2, 3]], ['getitem', 1]],
        expected: 2,
      },
      {
        input: [['MyList', [1, 2, 3]], ['contains', 2]],
        expected: true,
      },
    ],
    hints: [
      'Store items in internal list: self._items = items or []',
      'Delegate most operations to self._items',
      '__iter__ should return iter(self._items)',
      '__str__ can use str(self._items)',
    ],
    solution: `class MyList:
    def __init__(self, items=None):
        """Initialize with items."""
        self._items = items if items is not None else []
    
    def __len__(self):
        """Return length."""
        return len(self._items)
    
    def __getitem__(self, index):
        """Get item by index."""
        return self._items[index]
    
    def __setitem__(self, index, value):
        """Set item by index."""
        self._items[index] = value
    
    def __contains__(self, item):
        """Check if item exists."""
        return item in self._items
    
    def __iter__(self):
        """Make iterable."""
        return iter(self._items)
    
    def __str__(self):
        """String representation."""
        return str(self._items)
    
    def append(self, item):
        """Add item to end."""
        self._items.append(item)


# Test
mylist = MyList([1, 2, 3])
print(len(mylist))      # 3
print(mylist[0])        # 1
print(2 in mylist)      # True
mylist.append(4)
print(mylist)           # [1, 2, 3, 4]

for item in mylist:
    print(item)`,
    timeComplexity: 'O(1) for most operations, O(n) for __contains__',
    spaceComplexity: 'O(n)',
    order: 15,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'vector-comparison-magic',
    title: 'Vector with Comparison Magic Methods',
    difficulty: 'Medium',
    category: 'python-oop',
    description: `Create a \`Vector\` class that supports comparison operations based on magnitude (length).

Implement:
- \`__init__(x, y)\`: Initialize vector
- \`__eq__(other)\`: Equal if same magnitude
- \`__lt__(other)\`: Less than if smaller magnitude
- \`__le__(other)\`: Less than or equal
- \`__gt__(other)\`: Greater than
- \`__ge__(other)\`: Greater than or equal
- \`magnitude()\`: Return vector length
- \`__repr__()\`: Return "Vector(x, y)"

Use \`@functools.total_ordering\` to implement only \`__eq__\` and \`__lt__\`.

**Examples:**
\`\`\`python
v1 = Vector(3, 4)      # magnitude = 5
v2 = Vector(0, 5)      # magnitude = 5
v3 = Vector(1, 1)      # magnitude ≈ 1.41

print(v1 == v2)        # True (same magnitude)
print(v1 > v3)         # True (5 > 1.41)
print(sorted([v1, v3, v2]))  # [Vector(1, 1), Vector(3, 4), Vector(0, 5)]
\`\`\``,
    starterCode: `from functools import total_ordering
import math

@total_ordering
class Vector:
    def __init__(self, x, y):
        """Initialize vector."""
        pass
    
    def magnitude(self):
        """Calculate vector magnitude."""
        pass
    
    def __eq__(self, other):
        """Check if magnitudes are equal."""
        pass
    
    def __lt__(self, other):
        """Check if magnitude is less than."""
        pass
    
    def __repr__(self):
        """Return string representation."""
        pass`,
    testCases: [
      {
        input: [['Vector', 3, 4], ['Vector', 0, 5], ['equals']],
        expected: true,
      },
      {
        input: [['Vector', 3, 4], ['Vector', 1, 1], ['greater']],
        expected: true,
      },
      {
        input: [['Vector', 3, 4], ['magnitude']],
        expected: 5.0,
      },
    ],
    hints: [
      'magnitude = sqrt(x² + y²)',
      '@total_ordering auto-generates <=, >, >= from __eq__ and __lt__',
      'Compare using magnitude() in __eq__ and __lt__',
      'Check isinstance(other, Vector) before comparing',
    ],
    solution: `from functools import total_ordering
import math

@total_ordering
class Vector:
    def __init__(self, x, y):
        """Initialize vector."""
        self.x = x
        self.y = y
    
    def magnitude(self):
        """Calculate vector magnitude."""
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def __eq__(self, other):
        """Check if magnitudes are equal."""
        if not isinstance(other, Vector):
            return NotImplemented
        return self.magnitude() == other.magnitude()
    
    def __lt__(self, other):
        """Check if magnitude is less than."""
        if not isinstance(other, Vector):
            return NotImplemented
        return self.magnitude() < other.magnitude()
    
    def __repr__(self):
        """Return string representation."""
        return f"Vector({self.x}, {self.y})"


# Test
v1 = Vector(3, 4)      # magnitude = 5
v2 = Vector(0, 5)      # magnitude = 5
v3 = Vector(1, 1)      # magnitude ≈ 1.41

print(v1 == v2)        # True
print(v1 > v3)         # True
print(v1 <= v2)        # True (auto-generated!)
print(sorted([v1, v3, v2]))  # Sorted by magnitude`,
    timeComplexity: 'O(1) for all operations',
    spaceComplexity: 'O(1)',
    order: 16,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'counter-callable-magic',
    title: 'Callable Counter',
    difficulty: 'Easy',
    category: 'python-oop',
    description: `Create a \`Counter\` class that tracks how many times it has been called.

Make the instance callable using \`__call__\` magic method.

Implement:
- \`__init__()\`: Initialize count to 0
- \`__call__()\`: Increment count and return current count
- \`get_count()\`: Return current count
- \`reset()\`: Reset count to 0

**Examples:**
\`\`\`python
counter = Counter()
print(counter())        # 1
print(counter())        # 2
print(counter())        # 3
print(counter.get_count())  # 3
counter.reset()
print(counter())        # 1
\`\`\``,
    starterCode: `class Counter:
    def __init__(self):
        """Initialize counter."""
        pass
    
    def __call__(self):
        """Increment and return count."""
        pass
    
    def get_count(self):
        """Return current count."""
        pass
    
    def reset(self):
        """Reset count to 0."""
        pass`,
    testCases: [
      {
        input: [['Counter'], ['call'], ['call'], ['call']],
        expected: 3,
      },
      {
        input: [['Counter'], ['call'], ['get_count']],
        expected: 2,
      },
      {
        input: [['Counter'], ['call'], ['call'], ['reset'], ['call']],
        expected: 1,
      },
    ],
    hints: [
      '__call__ makes instances callable like functions',
      'Increment self.count in __call__',
      'Return the new count after incrementing',
    ],
    solution: `class Counter:
    def __init__(self):
        """Initialize counter."""
        self.count = 0
    
    def __call__(self):
        """Increment and return count."""
        self.count += 1
        return self.count
    
    def get_count(self):
        """Return current count."""
        return self.count
    
    def reset(self):
        """Reset count to 0."""
        self.count = 0


# Test
counter = Counter()
print(counter())        # 1
print(counter())        # 2
print(counter())        # 3
print(counter.get_count())  # 3
counter.reset()
print(counter())        # 1

# Can pass as function!
def apply_twice(func):
    func()
    func()

apply_twice(counter)  # counter is callable!`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 17,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'hashable-person',
    title: 'Hashable Person Class',
    difficulty: 'Medium',
    category: 'python-oop',
    description: `Create a \`Person\` class that can be used in sets and as dictionary keys.

Implement:
- \`__init__(name, age, email)\`: Initialize person
- \`__eq__(other)\`: Two people are equal if same email
- \`__hash__()\`: Hash based on email (immutable identifier)
- \`__repr__()\`: Return "Person(name, age, email)"

**Why this matters:** For objects to work in sets/dicts, they must be hashable. If you implement \`__eq__\`, you must implement \`__hash__\` such that equal objects have equal hashes.

**Examples:**
\`\`\`python
p1 = Person("Alice", 30, "alice@example.com")
p2 = Person("Alice Smith", 30, "alice@example.com")  # Same email
p3 = Person("Bob", 25, "bob@example.com")

print(p1 == p2)        # True (same email)
print(p1 is p2)        # False (different objects)

people = {p1, p2, p3}  # Set treats p1 and p2 as same
print(len(people))     # 2

lookup = {p1: "Manager", p3: "Engineer"}
print(lookup[p2])      # "Manager" (p2 treated same as p1)
\`\`\``,
    starterCode: `class Person:
    def __init__(self, name, age, email):
        """Initialize person."""
        pass
    
    def __eq__(self, other):
        """Check equality based on email."""
        pass
    
    def __hash__(self):
        """Return hash based on email."""
        pass
    
    def __repr__(self):
        """Return string representation."""
        pass`,
    testCases: [
      {
        input: [
          ['Person', 'Alice', 30, 'alice@example.com'],
          ['Person', 'Alice Smith', 30, 'alice@example.com'],
          ['equals'],
        ],
        expected: true,
      },
      {
        input: [
          ['Person', 'Alice', 30, 'alice@example.com'],
          ['Person', 'Alice', 30, 'alice@example.com'],
          ['set_len'],
        ],
        expected: 1,
      },
      {
        input: [['Person', 'Alice', 30, 'alice@example.com'], ['hash_equals_self']],
        expected: true,
      },
    ],
    hints: [
      'Email is the unique identifier (like SSN)',
      '__hash__ should return hash(self.email)',
      '__eq__ should check if emails are equal',
      'Always check isinstance(other, Person) in __eq__',
    ],
    solution: `class Person:
    def __init__(self, name, age, email):
        """Initialize person."""
        self.name = name
        self.age = age
        self.email = email
    
    def __eq__(self, other):
        """Check equality based on email."""
        if not isinstance(other, Person):
            return False
        return self.email == other.email
    
    def __hash__(self):
        """Return hash based on email (immutable identifier)."""
        return hash(self.email)
    
    def __repr__(self):
        """Return string representation."""
        return f"Person('{self.name}', {self.age}, '{self.email}')"


# Test
p1 = Person("Alice", 30, "alice@example.com")
p2 = Person("Alice Smith", 30, "alice@example.com")  # Same email!
p3 = Person("Bob", 25, "bob@example.com")

print(p1 == p2)        # True
print(hash(p1) == hash(p2))  # True

# Use in set
people = {p1, p2, p3}
print(len(people))     # 2 (p1 and p2 treated as one)

# Use as dict key
lookup = {p1: "Manager", p3: "Engineer"}
print(lookup[p2])      # "Manager" (p2 same as p1)`,
    timeComplexity: 'O(1) for all operations',
    spaceComplexity: 'O(1)',
    order: 18,
    topic: 'Python Object-Oriented Programming',
  },
];

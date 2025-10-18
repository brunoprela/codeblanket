/**
 * Vehicle Factory Pattern
 * Problem ID: oop-vehicle-factory
 * Order: 4
 */

import { Problem } from '../../../types';

export const vehicle_factoryProblem: Problem = {
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
};

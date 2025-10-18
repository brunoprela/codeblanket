/**
 * Design Parking Lot
 * Problem ID: parking-lot
 * Order: 11
 */

import { Problem } from '../../../types';

export const parking_lotProblem: Problem = {
  id: 'parking-lot',
  title: 'Design Parking Lot',
  difficulty: 'Hard',
  topic: 'Design Problems',
  description: `Design a parking lot system that can:

1. Park vehicles (cars, trucks, motorcycles) in available spots
2. Remove vehicles from parking spots
3. Track available spots by type (compact, large, handicapped, motorcycle)
4. Find nearest available spot efficiently

The parking lot has multiple levels, and each spot has a type. Vehicles can only park in compatible spot types:
- Motorcycles can park in motorcycle spots
- Cars can park in compact, large, or handicapped spots
- Trucks can only park in large spots

Implement the \`ParkingLot\` class with proper object-oriented design principles.`,
  hints: [
    'Use inheritance for vehicle types (Vehicle → Car/Truck/Motorcycle)',
    'Each spot type has different constraints',
    'Min heap to efficiently find nearest available spot (by level, row)',
    'HashMap to track which spot each vehicle is in',
    'Encapsulate spot state management within ParkingSpot class',
  ],
  approach: `## Intuition

This is an **Object-Oriented Design** problem testing:
- Class hierarchy (inheritance)
- Encapsulation
- Composition
- Data structure selection

---

## Class Design

### 1. Vehicle Hierarchy

\`\`\`python
class Vehicle (ABC):
    - license_plate
    - type
    - can_fit_in(spot) → bool  # Abstract method

class Car(Vehicle):
    can_fit_in() → compact, large, handicapped spots

class Truck(Vehicle):
    can_fit_in() → only large spots

class Motorcycle(Vehicle):
    can_fit_in() → only motorcycle spots
\`\`\`

**Polymorphism**: \`vehicle.can_fit_in(spot)\` works for any vehicle type.

### 2. ParkingSpot

\`\`\`python
class ParkingSpot:
    - id, type, level, row
    - vehicle (currently parked)
    
    - is_available() → bool
    - park_vehicle(vehicle) → bool
    - remove_vehicle()
\`\`\`

**Encapsulation**: Spot manages its own state.

### 3. ParkingLot

\`\`\`python
class ParkingLot:
    - spots: HashMap(spot_id → ParkingSpot)
    - vehicle_to_spot: HashMap(license_plate → spot_id)
    - available_spots: HashMap(spot_type → MinHeap)
    
    - park_vehicle(vehicle) → spot_id
    - remove_vehicle(license_plate) → bool
\`\`\`

**Key Design**: Min heap per spot type for O(log N) nearest spot.

---

## Why Min Heap?

Need to find **nearest** available spot (by level, then row).

- Without heap: O(N) linear search through all spots
- With heap: O(log N) to get nearest spot

Heap maintains spots sorted by (level, row), so \`heap[0]\` is always closest.

---

## Time Complexity:
- park_vehicle: O(log N) where N = spots of compatible type
- remove_vehicle: O(log N) to re-add spot to heap

## Space Complexity: O(S + V) where S = spots, V = vehicles`,
  testCases: [
    {
      input: [
        ['ParkingLot'],
        ['add_spot', 'compact', 1, 1],
        ['add_spot', 'large', 1, 2],
        ['park_vehicle', 'car'],
        ['park_vehicle', 'truck'],
        ['remove_vehicle', 'car'],
      ],
      expected:
        'Creates parking lot with 2 spots, parks car in compact, truck in large, removes car',
    },
  ],
  solution: `from enum import Enum
from abc import ABC, abstractmethod
import heapq

class VehicleType(Enum):
    COMPACT = 1
    LARGE = 2
    MOTORCYCLE = 3

class SpotType(Enum):
    COMPACT = 1
    LARGE = 2
    HANDICAPPED = 3
    MOTORCYCLE = 4

class Vehicle(ABC):
    """Abstract vehicle class"""
    def __init__(self, license_plate: str):
        self.license_plate = license_plate
        self.type = None
    
    @abstractmethod
    def can_fit_in(self, spot) -> bool:
        """Check if vehicle can fit in given spot"""
        pass

class Car(Vehicle):
    def __init__(self, license_plate: str):
        super().__init__(license_plate)
        self.type = VehicleType.COMPACT
    
    def can_fit_in(self, spot) -> bool:
        return spot.type in [SpotType.COMPACT, SpotType.LARGE, SpotType.HANDICAPPED]

class Truck(Vehicle):
    def __init__(self, license_plate: str):
        super().__init__(license_plate)
        self.type = VehicleType.LARGE
    
    def can_fit_in(self, spot) -> bool:
        return spot.type == SpotType.LARGE

class Motorcycle(Vehicle):
    def __init__(self, license_plate: str):
        super().__init__(license_plate)
        self.type = VehicleType.MOTORCYCLE
    
    def can_fit_in(self, spot) -> bool:
        return spot.type == SpotType.MOTORCYCLE

class ParkingSpot:
    """Parking spot that manages its own state"""
    def __init__(self, spot_id: int, spot_type: SpotType, level: int, row: int):
        self.id = spot_id
        self.type = spot_type
        self.level = level
        self.row = row
        self.vehicle = None
    
    def is_available(self) -> bool:
        return self.vehicle is None
    
    def park_vehicle(self, vehicle: Vehicle) -> bool:
        """Park vehicle if compatible and spot available"""
        if not self.is_available():
            return False
        if not vehicle.can_fit_in(self):
            return False
        self.vehicle = vehicle
        return True
    
    def remove_vehicle(self) -> None:
        self.vehicle = None

class ParkingLot:
    """Main parking lot system"""
    def __init__(self):
        self.spots = {}  # spot_id -> ParkingSpot
        self.vehicle_to_spot = {}  # license_plate -> spot_id
        # Min heaps for each spot type (sorted by level, row)
        self.available_spots = {
            SpotType.COMPACT: [],
            SpotType.LARGE: [],
            SpotType.HANDICAPPED: [],
            SpotType.MOTORCYCLE: []
        }
    
    def add_spot(self, spot: ParkingSpot) -> None:
        """Add spot to parking lot"""
        self.spots[spot.id] = spot
        heapq.heappush(
            self.available_spots[spot.type],
            (spot.level, spot.row, spot.id)
        )
    
    def park_vehicle(self, vehicle: Vehicle):
        """Park vehicle, returns spot_id or None"""
        spot_id = self._find_available_spot(vehicle)
        if not spot_id:
            return None  # No compatible spots available
        
        spot = self.spots[spot_id]
        if spot.park_vehicle(vehicle):
            self.vehicle_to_spot[vehicle.license_plate] = spot_id
            return spot_id
        return None
    
    def _find_available_spot(self, vehicle: Vehicle):
        """Find nearest available compatible spot"""
        # Check all compatible spot types
        compatible_types = []
        if isinstance(vehicle, Car):
            compatible_types = [SpotType.COMPACT, SpotType.LARGE, SpotType.HANDICAPPED]
        elif isinstance(vehicle, Truck):
            compatible_types = [SpotType.LARGE]
        elif isinstance(vehicle, Motorcycle):
            compatible_types = [SpotType.MOTORCYCLE]
        
        # Try each compatible type, find nearest
        for spot_type in compatible_types:
            heap = self.available_spots[spot_type]
            while heap:
                level, row, spot_id = heap[0]
                spot = self.spots[spot_id]
                if spot.is_available() and vehicle.can_fit_in(spot):
                    heapq.heappop(heap)
                    return spot_id
                else:
                    heapq.heappop(heap)  # Remove stale entry
        return None
    
    def remove_vehicle(self, license_plate: str) -> bool:
        """Remove vehicle from parking lot"""
        if license_plate not in self.vehicle_to_spot:
            return False
        
        spot_id = self.vehicle_to_spot[license_plate]
        spot = self.spots[spot_id]
        spot.remove_vehicle()
        
        # Return spot to available pool
        heapq.heappush(
            self.available_spots[spot.type],
            (spot.level, spot.row, spot_id)
        )
        
        del self.vehicle_to_spot[license_plate]
        return True

# Example usage:
# lot = ParkingLot()
# lot.add_spot(ParkingSpot(1, SpotType.COMPACT, level=1, row=1))
# lot.add_spot(ParkingSpot(2, SpotType.LARGE, level=1, row=2))
# car = Car("ABC123")
# spot_id = lot.park_vehicle(car)  # Parks in spot 1
# lot.remove_vehicle("ABC123")     # Frees spot 1`,
  timeComplexity:
    'park_vehicle: O(log N), remove_vehicle: O(log N) where N = spots of type',
  spaceComplexity:
    'O(S + V) where S = total spots, V = currently parked vehicles',
  patterns: ['OOP Design', 'Inheritance', 'Heap', 'HashMap'],
  companies: ['Amazon', 'Microsoft', 'Google', 'Uber'],
};

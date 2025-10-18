/**
 * System Design Basics Section
 */

export const systemdesignbasicsSection = {
  id: 'system-design-basics',
  title: 'System Design Basics',
  content: `System design problems go beyond single machines, asking you to design **scalable, distributed systems**. While full system design is beyond scope here, these problems introduce key concepts.

---

## Design Parking Lot

**Problem**: Design a parking lot system that can:
- Park cars
- Remove cars  
- Track available spots
- Support different spot types (compact, large, handicapped)

This is an **Object-Oriented Design** problem testing class structure and relationships.

### Class Design

\`\`\`python
from enum import Enum
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
    def __init__(self, license_plate):
        self.license_plate = license_plate
        self.type = None
    
    @abstractmethod
    def can_fit_in(self, spot):
        pass

class Car(Vehicle):
    def __init__(self, license_plate):
        super().__init__(license_plate)
        self.type = VehicleType.COMPACT
    
    def can_fit_in(self, spot):
        return spot.type in [SpotType.COMPACT, SpotType.LARGE, 
                             SpotType.HANDICAPPED]

class Truck(Vehicle):
    def __init__(self, license_plate):
        super().__init__(license_plate)
        self.type = VehicleType.LARGE
    
    def can_fit_in(self, spot):
        return spot.type == SpotType.LARGE

class ParkingSpot:
    def __init__(self, spot_id, spot_type, level, row):
        self.id = spot_id
        self.type = spot_type
        self.level = level
        self.row = row
        self.vehicle = None
    
    def is_available(self):
        return self.vehicle is None
    
    def park_vehicle(self, vehicle):
        if not self.is_available():
            return False
        if not vehicle.can_fit_in(self):
            return False
        self.vehicle = vehicle
        return True
    
    def remove_vehicle(self):
        self.vehicle = None

class ParkingLot:
    def __init__(self):
        self.spots = {}  # spot_id -> ParkingSpot
        self.vehicle_to_spot = {}  # license_plate -> spot_id
        # Min heaps for each type (for O(1) nearest spot)
        self.available_spots = {
            SpotType.COMPACT: [],
            SpotType.LARGE: [],
            SpotType.HANDICAPPED: [],
            SpotType.MOTORCYCLE: []
        }
    
    def add_spot(self, spot):
        self.spots[spot.id] = spot
        heapq.heappush(self.available_spots[spot.type], 
                      (spot.level, spot.row, spot.id))
    
    def park_vehicle(self, vehicle):
        # Find best available spot
        spot_id = self._find_available_spot(vehicle)
        if not spot_id:
            return None  # No spots available
        
        spot = self.spots[spot_id]
        if spot.park_vehicle(vehicle):
            self.vehicle_to_spot[vehicle.license_plate] = spot_id
            return spot_id
        return None
    
    def _find_available_spot(self, vehicle):
        # Try to find spot (check each compatible type)
        for spot_type in [SpotType.COMPACT, SpotType.LARGE, 
                         SpotType.HANDICAPPED]:
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
    
    def remove_vehicle(self, license_plate):
        if license_plate not in self.vehicle_to_spot:
            return False
        
        spot_id = self.vehicle_to_spot[license_plate]
        spot = self.spots[spot_id]
        spot.remove_vehicle()
        
        # Return spot to available pool
        heapq.heappush(self.available_spots[spot.type],
                      (spot.level, spot.row, spot_id))
        
        del self.vehicle_to_spot[license_plate]
        return True
\`\`\`

**Key Concepts**:
- **Inheritance**: Vehicle -> Car/Truck
- **Composition**: ParkingLot has many ParkingSpots
- **Encapsulation**: Spot manages its own state
- **Min Heap**: O(log N) find nearest spot

---

## Design URL Shortener

**Problem**: Design a service like bit.ly that:
- Creates short URLs from long URLs
- Redirects short URLs to original URLs
- Tracks click counts

### Approach 1: Hash-based

\`\`\`python
import hashlib

class URLShortener:
    def __init__(self):
        self.url_to_short = {}  # long -> short
        self.short_to_url = {}  # short -> long
        self.base_url = "http://short.url/"
    
    def shorten(self, long_url):
        if long_url in self.url_to_short:
            return self.base_url + self.url_to_short[long_url]
        
        # Generate short code using hash
        hash_val = hashlib.md5(long_url.encode()).hexdigest()
        short_code = hash_val[:7]  # Take first 7 chars
        
        # Handle collisions
        counter = 0
        while short_code in self.short_to_url:
            short_code = hash_val[:7] + str(counter)
            counter += 1
        
        self.url_to_short[long_url] = short_code
        self.short_to_url[short_code] = long_url
        
        return self.base_url + short_code
    
    def expand(self, short_url):
        short_code = short_url.replace(self.base_url, "")
        return self.short_to_url.get(short_code)
\`\`\`

### Approach 2: Counter-based (Better)

\`\`\`python
class URLShortener:
    def __init__(self):
        self.counter = 0
        self.short_to_url = {}
        self.url_to_short = {}
        self.base62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    def encode_base62(self, num):
        """Convert number to base62 string"""
        if num == 0:
            return self.base62[0]
        
        result = []
        while num:
            result.append(self.base62[num % 62])
            num //= 62
        return ''.join(reversed(result))
    
    def shorten(self, long_url):
        if long_url in self.url_to_short:
            return self.url_to_short[long_url]
        
        self.counter += 1
        short_code = self.encode_base62(self.counter)
        
        short_url = "http://short.url/" + short_code
        self.url_to_short[long_url] = short_url
        self.short_to_url[short_code] = long_url
        
        return short_url
    
    def expand(self, short_url):
        short_code = short_url.split("/")[-1]
        return self.short_to_url.get(short_code)
\`\`\`

**Why Base62?**
- 62 characters: 0-9, a-z, A-Z
- 62^7 = 3.5 trillion possible URLs with 7 characters
- Short and URL-safe (no special characters)

**Time**: O(1) for both shorten and expand  
**Space**: O(N) where N is number of URLs

### Production Considerations

1. **Database**: Store URLs in database (Redis for cache, SQL for persistence)

2. **Distributed Counter**: Use Redis INCR or database auto-increment

3. **Custom Short Codes**: Allow users to choose (e.g., bit.ly/mylink)

4. **Expiration**: Auto-delete old URLs

5. **Analytics**: Track clicks, referrers, locations

6. **Caching**: Cache popular URLs in memory/CDN

7. **Rate Limiting**: Prevent abuse (use Token Bucket!)

---

## Key System Design Concepts

### Scalability
- **Vertical**: Bigger machine (limited, expensive)
- **Horizontal**: More machines (unlimited, complex)

### Load Balancing
- Distribute requests across servers
- Round-robin, least connections, consistent hashing

### Caching
- Application cache (Redis, Memcached)
- CDN (CloudFlare, Akamai)
- Browser cache

### Database
- **SQL**: Relational, ACID, complex queries (PostgreSQL, MySQL)
- **NoSQL**: Scalable, eventual consistency, simple queries (MongoDB, Cassandra)
- **Key-Value**: Ultra-fast, simple (Redis, DynamoDB)

### CAP Theorem
You can only have 2 of 3:
- **Consistency**: All nodes see same data
- **Availability**: System always responds
- **Partition Tolerance**: System works despite network failures

---

## Interview Strategy

1. **Clarify requirements**:
   - Scale: 100 users or 100M users?
   - Read-heavy or write-heavy?
   - Consistency requirements?

2. **Start simple**: Single server, then discuss scaling

3. **Identify bottlenecks**: "Database would be bottleneck at 100M users..."

4. **Propose solutions**: "We could shard the database by user ID..."

5. **Discuss trade-offs**: "NoSQL is faster but less consistent than SQL"

6. **Draw diagrams**: Show client -> load balancer -> servers -> database

**Common Mistakes**:
- Jumping to distributed system without starting simple
- Not asking clarifying questions
- Ignoring trade-offs (everything is perfect!)
- Not considering failure modes`,
};

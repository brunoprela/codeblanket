/**
 * Property Decorators Deep-Dive Section
 */

export const propertiesSection = {
  id: 'properties',
  title: 'Property Decorators Deep-Dive',
  content: `**What are Properties?**
Properties allow you to define methods that behave like attributes, enabling controlled access, validation, and computed values while maintaining a clean interface.

**Basic Property:**
\`\`\`python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius (self):
        """Getter for celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius (self, value):
        """Setter with validation"""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @celsius.deleter
    def celsius (self, value):
        """Deleter (rarely used)"""
        del self._celsius

# Usage - looks like attribute access
temp = Temperature(25)
print(temp.celsius)  # Calls getter
temp.celsius = 30    # Calls setter
# temp.celsius = -300  # Raises ValueError
\`\`\`

**Computed Properties:**
\`\`\`python
class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    @property
    def diameter (self):
        """Computed on-the-fly"""
        return self.radius * 2
    
    @property
    def area (self):
        """No storage, calculated when accessed"""
        return 3.14159 * self.radius ** 2
    
    @property
    def circumference (self):
        return 2 * 3.14159 * self.radius

circle = Circle(5)
print(circle.area)  # 78.53975 (calculated)
circle.radius = 10
print(circle.area)  # 314.159 (recalculated)
\`\`\`

**Property for Validation:**
\`\`\`python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age  # Uses setter
    
    @property
    def age (self):
        return self._age
    
    @age.setter
    def age (self, value):
        if not isinstance (value, int):
            raise TypeError("Age must be an integer")
        if value < 0 or value > 150:
            raise ValueError("Age must be between 0 and 150")
        self._age = value

# Validation happens automatically
person = Person("Alice", 30)
# person.age = -5  # Raises ValueError
# person.age = "30"  # Raises TypeError
\`\`\`

**Read-Only Properties:**
\`\`\`python
class BankAccount:
    def __init__(self, account_number, balance):
        self._account_number = account_number
        self._balance = balance
    
    @property
    def account_number (self):
        """Read-only property (no setter)"""
        return self._account_number
    
    @property
    def balance (self):
        """Read-only balance"""
        return self._balance
    
    def deposit (self, amount):
        """Controlled balance modification"""
        if amount > 0:
            self._balance += amount

account = BankAccount("12345", 1000)
print(account.balance)  # OK
# account.balance = 5000  # AttributeError!
\`\`\`

**Lazy Evaluation with Properties:**
\`\`\`python
class DataProcessor:
    def __init__(self, filename):
        self.filename = filename
        self._data = None  # Not loaded yet
    
    @property
    def data (self):
        """Load data only when first accessed"""
        if self._data is None:
            print(f"Loading {self.filename}...")
            self._data = self._load_data()
        return self._data
    
    def _load_data (self):
        # Expensive operation
        return "Loaded data"

processor = DataProcessor("huge_file.txt")
# Data not loaded yet
print(processor.data)  # Loads now
print(processor.data)  # Uses cached version
\`\`\`

**Property with Type Conversion:**
\`\`\`python
class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price
    
    @property
    def price (self):
        return self._price
    
    @price.setter
    def price (self, value):
        """Auto-convert to float and validate"""
        try:
            value = float (value)
        except (TypeError, ValueError):
            raise ValueError("Price must be a number")
        
        if value < 0:
            raise ValueError("Price cannot be negative")
        
        self._price = round (value, 2)  # Store as 2 decimals

product = Product("Widget", "19.99")  # String accepted
print(product.price)  # 19.99 (float)
product.price = "25"  # Auto-converts
\`\`\`

**Dependent Properties:**
\`\`\`python
class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height
    
    @property
    def width (self):
        return self._width
    
    @width.setter
    def width (self, value):
        if value <= 0:
            raise ValueError("Width must be positive")
        self._width = value
    
    @property
    def height (self):
        return self._height
    
    @height.setter
    def height (self, value):
        if value <= 0:
            raise ValueError("Height must be positive")
        self._height = value
    
    @property
    def area (self):
        """Depends on width and height"""
        return self._width * self._height
    
    @property
    def aspect_ratio (self):
        """Depends on width and height"""
        return self._width / self._height

rect = Rectangle(10, 5)
print(rect.area)  # 50
rect.width = 20
print(rect.area)  # 100 (automatically recalculated)
\`\`\`

**Property vs Regular Method:**

**Use Property When:**
- Getting/setting feels like attribute access
- No parameters needed
- Fast computation (< 0.1s)
- No side effects beyond validation

**Use Method When:**
- Operation is expensive
- Parameters needed
- Operation has side effects
- Operation might fail frequently

\`\`\`python
class User:
    @property
    def full_name (self):
        """Property: simple, no parameters"""
        return f"{self.first_name} {self.last_name}"
    
    def send_email (self, subject, body):
        """Method: has side effects, needs parameters"""
        # Send email logic
        pass
\`\`\`

**Common Patterns:**

**1. Calculated Field:**
\`\`\`python
@property
def bmi (self):
    return self.weight / (self.height ** 2)
\`\`\`

**2. Cached Property (Python 3.8+):**
\`\`\`python
from functools import cached_property

class DataAnalyzer:
    @cached_property
    def statistics (self):
        """Computed once, then cached"""
        print("Computing statistics...")
        return self._compute_stats()
\`\`\`

**3. Aliasing:**
\`\`\`python
@property
def username (self):
    return self.email.split('@')[0]
\`\`\`

**Best Practices:**
- Use properties for simple attribute access
- Keep property getters fast (no heavy computation)
- Validate in setters, not getters
- Don't change object state in getters
- Use \`_underscore\` for internal attributes
- Document property behavior in docstrings`,
};

/**
 * Classes and Objects Section
 */

export const classesobjectsSection = {
  id: 'classes-objects',
  title: 'Classes and Objects',
  content: `**Defining Classes:**
\`\`\`python
class Dog:
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    def __init__(self, name, age):
        # Instance attributes (unique to each instance)
        self.name = name
        self.age = age
    
    def bark(self):
        return f"{self.name} says woof!"
    
    def __str__(self):
        return f"{self.name} is {self.age} years old"

# Creating objects
buddy = Dog("Buddy", 5)
miles = Dog("Miles", 4)

print(buddy.bark())  # "Buddy says woof!"
print(miles)  # "Miles is 4 years old"
\`\`\`

**Instance vs Class Attributes:**
\`\`\`python
class Employee:
    company = "TechCorp"  # Class attribute
    
    def __init__(self, name):
        self.name = name  # Instance attribute

emp1 = Employee("Alice")
emp2 = Employee("Bob")

# Class attribute shared
print(emp1.company)  # "TechCorp"
print(emp2.company)  # "TechCorp"

# Change class attribute affects all
Employee.company = "NewCorp"
print(emp1.company)  # "NewCorp"
\`\`\`

**Special Methods (Dunder Methods):**
\`\`\`python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Point({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

p1 = Point(1, 2)
p2 = Point(3, 4)
print(p1 + p2)  # Point(4, 6)
print(p1 == p2)  # False
\`\`\`

**Properties:**
\`\`\`python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius must be positive")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2

circle = Circle(5)
print(circle.area)  # 78.53975
circle.radius = 10  # Uses setter
\`\`\`

**Class Methods and Static Methods:**
\`\`\`python
class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients
    
    @classmethod
    def margherita(cls):
        return cls(['mozzarella', 'tomatoes'])
    
    @staticmethod
    def is_valid_topping(topping):
        return topping in ['mozzarella', 'tomatoes', 'pepperoni']

# Use class method as factory
pizza = Pizza.margherita()

# Use static method
Pizza.is_valid_topping('mozzarella')  # True
\`\`\``,
};

/**
 * Inheritance and Polymorphism Section
 */

export const inheritanceSection = {
  id: 'inheritance',
  title: 'Inheritance and Polymorphism',
  content: `**Basic Inheritance:**
\`\`\`python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak (self):
        pass

class Dog(Animal):
    def speak (self):
        return f"{self.name} barks"

class Cat(Animal):
    def speak (self):
        return f"{self.name} meows"

dog = Dog("Buddy")
cat = Cat("Whiskers")
print(dog.speak())  # "Buddy barks"
print(cat.speak())  # "Whiskers meows"
\`\`\`

**Method Resolution Order (MRO):**
\`\`\`python
class A:
    def method (self):
        return "A"

class B(A):
    def method (self):
        return "B"

class C(A):
    def method (self):
        return "C"

class D(B, C):
    pass

d = D()
print(d.method())  # "B"
print(D.mro())  # [D, B, C, A, object]
\`\`\`

**super() Function:**
\`\`\`python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area (self):
        return self.width * self.height

class Square(Rectangle):
    def __init__(self, size):
        # Call parent's __init__
        super().__init__(size, size)

square = Square(5)
print(square.area())  # 25
\`\`\`

**Polymorphism:**
\`\`\`python
def make_animals_speak (animals):
    for animal in animals:
        print(animal.speak())

animals = [Dog("Buddy"), Cat("Whiskers"), Dog("Max")]
make_animals_speak (animals)
# Buddy barks
# Whiskers meows  
# Max barks
\`\`\`

**Abstract Base Classes:**
\`\`\`python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area (self):
        pass
    
    @abstractmethod
    def perimeter (self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area (self):
        return 3.14159 * self.radius ** 2
    
    def perimeter (self):
        return 2 * 3.14159 * self.radius

# Cannot instantiate abstract class
# shape = Shape()  # TypeError

circle = Circle(5)  # OK
\`\`\`

**Multiple Inheritance:**
\`\`\`python
class Flyable:
    def fly (self):
        return "Flying!"

class Swimmable:
    def swim (self):
        return "Swimming!"

class Duck(Animal, Flyable, Swimmable):
    def speak (self):
        return f"{self.name} quacks"

duck = Duck("Donald")
print(duck.speak())  # "Donald quacks"
print(duck.fly())    # "Flying!"
print(duck.swim())   # "Swimming!"
\`\`\`

**Composition vs Inheritance:**
\`\`\`python
# Inheritance (is-a)
class Employee(Person):
    pass

# Composition (has-a)
class Car:
    def __init__(self):
        self.engine = Engine()
        self.wheels = [Wheel() for _ in range(4)]
    
    def start (self):
        self.engine.start()
\`\`\``,
};

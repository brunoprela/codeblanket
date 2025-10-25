/**
 * Composition Over Inheritance Section
 */

export const compositionSection = {
  id: 'composition',
  title: 'Composition Over Inheritance',
  content: `**What is Composition?**
Composition is a design principle where you build complex objects by combining simpler objects (has-a relationships) rather than inheriting from them (is-a relationships).

**Why Composition Matters:**
- More flexible than inheritance
- Avoids deep inheritance hierarchies
- Easier to test and modify
- Components can be reused independently
- Reduces coupling between classes

**Inheritance vs Composition:**

**❌ Bad: Inheritance Abuse**
\`\`\`python
class Engine:
    def start (self):
        return "Engine starting..."

class Wheels:
    def rotate (self):
        return "Wheels rotating..."

class Stereo:
    def play (self):
        return "Music playing..."

# Wrong: Car "is-a" Engine? No!
class Car(Engine, Wheels, Stereo):
    pass

# Problems:
# 1. Car inherits methods it might not need
# 2. Can't easily swap engine types
# 3. Tight coupling
# 4. Multiple inheritance complexity
\`\`\`

**✅ Good: Composition**
\`\`\`python
class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower
    
    def start (self):
        return f"{self.horsepower}hp engine starting..."

class Wheels:
    def __init__(self, count=4):
        self.count = count
    
    def rotate (self):
        return f"{self.count} wheels rotating..."

class Stereo:
    def __init__(self, brand):
        self.brand = brand
    
    def play (self):
        return f"{self.brand} stereo playing..."

# Right: Car "has-a" Engine, Wheels, Stereo
class Car:
    def __init__(self, engine, wheels, stereo):
        self.engine = engine  # Composition
        self.wheels = wheels  # Composition
        self.stereo = stereo  # Composition
    
    def start (self):
        """Delegates to components"""
        return f"{self.engine.start()}\\n{self.wheels.rotate()}"
    
    def play_music (self):
        return self.stereo.play()

# Easy to swap components!
v6_engine = Engine(300)
wheels = Wheels(4)
bose_stereo = Stereo("Bose")

car = Car (v6_engine, wheels, bose_stereo)
print(car.start())
print(car.play_music())

# Can easily create different configurations
electric_engine = Engine(400)
sport_car = Car (electric_engine, wheels, bose_stereo)
\`\`\`

**Delegation Pattern:**
\`\`\`python
class Logger:
    """Handles all logging"""
    def log (self, message):
        print(f"[LOG] {message}")

class Database:
    """Handles database operations"""
    def save (self, data):
        print(f"Saving {data} to database")

class UserService:
    """Composes logger and database"""
    def __init__(self):
        self.logger = Logger()  # Composition
        self.db = Database()    # Composition
    
    def create_user (self, name):
        """Delegates to composed objects"""
        self.logger.log (f"Creating user: {name}")
        self.db.save({'user': name})
        self.logger.log("User created successfully")
        return name

service = UserService()
service.create_user("Alice")
\`\`\`

**Strategy Pattern with Composition:**
\`\`\`python
class PaymentProcessor:
    """Different payment strategies"""
    def pay (self, amount):
        raise NotImplementedError

class CreditCardPayment(PaymentProcessor):
    def pay (self, amount):
        return f"Paid \${amount} with credit card"

class PayPalPayment(PaymentProcessor):
    def pay (self, amount):
        return f"Paid \${amount} with PayPal"

class CryptoPayment(PaymentProcessor):
    def pay (self, amount):
        return f"Paid \${amount} with crypto"

class ShoppingCart:
    def __init__(self, payment_processor):
        self.items = []
        self.payment_processor = payment_processor  # Composition
    
    def add_item (self, item, price):
        self.items.append((item, price))
    
    def checkout (self):
        total = sum (price for _, price in self.items)
        return self.payment_processor.pay (total)

# Easy to swap payment methods!
cart = ShoppingCart(CreditCardPayment())
cart.add_item("Book", 25)
cart.add_item("Pen", 5)
print(cart.checkout())  # Credit card payment

# Change payment method
cart.payment_processor = PayPalPayment()
print(cart.checkout())  # PayPal payment
\`\`\`

**Mixin Pattern (Composition via Inheritance):**
\`\`\`python
class JSONSerializableMixin:
    """Adds JSON serialization capability"""
    def to_json (self):
        import json
        return json.dumps (self.__dict__)

class LoggableMixin:
    """Adds logging capability"""
    def log (self, message):
        print(f"[{self.__class__.__name__}] {message}")

class User(JSONSerializableMixin, LoggableMixin):
    """Composes behaviors via mixins"""
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def save (self):
        self.log("Saving user...")
        print(self.to_json())
        self.log("User saved")

user = User("Alice", "alice@example.com")
user.save()
\`\`\`

**When to Use Each:**

**Use Inheritance When:**
- True "is-a" relationship (Dog is an Animal)
- Shared interface and behavior
- Polymorphism needed
- Liskov Substitution Principle holds

**Use Composition When:**
- "Has-a" or "uses-a" relationship
- Need flexibility to swap components
- Want to avoid deep hierarchies
- Multiple capabilities from different sources
- Want easier testing (can mock components)

**Real-World Example - Game Character:**
\`\`\`python
# ❌ Bad: Deep inheritance
class Character: pass
class Warrior(Character): pass
class MagicWarrior(Warrior): pass
class HealingMagicWarrior(MagicWarrior): pass  # Too specific!

# ✅ Good: Composition
class Weapon:
    def __init__(self, damage):
        self.damage = damage
    
    def attack (self):
        return f"Deals {self.damage} damage"

class Armor:
    def __init__(self, defense):
        self.defense = defense
    
    def protect (self):
        return f"Blocks {self.defense} damage"

class Spell:
    def __init__(self, name, effect):
        self.name = name
        self.effect = effect
    
    def cast (self):
        return f"{self.name}: {self.effect}"

class Character:
    def __init__(self, name):
        self.name = name
        self.weapon = None
        self.armor = None
        self.spells = []
    
    def equip_weapon (self, weapon):
        self.weapon = weapon
    
    def equip_armor (self, armor):
        self.armor = armor
    
    def learn_spell (self, spell):
        self.spells.append (spell)
    
    def attack (self):
        if self.weapon:
            return self.weapon.attack()
        return "Punches for 1 damage"
    
    def cast_spell (self, spell_index):
        if spell_index < len (self.spells):
            return self.spells[spell_index].cast()
        return "No spell in that slot"

# Flexible character customization!
warrior = Character("Conan")
warrior.equip_weapon(Weapon(50))
warrior.equip_armor(Armor(30))

mage = Character("Gandalf")
mage.learn_spell(Spell("Fireball", "Burns enemy"))
mage.learn_spell(Spell("Heal", "Restores HP"))
mage.equip_weapon(Weapon(10))  # Staff

# Same Character class, different configurations!
\`\`\`

**Best Practices:**
- Prefer composition over inheritance (general rule)
- Use inheritance for true "is-a" relationships only
- Keep inheritance hierarchies shallow (2-3 levels max)
- Components should have single responsibility
- Use dependency injection for flexibility
- Make components replaceable (interfaces/protocols)`,
};

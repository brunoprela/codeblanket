/**
 * Python OOP - New Problems Batch 3 (31-40)
 * 10 problems
 */

import { Problem } from '../types';

export const pythonOOPBatch3: Problem[] = [
  {
    id: 'oop-builder-pattern-pizza',
    title: 'Builder Pattern',
    difficulty: 'Medium',
    description: `Implement builder pattern for complex object construction.

**Pattern:**
- Separates construction from representation
- Step-by-step building
- Fluent interface
- Final build() method

This tests:
- Builder pattern
- Method chaining
- Complex initialization`,
    examples: [
      {
        input: 'Builder().set_x().set_y().build()',
        output: 'Constructs object step by step',
      },
    ],
    constraints: ['Use builder pattern', 'Support chaining'],
    hints: [
      'Return self from setters',
      'build() returns final object',
      'Validate in build()',
    ],
    starterCode: `class Pizza:
    """Pizza class"""
    def __init__(self, size, crust, toppings):
        self.size = size
        self.crust = crust
        self.toppings = toppings


class PizzaBuilder:
    """Builder for Pizza"""
    def __init__(self):
        self.size = None
        self.crust = "regular"
        self.toppings = []
    
    def set_size(self, size):
        """Set pizza size"""
        self.size = size
        return self
    
    def set_crust(self, crust):
        """Set crust type"""
        self.crust = crust
        return self
    
    def add_topping(self, topping):
        """Add a topping"""
        self.toppings.append(topping)
        return self
    
    def build(self):
        """Build final pizza"""
        if not self.size:
            raise ValueError("Size required")
        return Pizza(self.size, self.crust, self.toppings)


def test_builder():
    """Test builder pattern"""
    pizza = (PizzaBuilder()
             .set_size("large")
             .set_crust("thin")
             .add_topping("pepperoni")
             .add_topping("mushrooms")
             .build())
    
    return len(pizza.toppings) + len(pizza.size)
`,
    testCases: [
      {
        input: [],
        expected: 7,
        functionName: 'test_builder',
      },
    ],
    solution: `class Pizza:
    def __init__(self, size, crust, toppings):
        self.size = size
        self.crust = crust
        self.toppings = toppings


class PizzaBuilder:
    def __init__(self):
        self.size = None
        self.crust = "regular"
        self.toppings = []
    
    def set_size(self, size):
        self.size = size
        return self
    
    def set_crust(self, crust):
        self.crust = crust
        return self
    
    def add_topping(self, topping):
        self.toppings.append(topping)
        return self
    
    def build(self):
        if not self.size:
            raise ValueError("Size required")
        return Pizza(self.size, self.crust, self.toppings)


def test_builder():
    pizza = (PizzaBuilder()
             .set_size("large")
             .set_crust("thin")
             .add_topping("pepperoni")
             .add_topping("mushrooms")
             .build())
    
    return len(pizza.toppings) + len(pizza.size)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(n) for toppings',
    order: 31,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-strategy-pattern-sorter',
    title: 'Strategy Pattern',
    difficulty: 'Medium',
    description: `Implement strategy pattern to swap algorithms dynamically.

**Pattern:**
- Define family of algorithms
- Encapsulate each one
- Make them interchangeable
- Client chooses strategy

This tests:
- Strategy pattern
- Algorithm swapping
- Polymorphism`,
    examples: [
      {
        input: 'context.set_strategy(new_strategy)',
        output: 'Changes behavior dynamically',
      },
    ],
    constraints: ['Define strategy interface', 'Swap strategies'],
    hints: [
      'Strategy base class/interface',
      'Context holds strategy',
      'Delegate to strategy',
    ],
    starterCode: `class SortStrategy:
    """Base strategy"""
    def sort(self, data):
        raise NotImplementedError


class QuickSort(SortStrategy):
    """Quick sort strategy"""
    def sort(self, data):
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)


class BubbleSort(SortStrategy):
    """Bubble sort strategy"""
    def sort(self, data):
        data = list(data)
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        return data


class Sorter:
    """Context that uses strategy"""
    def __init__(self, strategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy):
        """Change strategy"""
        self.strategy = strategy
    
    def sort(self, data):
        """Delegate to strategy"""
        return self.strategy.sort(data)


def test_strategy():
    """Test strategy pattern"""
    data = [5, 2, 8, 1, 9]
    
    # Use quick sort
    sorter = Sorter(QuickSort())
    result1 = sorter.sort(data)
    
    # Switch to bubble sort
    sorter.set_strategy(BubbleSort())
    result2 = sorter.sort(data)
    
    return result1[0] + result2[-1]
`,
    testCases: [
      {
        input: [],
        expected: 10,
        functionName: 'test_strategy',
      },
    ],
    solution: `class SortStrategy:
    def sort(self, data):
        raise NotImplementedError


class QuickSort(SortStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)


class BubbleSort(SortStrategy):
    def sort(self, data):
        data = list(data)
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        return data


class Sorter:
    def __init__(self, strategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy):
        self.strategy = strategy
    
    def sort(self, data):
        return self.strategy.sort(data)


def test_strategy():
    data = [5, 2, 8, 1, 9]
    sorter = Sorter(QuickSort())
    result1 = sorter.sort(data)
    sorter.set_strategy(BubbleSort())
    result2 = sorter.sort(data)
    return result1[0] + result2[-1]`,
    timeComplexity: 'Depends on strategy',
    spaceComplexity: 'Depends on strategy',
    order: 32,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-template-method-processor',
    title: 'Template Method Pattern',
    difficulty: 'Medium',
    description: `Implement template method pattern with algorithm skeleton.

**Pattern:**
- Base class defines algorithm structure
- Subclasses fill in details
- Template method not overridden
- Hook methods can be overridden

This tests:
- Template method pattern
- Inheritance
- Algorithm structure`,
    examples: [
      {
        input: 'Base defines steps, subclass implements',
        output: 'Consistent algorithm flow',
      },
    ],
    constraints: [
      'Define template in base class',
      'Override steps in subclass',
    ],
    hints: [
      'Template method calls steps',
      'Steps are abstract or have defaults',
      'Subclasses override steps',
    ],
    starterCode: `from abc import ABC, abstractmethod

class DataProcessor(ABC):
    """Template for data processing"""
    def process(self):
        """Template method - defines algorithm structure"""
        data = self.read_data()
        processed = self.process_data(data)
        self.save_data(processed)
        return processed
    
    @abstractmethod
    def read_data(self):
        """Step 1: Read data"""
        pass
    
    @abstractmethod
    def process_data(self, data):
        """Step 2: Process data"""
        pass
    
    def save_data(self, data):
        """Step 3: Save (has default implementation)"""
        pass


class CSVProcessor(DataProcessor):
    """Concrete processor for CSV"""
    def __init__(self, data):
        self.data = data
    
    def read_data(self):
        """Read CSV data"""
        return self.data
    
    def process_data(self, data):
        """Process: uppercase strings"""
        return [item.upper() if isinstance(item, str) else item for item in data]


def test_template():
    """Test template method"""
    processor = CSVProcessor(["hello", "world"])
    result = processor.process()
    
    return len(result[0])
`,
    testCases: [
      {
        input: [],
        expected: 5,
        functionName: 'test_template',
      },
    ],
    solution: `from abc import ABC, abstractmethod

class DataProcessor(ABC):
    def process(self):
        data = self.read_data()
        processed = self.process_data(data)
        self.save_data(processed)
        return processed
    
    @abstractmethod
    def read_data(self):
        pass
    
    @abstractmethod
    def process_data(self, data):
        pass
    
    def save_data(self, data):
        pass


class CSVProcessor(DataProcessor):
    def __init__(self, data):
        self.data = data
    
    def read_data(self):
        return self.data
    
    def process_data(self, data):
        return [item.upper() if isinstance(item, str) else item for item in data]


def test_template():
    processor = CSVProcessor(["hello", "world"])
    result = processor.process()
    return len(result[0])`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 33,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-dependency-injection',
    title: 'Dependency Injection',
    difficulty: 'Medium',
    description: `Use dependency injection for loose coupling.

**Pattern:**
- Dependencies passed to class
- Not created internally
- Easier testing
- More flexible

This tests:
- Dependency injection
- Loose coupling
- Design principles`,
    examples: [
      {
        input: 'Pass dependencies via constructor',
        output: "Class doesn't create dependencies",
      },
    ],
    constraints: ['Inject dependencies', "Don't create internally"],
    hints: [
      'Pass via __init__',
      'Store as attributes',
      'Use interfaces/protocols',
    ],
    starterCode: `class EmailService:
    """Service to send emails"""
    def send(self, to, message):
        return f"Email sent to {to}: {message}"


class SMSService:
    """Service to send SMS"""
    def send(self, to, message):
        return f"SMS sent to {to}: {message}"


class NotificationManager:
    """Manages notifications using injected service"""
    def __init__(self, notification_service):
        # Dependency injected
        self.service = notification_service
    
    def notify(self, user, message):
        """Send notification using injected service"""
        return self.service.send(user, message)


def test_dependency_injection():
    """Test dependency injection"""
    # Inject email service
    email_service = EmailService()
    manager1 = NotificationManager(email_service)
    result1 = manager1.notify("alice@example.com", "Hello")
    
    # Inject SMS service (same manager interface)
    sms_service = SMSService()
    manager2 = NotificationManager(sms_service)
    result2 = manager2.notify("555-1234", "Hi")
    
    return len(result1) + len(result2)
`,
    testCases: [
      {
        input: [],
        expected: 70,
        functionName: 'test_dependency_injection',
      },
    ],
    solution: `class EmailService:
    def send(self, to, message):
        return f"Email sent to {to}: {message}"


class SMSService:
    def send(self, to, message):
        return f"SMS sent to {to}: {message}"


class NotificationManager:
    def __init__(self, notification_service):
        self.service = notification_service
    
    def notify(self, user, message):
        return self.service.send(user, message)


def test_dependency_injection():
    email_service = EmailService()
    manager1 = NotificationManager(email_service)
    result1 = manager1.notify("alice@example.com", "Hello")
    
    sms_service = SMSService()
    manager2 = NotificationManager(sms_service)
    result2 = manager2.notify("555-1234", "Hi")
    
    return len(result1) + len(result2)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 34,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-state-pattern',
    title: 'State Pattern',
    difficulty: 'Hard',
    description: `Implement state pattern to change behavior based on internal state.

**Pattern:**
- Object behavior depends on state
- State encapsulated in separate classes
- State transitions
- Delegates to state object

This tests:
- State pattern
- Behavior switching
- State machines`,
    examples: [
      {
        input: 'Connection state: disconnected → connecting → connected',
        output: 'Different behavior per state',
      },
    ],
    constraints: ['Separate state classes', 'Delegate to state'],
    hints: [
      'State interface',
      'Concrete state classes',
      'Context delegates to state',
    ],
    starterCode: `class State:
    """Base state"""
    def handle(self, context):
        raise NotImplementedError


class OffState(State):
    """Off state"""
    def handle(self, context):
        context.state = OnState()
        return "Turning on..."


class OnState(State):
    """On state"""
    def handle(self, context):
        context.state = OffState()
        return "Turning off..."


class Switch:
    """Context with state"""
    def __init__(self):
        self.state = OffState()
    
    def press(self):
        """Delegate to current state"""
        return self.state.handle(self)


def test_state():
    """Test state pattern"""
    switch = Switch()
    
    # Press 1: off -> on
    result1 = switch.press()
    
    # Press 2: on -> off
    result2 = switch.press()
    
    # Press 3: off -> on
    result3 = switch.press()
    
    return len(result1 + result2 + result3)
`,
    testCases: [
      {
        input: [],
        expected: 39,
        functionName: 'test_state',
      },
    ],
    solution: `class State:
    def handle(self, context):
        raise NotImplementedError


class OffState(State):
    def handle(self, context):
        context.state = OnState()
        return "Turning on..."


class OnState(State):
    def handle(self, context):
        context.state = OffState()
        return "Turning off..."


class Switch:
    def __init__(self):
        self.state = OffState()
    
    def press(self):
        return self.state.handle(self)


def test_state():
    switch = Switch()
    result1 = switch.press()
    result2 = switch.press()
    result3 = switch.press()
    return len(result1 + result2 + result3)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 35,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-adapter-pattern',
    title: 'Adapter Pattern',
    difficulty: 'Medium',
    description: `Implement adapter pattern to make incompatible interfaces work together.

**Pattern:**
- Wrap incompatible interface
- Provide compatible interface
- Translate requests
- Bridge between systems

This tests:
- Adapter pattern
- Interface compatibility
- Wrapper pattern`,
    examples: [
      {
        input: 'Old API → Adapter → New API',
        output: 'Makes interfaces compatible',
      },
    ],
    constraints: ['Wrap incompatible class', 'Provide new interface'],
    hints: ['Adapter wraps adaptee', 'Translates interface', 'Delegation'],
    starterCode: `class EuropeanSocket:
    """European power socket (220V)"""
    def provide_power(self):
        return "220V power"


class USASocket:
    """USA power socket (110V)"""
    def supply_electricity(self):
        return "110V power"


class PowerAdapter:
    """Adapter for European device to USA socket"""
    def __init__(self, usa_socket):
        self.socket = usa_socket
    
    def provide_power(self):
        """Translate USA interface to European interface"""
        usa_power = self.socket.supply_electricity()
        # Convert 110V to 220V (simplified)
        return f"{usa_power} (converted to 220V)"


class EuropeanDevice:
    """Device expecting European socket"""
    def __init__(self, socket):
        self.socket = socket
    
    def charge(self):
        power = self.socket.provide_power()
        return f"Charging with {power}"


def test_adapter():
    """Test adapter pattern"""
    # European device with European socket
    euro_socket = EuropeanSocket()
    device1 = EuropeanDevice(euro_socket)
    result1 = device1.charge()
    
    # European device with USA socket via adapter
    usa_socket = USASocket()
    adapter = PowerAdapter(usa_socket)
    device2 = EuropeanDevice(adapter)
    result2 = device2.charge()
    
    return len(result1) + len(result2)
`,
    testCases: [
      {
        input: [],
        expected: 81,
        functionName: 'test_adapter',
      },
    ],
    solution: `class EuropeanSocket:
    def provide_power(self):
        return "220V power"


class USASocket:
    def supply_electricity(self):
        return "110V power"


class PowerAdapter:
    def __init__(self, usa_socket):
        self.socket = usa_socket
    
    def provide_power(self):
        usa_power = self.socket.supply_electricity()
        return f"{usa_power} (converted to 220V)"


class EuropeanDevice:
    def __init__(self, socket):
        self.socket = socket
    
    def charge(self):
        power = self.socket.provide_power()
        return f"Charging with {power}"


def test_adapter():
    euro_socket = EuropeanSocket()
    device1 = EuropeanDevice(euro_socket)
    result1 = device1.charge()
    
    usa_socket = USASocket()
    adapter = PowerAdapter(usa_socket)
    device2 = EuropeanDevice(adapter)
    result2 = device2.charge()
    
    return len(result1) + len(result2)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 36,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-decorator-pattern',
    title: 'Decorator Pattern (not @decorator)',
    difficulty: 'Medium',
    description: `Implement decorator pattern to add functionality dynamically.

**Pattern:**
- Wrap object to extend behavior
- Same interface as wrapped object
- Can stack decorators
- Different from @decorator syntax

This tests:
- Decorator pattern
- Wrapper objects
- Dynamic behavior addition`,
    examples: [
      {
        input: 'Decorator wraps component',
        output: 'Adds functionality without modifying original',
      },
    ],
    constraints: ['Wrap objects', 'Maintain interface'],
    hints: [
      'Decorator wraps component',
      'Delegates to wrapped object',
      'Adds extra behavior',
    ],
    starterCode: `class Coffee:
    """Base coffee"""
    def cost(self):
        return 5
    
    def description(self):
        return "Coffee"


class CoffeeDecorator:
    """Base decorator"""
    def __init__(self, coffee):
        self._coffee = coffee
    
    def cost(self):
        return self._coffee.cost()
    
    def description(self):
        return self._coffee.description()


class Milk(CoffeeDecorator):
    """Milk decorator"""
    def cost(self):
        return self._coffee.cost() + 1
    
    def description(self):
        return self._coffee.description() + " + Milk"


class Sugar(CoffeeDecorator):
    """Sugar decorator"""
    def cost(self):
        return self._coffee.cost() + 0.5
    
    def description(self):
        return self._coffee.description() + " + Sugar"


def test_decorator_pattern():
    """Test decorator pattern"""
    # Plain coffee
    coffee = Coffee()
    
    # Add milk
    coffee_with_milk = Milk(coffee)
    
    # Add sugar to coffee with milk
    coffee_with_milk_and_sugar = Sugar(coffee_with_milk)
    
    # Get final cost
    return int(coffee_with_milk_and_sugar.cost() * 2)
`,
    testCases: [
      {
        input: [],
        expected: 13,
        functionName: 'test_decorator_pattern',
      },
    ],
    solution: `class Coffee:
    def cost(self):
        return 5
    
    def description(self):
        return "Coffee"


class CoffeeDecorator:
    def __init__(self, coffee):
        self._coffee = coffee
    
    def cost(self):
        return self._coffee.cost()
    
    def description(self):
        return self._coffee.description()


class Milk(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 1
    
    def description(self):
        return self._coffee.description() + " + Milk"


class Sugar(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 0.5
    
    def description(self):
        return self._coffee.description() + " + Sugar"


def test_decorator_pattern():
    coffee = Coffee()
    coffee_with_milk = Milk(coffee)
    coffee_with_milk_and_sugar = Sugar(coffee_with_milk)
    return int(coffee_with_milk_and_sugar.cost() * 2)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(n) for n decorators',
    order: 37,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-command-pattern',
    title: 'Command Pattern',
    difficulty: 'Medium',
    description: `Implement command pattern to encapsulate requests as objects.

**Pattern:**
- Request as object
- Parameterize clients
- Queue/log requests
- Support undo

This tests:
- Command pattern
- Request encapsulation
- Undo/redo support`,
    examples: [
      {
        input: 'Command objects execute and undo',
        output: 'Requests as first-class objects',
      },
    ],
    constraints: ['Encapsulate as objects', 'Support execute and undo'],
    hints: [
      'Command interface',
      'execute() and undo() methods',
      'Store state for undo',
    ],
    starterCode: `class Command:
    """Base command"""
    def execute(self):
        raise NotImplementedError
    
    def undo(self):
        raise NotImplementedError


class Light:
    """Receiver"""
    def __init__(self):
        self.is_on = False
    
    def turn_on(self):
        self.is_on = True
        return "Light on"
    
    def turn_off(self):
        self.is_on = False
        return "Light off"


class LightOnCommand(Command):
    """Concrete command"""
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        return self.light.turn_on()
    
    def undo(self):
        return self.light.turn_off()


class LightOffCommand(Command):
    """Concrete command"""
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        return self.light.turn_off()
    
    def undo(self):
        return self.light.turn_on()


class RemoteControl:
    """Invoker"""
    def __init__(self):
        self.history = []
    
    def execute_command(self, command):
        result = command.execute()
        self.history.append(command)
        return result
    
    def undo_last(self):
        if self.history:
            command = self.history.pop()
            return command.undo()


def test_command():
    """Test command pattern"""
    light = Light()
    remote = RemoteControl()
    
    # Turn on
    on_cmd = LightOnCommand(light)
    remote.execute_command(on_cmd)
    
    # Turn off
    off_cmd = LightOffCommand(light)
    remote.execute_command(off_cmd)
    
    # Undo (turns back on)
    remote.undo_last()
    
    return 1 if light.is_on else 0
`,
    testCases: [
      {
        input: [],
        expected: 1,
        functionName: 'test_command',
      },
    ],
    solution: `class Command:
    def execute(self):
        raise NotImplementedError
    
    def undo(self):
        raise NotImplementedError


class Light:
    def __init__(self):
        self.is_on = False
    
    def turn_on(self):
        self.is_on = True
        return "Light on"
    
    def turn_off(self):
        self.is_on = False
        return "Light off"


class LightOnCommand(Command):
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        return self.light.turn_on()
    
    def undo(self):
        return self.light.turn_off()


class LightOffCommand(Command):
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        return self.light.turn_off()
    
    def undo(self):
        return self.light.turn_on()


class RemoteControl:
    def __init__(self):
        self.history = []
    
    def execute_command(self, command):
        result = command.execute()
        self.history.append(command)
        return result
    
    def undo_last(self):
        if self.history:
            command = self.history.pop()
            return command.undo()


def test_command():
    light = Light()
    remote = RemoteControl()
    
    on_cmd = LightOnCommand(light)
    remote.execute_command(on_cmd)
    
    off_cmd = LightOffCommand(light)
    remote.execute_command(off_cmd)
    
    remote.undo_last()
    
    return 1 if light.is_on else 0`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(n) for history',
    order: 38,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-facade-pattern',
    title: 'Facade Pattern',
    difficulty: 'Easy',
    description: `Implement facade pattern to provide simplified interface to complex system.

**Pattern:**
- Unified interface to subsystems
- Hides complexity
- Easier to use
- Decouples client from subsystems

This tests:
- Facade pattern
- Interface simplification
- System decoupling`,
    examples: [
      {
        input: 'Complex subsystems → Simple facade',
        output: 'One method instead of many',
      },
    ],
    constraints: ['Simplify complex interface', 'Hide subsystem details'],
    hints: [
      'Facade delegates to subsystems',
      'Provides high-level operations',
      'Clients use facade only',
    ],
    starterCode: `class CPU:
    """Subsystem"""
    def freeze(self):
        return "CPU frozen"
    
    def execute(self):
        return "CPU executing"


class Memory:
    """Subsystem"""
    def load(self):
        return "Memory loaded"


class HardDrive:
    """Subsystem"""
    def read(self):
        return "Data read from disk"


class ComputerFacade:
    """Facade for complex computer startup"""
    def __init__(self):
        self.cpu = CPU()
        self.memory = Memory()
        self.hard_drive = HardDrive()
    
    def start(self):
        """Simple interface to complex operation"""
        results = []
        results.append(self.cpu.freeze())
        results.append(self.memory.load())
        results.append(self.hard_drive.read())
        results.append(self.cpu.execute())
        return " → ".join(results)


def test_facade():
    """Test facade pattern"""
    # Instead of calling multiple subsystems,
    # client uses simple facade
    computer = ComputerFacade()
    result = computer.start()
    
    return len(result)
`,
    testCases: [
      {
        input: [],
        expected: 64,
        functionName: 'test_facade',
      },
    ],
    solution: `class CPU:
    def freeze(self):
        return "CPU frozen"
    
    def execute(self):
        return "CPU executing"


class Memory:
    def load(self):
        return "Memory loaded"


class HardDrive:
    def read(self):
        return "Data read from disk"


class ComputerFacade:
    def __init__(self):
        self.cpu = CPU()
        self.memory = Memory()
        self.hard_drive = HardDrive()
    
    def start(self):
        results = []
        results.append(self.cpu.freeze())
        results.append(self.memory.load())
        results.append(self.hard_drive.read())
        results.append(self.cpu.execute())
        return " → ".join(results)


def test_facade():
    computer = ComputerFacade()
    result = computer.start()
    return len(result)`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 39,
    topic: 'Python Object-Oriented Programming',
  },
  {
    id: 'oop-proxy-pattern',
    title: 'Proxy Pattern',
    difficulty: 'Medium',
    description: `Implement proxy pattern to control access to an object.

**Pattern:**
- Surrogate/placeholder
- Controls access
- Add functionality (caching, logging, access control)
- Same interface as real object

This tests:
- Proxy pattern
- Access control
- Lazy loading`,
    examples: [
      {
        input: 'Proxy controls access to real object',
        output: 'Can add caching, logging, etc.',
      },
    ],
    constraints: ['Same interface as real object', 'Control access'],
    hints: [
      'Proxy wraps real object',
      'Delegates requests',
      'Can add checks/caching',
    ],
    starterCode: `class Image:
    """Real object"""
    def __init__(self, filename):
        self.filename = filename
        self._load_from_disk()
    
    def _load_from_disk(self):
        self.data = f"Image data from {self.filename}"
    
    def display(self):
        return f"Displaying {self.filename}"


class ImageProxy:
    """Proxy for lazy loading"""
    def __init__(self, filename):
        self.filename = filename
        self._image = None
    
    def display(self):
        """Lazy load on first access"""
        if self._image is None:
            self._image = Image(self.filename)
        return self._image.display()


def test_proxy():
    """Test proxy pattern"""
    # Proxy doesn't load until needed
    proxy = ImageProxy("photo.jpg")
    
    # First display loads image
    result1 = proxy.display()
    
    # Second display uses loaded image
    result2 = proxy.display()
    
    return len(result1 + result2)
`,
    testCases: [
      {
        input: [],
        expected: 42,
        functionName: 'test_proxy',
      },
    ],
    solution: `class Image:
    def __init__(self, filename):
        self.filename = filename
        self._load_from_disk()
    
    def _load_from_disk(self):
        self.data = f"Image data from {self.filename}"
    
    def display(self):
        return f"Displaying {self.filename}"


class ImageProxy:
    def __init__(self, filename):
        self.filename = filename
        self._image = None
    
    def display(self):
        if self._image is None:
            self._image = Image(self.filename)
        return self._image.display()


def test_proxy():
    proxy = ImageProxy("photo.jpg")
    result1 = proxy.display()
    result2 = proxy.display()
    return len(result1 + result2)`,
    timeComplexity: 'O(1) after first access',
    spaceComplexity: 'O(1)',
    order: 40,
    topic: 'Python Object-Oriented Programming',
  },
];

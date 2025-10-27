/**
 * Metaclasses & Class Creation Section
 */

export const metaclassesSection = {
  id: 'metaclasses',
  title: 'Metaclasses & Class Creation',
  content: `**What are Metaclasses?**
Metaclasses are "classes of classes" that control how classes are created.

**Basic Concept:**
\`\`\`python
# type is the default metaclass
class MyClass:
    pass

# Equivalent to:
MyClass = type('MyClass', (), {})

# Everything is an object
isinstance(MyClass, type)  # True
isinstance(5, int)  # True
isinstance (int, type)  # True
\`\`\`

**Custom Metaclass:**
\`\`\`python
class Meta (type):
    def __new__(mcs, name, bases, attrs):
        # Modify class before creation
        attrs['created_by'] = 'Meta'
        return super().__new__(mcs, name, bases, attrs)

class MyClass (metaclass=Meta):
    pass

print(MyClass.created_by)  # 'Meta'
\`\`\`

**Real-World Use Cases:**1. **ORM (like Django):**
\`\`\`python
class ModelMeta (type):
    def __new__(mcs, name, bases, attrs):
        # Auto-create database fields
        fields = {}
        for key, value in attrs.items():
            if isinstance (value, Field):
                fields[key] = value
        attrs['_fields'] = fields
        return super().__new__(mcs, name, bases, attrs)

class Model (metaclass=ModelMeta):
    pass

class User(Model):
    name = CharField()
    email = EmailField()
\`\`\`

2. **Singleton Pattern:**
\`\`\`python
class Singleton (type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database (metaclass=Singleton):
    pass

db1 = Database()
db2 = Database()
assert db1 is db2  # Same instance
\`\`\`

3. **API Client Registration:**
\`\`\`python
class APIRegistry (type):
    _apis = {}
    
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if 'endpoint' in attrs:
            mcs._apis[attrs['endpoint']] = cls
        return cls

class API(metaclass=APIRegistry):
    pass

class UsersAPI(API):
    endpoint = '/users'
\`\`\`

**__init__ vs __new__:**
\`\`\`python
class Meta (type):
    def __new__(mcs, name, bases, attrs):
        # Called before class is created
        # Can modify class definition
        return super().__new__(mcs, name, bases, attrs)
    
    def __init__(cls, name, bases, attrs):
        # Called after class is created
        # Can initialize class attributes
        super().__init__(name, bases, attrs)
\`\`\`

**When to Use Metaclasses:**
- Framework and library development
- Enforcing coding standards at class level
- Automatic registration patterns
- ORM implementations
- Plugin systems

**Alternatives to Consider:**
- Class decorators (simpler, often sufficient)
- __init_subclass__ (Python 3.6+, cleaner)
- Descriptors for attribute access control

**Best Practices:**
- Metaclasses are powerful but complex—use sparingly
- Consider simpler alternatives first
- Document metaclass behavior clearly
- Used mainly in frameworks, not application code`,
};

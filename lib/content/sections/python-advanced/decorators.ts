/**
 * Decorators & Function Wrapping Section
 */

export const decoratorsSection = {
  id: 'decorators',
  title: 'Decorators & Function Wrapping',
  content: `**What are Decorators?**
Decorators are a powerful way to modify or enhance functions and classes without changing their source code.

**Basic Decorator Pattern:**
\`\`\`python
def my_decorator (func):
    def wrapper(*args, **kwargs):
        # Do something before
        result = func(*args, **kwargs)
        # Do something after
        return result
    return wrapper

@my_decorator
def my_function():
    pass
\`\`\`

**Common Decorator Use Cases:**

1. **Timing/Profiling:**
\`\`\`python
import time
def timer (func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.2f}s")
        return result
    return wrapper
\`\`\`

2. **Caching/Memoization:**
\`\`\`python
from functools import lru_cache

@lru_cache (maxsize=128)
def fibonacci (n):
    if n < 2:
        return n
    return fibonacci (n-1) + fibonacci (n-2)
\`\`\`

3. **Authentication/Authorization:**
\`\`\`python
def require_auth (func):
    def wrapper (user, *args, **kwargs):
        if not user.is_authenticated:
            raise PermissionError("Not authenticated")
        return func (user, *args, **kwargs)
    return wrapper
\`\`\`

**Decorators with Arguments:**
\`\`\`python
def repeat (times):
    def decorator (func):
        def wrapper(*args, **kwargs):
            for _ in range (times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet():
    print("Hello!")
\`\`\`

**Class Decorators:**
\`\`\`python
def singleton (cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    pass
\`\`\`

**Best Practices:**
- Use functools.wraps to preserve function metadata
- Keep decorators simple and focused
- Chain decorators carefully (order matters)
- Consider using parameterized decorators for flexibility`,
};

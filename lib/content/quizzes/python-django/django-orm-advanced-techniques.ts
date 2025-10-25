export const djangoOrmAdvancedTechniquesQuiz = {
  title: 'Django ORM Advanced Techniques - Discussion Questions',
  questions: [
    {
      question:
        'Explain the difference between F() expressions, Q() objects, and annotations in Django ORM. Provide real-world examples where each would be essential for building efficient queries.',
      answer: `
**F() Expressions:**

F() expressions reference model field values directly in the database, avoiding race conditions and improving performance.

**Use Cases:**
1. **Atomic Updates** (avoid race conditions):
\`\`\`python
from django.db.models import F

# ❌ Race condition (read-modify-write)
article = Article.objects.get(id=1)
article.view_count += 1
article.save()

# ✅ Atomic update (database-level increment)
Article.objects.filter(id=1).update(view_count=F('view_count') + 1)
\`\`\`

2. **Field Comparisons**:
\`\`\`python
# Find products where discount > markup
Product.objects.filter(discount__gt=F('markup'))

# Find articles published more than 30 days after creation
from datetime.timedelta import timedelta
Article.objects.filter(
    published_at__gt=F('created_at') + timedelta(days=30)
)
\`\`\`

3. **Mathematical Operations**:
\`\`\`python
# Calculate profit margin in database
Product.objects.annotate(
    profit_margin=F('price') - F('cost')
).filter(profit_margin__gt=10)
\`\`\`

**Q() Objects:**

Q() objects enable complex query logic with AND, OR, NOT operations.

**Use Cases:**
1. **OR Queries**:
\`\`\`python
from django.db.models import Q

# Search in multiple fields
search_term = "django"
Article.objects.filter(
    Q(title__icontains=search_term) | 
    Q(content__icontains=search_term) |
    Q(tags__name__icontains=search_term)
)
\`\`\`

2. **Complex Conditions**:
\`\`\`python
# (status='published' AND featured=True) OR (status='sponsored')
Article.objects.filter(
    (Q(status='published') & Q(featured=True)) | Q(status='sponsored')
)
\`\`\`

3. **Negation**:
\`\`\`python
# Articles NOT by specific author
Article.objects.filter(~Q(author__username='admin'))
\`\`\`

4. **Dynamic Queries**:
\`\`\`python
def build_filter(filters):
    query = Q()
    for key, value in filters.items():
        query &= Q(**{key: value})
    return Article.objects.filter(query)
\`\`\`

**Annotations:**

Annotations add computed fields to QuerySets using aggregation functions.

**Use Cases:**
1. **Count Relationships**:
\`\`\`python
from django.db.models import Count

# Add comment count to each article
Article.objects.annotate(
    comment_count=Count('comments')
).filter(comment_count__gt=10)
\`\`\`

2. **Aggregate Calculations**:
\`\`\`python
from django.db.models import Avg, Sum, Max

# Add average rating
Product.objects.annotate(
    avg_rating=Avg('reviews__rating'),
    total_sales=Sum('orders__quantity'),
    highest_price=Max('variants__price')
)
\`\`\`

3. **Conditional Annotations**:
\`\`\`python
from django.db.models import Case, When, Value, IntegerField

# Categorize articles by view count
Article.objects.annotate(
    popularity=Case(
        When(view_count__gte=10000, then=Value('viral')),
        When(view_count__gte=1000, then=Value('popular')),
        When(view_count__gte=100, then=Value('moderate')),
        default=Value('low'),
        output_field=CharField(),
    )
)
\`\`\`

**Real-World Combined Example - E-commerce Dashboard:**

\`\`\`python
from django.db.models import F, Q, Count, Sum, Avg, Case, When

# Complex product analytics query
products = Product.objects.annotate(
    # Count total orders
    order_count=Count('orders'),
    
    # Sum total revenue
    total_revenue=Sum(F('orders__quantity') * F('price')),
    
    # Average rating
    avg_rating=Avg('reviews__rating'),
    
    # Profit margin
    profit_per_unit=F('price') - F('cost'),
    
    # Performance category
    performance=Case(
        When(order_count__gte=100, then=Value('bestseller')),
        When(order_count__gte=50, then=Value('popular')),
        default=Value('regular'),
        output_field=CharField(),
    )
).filter(
    # Complex filtering with Q objects
    Q(stock__gt=0) &  # In stock
    (Q(avg_rating__gte=4.0) | Q(order_count__gte=50)) &  # Highly rated OR popular
    ~Q(status='discontinued')  # Not discontinued
).select_related('category').order_by('-total_revenue')
\`\`\`

This single query efficiently computes all metrics at the database level, avoiding N+1 queries and reducing memory usage.
      `,
    },
    {
      question:
        "Describe Django's transaction management system. Explain the difference between ATOMIC_REQUESTS, atomic() decorator, and manual transaction control. When would you use each approach, and what are the potential pitfalls?",
      answer: `
**Django Transaction Management:**

Django provides multiple ways to control database transactions, each suited for different scenarios.

**1. ATOMIC_REQUESTS (settings.py):**

\`\`\`python
# settings.py
DATABASES = {
    'default': {
        'ATOMIC_REQUESTS': True,
    }
}
\`\`\`

**How it works:**
- Wraps each view in a transaction
- Commits on success, rolls back on exception
- Simplest approach for ensuring data consistency

**When to use:**
- Simple applications
- When most views modify data
- When you want automatic transaction management

**Pitfalls:**
1. **Performance**: Long-running views keep transactions open, blocking database
2. **External APIs**: API calls inside transaction waste connection time
3. **Granularity**: All-or-nothing per view (can't have partial commits)

\`\`\`python
# ❌ Problem: Long transaction
@require_POST
def process_order(request):  # Transaction starts
    order = Order.objects.create(...)  # DB write
    
    # External API call - transaction still open!
    send_payment_request(order)  # 5+ seconds
    
    # More DB writes
    order.status = 'paid'
    order.save()  # Transaction ends (commits/rolls back)
\`\`\`

**2. atomic() Decorator/Context Manager:**

\`\`\`python
from django.db import transaction

# As decorator
@transaction.atomic
def create_article(title, content, author):
    article = Article.objects.create(title=title, content=content, author=author)
    Tag.objects.create(name='auto-tag', article=article)
    return article

# As context manager
def transfer_money(from_account, to_account, amount):
    with transaction.atomic():
        from_account.balance -= amount
        from_account.save()
        
        to_account.balance += amount
        to_account.save()
\`\`\`

**When to use:**
- Specific operations need atomicity
- ATOMIC_REQUESTS is disabled
- Need fine-grained control
- Wrapping multiple operations

**Advantages:**
- Granular control
- Can mix transactional and non-transactional code
- Explicit intent in code

**Example - Avoiding Long Transactions:**

\`\`\`python
def process_order(request):
    # Create order in transaction
    with transaction.atomic():
        order = Order.objects.create(...)
        OrderItem.objects.bulk_create(items)
    
    # External API call OUTSIDE transaction
    payment_result = send_payment_request(order)
    
    # Update order status in new transaction
    with transaction.atomic():
        order.status = 'paid' if payment_result else 'failed'
        order.save()
\`\`\`

**3. Manual Transaction Control:**

\`\`\`python
from django.db import transaction

def complex_operation():
    # Disable auto-commit
    transaction.set_autocommit(False)
    
    try:
        # Multiple operations
        article = Article.objects.create(...)
        Tag.objects.create(...)
        
        # Manual commit
        transaction.commit()
    except Exception:
        # Manual rollback
        transaction.rollback()
    finally:
        # Re-enable auto-commit
        transaction.set_autocommit(True)
\`\`\`

**When to use:**
- Very complex workflows
- Need explicit control over commit points
- Advanced use cases (rarely needed)

**Pitfall: Forgetting to re-enable auto-commit**

**Nested Transactions (Savepoints):**

\`\`\`python
from django.db import transaction

def create_blog_post():
    with transaction.atomic():  # Outer transaction
        article = Article.objects.create(title='Main Article')
        
        # Try to add tags (nested transaction = savepoint)
        try:
            with transaction.atomic():  # Savepoint
                Tag.objects.create(name='invalid//tag')  # Validation error
        except ValidationError:
            # Savepoint rolled back, article creation continues
            pass
        
        # Article still saved
        return article
\`\`\`

**select_for_update() for Locking:**

\`\`\`python
def transfer_money(from_id, to_id, amount):
    with transaction.atomic():
        # Lock rows to prevent concurrent modifications
        from_account = Account.objects.select_for_update().get(id=from_id)
        to_account = Account.objects.select_for_update().get(id=to_id)
        
        if from_account.balance < amount:
            raise InsufficientFundsError()
        
        from_account.balance -= amount
        to_account.balance += amount
        
        from_account.save()
        to_account.save()
\`\`\`

**Best Practices:**

1. ✅ Keep transactions short
2. ✅ Don't call external APIs inside transactions
3. ✅ Use atomic() for multi-step operations
4. ✅ Use savepoints for partial rollbacks
5. ✅ Use select_for_update() to prevent race conditions
6. ❌ Don't nest transactions unnecessarily
7. ❌ Don't perform long-running tasks in transactions
8. ❌ Don't forget error handling

**Performance Comparison:**

\`\`\`python
# ❌ Slow: Many small transactions
def import_data(rows):
    for row in rows:  # 10,000 rows
        Article.objects.create(**row)  # 10,000 transactions

# ✅ Fast: Single transaction
def import_data(rows):
    with transaction.atomic():
        for row in rows:
            Article.objects.create(**row)  # 1 transaction

# ✅ Fastest: Bulk operation in transaction
def import_data(rows):
    with transaction.atomic():
        Article.objects.bulk_create(
            [Article(**row) for row in rows]
        )  # 1 transaction, 1 query
\`\`\`

Choose the right transaction strategy based on your specific use case, balancing between data consistency, performance, and code clarity.
      `,
    },
    {
      question:
        "Explain Django's database routing and multi-database support. How would you implement a read-replica setup with automatic routing, and what considerations are important for data consistency?",
      answer: `
**Multi-Database Setup:**

Django supports multiple databases for horizontal scaling, separating read/write traffic, and data isolation.

**Configuration (settings.py):**

\`\`\`python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'primary_db',
        'HOST': 'primary-db.example.com',
        'USER': 'db_user',
        'PASSWORD': 'password',
    },
    'replica1': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'primary_db',  # Same database, different server
        'HOST': 'replica1-db.example.com',
        'USER': 'readonly_user',
        'PASSWORD': 'password',
    },
    'replica2': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'primary_db',
        'HOST': 'replica2-db.example.com',
        'USER': 'readonly_user',
        'PASSWORD': 'password',
    },
    'analytics': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'analytics_db',
        'HOST': 'analytics-db.example.com',
    },
}
\`\`\`

**Database Router:**

\`\`\`python
# config/database_router.py
import random

class PrimaryReplicaRouter:
    """
    Route reads to replicas, writes to primary
    """
    
    def db_for_read(self, model, **hints):
        """
        Reads go to random replica
        """
        return random.choice(['replica1', 'replica2'])
    
    def db_for_write(self, model, **hints):
        """
        Writes always go to primary
        """
        return 'default'
    
    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if both objects in same database
        """
        db_set = {'default', 'replica1', 'replica2'}
        if obj1._state.db in db_set and obj2._state.db in db_set:
            return True
        return None
    
    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Only run migrations on primary
        """
        return db == 'default'

# settings.py
DATABASE_ROUTERS = ['config.database_router.PrimaryReplicaRouter']
\`\`\`

**Advanced Router with Sticky Writes:**

\`\`\`python
import threading
from django.conf import settings

# Thread-local storage for tracking recent writes
_thread_local = threading.local()

class StickyPrimaryRouter:
    """
    Route reads to replicas, but use primary for short time after writes
    to avoid replication lag issues
    """
    
    def __init__(self):
        self.primary_db = 'default'
        self.replica_dbs = ['replica1', 'replica2']
        self.sticky_seconds = 5  # Use primary for 5 seconds after write
    
    def db_for_read(self, model, **hints):
        """Read from primary if recent write, otherwise replica"""
        if self._should_use_primary():
            return self.primary_db
        return random.choice(self.replica_dbs)
    
    def db_for_write(self, model, **hints):
        """All writes go to primary"""
        self._mark_write()
        return self.primary_db
    
    def _should_use_primary(self):
        """Check if we should use primary due to recent write"""
        import time
        last_write = getattr(_thread_local, 'last_write_time', 0)
        return (time.time() - last_write) < self.sticky_seconds
    
    def _mark_write(self):
        """Mark that a write occurred"""
        import time
        _thread_local.last_write_time = time.time()
    
    def allow_relation(self, obj1, obj2, **hints):
        db_set = {self.primary_db} | set(self.replica_dbs)
        if obj1._state.db in db_set and obj2._state.db in db_set:
            return True
        return None
    
    def allow_migrate(self, db, app_label, model_name=None, **hints):
        return db == self.primary_db
\`\`\`

**App-Specific Routing:**

\`\`\`python
class AnalyticsRouter:
    """
    Route analytics app to separate database
    """
    analytics_apps = {'analytics', 'reports'}
    
    def db_for_read(self, model, **hints):
        if model._meta.app_label in self.analytics_apps:
            return 'analytics'
        return None  # Let other routers decide
    
    def db_for_write(self, model, **hints):
        if model._meta.app_label in self.analytics_apps:
            return 'analytics'
        return None
    
    def allow_relation(self, obj1, obj2, **hints):
        if (obj1._meta.app_label in self.analytics_apps or
            obj2._meta.app_label in self.analytics_apps):
            return obj1._state.db == obj2._state.db
        return None
    
    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if app_label in self.analytics_apps:
            return db == 'analytics'
        return db == 'default'

# Use multiple routers
DATABASE_ROUTERS = [
    'config.database_router.AnalyticsRouter',
    'config.database_router.StickyPrimaryRouter',
]
\`\`\`

**Manual Database Selection:**

\`\`\`python
# Query specific database
Article.objects.using('replica1').all()

# Save to specific database
article = Article(title='Test')
article.save(using='default')

# Transactions on specific database
from django.db import transaction

with transaction.atomic(using='default'):
    article = Article.objects.create(...)
    Tag.objects.create(...)
\`\`\`

**Data Consistency Considerations:**

1. **Replication Lag:**
\`\`\`python
# Problem: Read-after-write inconsistency
def create_and_display(request):
    # Write to primary
    article = Article.objects.create(title='New Article')
    
    # Read from replica (might not have new data yet!)
    articles = Article.objects.all()  # Might not include new article
    
    return render(request, 'articles.html', {'articles': articles})

# Solution 1: Force primary for this read
articles = Article.objects.using('default').all()

# Solution 2: Use sticky writes (router handles it automatically)
\`\`\`

2. **Foreign Key Integrity:**
\`\`\`python
# Ensure related objects use same database
article = Article.objects.using('default').get(id=1)
comment = Comment(article=article, text='Great!')
comment.save(using='default')  # Must use same database
\`\`\`

3. **Cross-Database Queries:**
\`\`\`python
# ❌ Can't join across databases
Article.objects.select_related('author')  # Fails if Author on different DB

# ✅ Fetch separately
articles = Article.objects.all()
author_ids = articles.values_list('author_id', flat=True)
authors = User.objects.filter(id__in=author_ids)
# Combine in Python
\`\`\`

**Monitoring and Health Checks:**

\`\`\`python
from django.db import connections
from django.core.management import call_command

def check_database_health():
    """Check all databases are accessible"""
    results = {}
    for db_alias in connections:
        try:
            with connections[db_alias].cursor() as cursor:
                cursor.execute("SELECT 1")
            results[db_alias] = "OK"
        except Exception as e:
            results[db_alias] = f"ERROR: {e}"
    return results
\`\`\`

**Best Practices:**

1. ✅ Use primary for reads immediately after writes (sticky writes)
2. ✅ Monitor replication lag
3. ✅ Use connection pooling (pgbouncer)
4. ✅ Keep related objects in same database
5. ✅ Test failover scenarios
6. ✅ Use database-level read-only users for replicas
7. ❌ Don't assume immediate consistency
8. ❌ Don't join across databases
9. ❌ Don't forget to handle database failures gracefully

**Scaling Pattern:**

\`\`\`
┌─────────┐
│  Write  │ ────────┐
└─────────┘         │
                    ▼
              ┌──────────┐
              │ Primary  │
              │ Database │
              └──────────┘
                    │ Replication
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │Replica 1│ │Replica 2│ │Replica 3│
   └─────────┘ └─────────┘ └─────────┘
        │           │           │
        └───────────┼───────────┘
                    ▼
              Load Balancer
                    │
              ┌─────┴─────┐
              │   Reads   │
              └───────────┘
\`\`\`

This architecture allows horizontal scaling of read traffic while maintaining data consistency for writes.
      `,
    },
  ],
};

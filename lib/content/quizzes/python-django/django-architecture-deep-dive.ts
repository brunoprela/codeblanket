export const djangoArchitectureDeepDiveQuiz = {
  title: 'Django Architecture Deep Dive - Discussion Questions',
  questions: [
    {
      question:
        'Explain the Django MVT (Model-View-Template) architecture and how it differs from traditional MVC. In your explanation, describe the request/response lifecycle in detail, including middleware execution, URL routing, view processing, and template rendering.',
      answer: `
**Django MVT vs MVC:**

Django follows the MVT (Model-View-Template) pattern, which is conceptually similar to MVC but with different naming:

1. **Model (M)**: Same as MVC - represents data structure and database schema
2. **View (V)**: Equivalent to MVC's Controller - contains business logic and handles requests
3. **Template (T)**: Equivalent to MVC's View - handles presentation logic

**Key Difference**: In Django, the "View" is actually the controller (business logic), and "Template" is the view (presentation). Django's framework itself acts as the controller that routes requests.

**Request/Response Lifecycle:**

1. **Client Request**: Browser sends HTTP request to server
2. **WSGI Handler**: Request enters Django via WSGI server (Gunicorn/uWSGI)
3. **Middleware (Request Phase)**: Request passes through middleware stack (authentication, session, CSRF, etc.)
4. **URL Resolver**: Django matches URL pattern to view function/class
5. **View Execution**: View processes request, queries models, prepares context
6. **Model Layer**: ORM queries database if needed, returns Python objects
7. **Template Rendering**: View passes context to template, which renders HTML
8. **Middleware (Response Phase)**: Response passes back through middleware stack
9. **WSGI Response**: Final HTTP response sent to client

**Middleware Execution Order:**
- Request: Top to bottom in MIDDLEWARE setting
- Response: Bottom to top in MIDDLEWARE setting

**Example Flow:**
\`\`\`
Request → SecurityMiddleware → SessionMiddleware → AuthenticationMiddleware 
→ URL Resolver → View (queries Model) → Template Rendering 
→ AuthenticationMiddleware → SessionMiddleware → SecurityMiddleware → Response
\`\`\`

This architecture provides clean separation of concerns, making Django applications maintainable and scalable.
      `,
    },
    {
      question:
        'Describe Django\'s app structure and the principle of "loose coupling" in Django applications. How would you design a multi-app project for an e-commerce platform, and what strategies would you use to maintain clear boundaries between apps?',
      answer: `
**Django App Structure:**

A Django "app" is a self-contained module that serves a specific purpose. Apps should be:
- **Reusable**: Can be moved to other projects
- **Single-responsibility**: Focused on one domain
- **Loosely coupled**: Minimal dependencies on other apps
- **Highly cohesive**: Related functionality grouped together

**E-Commerce Platform Design:**

\`\`\`
ecommerce_project/
├── accounts/          # User authentication and profiles
│   ├── models.py      # User, Profile models
│   ├── views.py       # Registration, login views
│   └── signals.py     # Create profile on user creation
├── products/          # Product catalog
│   ├── models.py      # Product, Category, Image
│   ├── views.py       # Product list/detail
│   └── managers.py    # Custom querysets
├── cart/              # Shopping cart
│   ├── models.py      # Cart, CartItem
│   ├── views.py       # Add/remove from cart
│   └── cart.py        # Cart session management
├── orders/            # Order processing
│   ├── models.py      # Order, OrderItem
│   ├── views.py       # Checkout, order creation
│   └── tasks.py       # Celery tasks for order processing
├── payments/          # Payment processing
│   ├── models.py      # Payment, Transaction
│   ├── services.py    # Payment gateway integration
│   └── webhooks.py    # Payment provider webhooks
├── inventory/         # Inventory management
│   ├── models.py      # Stock, Warehouse
│   └── managers.py    # Stock queries
└── core/              # Shared utilities
    ├── models.py      # Abstract base models
    ├── mixins.py      # Reusable mixins
    └── utils.py       # Common utilities
\`\`\`

**Maintaining Loose Coupling:**

1. **Use Signals for Cross-App Communication:**
\`\`\`python
# orders/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Order

@receiver(post_save, sender=Order)
def update_inventory_on_order(sender, instance, created, **kwargs):
    if created:
        # Signal sent to inventory app
        from inventory.services import reserve_stock
        reserve_stock(instance)
\`\`\`

2. **Abstract Base Classes in Core App:**
\`\`\`python
# core/models.py
class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        abstract = True
\`\`\`

3. **Foreign Keys with Dependency Direction:**
- Lower-level apps (products) shouldn't import from higher-level apps (orders)
- Higher-level apps (orders) can import from lower-level apps (products)

4. **Service Layer Pattern:**
\`\`\`python
# orders/services.py
from products.models import Product
from inventory.services import check_stock

def create_order(user, items):
    # Business logic spanning multiple apps
    for item in items:
        if not check_stock(item['product_id'], item['quantity']):
            raise InsufficientStockError()
    # Create order...
\`\`\`

5. **Settings-Based Configuration:**
\`\`\`python
# settings.py
PRODUCT_IMAGE_UPLOAD_PATH = 'products/images/'
ORDER_EXPIRY_HOURS = 24
\`\`\`

**Benefits:**
- Apps can be tested independently
- Easy to replace/modify one app without affecting others
- Clear separation of concerns
- Better team collaboration (different devs work on different apps)
      `,
    },
    {
      question:
        "Explain Django's ORM lazy evaluation and query optimization. How do select_related() and prefetch_related() differ, and when would you use each? Provide examples of N+1 query problems and how to solve them.",
      answer: `
**Lazy Evaluation:**

Django QuerySets are lazy - they don't hit the database until evaluated. Evaluation happens when you:
- Iterate over QuerySet
- Slice with step parameter
- Call len(), list(), bool()
- Call exists(), count()
- Access specific items

**Example:**
\`\`\`python
# No database hit yet
articles = Article.objects.filter(status='published')

# Database hit happens here
for article in articles:  # Iteration triggers query
    print(article.title)
\`\`\`

**select_related() vs prefetch_related():**

**select_related():**
- Uses SQL JOIN
- For ForeignKey and OneToOneField relationships
- Returns single complex query
- More efficient for "to-one" relationships

\`\`\`python
# WITHOUT select_related (N+1 problem)
articles = Article.objects.all()
for article in articles:  # 1 query for articles
    print(article.author.name)  # N queries for authors

# WITH select_related (1 query)
articles = Article.objects.select_related('author').all()
for article in articles:  # Single JOIN query
    print(article.author.name)  # No additional query

# SQL generated:
# SELECT * FROM articles INNER JOIN users ON articles.author_id = users.id
\`\`\`

**prefetch_related():**
- Uses separate queries + Python joining
- For ManyToManyField and reverse ForeignKey
- Returns multiple simple queries
- More efficient for "to-many" relationships

\`\`\`python
# WITHOUT prefetch_related (N+1 problem)
articles = Article.objects.all()
for article in articles:  # 1 query for articles
    for tag in article.tags.all():  # N queries for tags
        print(tag.name)

# WITH prefetch_related (2 queries)
articles = Article.objects.prefetch_related('tags').all()
for article in articles:  # Query 1: articles
    for tag in article.tags.all():  # Query 2: all tags (Python joins them)
        print(tag.name)

# SQL generated:
# Query 1: SELECT * FROM articles
# Query 2: SELECT * FROM tags WHERE article_id IN (1, 2, 3, ...)
\`\`\`

**N+1 Query Problem Examples:**

**Problem 1: Article with Author and Category**
\`\`\`python
# ❌ N+1+1 queries
articles = Article.objects.all()  # 1 query
for article in articles:
    print(article.author.name)  # N queries
    print(article.category.name)  # N queries

# ✅ Solution: 1 query with multiple JOINs
articles = Article.objects.select_related('author', 'category').all()
\`\`\`

**Problem 2: Article with Tags and Comments**
\`\`\`python
# ❌ N+N+1 queries
articles = Article.objects.all()  # 1 query
for article in articles:
    for tag in article.tags.all():  # N queries
        print(tag.name)
    for comment in article.comments.all():  # N queries
        print(comment.text)

# ✅ Solution: 3 queries total
articles = Article.objects.prefetch_related('tags', 'comments').all()
\`\`\`

**Problem 3: Nested Relationships**
\`\`\`python
# Article -> Author -> Profile
# ❌ N+1 queries
articles = Article.objects.select_related('author').all()
for article in articles:
    print(article.author.profile.bio)  # N queries for profiles

# ✅ Solution: Nested select_related
articles = Article.objects.select_related('author__profile').all()
\`\`\`

**Problem 4: Mixed Relationships**
\`\`\`python
# Article -> Author (FK), Tags (M2M), Comments (reverse FK) -> User (FK)
# ✅ Combine both techniques
from django.db.models import Prefetch

articles = Article.objects.select_related('author').prefetch_related(
    'tags',
    Prefetch('comments', queryset=Comment.objects.select_related('user'))
)
\`\`\`

**Query Optimization Checklist:**
1. Use select_related() for FK and O2O
2. Use prefetch_related() for M2M and reverse FK
3. Chain them for complex relationships
4. Use only() and defer() to limit fields
5. Use count() instead of len(queryset)
6. Use exists() instead of bool(queryset)
7. Monitor with django-debug-toolbar

This optimization can reduce hundreds of queries to just a handful, dramatically improving performance.
      `,
    },
  ],
};

export const djangoArchitectureDeepDive = {
  title: 'Django Architecture Deep Dive',
  id: 'django-architecture-deep-dive',
  content: `
# Django Architecture Deep Dive

## Introduction

Django is a **high-level Python web framework** that encourages rapid development and clean, pragmatic design. Built by experienced developers, it takes care of much of the hassle of web development, so you can focus on writing your app without needing to reinvent the wheel.

### Why Django?

- **Batteries included**: ORM, admin interface, authentication, forms, templates
- **MTV pattern**: Model-Template-View architecture (similar to MVC)
- **Production-ready**: Used by Instagram, Pinterest, NASA, National Geographic
- **Security**: Built-in protection against CSRF, XSS, SQL injection, clickjacking
- **Scalable**: From small projects to billions of requests per day

By the end of this section, you'll understand:
- Django\'s MTV (Model-Template-View) architecture
- The request/response lifecycle
- Django's app structure and design philosophy
- Settings configuration and environment management
- URL routing and view resolution
- How Django compares to other frameworks

---

## MTV Architecture Pattern

### Django's MTV (Not MVC)

Django uses **MTV** (Model-Template-View), which differs slightly from traditional MVC:

\`\`\`
Traditional MVC        Django MTV
--------------        -----------
Model          →      Model (same - data layer)
View           →      Template (presentation)
Controller     →      View (business logic)
\`\`\`

**Why the difference?**
- In Django, the **framework itself** acts as the controller (URL routing)
- **Views** contain business logic (like controllers in MVC)
- **Templates** handle presentation (like views in MVC)

### Django Architecture Diagram

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                      HTTP Request                            │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. WSGI/ASGI Handler (entry point)                         │
│     - gunicorn/uvicorn in production                         │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Middleware Stack (process request)                      │
│     - SecurityMiddleware                                     │
│     - SessionMiddleware                                      │
│     - AuthenticationMiddleware                               │
│     - CsrfViewMiddleware                                     │
│     - Custom middleware...                                   │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. URL Resolver (urls.py)                                  │
│     - Match URL pattern                                      │
│     - Extract URL parameters                                 │
│     - Route to view function/class                           │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. View (views.py)                                         │
│     - Execute business logic                                 │
│     - Query models (ORM)                                     │
│     - Prepare context data                                   │
│     - Return response or render template                     │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Model (models.py) - if data access needed               │
│     - Database queries                                       │
│     - Business logic validation                              │
│     - Return QuerySet or instances                           │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Template (templates/*.html) - if HTML rendering         │
│     - Render template with context                           │
│     - Template tags and filters                              │
│     - Generate HTML                                          │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Middleware Stack (process response)                     │
│     - Reverse order through middleware                       │
│     - Add headers, compress, etc.                            │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                     HTTP Response                            │
└─────────────────────────────────────────────────────────────┘
\`\`\`

---

## Request/Response Lifecycle

### Complete Request Flow Example

Let\'s trace a request from client to response:

**User requests**: \`GET /articles/2024/django-guide/\`

#### Step 1: WSGI Handler

\`\`\`python
# Django's WSGI application (wsgi.py)
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
application = get_wsgi_application()

# Production: gunicorn myproject.wsgi:application
\`\`\`

#### Step 2: Middleware Processing

\`\`\`python
# Middleware processes request in order
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',  # Security headers
    'django.contrib.sessions.middleware.SessionMiddleware',  # Session handling
    'django.middleware.common.CommonMiddleware',  # Common operations
    'django.middleware.csrf.CsrfViewMiddleware',  # CSRF protection
    'django.contrib.auth.middleware.AuthenticationMiddleware',  # Auth
    'django.contrib.messages.middleware.MessageMiddleware',  # Messages
    'django.middleware.clickjacking.XFrameOptionsMiddleware',  # Clickjacking
]

# Each middleware has process_request() method
# They execute in order: top to bottom
\`\`\`

#### Step 3: URL Routing

\`\`\`python
# urls.py - Root URL configuration
from django.urls import path, include
from django.contrib import admin

urlpatterns = [
    path('admin/', admin.site.urls),
    path('articles/', include('articles.urls')),  # Delegate to app URLs
    path('api/', include('api.urls')),
]

# articles/urls.py - App-specific URLs
from django.urls import path
from . import views

app_name = 'articles'

urlpatterns = [
    path('', views.ArticleListView.as_view(), name='list'),
    path('<int:year>/<slug:slug>/', views.ArticleDetailView.as_view(), name='detail'),
    path('create/', views.ArticleCreateView.as_view(), name='create'),
]

# URL pattern matches: /articles/2024/django-guide/
# Extracted parameters: year=2024, slug='django-guide'
# Routes to: ArticleDetailView
\`\`\`

#### Step 4: View Execution

\`\`\`python
# articles/views.py
from django.views.generic import DetailView
from django.shortcuts import get_object_or_404
from django.utils import timezone
from .models import Article

class ArticleDetailView(DetailView):
    """
    Display a single article
    Class-based view with built-in functionality
    """
    model = Article
    template_name = 'articles/article_detail.html'
    context_object_name = 'article'
    
    def get_object (self):
        """Custom object retrieval with year and slug"""
        year = self.kwargs['year']
        slug = self.kwargs['slug']
        
        # Query the database
        article = get_object_or_404(
            Article,
            published_at__year=year,
            slug=slug,
            status='published'
        )
        
        # Track view count (side effect)
        article.view_count += 1
        article.save (update_fields=['view_count'])
        
        return article
    
    def get_context_data (self, **kwargs):
        """Add extra context for template"""
        context = super().get_context_data(**kwargs)
        
        # Add related articles
        context['related_articles'] = Article.objects.filter(
            category=self.object.category,
            status='published'
        ).exclude(
            id=self.object.id
        )[:5]
        
        # Add reading time estimate
        words = len (self.object.content.split())
        context['reading_time'] = words // 200  # ~200 words per minute
        
        return context
\`\`\`

#### Step 5: Model Layer

\`\`\`python
# articles/models.py
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils.text import slugify

class Article (models.Model):
    """
    Article model with automatic slug generation
    Demonstrates Django ORM features
    """
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('published', 'Published'),
        ('archived', 'Archived'),
    ]
    
    title = models.CharField (max_length=200)
    slug = models.SlugField (max_length=200, unique=True)
    author = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        related_name='articles'
    )
    content = models.TextField()
    excerpt = models.TextField (max_length=500, blank=True)
    category = models.ForeignKey(
        'Category',
        on_delete=models.SET_NULL,
        null=True,
        related_name='articles'
    )
    tags = models.ManyToManyField('Tag', related_name='articles', blank=True)
    status = models.CharField (max_length=10, choices=STATUS_CHOICES, default='draft')
    view_count = models.IntegerField (default=0)
    featured = models.BooleanField (default=False)
    
    created_at = models.DateTimeField (auto_now_add=True)
    updated_at = models.DateTimeField (auto_now=True)
    published_at = models.DateTimeField (null=True, blank=True)
    
    class Meta:
        ordering = ['-published_at']
        indexes = [
            models.Index (fields=['slug']),
            models.Index (fields=['status', 'published_at']),
            models.Index (fields=['author', 'status']),
        ]
        verbose_name_plural = 'articles'
    
    def __str__(self):
        return self.title
    
    def save (self, *args, **kwargs):
        """Auto-generate slug if not provided"""
        if not self.slug:
            self.slug = slugify (self.title)
        super().save(*args, **kwargs)
    
    def get_absolute_url (self):
        """Return canonical URL for article"""
        return reverse('articles:detail', kwargs={
            'year': self.published_at.year,
            'slug': self.slug
        })
    
    @property
    def is_published (self):
        """Check if article is published"""
        return self.status == 'published' and self.published_at is not None

class Category (models.Model):
    """Article category"""
    name = models.CharField (max_length=100, unique=True)
    slug = models.SlugField (max_length=100, unique=True)
    description = models.TextField (blank=True)
    
    class Meta:
        verbose_name_plural = 'categories'
    
    def __str__(self):
        return self.name

class Tag (models.Model):
    """Article tag"""
    name = models.CharField (max_length=50, unique=True)
    slug = models.SlugField (max_length=50, unique=True)
    
    def __str__(self):
        return self.name
\`\`\`

#### Step 6: Template Rendering

\`\`\`django
{# articles/templates/articles/article_detail.html #}
{% extends "base.html" %}
{% load static %}

{% block title %}{{ article.title }} - My Blog{% endblock %}

{% block content %}
<article class="article-detail">
    <header class="article-header">
        <h1>{{ article.title }}</h1>
        
        <div class="article-meta">
            <span class="author">
                By <a href="{% url 'user_profile' article.author.username %}">
                    {{ article.author.get_full_name }}
                </a>
            </span>
            <span class="date">{{ article.published_at|date:"F j, Y" }}</span>
            <span class="reading-time">{{ reading_time }} min read</span>
            <span class="views">{{ article.view_count }} views</span>
        </div>
        
        <div class="article-tags">
            {% for tag in article.tags.all %}
                <a href="{% url 'articles:tag' tag.slug %}" class="tag">
                    #{{ tag.name }}
                </a>
            {% endfor %}
        </div>
    </header>
    
    <div class="article-content">
        {{ article.content|safe }}
    </div>
    
    <footer class="article-footer">
        <div class="category">
            Category: 
            <a href="{% url 'articles:category' article.category.slug %}">
                {{ article.category.name }}
            </a>
        </div>
    </footer>
</article>

{% if related_articles %}
<section class="related-articles">
    <h2>Related Articles</h2>
    <div class="article-grid">
        {% for related in related_articles %}
            <div class="article-card">
                <h3>
                    <a href="{{ related.get_absolute_url }}">
                        {{ related.title }}
                    </a>
                </h3>
                <p>{{ related.excerpt }}</p>
            </div>
        {% endfor %}
    </div>
</section>
{% endif %}
{% endblock %}
\`\`\`

#### Step 7: Middleware Response Processing

\`\`\`python
# Middleware processes response in REVERSE order
# Each middleware can modify the response

# Example: Custom timing middleware
class TimingMiddleware:
    """Measure request processing time"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Process request
        import time
        start_time = time.time()
        
        # Get response from next middleware/view
        response = self.get_response (request)
        
        # Process response
        duration = time.time() - start_time
        response['X-Request-Duration'] = f"{duration:.3f}s"
        
        return response
\`\`\`

---

## Django Project Structure

### Standard Django Project Layout

\`\`\`
myproject/                      # Project root
├── manage.py                   # Django management script
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not committed)
├── .gitignore                 # Git ignore patterns
├── README.md                  # Project documentation
│
├── myproject/                  # Project configuration package
│   ├── __init__.py
│   ├── settings/               # Settings split by environment
│   │   ├── __init__.py
│   │   ├── base.py            # Base settings
│   │   ├── development.py     # Dev settings
│   │   ├── production.py      # Production settings
│   │   └── test.py            # Test settings
│   ├── urls.py                # Root URL configuration
│   ├── wsgi.py                # WSGI entry point
│   └── asgi.py                # ASGI entry point (async)
│
├── apps/                       # Django applications
│   ├── articles/               # Article management app
│   │   ├── __init__.py
│   │   ├── admin.py           # Admin configuration
│   │   ├── apps.py            # App configuration
│   │   ├── models.py          # Data models
│   │   ├── views.py           # View logic
│   │   ├── urls.py            # App URL patterns
│   │   ├── forms.py           # Form definitions
│   │   ├── serializers.py     # DRF serializers
│   │   ├── managers.py        # Custom model managers
│   │   ├── signals.py         # Signal handlers
│   │   ├── tasks.py           # Celery tasks
│   │   ├── utils.py           # Utility functions
│   │   ├── migrations/        # Database migrations
│   │   │   ├── __init__.py
│   │   │   └── 0001_initial.py
│   │   ├── templates/         # App templates
│   │   │   └── articles/
│   │   │       ├── article_list.html
│   │   │       └── article_detail.html
│   │   ├── static/            # App static files
│   │   │   └── articles/
│   │   │       ├── css/
│   │   │       ├── js/
│   │   │       └── images/
│   │   └── tests/             # Test files
│   │       ├── __init__.py
│   │       ├── test_models.py
│   │       ├── test_views.py
│   │       └── test_api.py
│   │
│   ├── users/                  # User management app
│   │   └── ... (same structure)
│   │
│   └── core/                   # Core/common functionality
│       └── ... (same structure)
│
├── templates/                  # Global templates
│   ├── base.html              # Base template
│   ├── 404.html               # Error pages
│   └── 500.html
│
├── static/                     # Global static files
│   ├── css/
│   ├── js/
│   └── images/
│
├── media/                      # User-uploaded files
│   └── uploads/
│
├── locale/                     # Internationalization
│   ├── en/
│   └── es/
│
└── tests/                      # Integration tests
    └── ... (test files)
\`\`\`

### App vs Project

**Project**: The entire Django application
- Contains settings, URL config, WSGI/ASGI
- One per website/service

**App**: A Python package with specific functionality
- Reusable component (articles, users, payments)
- Multiple apps per project
- Each app should do ONE thing well

\`\`\`bash
# Create a new app
python manage.py startapp articles
\`\`\`

---

## Django Settings Configuration

### Environment-Based Settings

\`\`\`python
# settings/base.py - Base settings for all environments
import os
from pathlib import Path
from decouple import config  # python-decouple for env vars

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Security
SECRET_KEY = config('SECRET_KEY')
ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='').split(',')

# Application definition
INSTALLED_APPS = [
    # Django apps
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third-party apps
    'rest_framework',
    'rest_framework.authtoken',
    'django_filters',
    'corsheaders',
    'celery',
    'django_celery_beat',
    'debug_toolbar',  # Dev only (conditionally added)
    
    # Local apps
    'apps.core',
    'apps.users',
    'apps.articles',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # Static files
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'myproject.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'myproject.wsgi.application'

# Database (overridden in environment-specific settings)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DB_NAME'),
        'USER': config('DB_USER'),
        'PASSWORD': config('DB_PASSWORD'),
        'HOST': config('DB_HOST', default='localhost'),
        'PORT': config('DB_PORT', default='5432'),
        'CONN_MAX_AGE': 600,  # Connection pooling
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

# Media files (user uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
\`\`\`

\`\`\`python
# settings/development.py - Development settings
from .base import *

DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# Development-only apps
INSTALLED_APPS += [
    'debug_toolbar',
]

MIDDLEWARE += [
    'debug_toolbar.middleware.DebugToolbarMiddleware',
]

# Debug toolbar configuration
INTERNAL_IPS = ['127.0.0.1']

# Email backend (console for development)
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Disable caching in development
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
    }
}

# Celery (use synchronous execution in dev)
CELERY_TASK_ALWAYS_EAGER = True
\`\`\`

\`\`\`python
# settings/production.py - Production settings
from .base import *
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

DEBUG = False

# Security settings
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
X_FRAME_OPTIONS = 'DENY'

# Static files (use WhiteNoise or CDN)
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Caching (Redis)
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': config('REDIS_URL'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Session backend (Redis)
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'

# Email backend (production mail service)
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = config('EMAIL_HOST')
EMAIL_PORT = config('EMAIL_PORT', cast=int)
EMAIL_USE_TLS = True
EMAIL_HOST_USER = config('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD')

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': BASE_DIR / 'logs' / 'django.log',
            'maxBytes': 1024 * 1024 * 10,  # 10 MB
            'backupCount': 10,
            'formatter': 'verbose',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# Sentry error tracking
sentry_sdk.init(
    dsn=config('SENTRY_DSN'),
    integrations=[DjangoIntegration()],
    traces_sample_rate=0.1,
    send_default_pii=True,
    environment='production',
)
\`\`\`

### Using Environment Variables

\`\`\`bash
# .env file (not committed to git)
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

DB_NAME=myproject_db
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432

REDIS_URL=redis://localhost:6379/0

EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-password

SENTRY_DSN=https://your-sentry-dsn
\`\`\`

\`\`\`python
# Load environment-specific settings
# manage.py or wsgi.py
import os
from django.core.wsgi import get_wsgi_application

# Set settings module based on environment
environment = os.environ.get('DJANGO_ENVIRONMENT', 'development')
os.environ.setdefault(
    'DJANGO_SETTINGS_MODULE',
    f'myproject.settings.{environment}'
)

application = get_wsgi_application()
\`\`\`

---

## URL Configuration Deep Dive

### URL Patterns and Routing

\`\`\`python
# myproject/urls.py - Root URL configuration
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # Apps
    path('', TemplateView.as_view (template_name='home.html'), name='home'),
    path('articles/', include('apps.articles.urls')),
    path('users/', include('apps.users.urls')),
    
    # API
    path('api/v1/', include('apps.api.urls')),
    
    # Authentication
    path('accounts/', include('django.contrib.auth.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static (settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static (settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    
    # Debug toolbar
    import debug_toolbar
    urlpatterns += [path('__debug__/', include (debug_toolbar.urls))]
\`\`\`

### URL Patterns with Parameters

\`\`\`python
# articles/urls.py
from django.urls import path, re_path
from . import views

app_name = 'articles'

urlpatterns = [
    # Simple patterns
    path('', views.ArticleListView.as_view(), name='list'),
    path('create/', views.ArticleCreateView.as_view(), name='create'),
    
    # Integer parameter
    path('<int:pk>/', views.ArticleDetailView.as_view(), name='detail'),
    path('<int:pk>/edit/', views.ArticleUpdateView.as_view(), name='edit'),
    path('<int:pk>/delete/', views.ArticleDeleteView.as_view(), name='delete'),
    
    # Slug parameter
    path('category/<slug:slug>/', views.CategoryArticlesView.as_view(), name='category'),
    path('tag/<slug:slug>/', views.TagArticlesView.as_view(), name='tag'),
    
    # Multiple parameters
    path('<int:year>/<slug:slug>/', views.ArticleDetailView.as_view(), name='detail_by_date'),
    
    # UUID parameter
    path('draft/<uuid:draft_id>/', views.DraftDetailView.as_view(), name='draft'),
    
    # Regex pattern (more complex)
    re_path (r'^archive/(? P<year>[0-9]{4})/(? P<month>[0-9]{2})/$', 
            views.ArchiveView.as_view(), 
            name='archive'),
    
    # Optional parameters with defaults
    path('search/', views.ArticleSearchView.as_view(), name='search'),
    path('search/<str:query>/', views.ArticleSearchView.as_view(), name='search_query'),
]

# Usage in views: self.kwargs['year'], self.kwargs['slug']
# Usage in templates: {% url 'articles:detail' pk=article.pk %}
# Usage in reverse: reverse('articles:detail', kwargs={'pk': 1})
\`\`\`

### URL Converters

Built-in converters:
- **str**: Matches any non-empty string, excluding '/'
- **int**: Matches zero or any positive integer
- **slug**: Matches any slug string (letters, numbers, hyphens, underscores)
- **uuid**: Matches a formatted UUID
- **path**: Matches any non-empty string, including '/'

\`\`\`python
# Custom URL converter
class YearConverter:
    """Custom converter for 4-digit years"""
    regex = '[0-9]{4}'
    
    def to_python (self, value):
        return int (value)
    
    def to_url (self, value):
        return f'{value:04d}'

# Register converter
from django.urls import register_converter
register_converter(YearConverter, 'year')

# Use in URL pattern
path('articles/<year:year>/', views.YearArchiveView.as_view(), name='year_archive')
\`\`\`

---

## Django vs Other Frameworks

### Comparison Matrix

| Feature | Django | FastAPI | Flask | Express (Node) |
|---------|--------|---------|-------|----------------|
| **Type** | Full-stack | API-first | Micro | Micro |
| **Philosophy** | Batteries included | Modern async | Minimalist | Minimalist |
| **ORM** | Built-in (powerful) | None (use SQLAlchemy) | None | None |
| **Admin** | Built-in | None | None | None |
| **Auth** | Built-in | Manual | Flask-Login | Passport.js |
| **Templates** | Built-in | None (API-first) | Jinja2 | Various |
| **Forms** | Built-in | Pydantic | WTForms | Manual |
| **Async** | Partial (3.1+) | Full | Partial | Full |
| **Speed** | Medium | Very fast | Medium | Very fast |
| **Learning curve** | Steep | Medium | Easy | Easy |
| **Production use** | Very high | Growing | High | Very high |

### When to Choose Django

**Choose Django if:**
✅ Building a full-stack web application (not just API)
✅ Need admin interface out of the box
✅ Want comprehensive authentication/authorization
✅ Prefer convention over configuration
✅ Need ORM with migrations
✅ Building a CMS, blog, e-commerce site
✅ Team prefers monolithic architecture

**Choose something else if:**
✅ Building microservices (FastAPI)
✅ Need extreme performance (FastAPI, Go)
✅ Want full control/flexibility (Flask)
✅ Building real-time apps (Node.js + Socket.io)
✅ Already have frontend framework (FastAPI for API only)

---

## Production Django Examples

### Instagram (Django at Scale)

Instagram runs on Django and serves billions of requests per day:

- **Challenge**: Scale from thousands to billions of users
- **Solutions**:
  - Async views for I/O-bound operations
  - Extensive caching (Memcached, Redis)
  - Database sharding
  - CDN for static/media files
  - Custom middleware for monitoring
  - Celery for background jobs

### NASA

NASA uses Django for public-facing websites:

- **Why Django**: Security, reliability, rapid development
- **Features**: Content management, data visualization, API endpoints

---

## Best Practices

### 1. Settings Organization

- **Split settings** by environment (base, dev, prod, test)
- **Use environment variables** for secrets
- **Never commit** .env files or secrets

### 2. App Design

- **Single responsibility**: Each app does ONE thing
- **Reusable**: Design apps to be reusable across projects
- **Minimal coupling**: Apps should be loosely coupled

### 3. URL Naming

- **Use app namespaces**: \`app_name = 'articles'\`
- **Name all URLs**: \`name='detail'\`
- **Use reverse()**: Never hardcode URLs

### 4. Security

- **Keep SECRET_KEY secret**: Use environment variables
- **Set DEBUG=False** in production
- **Use HTTPS**: Set SECURE_SSL_REDIRECT=True
- **Enable security middleware**: All of them

### 5. Database

- **Use migrations**: Never edit database directly
- **Index frequently queried fields**: Add db_index=True
- **Use select_related/prefetch_related**: Avoid N+1 queries

---

## Common Patterns

### Function-Based Views (FBVs)

\`\`\`python
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .models import Article
from .forms import ArticleForm

def article_list (request):
    """List all published articles"""
    articles = Article.objects.filter (status='published').order_by('-published_at')
    return render (request, 'articles/article_list.html', {'articles': articles})

def article_detail (request, pk):
    """Display single article"""
    article = get_object_or_404(Article, pk=pk, status='published')
    return render (request, 'articles/article_detail.html', {'article': article})

@login_required
def article_create (request):
    """Create new article"""
    if request.method == 'POST':
        form = ArticleForm (request.POST)
        if form.is_valid():
            article = form.save (commit=False)
            article.author = request.user
            article.save()
            return redirect('articles:detail', pk=article.pk)
    else:
        form = ArticleForm()
    
    return render (request, 'articles/article_form.html', {'form': form})
\`\`\`

### Class-Based Views (CBVs)

\`\`\`python
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from .models import Article

class ArticleListView(ListView):
    """List all published articles"""
    model = Article
    template_name = 'articles/article_list.html'
    context_object_name = 'articles'
    paginate_by = 20
    
    def get_queryset (self):
        return Article.objects.filter (status='published').order_by('-published_at')

class ArticleDetailView(DetailView):
    """Display single article"""
    model = Article
    template_name = 'articles/article_detail.html'
    context_object_name = 'article'

class ArticleCreateView(LoginRequiredMixin, CreateView):
    """Create new article"""
    model = Article
    form_class = ArticleForm
    template_name = 'articles/article_form.html'
    
    def form_valid (self, form):
        form.instance.author = self.request.user
        return super().form_valid (form)
    
    def get_success_url (self):
        return reverse_lazy('articles:detail', kwargs={'pk': self.object.pk})
\`\`\`

---

## Summary

Django is a **full-featured web framework** that follows the MTV pattern:

**Key Architecture Components:**
1. **Models**: Data layer (ORM)
2. **Templates**: Presentation layer (HTML rendering)
3. **Views**: Business logic (request handling)
4. **URLs**: Routing layer (pattern matching)
5. **Middleware**: Request/response processing
6. **Settings**: Configuration management

**Django Strengths:**
- Batteries included (ORM, admin, auth, forms)
- Security by default
- Excellent documentation
- Large ecosystem
- Production-proven (Instagram, Pinterest, NASA)

**When to Use Django:**
- Full-stack web applications
- Content management systems
- E-commerce platforms
- Social networks
- Any project needing rapid development with built-in features

In the next sections, we'll dive deep into advanced ORM techniques, custom managers, signals, middleware, and Django REST Framework for building production APIs.
`,
};

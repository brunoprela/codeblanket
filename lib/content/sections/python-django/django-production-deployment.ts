export const djangoProductionDeployment = {
  title: 'Django Production Deployment',
  id: 'django-production-deployment',
  content: `
# Django Production Deployment

## Introduction

**Production deployment** transforms your Django application from a development environment to a scalable, secure, high-performance system serving real users.

### Production Requirements

- **Performance**: Fast response times under load
- **Reliability**: High uptime and fault tolerance
- **Security**: Protected against attacks
- **Scalability**: Handle growing traffic
- **Monitoring**: Track performance and errors

**Industry Standards:**
- Instagram: Serves billions of requests daily
- Pinterest: Handles massive concurrent users
- Spotify: 99.9% uptime with Django backend

By the end of this section, you'll master:
- Production settings configuration
- WSGI servers (Gunicorn, uWSGI)
- Reverse proxy (NGINX)
- Docker containerization
- Database optimization
- Static file serving
- Monitoring and logging
- CI/CD pipelines

---

## Production Settings

### Environment-Based Configuration

\`\`\`python
# config/settings/__init__.py
import os

ENV = os.environ.get('DJANGO_ENV', 'development')

if ENV == 'production':
    from .production import *
elif ENV == 'staging':
    from .staging import *
else:
    from .development import *
\`\`\`

\`\`\`python
# config/settings/base.py
"""Base settings shared across environments"""
import os
from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = config('SECRET_KEY')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third party
    'rest_framework',
    'django_celery_beat',
    
    # Local apps
    'articles',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

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
\`\`\`

\`\`\`python
# config/settings/production.py
"""Production settings"""
from .base import *

DEBUG = False

ALLOWED_HOSTS = config('ALLOWED_HOSTS', cast=lambda v: [s.strip() for s in v.split(',')])

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DB_NAME'),
        'USER': config('DB_USER'),
        'PASSWORD': config('DB_PASSWORD'),
        'HOST': config('DB_HOST'),
        'PORT': config('DB_PORT', default='5432'),
        'CONN_MAX_AGE': 60,
        'OPTIONS': {
            'connect_timeout': 10,
        }
    }
}

# Security
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = 'DENY'

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Cache
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': config('REDIS_URL'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Celery
CELERY_BROKER_URL = config('CELERY_BROKER_URL')
CELERY_RESULT_BACKEND = config('CELERY_RESULT_BACKEND')

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
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/django/app.log',
            'maxBytes': 1024 * 1024 * 10,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['file', 'console'],
        'level': 'INFO',
    },
}
\`\`\`

---

## WSGI Servers

### Gunicorn Configuration

\`\`\`bash
# Install Gunicorn
pip install gunicorn

# Run Gunicorn
gunicorn config.wsgi:application --bind 0.0.0.0:8000
\`\`\`

\`\`\`python
# config/gunicorn.py
"""Gunicorn configuration"""
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"

# Process naming
proc_name = "django_app"

# Server mechanics
daemon = False
pidfile = "/var/run/gunicorn.pid"
user = "www-data"
group = "www-data"
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None
\`\`\`

\`\`\`bash
# Run with config file
gunicorn config.wsgi:application -c config/gunicorn.py
\`\`\`

### Async Workers with Uvicorn

\`\`\`bash
# Install
pip install uvicorn gunicorn

# Run with Uvicorn workers
gunicorn config.asgi:application \\
    -k uvicorn.workers.UvicornWorker \\
    --workers 4 \\
    --bind 0.0.0.0:8000
\`\`\`

---

## NGINX Configuration

### Basic NGINX Setup

\`\`\`nginx
# /etc/nginx/sites-available/myapp
upstream app_server {
    server 127.0.0.1:8000 fail_timeout=0;
}

server {
    listen 80;
    server_name example.com www.example.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name example.com www.example.com;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Logs
    access_log /var/log/nginx/myapp-access.log;
    error_log /var/log/nginx/myapp-error.log;
    
    # Max upload size
    client_max_body_size 20M;
    
    # Static files
    location /static/ {
        alias /path/to/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Media files
    location /media/ {
        alias /path/to/media/;
        expires 7d;
    }
    
    # Django application
    location / {
        proxy_pass http://app_server;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
    }
}
\`\`\`

\`\`\`bash
# Enable site
sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
\`\`\`

---

## Docker Deployment

### Dockerfile

\`\`\`dockerfile
# Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=off \\
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    postgresql-client \\
    build-essential \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \\
    pip install -r requirements.txt

# Copy project
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Create non-root user
RUN useradd -m -u 1000 appuser && \\
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run gunicorn
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "4"]
\`\`\`

### Docker Compose

\`\`\`yaml
# docker-compose.yml
version: '3.8'

services:
  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=myapp_db
      - POSTGRES_USER=myapp_user
      - POSTGRES_PASSWORD=secret_password
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U myapp_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  web:
    build: .
    command: gunicorn config.wsgi:application --bind 0.0.0.0:8000
    volumes:
      - .:/app
      - static_volume:/app/staticfiles
      - media_volume:/app/media
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started

  celery:
    build: .
    command: celery -A config worker -l info
    volumes:
      - .:/app
    env_file:
      - .env
    depends_on:
      - db
      - redis

  celery-beat:
    build: .
    command: celery -A config beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler
    volumes:
      - .:/app
    env_file:
      - .env
    depends_on:
      - db
      - redis

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
      - static_volume:/app/staticfiles
      - media_volume:/app/media
      - /etc/letsencrypt:/etc/letsencrypt
    depends_on:
      - web

volumes:
  postgres_data:
  redis_data:
  static_volume:
  media_volume:
\`\`\`

---

## Database Optimization

### Connection Pooling

\`\`\`python
# Install pgbouncer or use connection pooling
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'myapp_db',
        'USER': 'myapp_user',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '6432',  # pgbouncer port
        'CONN_MAX_AGE': 60,  # Persistent connections
        'OPTIONS': {
            'connect_timeout': 10,
            'options': '-c statement_timeout=30000',  # 30 seconds
        }
    }
}
\`\`\`

### Database Indexes

\`\`\`python
from django.db import models

class Article (models.Model):
    title = models.CharField (max_length=200, db_index=True)
    slug = models.SlugField (unique=True, db_index=True)
    status = models.CharField (max_length=10, db_index=True)
    created_at = models.DateTimeField (auto_now_add=True, db_index=True)
    
    class Meta:
        indexes = [
            models.Index (fields=['status', '-created_at']),
            models.Index (fields=['author', 'status']),
        ]
\`\`\`

---

## Monitoring and Logging

### Sentry Integration

\`\`\`python
# Install
pip install sentry-sdk

# settings.py
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

sentry_sdk.init(
    dsn=config('SENTRY_DSN'),
    integrations=[DjangoIntegration()],
    traces_sample_rate=0.1,
    send_default_pii=True,
    environment=config('ENVIRONMENT', default='production'),
)
\`\`\`

### Prometheus Metrics

\`\`\`python
# Install
pip install django-prometheus

# settings.py
INSTALLED_APPS = [
    'django_prometheus',
    ...
]

MIDDLEWARE = [
    'django_prometheus.middleware.PrometheusBeforeMiddleware',
    ...
    'django_prometheus.middleware.PrometheusAfterMiddleware',
]

# urls.py
urlpatterns = [
    path('', include('django_prometheus.urls')),
]
\`\`\`

---

## CI/CD Pipeline

### GitHub Actions

\`\`\`yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          python manage.py test
      
      - name: Check security
        run: |
          python manage.py check --deploy
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: \${{ secrets.SERVER_HOST }}
          username: \${{ secrets.SERVER_USER }}
          key: \${{ secrets.SSH_KEY }}
          script: |
            cd /var/www/myapp
            git pull origin main
            source venv/bin/activate
            pip install -r requirements.txt
            python manage.py migrate
            python manage.py collectstatic --noinput
            sudo systemctl restart gunicorn
\`\`\`

---

## Summary

**Production Deployment Checklist:**
- ✅ Environment-based settings
- ✅ SECRET_KEY in environment
- ✅ DEBUG = False
- ✅ ALLOWED_HOSTS configured
- ✅ Database with connection pooling
- ✅ Static files with WhiteNoise/CDN
- ✅ HTTPS with SSL certificates
- ✅ Gunicorn WSGI server
- ✅ NGINX reverse proxy
- ✅ Redis for caching
- ✅ Celery for background tasks
- ✅ Docker containerization
- ✅ Database backups
- ✅ Monitoring (Sentry, Prometheus)
- ✅ Logging configured
- ✅ CI/CD pipeline
- ✅ Security headers
- ✅ Rate limiting
- ✅ Regular updates

**Performance Optimization:**
- Use CDN for static files
- Enable caching (Redis)
- Database connection pooling
- Query optimization
- Async task processing
- Load balancing
- Auto-scaling

Django is production-ready out of the box, but proper configuration and infrastructure are essential for scalable, secure deployments.
`,
};

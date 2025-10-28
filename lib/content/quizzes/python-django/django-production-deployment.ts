export const djangoProductionDeploymentQuiz = [
  {
    id: 1,
    question:
      'Explain a complete Django production deployment stack with Gunicorn, NGINX, PostgreSQL, and Redis. Include configuration, process management, and load balancing.',
    answer: `
**Production Stack Architecture:**

\`\`\`
Client → NGINX (reverse proxy) → Gunicorn (WSGI) → Django
                                            ↓
                                        PostgreSQL
                                            ↓
                                          Redis
\`\`\`

**Gunicorn Configuration:**

\`\`\`python
# gunicorn.conf.py
import multiprocessing

bind = "127.0.0.1:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"  # or "gevent" for async
worker_connections = 1000
max_requests = 1000  # Restart workers after N requests
max_requests_jitter = 50
timeout = 30
keepalive = 2

# Logging
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"

# Process naming
proc_name = "myproject"

# Run
# gunicorn myproject.wsgi:application -c gunicorn.conf.py
\`\`\`

**NGINX Configuration:**

\`\`\`nginx
upstream myproject {
    server 127.0.0.1:8000;
    # Load balancing multiple Gunicorn instances
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name example.com www.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name example.com www.example.com;
    
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    client_max_body_size 100M;
    
    # Static files
    location /static/ {
        alias /var/www/myproject/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Media files
    location /media/ {
        alias /var/www/myproject/media/;
        expires 7d;
    }
    
    # Proxy to Django
    location / {
        proxy_pass http://myproject;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        proxy_buffering off;
    }
}
\`\`\`

**PostgreSQL Configuration:**

\`\`\`python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'myproject',
        'USER': 'myproject_user',
        'PASSWORD': os.environ['DB_PASSWORD'],
        'HOST': 'localhost',
        'PORT': '5432',
        'CONN_MAX_AGE': 600,  # Connection pooling
        'OPTIONS': {
            'connect_timeout': 10,
        }
    }
}

# Use pgbouncer for connection pooling
# HOST': '/var/run/postgresql',  # Unix socket
\`\`\`

**Redis Configuration:**

\`\`\`python
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            }
        }
    }
}

CELERY_BROKER_URL = 'redis://127.0.0.1:6379/0'
CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379/0'
\`\`\`

**Process Management (systemd):**

\`\`\`ini
# /etc/systemd/system/gunicorn.service
[Unit]
Description=Gunicorn daemon for Django
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/myproject
Environment="PATH=/var/www/myproject/venv/bin"
ExecStart=/var/www/myproject/venv/bin/gunicorn \\
    --workers 3 \\
    --bind unix:/run/gunicorn.sock \\
    myproject.wsgi:application

[Install]
WantedBy=multi-user.target

# Enable and start
# systemctl enable gunicorn
# systemctl start gunicorn
\`\`\`
      `,
  },
  {
    question:
      'Describe Django application containerization with Docker. Include multi-stage builds, docker-compose for development, and production Kubernetes deployment.',
    answer: `
**Multi-Stage Dockerfile:**

\`\`\`dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    build-essential \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    libpq5 \\
    && rm -rf /var/lib/apt/lists/* \\
    && useradd -m -u 1000 appuser

COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . .

USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH

RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "myproject.wsgi:application"]
\`\`\`

**docker-compose.yml (Development):**

\`\`\`yaml
version: '3.8'

services:
  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: myproject
      POSTGRES_USER: myproject
      POSTGRES_PASSWORD: devpassword
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  web:
    build: .
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - db
      - redis
  
  celery:
    build: .
    command: celery -A myproject worker -l info
    volumes:
      - .:/app
    env_file:
      - .env
    depends_on:
      - db
      - redis
  
  celery-beat:
    build: .
    command: celery -A myproject beat -l info
    volumes:
      - .:/app
    env_file:
      - .env
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
\`\`\`

**Kubernetes Deployment:**

\`\`\`yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: django-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: django
  template:
    metadata:
      labels:
        app: django
    spec:
      containers:
      - name: django
        image: myregistry.com/myproject:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: django-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: django-secrets
              key: secret-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health/
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready/
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: django-service
spec:
  selector:
    app: django
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: django-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: django-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
\`\`\`

**Health Check Views:**

\`\`\`python
from django.http import JsonResponse
from django.db import connection

def health_check (request):
    return JsonResponse({'status': 'healthy'})

def readiness_check (request):
    try:
        # Check database
        connection.ensure_connection()
        # Check Redis
        cache.set('health_check', 'ok', 1)
        return JsonResponse({'status': 'ready'})
    except Exception as e:
        return JsonResponse({'status': 'not ready', 'error': str (e)}, status=503)
\`\`\`
      `,
  },
  {
    question:
      'Explain production settings, environment variables, secret management, monitoring, and logging for Django applications. Include CI/CD best practices.',
    answer: `
**Production Settings:**

\`\`\`python
# settings/production.py
import os
from .base import *

DEBUG = False
ALLOWED_HOSTS = os.environ['ALLOWED_HOSTS'].split(',')

# Security
SECRET_KEY = os.environ['SECRET_KEY']
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# Static/Media
STATIC_ROOT = '/var/www/myproject/static/'
MEDIA_ROOT = '/var/www/myproject/media/'

# Use S3 for static files
STATICFILES_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
AWS_STORAGE_BUCKET_NAME = os.environ['AWS_BUCKET']

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
            'filename': '/var/log/django/django.log',
            'maxBytes': 1024*1024*15,  # 15MB
            'backupCount': 10,
            'formatter': 'verbose',
        },
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/django/error.log',
            'maxBytes': 1024*1024*15,
            'backupCount': 10,
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
        'django.request': {
            'handlers': ['error_file'],
            'level': 'ERROR',
            'propagate': False,
        },
    },
}
\`\`\`

**Environment Variables (.env):**

\`\`\`bash
# .env
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=example.com,www.example.com

DATABASE_URL=postgresql://user:pass@localhost/dbname
REDIS_URL=redis://localhost:6379/0

AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_STORAGE_BUCKET_NAME=mybucket

SENTRY_DSN=https://xxx@sentry.io/xxx
\`\`\`

**Loading Environment Variables:**

\`\`\`python
import os
from pathlib import Path
import environ

env = environ.Env()
BASE_DIR = Path(__file__).resolve().parent.parent

# Read .env file
environ.Env.read_env (os.path.join(BASE_DIR, '.env'))

SECRET_KEY = env('SECRET_KEY')
DEBUG = env.bool('DEBUG', default=False)
DATABASES = {'default': env.db()}
\`\`\`

**Secret Management (Kubernetes):**

\`\`\`yaml
apiVersion: v1
kind: Secret
metadata:
  name: django-secrets
type: Opaque
data:
  secret-key: <base64-encoded>
  database-url: <base64-encoded>
\`\`\`

**Monitoring with Sentry:**

\`\`\`python
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

sentry_sdk.init(
    dsn=os.environ['SENTRY_DSN'],
    integrations=[DjangoIntegration()],
    traces_sample_rate=0.1,
    send_default_pii=False,
)
\`\`\`

**Application Monitoring:**

\`\`\`python
# Custom middleware for metrics
import time
from django.utils.deprecation import MiddlewareMixin

class MetricsMiddleware(MiddlewareMixin):
    def process_request (self, request):
        request.start_time = time.time()
    
    def process_response (self, request, response):
        if hasattr (request, 'start_time'):
            duration = time.time() - request.start_time
            logger.info (f'{request.method} {request.path} - {response.status_code} - {duration:.2f}s')
        return response
\`\`\`

**CI/CD Pipeline (.github/workflows/deploy.yml):**

\`\`\`yaml
name: Deploy

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest --cov=myapp
    - name: Lint
      run: |
        ruff check .
        mypy .
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      run: |
        # Build Docker image
        docker build -t myregistry.com/myproject:\${{ github.sha }} .
        docker push myregistry.com/myproject:\${{ github.sha }}
        
        # Update Kubernetes
        kubectl set image deployment/django-app \\
          django=myregistry.com/myproject:\${{ github.sha }}
\`\`\`
      `,
  },
].map(({ id: _id, ...q }, idx) => ({
  id: `django-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));

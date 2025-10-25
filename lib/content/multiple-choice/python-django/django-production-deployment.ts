import { MultipleChoiceQuestion } from '@/lib/types';

export const DjangoProductionDeploymentMultipleChoice = [
  {
    id: 1,
    question: 'What is the role of Gunicorn in Django production deployment?',
    options: [
      'A) Database connection pooling',
      'B) WSGI HTTP server for running Django',
      'C) Static file serving',
      'D) Task queue management',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) WSGI HTTP server for running Django**

\`\`\`bash
gunicorn myproject.wsgi:application --workers 3 --bind 0.0.0.0:8000
\`\`\`

Gunicorn is a production-grade WSGI server that runs Django applications, handling multiple workers and requests efficiently.
      `,
  },
  {
    question: 'Why should you use NGINX in front of Gunicorn?',
    options: [
      'A) To run Python code faster',
      'B) To serve static files efficiently and act as reverse proxy',
      'C) To manage database connections',
      'D) To compile Django templates',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) To serve static files efficiently and act as reverse proxy**

\`\`\`nginx
location /static/ {
    alias /var/www/myproject/static/;
}

location / {
    proxy_pass http://127.0.0.1:8000;
}
\`\`\`

NGINX excels at serving static files and acts as a reverse proxy, load balancer, and SSL terminator.
      `,
  },
  {
    question: 'Which setting should NEVER be True in production?',
    options: [
      'A) SECURE_SSL_REDIRECT',
      'B) DEBUG',
      'C) SESSION_COOKIE_SECURE',
      'D) CSRF_COOKIE_SECURE',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) DEBUG**

\`\`\`python
# Production settings
DEBUG = False  # MUST be False!

# DEBUG=True in production exposes:
# - Detailed error pages with code
# - Stack traces
# - Environment variables
# - Security vulnerabilities
\`\`\`

DEBUG=True exposes sensitive information and should never be enabled in production.
      `,
  },
  {
    question: 'What is the purpose of collectstatic command?',
    options: [
      'A) To compress CSS/JS files',
      'B) To gather all static files into STATIC_ROOT for serving',
      'C) To minify static assets',
      'D) To upload files to CDN',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) To gather all static files into STATIC_ROOT for serving**

\`\`\`bash
python manage.py collectstatic --noinput
\`\`\`

Collects static files from all apps into a single directory (STATIC_ROOT) for efficient serving by NGINX or CDN.
      `,
  },
  {
    question: 'How should secrets be managed in containerized Django apps?',
    options: [
      'A) Hard-code in settings.py',
      'B) Store in .env file in Docker image',
      'C) Use environment variables or secret management systems',
      'D) Store in database',
    ],
    correctAnswer: 2,
    explanation: `
**Correct Answer: C) Use environment variables or secret management systems**

\`\`\`python
# Load from environment
SECRET_KEY = os.environ['SECRET_KEY']

# Kubernetes secrets
env:
- name: SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: django-secrets
      key: secret-key
\`\`\`

Never hard-code secrets or include them in Docker images. Use environment variables or secret management.
      `,
  },
].map(({ id, ...q }, idx) => ({ id: `django-mc-${idx + 1}`, ...q }));

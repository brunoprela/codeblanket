import { MultipleChoiceQuestion } from '@/lib/types';

export const TestingDjangoApplicationsMultipleChoice = [
  {
    id: 1,
    question:
      'Which decorator is required to test database operations with pytest-django?',
    options: [
      'A) @pytest.database',
      'B) @pytest.mark.django_db',
      'C) @pytest.db_access',
      'D) @django.test.db',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) @pytest.mark.django_db**

\`\`\`python
@pytest.mark.django_db
def test_user_creation():
    user = User.objects.create(username='test')
    assert User.objects.count() == 1
\`\`\`

This marker enables database access for the test.
      `,
  },
  {
    question:
      'How do you test an API endpoint that requires authentication in DRF?',
    options: [
      'A) client.login(username, password)',
      'B) client.force_authenticate(user=user)',
      'C) client.set_auth(user)',
      'D) client.authenticate(user)',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) client.force_authenticate(user=user)**

\`\`\`python
from rest_framework.test import APIClient

client = APIClient()
client.force_authenticate(user=user)
response = client.get('/api/articles/')
\`\`\`

force_authenticate() bypasses authentication for testing.
      `,
  },
  {
    question: 'What is the purpose of factory_boy in Django testing?',
    options: [
      'A) To create test database',
      'B) To generate test model instances with realistic data',
      'C) To mock API calls',
      'D) To run tests in parallel',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) To generate test model instances with realistic data**

\`\`\`python
class UserFactory(DjangoModelFactory):
    class Meta:
        model = User
    username = factory.Faker('username')

# Create instances easily
user = UserFactory()
users = UserFactory.create_batch(10)
\`\`\`

Factories make test data creation easier and more realistic.
      `,
  },
  {
    question:
      'How do you test a Celery task without executing it asynchronously?',
    options: [
      'A) Call task() directly without .delay()',
      'B) Use task.sync()',
      'C) Set CELERY_TASK_ALWAYS_EAGER = True',
      'D) Use @mock.celery decorator',
    ],
    correctAnswer: 0,
    explanation: `
**Correct Answer: A) Call task() directly without .delay()**

\`\`\`python
# Test task directly
result = send_email_task(user_id, subject, message)

# Or set eager mode for all tasks
CELERY_TASK_ALWAYS_EAGER = True  # Executes synchronously

# Mock the delay call
with patch('myapp.tasks.send_email_task.delay') as mock:
    trigger_email()
    mock.assert_called_once()
\`\`\`

Calling without .delay() executes synchronously for testing.
      `,
  },
  {
    question: 'What does the --reuse-db flag do in pytest-django?',
    options: [
      'A) Uses production database',
      'B) Reuses test database between test runs instead of recreating',
      'C) Shares database between tests',
      'D) Prevents database cleanup',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Reuses test database between test runs instead of recreating**

\`\`\`bash
pytest --reuse-db
\`\`\`

Significantly speeds up test runs by not dropping/recreating the test database each time.
      `,
  },
].map(({ id, ...q }, idx) => ({ id: `django-mc-${idx + 1}`, ...q }));

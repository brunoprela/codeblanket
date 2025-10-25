export const testingDjangoApplicationsQuiz = [
  {
    id: 1,
    question:
      'Explain comprehensive testing strategies for Django applications using pytest-django. Include fixtures, database handling, and test organization.',
    answer: `
**pytest-django Setup:**

\`\`\`python
# requirements.txt
pytest==7.4.0
pytest-django==4.5.2
pytest-cov==4.1.0
factory-boy==3.3.0

# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
DJANGO_SETTINGS_MODULE = "myproject.settings"
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--reuse-db --cov=myapp --cov-report=html"
\`\`\`

**Database Fixtures:**

\`\`\`python
import pytest
from django.contrib.auth import get_user_model

User = get_user_model()

@pytest.fixture
def user (db):
    return User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='testpass123'
    )

@pytest.fixture
def article (db, user):
    return Article.objects.create(
        title='Test Article',
        content='Content',
        author=user
    )

# Use fixtures
def test_article_creation (article):
    assert article.title == 'Test Article'
    assert article.author.username == 'testuser'
\`\`\`

**Factory Pattern:**

\`\`\`python
import factory
from factory.django import DjangoModelFactory

class UserFactory(DjangoModelFactory):
    class Meta:
        model = User
    
    username = factory.Sequence (lambda n: f'user{n}')
    email = factory.LazyAttribute (lambda obj: f'{obj.username}@example.com')
    
class ArticleFactory(DjangoModelFactory):
    class Meta:
        model = Article
    
    title = factory.Faker('sentence')
    content = factory.Faker('text')
    author = factory.SubFactory(UserFactory)

# Usage
def test_with_factories (db):
    user = UserFactory()
    articles = ArticleFactory.create_batch(5, author=user)
    assert user.articles.count() == 5
\`\`\`

**Test Organization:**

\`\`\`
tests/
├── conftest.py          # Shared fixtures
├── test_models.py
├── test_views.py
├── test_serializers.py
├── test_tasks.py
└── factories.py
\`\`\`

**Model Testing:**

\`\`\`python
def test_article_str_representation (article):
    assert str (article) == article.title

def test_article_published_manager (db):
    ArticleFactory.create_batch(3, status='published')
    ArticleFactory.create_batch(2, status='draft')
    assert Article.published.count() == 3

@pytest.mark.django_db
def test_article_slug_generation():
    article = Article.objects.create (title='Test Article')
    assert article.slug == 'test-article'
\`\`\`

**Database Strategies:**

\`\`\`python
# Transactional test (default, rolled back)
def test_with_db (db):
    user = User.objects.create (username='test')
    assert User.objects.count() == 1
    # Auto rollback

# Non-transactional (for testing transactions)
@pytest.mark.django_db (transaction=True)
def test_with_transaction():
    with transaction.atomic():
        user = User.objects.create (username='test')
\`\`\`
      `,
  },
  {
    question:
      'Describe testing DRF APIs including authentication, permissions, and response validation. Include examples of testing ViewSets and custom actions.',
    answer: `
**API Testing Setup:**

\`\`\`python
import pytest
from rest_framework.test import APIClient
from rest_framework import status

@pytest.fixture
def api_client():
    return APIClient()

@pytest.fixture
def authenticated_client (user):
    client = APIClient()
    client.force_authenticate (user=user)
    return client
\`\`\`

**Testing ViewSets:**

\`\`\`python
@pytest.mark.django_db
class TestArticleViewSet:
    def test_list_articles (self, api_client):
        ArticleFactory.create_batch(5, status='published')
        response = api_client.get('/api/articles/')
        
        assert response.status_code == status.HTTP_200_OK
        assert len (response.data['results']) == 5
    
    def test_create_article_authenticated (self, authenticated_client, user):
        data = {
            'title': 'New Article',
            'content': 'Content here',
        }
        response = authenticated_client.post('/api/articles/', data)
        
        assert response.status_code == status.HTTP_201_CREATED
        assert response.data['title'] == 'New Article'
        assert Article.objects.filter (author=user).count() == 1
    
    def test_create_article_unauthenticated (self, api_client):
        data = {'title': 'Test'}
        response = api_client.post('/api/articles/', data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_update_own_article (self, authenticated_client, user):
        article = ArticleFactory (author=user)
        data = {'title': 'Updated Title'}
        response = authenticated_client.patch(
            f'/api/articles/{article.id}/', data
        )
        
        assert response.status_code == status.HTTP_200_OK
        article.refresh_from_db()
        assert article.title == 'Updated Title'
    
    def test_cannot_update_others_article (self, authenticated_client):
        other_user = UserFactory()
        article = ArticleFactory (author=other_user)
        response = authenticated_client.patch(
            f'/api/articles/{article.id}/', {'title': 'Hacked'}
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
\`\`\`

**Testing Custom Actions:**

\`\`\`python
def test_publish_action (self, authenticated_client, user):
    article = ArticleFactory (author=user, status='draft')
    response = authenticated_client.post(
        f'/api/articles/{article.id}/publish/'
    )
    
    assert response.status_code == status.HTTP_200_OK
    article.refresh_from_db()
    assert article.status == 'published'

def test_featured_list (self, api_client):
    ArticleFactory.create_batch(3, featured=True)
    ArticleFactory.create_batch(2, featured=False)
    
    response = api_client.get('/api/articles/featured/')
    
    assert response.status_code == status.HTTP_200_OK
    assert len (response.data) == 3
\`\`\`

**Testing Permissions:**

\`\`\`python
@pytest.fixture
def admin_client (admin_user):
    client = APIClient()
    client.force_authenticate (user=admin_user)
    return client

def test_delete_requires_admin (self, authenticated_client, user):
    article = ArticleFactory (author=user)
    response = authenticated_client.delete (f'/api/articles/{article.id}/')
    
    assert response.status_code == status.HTTP_403_FORBIDDEN

def test_admin_can_delete (self, admin_client):
    article = ArticleFactory()
    response = admin_client.delete (f'/api/articles/{article.id}/')
    
    assert response.status_code == status.HTTP_204_NO_CONTENT
    assert not Article.objects.filter (id=article.id).exists()
\`\`\`

**Response Validation:**

\`\`\`python
def test_article_detail_response_structure (self, api_client):
    article = ArticleFactory (status='published')
    response = api_client.get (f'/api/articles/{article.id}/')
    
    assert response.status_code == status.HTTP_200_OK
    data = response.data
    
    # Validate response structure
    assert 'id' in data
    assert 'title' in data
    assert 'author' in data
    assert data['title'] == article.title
    assert data['author']['username'] == article.author.username
\`\`\`
      `,
  },
  {
    question:
      'Explain testing async views, middleware, signals, and Celery tasks in Django. Include mocking strategies and integration testing.',
    answer: `
**Testing Async Views:**

\`\`\`python
import pytest
from asgiref.sync import sync_to_async

@pytest.mark.django_db
@pytest.mark.asyncio
async def test_async_article_list (api_client):
    await sync_to_async(ArticleFactory.create_batch)(5)
    response = await sync_to_async (api_client.get)('/api/articles/')
    
    assert response.status_code == 200

# Or use async client
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_with_async_client():
    async with AsyncClient (app=app, base_url="http://test") as client:
        response = await client.get("/api/articles/")
        assert response.status_code == 200
\`\`\`

**Testing Middleware:**

\`\`\`python
from django.test import RequestFactory

@pytest.fixture
def rf():
    return RequestFactory()

def test_custom_middleware (rf):
    request = rf.get('/api/articles/')
    request.user = UserFactory()
    
    middleware = CustomMiddleware (get_response=lambda r: r)
    response = middleware (request)
    
    assert hasattr (request, 'custom_attr')

# Integration test
@pytest.mark.django_db
def test_middleware_integration (client):
    response = client.get('/api/articles/')
    assert 'X-Custom-Header' in response
\`\`\`

**Testing Signals:**

\`\`\`python
from unittest.mock import Mock, patch

def test_post_save_signal_sends_email (db):
    with patch('myapp.signals.send_email_task.delay') as mock_task:
        article = Article.objects.create (title='Test', status='published')
        
        mock_task.assert_called_once_with (article.id)

def test_signal_handler_directly():
    mock_instance = Mock()
    mock_instance.status = 'published'
    
    # Call handler directly
    article_published_handler (sender=Article, instance=mock_instance)
    
    # Verify side effects
    assert mock_instance.published_at is not None
\`\`\`

**Testing Celery Tasks:**

\`\`\`python
from unittest.mock import patch

@pytest.mark.django_db
def test_send_email_task():
    user = UserFactory()
    
    # Test task directly
    result = send_email_task (user.id, 'Subject', 'Message')
    
    assert 'Email sent' in result
    assert len (mail.outbox) == 1
    assert mail.outbox[0].subject == 'Subject'

def test_task_retry_on_failure():
    with patch('myapp.tasks.external_api_call', side_effect=Exception('API down')):
        with pytest.raises(Exception):
            api_task.apply().get()

# Mock Celery execution
def test_task_called (authenticated_client):
    with patch('myapp.tasks.process_article.delay') as mock_task:
        response = authenticated_client.post('/api/articles/', {...})
        
        assert response.status_code == 201
        mock_task.assert_called_once_with (response.data['id'])
\`\`\`

**Integration Testing:**

\`\`\`python
@pytest.mark.django_db
class TestArticleWorkflow:
    def test_complete_article_lifecycle (self, authenticated_client, user):
        # Create
        data = {'title': 'Test', 'content': 'Content'}
        response = authenticated_client.post('/api/articles/', data)
        article_id = response.data['id']
        
        # Retrieve
        response = authenticated_client.get (f'/api/articles/{article_id}/')
        assert response.data['status'] == 'draft'
        
        # Publish
        response = authenticated_client.post (f'/api/articles/{article_id}/publish/')
        assert response.status_code == 200
        
        # Verify published
        article = Article.objects.get (id=article_id)
        assert article.status == 'published'
        assert article.published_at is not None

# Test with real Celery (integration)
@pytest.mark.django_db
@pytest.mark.celery
def test_with_real_celery (celery_worker):
    result = send_email_task.delay (user_id=1)
    assert result.get (timeout=10) == 'Email sent'
\`\`\`

**Coverage Best Practices:**

\`\`\`bash
pytest --cov=myapp --cov-report=html --cov-report=term-missing
\`\`\`
      `,
  },
].map(({ id, ...q }, idx) => ({
  id: `django-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));

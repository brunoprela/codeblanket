export const testingDjangoApplications = {
  title: 'Testing Django Applications',
  id: 'testing-django-applications',
  content: `
# Testing Django Applications

## Introduction

**Testing** is crucial for building reliable Django applications. Comprehensive test suites catch bugs early, enable refactoring with confidence, and serve as living documentation.

### Why Test Django Applications?

- **Confidence**: Deploy changes without fear
- **Regression Prevention**: Catch breaking changes immediately
- **Documentation**: Tests describe expected behavior
- **Refactoring Safety**: Modify code confidently
- **Team Collaboration**: Tests prevent integration issues

**Industry Standards:**
- Stripe: 90%+ test coverage for payment processing
- Shopify: Comprehensive test suites for e-commerce reliability
- Instagram: Extensive testing for high-traffic scenarios

By the end of this section, you'll master:
- Unit testing with Django TestCase
- pytest for Django
- Testing views, models, APIs
- Fixtures and factories
- Mocking external services
- Test coverage analysis
- CI/CD integration

---

## Django's Testing Framework

### Basic Test Structure

\`\`\`python
# articles/tests.py
from django.test import TestCase
from .models import Article

class ArticleModelTest(TestCase):
    """Test the Article model"""
    
    def setUp(self):
        """Run before each test method"""
        self.article = Article.objects.create(
            title='Test Article',
            content='Test content',
            status='published'
        )
    
    def test_article_creation(self):
        """Test article is created correctly"""
        self.assertEqual(self.article.title, 'Test Article')
        self.assertEqual(self.article.status, 'published')
    
    def test_article_str(self):
        """Test string representation"""
        self.assertEqual(str(self.article), 'Test Article')
    
    def tearDown(self):
        """Run after each test method"""
        pass  # Cleanup if needed
\`\`\`

### Running Tests

\`\`\`bash
# Run all tests
python manage.py test

# Run tests for specific app
python manage.py test articles

# Run specific test class
python manage.py test articles.tests.ArticleModelTest

# Run specific test method
python manage.py test articles.tests.ArticleModelTest.test_article_creation

# Run with verbosity
python manage.py test --verbosity=2

# Keep test database
python manage.py test --keepdb

# Run in parallel
python manage.py test --parallel
\`\`\`

---

## Testing Models

### Model Creation and Validation

\`\`\`python
from django.test import TestCase
from django.core.exceptions import ValidationError
from .models import Article, Category

class ArticleModelTest(TestCase):
    
    def setUp(self):
        self.category = Category.objects.create(name='Tech')
    
    def test_create_article(self):
        """Test creating a valid article"""
        article = Article.objects.create(
            title='Django Testing',
            content='Content here',
            category=self.category,
            status='published'
        )
        
        self.assertEqual(Article.objects.count(), 1)
        self.assertEqual(article.title, 'Django Testing')
        self.assertEqual(article.category, self.category)
    
    def test_article_slug_generation(self):
        """Test slug is auto-generated from title"""
        article = Article.objects.create(
            title='Django Testing Guide',
            content='Content',
            category=self.category
        )
        
        self.assertEqual(article.slug, 'django-testing-guide')
    
    def test_article_validation(self):
        """Test article validation"""
        article = Article(
            title='',  # Invalid: title required
            content='Content',
            category=self.category
        )
        
        with self.assertRaises(ValidationError):
            article.full_clean()
    
    def test_get_absolute_url(self):
        """Test URL generation"""
        article = Article.objects.create(
            title='Test',
            slug='test',
            content='Content',
            category=self.category
        )
        
        expected_url = f'/articles/{article.slug}/'
        self.assertEqual(article.get_absolute_url(), expected_url)
\`\`\`

### Testing QuerySets and Managers

\`\`\`python
class ArticleQuerySetTest(TestCase):
    
    def setUp(self):
        self.category = Category.objects.create(name='Tech')
        
        # Create test articles
        Article.objects.create(
            title='Published 1',
            content='Content',
            category=self.category,
            status='published'
        )
        Article.objects.create(
            title='Draft 1',
            content='Content',
            category=self.category,
            status='draft'
        )
        Article.objects.create(
            title='Published 2',
            content='Content',
            category=self.category,
            status='published'
        )
    
    def test_published_manager(self):
        """Test custom manager returns only published articles"""
        published = Article.published.all()
        self.assertEqual(published.count(), 2)
        self.assertTrue(all(a.status == 'published' for a in published))
    
    def test_by_category(self):
        """Test filtering by category"""
        articles = Article.objects.filter(category=self.category)
        self.assertEqual(articles.count(), 3)
\`\`\`

---

## Testing Views

### Testing Function-Based Views

\`\`\`python
from django.test import TestCase, Client
from django.urls import reverse

class ArticleViewTest(TestCase):
    
    def setUp(self):
        self.client = Client()
        self.category = Category.objects.create(name='Tech')
        self.article = Article.objects.create(
            title='Test Article',
            slug='test-article',
            content='Content',
            category=self.category,
            status='published'
        )
    
    def test_article_list_view(self):
        """Test article list view"""
        response = self.client.get(reverse('article_list'))
        
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'articles/list.html')
        self.assertContains(response, 'Test Article')
        self.assertEqual(len(response.context['articles']), 1)
    
    def test_article_detail_view(self):
        """Test article detail view"""
        url = reverse('article_detail', kwargs={'slug': self.article.slug})
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'articles/detail.html')
        self.assertEqual(response.context['article'], self.article)
    
    def test_article_not_found(self):
        """Test 404 for non-existent article"""
        response = self.client.get(reverse('article_detail', kwargs={'slug': 'non-existent'}))
        self.assertEqual(response.status_code, 404)
\`\`\`

### Testing Authenticated Views

\`\`\`python
from django.contrib.auth.models import User

class AuthenticatedViewTest(TestCase):
    
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.category = Category.objects.create(name='Tech')
    
    def test_create_article_requires_login(self):
        """Test create view requires authentication"""
        response = self.client.get(reverse('article_create'))
        self.assertEqual(response.status_code, 302)  # Redirect to login
    
    def test_create_article_authenticated(self):
        """Test creating article when logged in"""
        self.client.login(username='testuser', password='testpass123')
        
        response = self.client.post(reverse('article_create'), {
            'title': 'New Article',
            'content': 'Content here',
            'category': self.category.id,
            'status': 'published'
        })
        
        self.assertEqual(response.status_code, 302)  # Redirect after success
        self.assertEqual(Article.objects.count(), 1)
        
        article = Article.objects.first()
        self.assertEqual(article.title, 'New Article')
        self.assertEqual(article.author, self.user)
\`\`\`

---

## Testing Django REST Framework

### Testing API Endpoints

\`\`\`python
from rest_framework.test import APITestCase, APIClient
from rest_framework import status

class ArticleAPITest(APITestCase):
    
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.category = Category.objects.create(name='Tech')
        self.article = Article.objects.create(
            title='Test Article',
            content='Content',
            category=self.category,
            author=self.user,
            status='published'
        )
    
    def test_get_article_list(self):
        """Test GET /api/articles/"""
        response = self.client.get('/api/articles/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)
        self.assertEqual(response.data['results'][0]['title'], 'Test Article')
    
    def test_get_article_detail(self):
        """Test GET /api/articles/{id}/"""
        response = self.client.get(f'/api/articles/{self.article.id}/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['title'], 'Test Article')
    
    def test_create_article_unauthenticated(self):
        """Test POST /api/articles/ without authentication"""
        data = {
            'title': 'New Article',
            'content': 'Content',
            'category': self.category.id
        }
        response = self.client.post('/api/articles/', data)
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_create_article_authenticated(self):
        """Test POST /api/articles/ with authentication"""
        self.client.force_authenticate(user=self.user)
        
        data = {
            'title': 'New Article',
            'content': 'Content',
            'category': self.category.id,
            'status': 'published'
        }
        response = self.client.post('/api/articles/', data)
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Article.objects.count(), 2)
        self.assertEqual(response.data['title'], 'New Article')
    
    def test_update_article(self):
        """Test PUT /api/articles/{id}/"""
        self.client.force_authenticate(user=self.user)
        
        data = {
            'title': 'Updated Article',
            'content': 'Updated content',
            'category': self.category.id,
            'status': 'published'
        }
        response = self.client.put(f'/api/articles/{self.article.id}/', data)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.article.refresh_from_db()
        self.assertEqual(self.article.title, 'Updated Article')
    
    def test_delete_article(self):
        """Test DELETE /api/articles/{id}/"""
        self.client.force_authenticate(user=self.user)
        
        response = self.client.delete(f'/api/articles/{self.article.id}/')
        
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(Article.objects.count(), 0)
\`\`\`

---

## pytest for Django

### Installation

\`\`\`bash
pip install pytest pytest-django pytest-cov
\`\`\`

### Configuration

\`\`\`ini
# pytest.ini
[pytest]
DJANGO_SETTINGS_MODULE = myproject.settings
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = --cov=. --cov-report=html --cov-report=term-missing
\`\`\`

### pytest Tests

\`\`\`python
# articles/test_models.py
import pytest
from articles.models import Article

@pytest.mark.django_db
def test_create_article():
    """Test article creation"""
    article = Article.objects.create(
        title='Test',
        content='Content',
        status='published'
    )
    
    assert article.title == 'Test'
    assert Article.objects.count() == 1

@pytest.mark.django_db
def test_article_slug():
    """Test slug generation"""
    article = Article.objects.create(
        title='My Test Article',
        content='Content'
    )
    
    assert article.slug == 'my-test-article'
\`\`\`

### pytest Fixtures

\`\`\`python
# conftest.py
import pytest
from django.contrib.auth.models import User
from articles.models import Article, Category

@pytest.fixture
def user():
    """Create test user"""
    return User.objects.create_user(
        username='testuser',
        password='testpass123'
    )

@pytest.fixture
def category():
    """Create test category"""
    return Category.objects.create(name='Tech')

@pytest.fixture
def article(user, category):
    """Create test article"""
    return Article.objects.create(
        title='Test Article',
        content='Content',
        category=category,
        author=user,
        status='published'
    )

# Usage in tests
@pytest.mark.django_db
def test_article_with_fixtures(article):
    """Test using fixtures"""
    assert article.title == 'Test Article'
    assert article.status == 'published'
\`\`\`

---

## Factory Boy

### Installation

\`\`\`bash
pip install factory_boy
\`\`\`

### Defining Factories

\`\`\`python
# articles/factories.py
import factory
from factory.django import DjangoModelFactory
from django.contrib.auth.models import User
from .models import Article, Category

class UserFactory(DjangoModelFactory):
    class Meta:
        model = User
    
    username = factory.Sequence(lambda n: f'user{n}')
    email = factory.LazyAttribute(lambda obj: f'{obj.username}@example.com')
    first_name = factory.Faker('first_name')
    last_name = factory.Faker('last_name')

class CategoryFactory(DjangoModelFactory):
    class Meta:
        model = Category
    
    name = factory.Faker('word')

class ArticleFactory(DjangoModelFactory):
    class Meta:
        model = Article
    
    title = factory.Faker('sentence', nb_words=4)
    content = factory.Faker('paragraph', nb_sentences=10)
    author = factory.SubFactory(UserFactory)
    category = factory.SubFactory(CategoryFactory)
    status = 'published'
\`\`\`

### Using Factories

\`\`\`python
# In tests
from articles.factories import ArticleFactory, UserFactory

@pytest.mark.django_db
def test_with_factory():
    """Test using factories"""
    # Create single article
    article = ArticleFactory()
    assert Article.objects.count() == 1
    
    # Create multiple articles
    articles = ArticleFactory.create_batch(5)
    assert Article.objects.count() == 6
    
    # Override attributes
    article = ArticleFactory(title='Custom Title', status='draft')
    assert article.title == 'Custom Title'
    assert article.status == 'draft'
    
    # Create related objects
    user = UserFactory()
    article = ArticleFactory(author=user)
    assert article.author == user
\`\`\`

---

## Mocking External Services

### Using unittest.mock

\`\`\`python
from unittest.mock import patch, Mock
from django.test import TestCase

class PaymentTest(TestCase):
    
    @patch('articles.services.payment_gateway.charge')
    def test_process_payment(self, mock_charge):
        """Test payment processing with mocked gateway"""
        # Configure mock
        mock_charge.return_value = {'status': 'success', 'transaction_id': '12345'}
        
        # Call function that uses payment gateway
        result = process_order_payment(order_id=1)
        
        # Assertions
        self.assertEqual(result['status'], 'success')
        mock_charge.assert_called_once_with(amount=100, currency='USD')
    
    @patch('articles.services.send_email')
    def test_send_notification(self, mock_send_email):
        """Test email notification"""
        send_article_notification(article_id=1)
        
        # Verify email was sent
        mock_send_email.assert_called_once()
        args, kwargs = mock_send_email.call_args
        self.assertIn('New Article', kwargs['subject'])
\`\`\`

---

## Test Coverage

\`\`\`bash
# Install coverage
pip install coverage

# Run tests with coverage
coverage run --source='.' manage.py test

# Generate report
coverage report

# Generate HTML report
coverage html

# View in browser
open htmlcov/index.html
\`\`\`

---

## Summary

**Key Testing Concepts:**
- **TestCase**: Django's base test class
- **APITestCase**: For testing DRF APIs
- **pytest**: Modern testing framework
- **Fixtures**: Reusable test data
- **Factories**: Generate test data
- **Mocking**: Isolate external dependencies

**Testing Best Practices:**
- ✅ Test models, views, and APIs
- ✅ Use factories for test data
- ✅ Mock external services
- ✅ Aim for 80%+ coverage
- ✅ Write fast, independent tests
- ✅ Test edge cases and errors
- ✅ Use descriptive test names
- ✅ Run tests in CI/CD

Comprehensive testing ensures Django applications are reliable, maintainable, and production-ready.
`,
};

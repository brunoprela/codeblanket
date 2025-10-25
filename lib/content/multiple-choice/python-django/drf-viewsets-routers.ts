import { MultipleChoiceQuestion } from '@/lib/types';

export const DrfViewsetsRoutersMultipleChoice = [
  {
    id: 1,
    question: 'What does detail=True mean in @action decorator?',
    options: [
      'A) The action returns detailed information',
      'B) The action requires an instance ID in the URL',
      'C) The action only works in detail views',
      'D) The action provides verbose output',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) The action requires an instance ID in the URL**

\`\`\`python
@action (detail=True, methods=['post'])
def publish (self, request, pk=None):
    # URL: /articles/{id}/publish/
    article = self.get_object()  # Gets specific article
    
@action (detail=False, methods=['get'])
def recent (self, request):
    # URL: /articles/recent/
    # No ID needed
\`\`\`

detail=True creates instance-level endpoints, detail=False creates collection-level endpoints.
      `,
  },
  {
    question: 'Which mixin provides the create() action in GenericViewSet?',
    options: [
      'A) CreateModelMixin',
      'B) CreateMixin',
      'C) AddModelMixin',
      'D) PostModelMixin',
    ],
    correctAnswer: 0,
    explanation: `
**Correct Answer: A) CreateModelMixin**

\`\`\`python
from rest_framework import mixins, viewsets

class ArticleViewSet (mixins.CreateModelMixin,
                      mixins.ListModelMixin,
                      viewsets.GenericViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    # Provides list() and create()
\`\`\`

Available mixins: CreateModelMixin, ListModelMixin, RetrieveModelMixin, UpdateModelMixin, DestroyModelMixin.
      `,
  },
  {
    question: 'What is the difference between SimpleRouter and DefaultRouter?',
    options: [
      'A) SimpleRouter is faster',
      'B) DefaultRouter includes API root view and format suffixes',
      'C) SimpleRouter only works with simple ViewSets',
      'D) DefaultRouter requires authentication',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) DefaultRouter includes API root view and format suffixes**

\`\`\`python
# DefaultRouter adds:
# - / (API root showing all endpoints)
# - Format suffixes (.json, .api)

router = DefaultRouter()  # Includes extras
router = SimpleRouter()   # Minimal, just the endpoints
\`\`\`

Use DefaultRouter for browsable API, SimpleRouter for minimal overhead.
      `,
  },
  {
    question: 'How do you specify a custom URL path for an action?',
    options: [
      'A) @action (path="custom-path")',
      'B) @action (url_path="custom-path")',
      'C) @action (route="custom-path")',
      'D) @action (endpoint="custom-path")',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) @action (url_path="custom-path")**

\`\`\`python
@action (detail=True, methods=['post'], url_path='mark-as-read')
def mark_read (self, request, pk=None):
    # URL: /articles/{id}/mark-as-read/
    # Method name can be different from URL
    pass
\`\`\`

url_path customizes the URL segment, url_name customizes the URL name for reverse().
      `,
  },
  {
    question: 'What does basename do when registering a ViewSet with a router?',
    options: [
      'A) Sets the database table name',
      'B) Defines the URL name prefix for reverse()',
      'C) Names the serializer class',
      'D) Sets the model class name',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Defines the URL name prefix for reverse()**

\`\`\`python
router.register (r'articles', ArticleViewSet, basename='article')

# Creates URL names:
# article-list, article-detail, article-publish, etc.

# Use in code:
from django.urls import reverse
url = reverse('article-detail', args=[article.id])
\`\`\`

Required when ViewSet doesn't have a queryset attribute.
      `,
  },
].map(({ id, ...q }, idx) => ({ id: `django-mc-${idx + 1}`, ...q }));

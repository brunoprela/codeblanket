export const FilteringSearchingPaginationMultipleChoice = {
  title: 'Filtering, Searching & Pagination - Multiple Choice Questions',
  questions: [
    {
      question:
        'Which filter backend should you use for complex query filtering in DRF?',
      options: [
        'A) SearchFilter',
        'B) OrderingFilter',
        'C) DjangoFilterBackend',
        'D) GenericFilter',
      ],
      correctAnswer: 2,
      explanation: `
**Correct Answer: C) DjangoFilterBackend**

\`\`\`python
from django_filters.rest_framework import DjangoFilterBackend

class ArticleViewSet(viewsets.ModelViewSet):
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['status', 'category', 'author']
    # /api/articles/?status=published&category=tech
\`\`\`

DjangoFilterBackend enables field-based filtering with lookups.
      `,
    },
    {
      question: 'What does the "^" prefix mean in DRF search_fields?',
      options: [
        'A) Case-insensitive search',
        'B) Starts-with search',
        'C) Exact match',
        'D) Regular expression',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) Starts-with search**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    search_fields = ['^title', 'content']
    # ^title - matches titles starting with search term
    # content - matches anywhere in content
\`\`\`

Other prefixes: '=' exact, '@' full-text, '$' regex.
      `,
    },
    {
      question:
        'Which pagination style is best for infinite scroll implementations?',
      options: [
        'A) PageNumberPagination',
        'B) LimitOffsetPagination',
        'C) CursorPagination',
        'D) SimplePagination',
      ],
      correctAnswer: 2,
      explanation: `
**Correct Answer: C) CursorPagination**

\`\`\`python
class CursorPagination(pagination.CursorPagination):
    page_size = 25
    ordering = '-created_at'
\`\`\`

Cursor pagination has constant performance and handles real-time data well, making it ideal for infinite scroll.
      `,
    },
    {
      question:
        'How do you enable client-controlled page size in DRF pagination?',
      options: [
        'A) Set allow_page_size = True',
        'B) Set page_size_query_param in pagination class',
        'C) Add ?page_size to filterset_fields',
        'D) Override get_page_size() method',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) Set page_size_query_param in pagination class**

\`\`\`python
class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = 'page_size'
    max_page_size = 100

# Client: /api/articles/?page=1&page_size=50
\`\`\`

This allows clients to control page size up to max_page_size.
      `,
    },
    {
      question:
        'What is the purpose of ordering parameter in CursorPagination?',
      options: [
        'A) To set default sort order',
        'B) To define the field used for cursor positioning (required)',
        'C) To enable ordering filter',
        'D) To sort results alphabetically',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) To define the field used for cursor positioning (required)**

\`\`\`python
class MyCursorPagination(CursorPagination):
    ordering = '-created_at'  # Required!
    # Cursor uses this field to track position
\`\`\`

CursorPagination requires a consistent ordering field to work properly.
      `,
    },
  ],
};

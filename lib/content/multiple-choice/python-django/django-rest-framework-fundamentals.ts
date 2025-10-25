import { MultipleChoiceQuestion } from '@/lib/types';

export const DjangoRestFrameworkFundamentalsMultipleChoice = [
  {
    id: 1,
    question:
      'What is the main difference between APIView and ViewSets in DRF?',
    options: [
      'A) APIView is faster than ViewSets',
      'B) ViewSets combine multiple related views and enable automatic URL routing',
      'C) APIView supports authentication while ViewSets do not',
      'D) ViewSets can only handle GET requests',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) ViewSets combine multiple related views and enable automatic URL routing**

ViewSets group list/create/retrieve/update/destroy operations into a single class and work with routers for automatic URL generation.

\`\`\`python
# APIView - explicit for each operation
class ArticleList(APIView):
    def get(self, request): ...
    def post(self, request): ...

# ViewSet - combines all CRUD
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    # Automatically provides list, create, retrieve, update, destroy
\`\`\`
      `,
  },
  {
    question: 'When should you use SerializerMethodField in DRF?',
    options: [
      'A) For write-only fields',
      'B) For computed read-only fields',
      'C) For ForeignKey relationships',
      'D) For required validation',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) For computed read-only fields**

SerializerMethodField is for read-only computed values based on the object.

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    word_count = serializers.SerializerMethodField()
    
    def get_word_count(self, obj):
        return len(obj.content.split())
\`\`\`

It's always read-only and calculated on-the-fly during serialization.
      `,
  },
  {
    question:
      'What does rest_framework.permissions.IsAuthenticatedOrReadOnly do?',
    options: [
      'A) Allows all users to read, requires authentication for write operations',
      'B) Allows authenticated users to read only',
      'C) Makes the API read-only for everyone',
      'D) Requires authentication for all operations',
    ],
    correctAnswer: 0,
    explanation: `
**Correct Answer: A) Allows all users to read, requires authentication for write operations**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticatedOrReadOnly]
    # GET (list, retrieve) - anyone
    # POST, PUT, PATCH, DELETE - authenticated users only
\`\`\`

Perfect for public read, authenticated write APIs.
      `,
  },
  {
    question: 'How do you pass extra context to a serializer in a DRF view?',
    options: [
      'A) serializer = MySerializer(data, context={"key": "value"})',
      'B) Override get_serializer_context()',
      'C) Set serializer.context directly',
      'D) Pass it in serializer_class',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Override get_serializer_context()**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['user_id'] = self.request.user.id
        return context
\`\`\`

The serializer can then access it via \`self.context['user_id']\`.
      `,
  },
  {
    question: 'What is the purpose of @action decorator in DRF ViewSets?',
    options: [
      'A) To add custom endpoints beyond standard CRUD operations',
      'B) To override existing CRUD methods',
      'C) To add admin actions',
      'D) To define serializer actions',
    ],
    correctAnswer: 0,
    explanation: `
**Correct Answer: A) To add custom endpoints beyond standard CRUD operations**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    @action(detail=True, methods=['post'])
    def publish(self, request, pk=None):
        article = self.get_object()
        article.status = 'published'
        article.save()
        return Response({'status': 'published'})

# Creates: POST /api/articles/{id}/publish/
\`\`\`

Allows adding custom endpoints like publish, archive, etc.
      `,
  },
].map(({ id, ...q }, idx) => ({ id: `django-mc-${idx + 1}`, ...q }));

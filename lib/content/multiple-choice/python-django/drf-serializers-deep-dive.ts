import { MultipleChoiceQuestion } from '@/lib/types';

export const DrfSerializersDeepDiveMultipleChoice = [
  {
    id: 1,
    question: 'How do you handle writable nested serializers in DRF?',
    options: [
      'A) Nested serializers are automatically writable',
      'B) Override create() and update() methods',
      'C) Set read_only=False on nested field',
      'D) Use depth=2 in Meta',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Override create() and update() methods**

\`\`\`python
class ArticleSerializer (serializers.ModelSerializer):
    comments = CommentSerializer (many=True)
    
    def create (self, validated_data):
        comments_data = validated_data.pop('comments')
        article = Article.objects.create(**validated_data)
        for comment_data in comments_data:
            Comment.objects.create (article=article, **comment_data)
        return article
\`\`\`

You must explicitly handle nested creation/updates.
      `,
  },
  {
    question: 'What is the purpose of to_representation() in serializers?',
    options: [
      'A) To validate incoming data',
      'B) To customize how data is converted to JSON',
      'C) To handle database queries',
      'D) To define required fields',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) To customize how data is converted to JSON**

\`\`\`python
class ArticleSerializer (serializers.ModelSerializer):
    def to_representation (self, instance):
        data = super().to_representation (instance)
        # Customize output
        data['url'] = f'/articles/{instance.slug}/'
        return data
\`\`\`

Controls the final JSON structure sent to clients.
      `,
  },
  {
    question: 'How do you implement field-level validation in DRF serializers?',
    options: [
      'A) Override validate() method',
      'B) Add validate_<field_name>() method',
      'C) Use validators parameter',
      'D) Set required=True',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Add validate_<field_name>() method**

\`\`\`python
class ArticleSerializer (serializers.ModelSerializer):
    def validate_title (self, value):
        if len (value) < 10:
            raise serializers.ValidationError(
                "Title must be at least 10 characters"
            )
        return value
\`\`\`

This validates a specific field independently.
      `,
  },
  {
    question: 'What does source parameter do in serializer fields?',
    options: [
      'A) Specifies the data source/model field to use',
      'B) Defines the database table',
      'C) Sets the API endpoint source',
      'D) Configures the serializer class',
    ],
    correctAnswer: 0,
    explanation: `
**Correct Answer: A) Specifies the data source/model field to use**

\`\`\`python
class ArticleSerializer (serializers.ModelSerializer):
    author_name = serializers.CharField (source='author.username')
    # Maps author_name in JSON to author.username on model
\`\`\`

Allows different names in API vs model.
      `,
  },
  {
    question: 'How do you make a serializer field write-only?',
    options: [
      'A) Set write_only=True',
      'B) Override to_representation()',
      'C) Use write=True parameter',
      'D) Add to write_only_fields in Meta',
    ],
    correctAnswer: 0,
    explanation: `
**Correct Answer: A) Set write_only=True**

\`\`\`python
class UserSerializer (serializers.ModelSerializer):
    password = serializers.CharField (write_only=True)
    
    class Meta:
        model = User
        fields = ['username', 'password', 'email']
\`\`\`

Field accepted in POST/PUT but not included in responses.
      `,
  },
].map(({ id, ...q }, idx) => ({ id: `django-mc-${idx + 1}`, ...q }));

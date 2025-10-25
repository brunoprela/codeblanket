export const AuthenticationPermissionsDrfMultipleChoice = {
  title: 'Authentication & Permissions (DRF) - Multiple Choice Questions',
  questions: [
    {
      question:
        'What is the main advantage of JWT over Token authentication in DRF?',
      options: [
        'A) JWT is faster to validate',
        'B) JWT tokens are self-contained and can expire without database lookups',
        'C) JWT works better with CSRF protection',
        'D) JWT is more secure than Token auth',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) JWT tokens are self-contained and can expire without database lookups**

JWT contains all user info and expiration in the token itself, eliminating database queries for validation.

\`\`\`python
# Token Auth - needs database lookup every request
token = Token.objects.get(key=request_token)

# JWT - validates signature, no database needed
decoded = jwt.decode(token, SECRET_KEY)
\`\`\`

JWTs are stateless and scale better for high-traffic APIs.
      `,
    },
    {
      question:
        'How do you apply different permissions to different actions in a ViewSet?',
      options: [
        'A) Set permissions in @action decorator',
        'B) Override get_permissions() method',
        'C) Use permission_classes_by_action dict',
        'D) Set different permission_classes for each method',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) Override get_permissions() method**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    def get_permissions(self):
        if self.action in ['list', 'retrieve']:
            return [AllowAny()]
        elif self.action == 'destroy':
            return [IsAdminUser()]
        return [IsAuthenticated()]
\`\`\`

This gives you full control over permissions per action.
      `,
    },
    {
      question:
        'What does has_object_permission() check that has_permission() does not?',
      options: [
        'A) User authentication status',
        'B) Specific object instance access rights',
        'C) View-level permissions',
        'D) HTTP method type',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) Specific object instance access rights**

\`\`\`python
class IsOwner(permissions.BasePermission):
    def has_permission(self, request, view):
        # View-level: Can user access this endpoint at all?
        return request.user.is_authenticated
    
    def has_object_permission(self, request, view, obj):
        # Object-level: Can user access THIS specific object?
        return obj.owner == request.user
\`\`\`

has_object_permission() checks access to specific instances.
      `,
    },
    {
      question: 'Which HTTP header should clients use to send JWT tokens?',
      options: [
        'A) X-Auth-Token: <token>',
        'B) Authorization: Bearer <token>',
        'C) JWT-Token: <token>',
        'D) X-JWT: <token>',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) Authorization: Bearer <token>**

\`\`\`python
# Client request
GET /api/articles/
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...

# DRF extracts and validates the token
\`\`\`

"Bearer" is the standard token type for JWT per RFC 6750.
      `,
    },
    {
      question:
        'How do you make an API endpoint accessible without authentication?',
      options: [
        'A) Remove authentication_classes',
        'B) Set permission_classes = [AllowAny]',
        'C) Add @public decorator',
        'D) Set authenticated = False',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) Set permission_classes = [AllowAny]**

\`\`\`python
from rest_framework.permissions import AllowAny

class PublicArticleView(APIView):
    permission_classes = [AllowAny]
    
    def get(self, request):
        # Anyone can access
        return Response({...})
\`\`\`

AllowAny explicitly allows unauthenticated access.
      `,
    },
  ],
};

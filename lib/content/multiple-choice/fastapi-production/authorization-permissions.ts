import { MultipleChoiceQuestion } from '@/lib/types';

export const authorizationPermissionsMultipleChoice = [
  {
      id: 1,
      question:
        'What is the primary difference between authentication and authorization in FastAPI applications?',
      options: [
        'Authentication verifies who the user is, while authorization determines what they can do',
        'Authentication is for external APIs, while authorization is for internal endpoints',
        'Authentication uses JWT tokens, while authorization uses OAuth2',
        'Authentication checks passwords, while authorization checks API keys',
      ],
      correctAnswer: 0,
      explanation:
        'Authentication answers "who are you?" by verifying identity (username/password, JWT token), while authorization answers "what can you do?" by checking permissions, roles, and access rights. They are complementary but distinct security layers: authentication happens first to identify the user, then authorization determines what that authenticated user can access.',
    },
    {
      id: 2,
      question:
        'In Role-Based Access Control (RBAC), you have an endpoint that should be accessible to both moderators and admins. Which implementation is most maintainable?',
      options: [
        'Use class RoleChecker([Role.MODERATOR, Role.ADMIN]) with OR logic to check if user has any of the allowed roles',
        'Create separate dependencies for moderator and admin, then use both in the endpoint',
        'Check if user.role == "moderator" or user.role == "admin" inside the endpoint function',
        'Store moderator permissions in the database and check at runtime for each request',
      ],
      correctAnswer: 0,
      explanation:
        "A RoleChecker class that accepts a list of allowed roles and uses OR logic (any() function) is the most maintainable approach. It's reusable across endpoints, declarative (clear from the dependency what roles are allowed), and centralized (role checking logic in one place). Checking inside endpoint functions spreads authorization logic throughout your codebase. The RoleChecker pattern: class RoleChecker: def __init__(self, allowed_roles): self.allowed_roles = allowed_roles; def __call__(self, user): if not any(user.has_role(r) for r in self.allowed_roles): raise HTTPException(403)",
    },
    {
      id: 3,
      question:
        'For a multi-tenant SaaS application, what is the most critical security measure to prevent data leakage between tenants?',
      options: [
        'Always include organization_id in WHERE clauses and validate the user belongs to that organization before any database operation',
        'Use different database connections for each tenant',
        'Encrypt all data with tenant-specific encryption keys',
        'Require authentication for all endpoints',
      ],
      correctAnswer: 0,
      explanation:
        "The most critical security measure for multi-tenancy is ensuring every database query filters by organization_id and validating user-organization membership. This prevents the most common vulnerability: accidentally returning data from the wrong tenant. While separate databases (option 2) provide strong isolation, they're expensive and complex to manage. Encryption (option 3) protects data at rest but doesn't prevent cross-tenant access if queries aren't properly scoped. Authentication (option 4) is necessary but insufficient—an authenticated user could still access another tenant's data without authorization checks. The defense-in-depth approach: middleware sets current org, dependencies validate membership, queries automatically filter by organization_id, and final double-checks before mutations.",
    },
    {
      id: 4,
      question:
        'You need to implement authorization where users can only update their own posts, but admins can update any post. What is the best dependency pattern?',
      options: [
        'Create a dependency require_post_owner_or_admin that checks if user is the post owner OR has admin role',
        'Create two separate endpoints: one for users (owner check) and one for admins (no check)',
        'Check user.id == post.author_id inside the endpoint function after fetching the post',
        'Store the post owner in the JWT token to avoid database lookups',
      ],
      correctAnswer: 0,
      explanation:
        "A dependency that implements OR logic (owner OR admin) keeps authorization declarative and reusable. The pattern: async def require_post_owner_or_admin(post: Post = Depends(get_post), user: User = Depends(get_current_user)): if post.author_id != user.id and not user.is_admin: raise HTTPException(403); return post. This dependency chain ensures the post exists (404 if not), validates ownership or admin privilege (403 if neither), and passes the validated post to the endpoint. Option 2 (separate endpoints) duplicates code. Option 3 (inline check) spreads authorization logic. Option 4 (JWT storage) is problematic: JWTs should be stateless and shouldn't contain mutable data like post ownership.",
    },
    {
      id: 5,
      question:
        'For HIPAA compliance, when should authorization audit logs be created?',
      options: [
        'For every access attempt, including both successful and denied attempts, with timestamp, user, resource, action, and result',
        'Only when access is denied, to track security incidents',
        'Only when sensitive data like SSN or diagnosis is accessed',
        'Only for admin users, since regular users are trusted',
      ],
      correctAnswer: 0,
      explanation:
        'HIPAA requires comprehensive audit trails that log ALL access attempts (success and failure) with detailed context. This provides: 1) Forensic investigation capability (who accessed what when), 2) Breach detection (unusual access patterns), 3) Compliance evidence (demonstrate due diligence), 4) Anomaly detection (multiple failed attempts may indicate attack). The audit log should include: timestamp, user_id, action (read_patient, update_diagnosis), resource (patient:123), result (success/denied), reason (why denied), and metadata (IP address, user agent). Logging only denials (option 2) misses the critical "who accessed what" data. Logging only sensitive fields (option 3) misses context. Logging only admins (option 4) violates least privilege—all users should be audited. The pattern: use a decorator or dependency that wraps every protected endpoint and logs before/after execution.',
    },
].map(({ id, ...q }, idx) => ({ id: `fastapi-mc-${idx + 1}`, ...q }));

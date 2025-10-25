export const authorizationPermissions = {
  title: 'Authorization & Permissions',
  id: 'authorization-permissions',
  content: `
# Authorization & Permissions

## Introduction

Authentication tells you **who** the user is. Authorization tells you **what** they can do. While authentication is often straightforward, authorization is where complexity emerges—different users have different permissions, roles change over time, and business rules govern access control.

**Why authorization matters:**
- **Security**: Prevent unauthorized access to resources
- **Compliance**: Meet regulatory requirements (GDPR, HIPAA, SOC 2)
- **Business logic**: Enforce organizational hierarchies and rules
- **Multi-tenancy**: Isolate data between customers
- **Audit trails**: Track who accessed what

In production, authorization solves:
- Role-Based Access Control (RBAC)
- Permission-Based Access Control (PBAC)
- Resource ownership validation
- Multi-tenant data isolation
- Fine-grained permissions
- Admin vs user vs moderator access

In this section, you'll master:
- Role-based authorization patterns
- Permission checking with dependencies
- Resource ownership validation
- Multi-tenant authorization
- Policy-based authorization
- Testing authorization logic
- Production patterns

### Authorization Models

\`\`\`
1. Role-Based (RBAC): User has roles (admin, moderator, user)
2. Permission-Based (PBAC): User has permissions (read:posts, write:posts)
3. Attribute-Based (ABAC): Rules based on attributes (department, time, location)
4. Resource-Based: User owns resource (can edit own posts only)
\`\`\`

---

## Role-Based Access Control (RBAC)

### Simple Role Checking

\`\`\`python
"""
Basic RBAC Implementation
"""

from fastapi import FastAPI, Depends, HTTPException, status
from enum import Enum

app = FastAPI()

class Role(str, Enum):
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"

# User model with roles
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    roles = Column(ARRAY(String))  # PostgreSQL array: ['user', 'admin']
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role"""
        return role in self.roles
    
    @property
    def is_admin(self) -> bool:
        return self.has_role(Role.ADMIN)
    
    @property
    def is_moderator(self) -> bool:
        return self.has_role(Role.MODERATOR)

# Role requirement dependency
def require_role(required_role: Role):
    """
    Dependency factory for role checking
    
    Usage: Depends(require_role(Role.ADMIN))
    """
    async def role_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        if not current_user.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        return current_user
    
    return role_checker

# Admin-only endpoint
@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: User = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db)
):
    """
    Only admins can delete users
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(user)
    db.commit()
    
    return {"detail": f"User {user_id} deleted"}

# Moderator or admin endpoint
def require_moderator_or_admin(current_user: User = Depends(get_current_user)):
    """Check if user is moderator OR admin"""
    if not (current_user.is_moderator or current_user.is_admin):
        raise HTTPException(
            status_code=403,
            detail="Moderator or admin role required"
        )
    return current_user

@app.post("/posts/{post_id}/moderate")
async def moderate_post(
    post_id: int,
    moderator: User = Depends(require_moderator_or_admin),
    db: Session = Depends(get_db)
):
    """
    Moderators and admins can moderate posts
    """
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404)
    
    post.status = "moderated"
    db.commit()
    
    return post
\`\`\`

### Flexible Role Checker

\`\`\`python
"""
Reusable Role Checker Class
"""

from typing import List

class RoleChecker:
    """
    Callable class for checking user roles
    
    Supports multiple roles (OR logic)
    """
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        """
        Check if user has any of the allowed roles
        """
        if not any(current_user.has_role(role) for role in self.allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of these roles required: {', '.join(self.allowed_roles)}"
            )
        return current_user

# Create role checkers
require_admin = RoleChecker([Role.ADMIN])
require_moderator = RoleChecker([Role.MODERATOR, Role.ADMIN])
require_authenticated = RoleChecker([Role.USER, Role.MODERATOR, Role.ADMIN])

# Use in endpoints
@app.get("/admin/dashboard")
async def admin_dashboard(admin: User = Depends(require_admin)):
    """Admin-only dashboard"""
    return {"message": "Admin dashboard"}

@app.post("/posts/{post_id}/approve")
async def approve_post(
    post_id: int,
    moderator: User = Depends(require_moderator)  # Moderator OR admin
):
    """Moderators and admins can approve posts"""
    return {"approved": post_id}
\`\`\`

---

## Permission-Based Access Control (PBAC)

### Permission System

\`\`\`python
"""
Permission-Based Authorization
"""

from typing import List, Set

# Permission model
class Permission(Base):
    __tablename__ = "permissions"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)  # e.g., "read:posts", "write:posts"
    description = Column(String)

# Role-Permission association
class RolePermission(Base):
    __tablename__ = "role_permissions"
    
    role_id = Column(Integer, ForeignKey("roles.id"), primary_key=True)
    permission_id = Column(Integer, ForeignKey("permissions.id"), primary_key=True)

# Enhanced User model
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String)
    roles = relationship("Role", secondary="user_roles", back_populates="users")
    
    def get_permissions(self) -> Set[str]:
        """Get all permissions from all roles"""
        permissions = set()
        for role in self.roles:
            permissions.update(perm.name for perm in role.permissions)
        return permissions
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in self.get_permissions()

# Permission checker
class PermissionChecker:
    """
    Check if user has required permissions
    """
    def __init__(self, required_permissions: List[str]):
        self.required_permissions = required_permissions
    
    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        """
        Check if user has ALL required permissions
        """
        user_permissions = current_user.get_permissions()
        
        for permission in self.required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
        
        return current_user

# Create permission checkers
require_read_posts = PermissionChecker(["read:posts"])
require_write_posts = PermissionChecker(["write:posts"])
require_delete_posts = PermissionChecker(["delete:posts"])
require_manage_users = PermissionChecker(["read:users", "write:users", "delete:users"])

# Use in endpoints
@app.get("/posts")
async def list_posts(user: User = Depends(require_read_posts)):
    """Requires 'read:posts' permission"""
    return []

@app.post("/posts")
async def create_post(
    post: PostCreate,
    user: User = Depends(require_write_posts)
):
    """Requires 'write:posts' permission"""
    return post

@app.delete("/posts/{post_id}")
async def delete_post(
    post_id: int,
    user: User = Depends(require_delete_posts)
):
    """Requires 'delete:posts' permission"""
    return {"deleted": post_id}
\`\`\`

---

## Resource Ownership

### Owner-Based Authorization

\`\`\`python
"""
Resource Ownership Validation
"""

async def get_post_or_404(
    post_id: int,
    db: Session = Depends(get_db)
) -> Post:
    """Get post or raise 404"""
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post

async def require_post_owner(
    post: Post = Depends(get_post_or_404),
    current_user: User = Depends(get_current_user)
) -> Post:
    """
    Ensure current user owns the post
    """
    if post.author_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to modify this post"
        )
    return post

@app.put("/posts/{post_id}")
async def update_post(
    post_id: int,
    post_update: PostUpdate,
    post: Post = Depends(require_post_owner),  # Must be owner
    db: Session = Depends(get_db)
):
    """
    Only post owner can update
    """
    for key, value in post_update.dict(exclude_unset=True).items():
        setattr(post, key, value)
    
    db.commit()
    db.refresh(post)
    
    return post

# Owner OR admin can delete
async def require_post_owner_or_admin(
    post: Post = Depends(get_post_or_404),
    current_user: User = Depends(get_current_user)
) -> Post:
    """
    Owner OR admin can perform action
    """
    is_owner = post.author_id == current_user.id
    is_admin = current_user.is_admin
    
    if not (is_owner or is_admin):
        raise HTTPException(
            status_code=403,
            detail="Must be post owner or admin"
        )
    
    return post

@app.delete("/posts/{post_id}")
async def delete_post(
    post_id: int,
    post: Post = Depends(require_post_owner_or_admin),
    db: Session = Depends(get_db)
):
    """
    Owner OR admin can delete
    """
    db.delete(post)
    db.commit()
    
    return {"deleted": post_id}
\`\`\`

---

## Multi-Tenant Authorization

### Tenant Isolation

\`\`\`python
"""
Multi-Tenant Authorization
"""

from fastapi import Request

# Tenant model
class Tenant(Base):
    __tablename__ = "tenants"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    subdomain = Column(String, unique=True)
    is_active = Column(Boolean, default=True)

# User-Tenant association
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"))
    tenant = relationship("Tenant")

# Get tenant from subdomain
async def get_tenant(request: Request, db: Session = Depends(get_db)) -> Tenant:
    """
    Extract tenant from subdomain
    
    tenant1.example.com → tenant1
    """
    hostname = request.url.hostname
    subdomain = hostname.split('.')[0]
    
    if subdomain in ['www', 'api']:
        raise HTTPException(status_code=400, detail="Invalid subdomain")
    
    tenant = db.query(Tenant).filter(
        Tenant.subdomain == subdomain,
        Tenant.is_active == True
    ).first()
    
    if not tenant:
        raise HTTPException(status_code=403, detail="Invalid tenant")
    
    return tenant

# Validate user belongs to tenant
async def validate_tenant_access(
    current_user: User = Depends(get_current_user),
    tenant: Tenant = Depends(get_tenant)
) -> User:
    """
    Ensure user belongs to current tenant
    """
    if current_user.tenant_id != tenant.id:
        raise HTTPException(
            status_code=403,
            detail="Access denied: wrong tenant"
        )
    
    return current_user

# Apply to all tenant routes
tenant_router = APIRouter(
    dependencies=[Depends(validate_tenant_access)]
)

@tenant_router.get("/users")
async def list_tenant_users(
    tenant: Tenant = Depends(get_tenant),
    db: Session = Depends(get_db)
):
    """
    List users in current tenant only
    Automatic tenant filtering
    """
    users = db.query(User).filter(User.tenant_id == tenant.id).all()
    return users

@tenant_router.get("/posts")
async def list_tenant_posts(
    tenant: Tenant = Depends(get_tenant),
    db: Session = Depends(get_db)
):
    """
    List posts in current tenant only
    """
    posts = db.query(Post).filter(Post.tenant_id == tenant.id).all()
    return posts
\`\`\`

---

## Policy-Based Authorization

### Complex Authorization Rules

\`\`\`python
"""
Policy-Based Authorization
"""

from typing import Callable
from datetime import datetime, time

class AuthorizationPolicy:
    """
    Policy-based authorization
    
    Allows complex rules beyond simple role checks
    """
    def __init__(
        self,
        policy_name: str,
        policy_func: Callable[[User, dict], bool]
    ):
        self.policy_name = policy_name
        self.policy_func = policy_func
    
    async def __call__(
        self,
        current_user: User = Depends(get_current_user),
        **kwargs
    ) -> User:
        """
        Evaluate policy
        """
        if not self.policy_func(current_user, kwargs):
            raise HTTPException(
                status_code=403,
                detail=f"Policy '{self.policy_name}' failed"
            )
        
        return current_user

# Define policies
def business_hours_policy(user: User, context: dict) -> bool:
    """
    User can only access during business hours (9 AM - 5 PM)
    Unless they're admin
    """
    if user.is_admin:
        return True
    
    now = datetime.now().time()
    return time(9, 0) <= now <= time(17, 0)

def same_department_policy(user: User, context: dict) -> bool:
    """
    User can only access resources in their department
    """
    resource = context.get('resource')
    if not resource:
        return False
    
    return user.department == resource.department

def subscription_active_policy(user: User, context: dict) -> bool:
    """
    User must have active subscription
    """
    return user.subscription_status == "active"

# Create policy checkers
require_business_hours = AuthorizationPolicy(
    "business_hours",
    business_hours_policy
)

require_same_department = AuthorizationPolicy(
    "same_department",
    same_department_policy
)

require_active_subscription = AuthorizationPolicy(
    "active_subscription",
    subscription_active_policy
)

# Use in endpoints
@app.get("/reports")
async def get_reports(user: User = Depends(require_business_hours)):
    """
    Only accessible during business hours (9 AM - 5 PM)
    Unless user is admin
    """
    return []

@app.get("/premium-features")
async def premium_features(user: User = Depends(require_active_subscription)):
    """
    Requires active subscription
    """
    return {"features": ["feature1", "feature2"]}
\`\`\`

---

## Testing Authorization

### Authorization Testing

\`\`\`python
"""
Testing Authorization Logic
"""

import pytest
from fastapi.testclient import TestClient

# Mock users with different roles
def mock_regular_user():
    return User(id=1, username="user", roles=["user"], is_admin=False)

def mock_admin_user():
    return User(id=2, username="admin", roles=["admin"], is_admin=True)

def mock_moderator_user():
    return User(id=3, username="mod", roles=["moderator"], is_admin=False)

@pytest.fixture
def client_regular_user():
    """Test client as regular user"""
    app.dependency_overrides[get_current_user] = mock_regular_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()

@pytest.fixture
def client_admin():
    """Test client as admin"""
    app.dependency_overrides[get_current_user] = mock_admin_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()

# Test admin-only endpoint
def test_admin_endpoint_as_admin(client_admin):
    """Admin can access admin endpoint"""
    response = client_admin.delete("/users/999")
    assert response.status_code in [200, 404]  # Access granted

def test_admin_endpoint_as_user(client_regular_user):
    """Regular user cannot access admin endpoint"""
    response = client_regular_user.delete("/users/999")
    assert response.status_code == 403
    assert "admin" in response.json()["detail"].lower()

# Test resource ownership
def test_update_own_post(client_regular_user, db):
    """User can update their own post"""
    # Create post owned by user
    post = Post(id=1, title="Test", author_id=1)
    db.add(post)
    db.commit()
    
    response = client_regular_user.put(
        "/posts/1",
        json={"title": "Updated"}
    )
    
    assert response.status_code == 200

def test_update_others_post(client_regular_user, db):
    """User cannot update someone else's post"""
    # Create post owned by different user
    post = Post(id=2, title="Test", author_id=999)
    db.add(post)
    db.commit()
    
    response = client_regular_user.put(
        "/posts/2",
        json={"title": "Updated"}
    )
    
    assert response.status_code == 403
\`\`\`

---

## Production Patterns

### Authorization Audit Logging

\`\`\`python
"""
Audit Logging for Authorization
"""

import logging
from functools import wraps

logger = logging.getLogger(__name__)

def audit_authorization(action: str):
    """
    Decorator for audit logging
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs
            current_user = kwargs.get('current_user')
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful authorization
                logger.info(
                    f"Authorization success: {action}",
                    extra={
                        "user_id": current_user.id if current_user else None,
                        "action": action,
                        "result": "success"
                    }
                )
                
                return result
                
            except HTTPException as e:
                # Log failed authorization
                logger.warning(
                    f"Authorization failed: {action}",
                    extra={
                        "user_id": current_user.id if current_user else None,
                        "action": action,
                        "result": "denied",
                        "reason": e.detail
                    }
                )
                
                raise
        
        return wrapper
    return decorator

@app.delete("/users/{user_id}")
@audit_authorization("delete_user")
async def delete_user(
    user_id: int,
    current_user: User = Depends(require_admin)
):
    """
    All authorization attempts logged
    """
    pass
\`\`\`

---

## Summary

### Key Takeaways

✅ **RBAC**: Role-based (admin, moderator, user) with RoleChecker  
✅ **PBAC**: Permission-based (read:posts, write:posts) with PermissionChecker  
✅ **Resource ownership**: Users can only modify their own resources  
✅ **Multi-tenant**: Automatic tenant isolation with get_tenant dependency  
✅ **Policy-based**: Complex rules (business hours, department, subscription)  
✅ **Testing**: Mock users with different roles, test all permission scenarios  
✅ **Audit logging**: Track all authorization decisions

### Best Practices

**1. Fail securely**:
- Default to deny access
- Explicit permission checks
- Return 403 Forbidden (not 404)

**2. Separation of concerns**:
- Authentication (who you are) separate from authorization (what you can do)
- Use dependency injection for clean code
- Reusable permission checkers

**3. Multi-layer authorization**:
- Role check (is_admin)
- Permission check (has permission "delete:posts")
- Resource ownership (owns the post)
- Policy check (business hours, subscription)

**4. Testing**:
- Test all role combinations
- Test resource ownership
- Test policy edge cases
- Test authorization failures

### Next Steps

In the next section, we'll explore **Background Tasks**: executing long-running operations asynchronously without blocking API responses, using FastAPI's built-in background tasks and Celery integration.

**Production mindset**: Authorization is where security meets business logic. Get it wrong, and users can access data they shouldn't. Test thoroughly!
`,
};

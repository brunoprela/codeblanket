export const authorizationPermissionsQuiz = [
        {
            id: 1,
            question:
                'Compare and contrast Role-Based Access Control (RBAC) and Permission-Based Access Control (PBAC). In what scenarios would you choose RBAC over PBAC, and vice versa? Design an authorization system for a content management platform where authors can create posts, editors can approve posts, and administrators can delete any post. Would you use RBAC, PBAC, or a hybrid approach? Justify your choice with code examples showing how you would implement the key authorization checks.',
            answer: `**RBAC vs PBAC Comparison**:

**RBAC (Role-Based Access Control)**:
- Users assigned to roles (admin, editor, author)
- Roles grant broad permissions
- Simpler to implement and understand
- Better for hierarchical organizations
- Role checks: \`if user.has_role("admin")\`

**PBAC (Permission-Based Access Control)**:
- Users granted specific permissions (read:posts, write:posts, delete:posts)
- Fine-grained control
- More flexible, easier to audit
- Better for complex permission matrices
- Permission checks: \`if user.has_permission("delete:posts")\`

**When to choose each**:

RBAC:
- Small number of roles (< 10)
- Clear hierarchies (admin > moderator > user)
- Permissions rarely change
- Example: Internal tools, admin panels

PBAC:
- Many permission types (> 20)
- Complex permission requirements
- Permissions frequently added/changed
- Example: Enterprise applications, multi-tenant SaaS

**Hybrid approach for CMS** (recommended):

\`\`\`python
# Roles grant default permissions
roles = {
    "author": ["read:posts", "write:own_posts"],
    "editor": ["read:posts", "write:own_posts", "approve:posts"],
    "admin": ["read:posts", "write:all_posts", "approve:posts", "delete:posts"]
}

# Plus individual permission overrides
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    roles = Column(ARRAY(String))  # ["author", "editor"]
    extra_permissions = Column(ARRAY(String))  # ["delete:featured_posts"]
    revoked_permissions = Column(ARRAY(String))  # ["approve:posts"]
    
    def get_permissions(self) -> Set[str]:
        """Get all effective permissions"""
        permissions = set()
        
        # Add permissions from roles
        for role in self.roles:
            permissions.update(roles.get(role, []))
        
        # Add extra permissions
        permissions.update(self.extra_permissions or [])
        
        # Remove revoked permissions
        permissions -= set(self.revoked_permissions or [])
        
        return permissions
    
    def has_permission(self, permission: str) -> bool:
        return permission in self.get_permissions()

# Authorization for creating posts
@app.post("/posts")
async def create_post(
    post: PostCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Authors, editors, and admins can create posts"""
    if not user.has_permission("write:own_posts"):
        raise HTTPException(status_code=403, detail="Cannot create posts")
    
    new_post = Post(**post.dict(), author_id=user.id, status="draft")
    db.add(new_post)
    db.commit()
    
    return new_post

# Authorization for approving posts
@app.post("/posts/{post_id}/approve")
async def approve_post(
    post_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Only editors and admins can approve"""
    if not user.has_permission("approve:posts"):
        raise HTTPException(status_code=403, detail="Cannot approve posts")
    
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404)
    
    post.status = "approved"
    db.commit()
    
    return post

# Authorization for deleting posts
async def require_delete_permission(
    post: Post = Depends(get_post_or_404),
    user: User = Depends(get_current_user)
) -> Post:
    """
    Can delete if:
    1. Admin with delete:posts permission, OR
    2. Author deleting own post
    """
    is_admin_delete = user.has_permission("delete:posts")
    is_own_post = post.author_id == user.id
    
    if not (is_admin_delete or is_own_post):
        raise HTTPException(
            status_code=403,
            detail="Cannot delete this post"
        )
    
    return post

@app.delete("/posts/{post_id}")
async def delete_post(
    post_id: int,
    post: Post = Depends(require_delete_permission),
    db: Session = Depends(get_db)
):
    """Delete post with permission check"""
    db.delete(post)
    db.commit()
    
    return {"deleted": post_id}
\`\`\`

**Why hybrid approach**:
- Roles provide default permissions (easy to assign)
- Extra permissions for special cases (guest editor can approve specific category)
- Revoked permissions for exceptions (editor temporarily suspended from approving)
- Combines simplicity of RBAC with flexibility of PBAC`,
        },
        {
            id: 2,
            question:
                'Design a comprehensive multi-tenant authorization system where each organization (tenant) has its own users, and users can belong to multiple organizations with different roles in each. For example, Alice is an admin in Org A but just a regular user in Org B. How would you model this in the database? How would you ensure complete data isolation between tenants? Write the FastAPI dependencies and middleware needed to automatically filter all database queries by the current tenant. What security considerations must you address to prevent tenant data leakage?',
            answer: `**Multi-Tenant Authorization Design**:

**Database Schema**:

\`\`\`python
class Organization(Base):
    """Tenant/Organization"""
    __tablename__ = "organizations"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    subdomain = Column(String, unique=True)  # tenant1.example.com
    is_active = Column(Boolean, default=True)
    
    # Relationships
    memberships = relationship("OrganizationMember", back_populates="organization")
    posts = relationship("Post", back_populates="organization")

class User(Base):
    """User can belong to multiple orgs"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    password_hash = Column(String)
    
    # Relationships
    memberships = relationship("OrganizationMember", back_populates="user")

class OrganizationMember(Base):
    """User's membership in an organization with role"""
    __tablename__ = "organization_members"
    
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), primary_key=True)
    role = Column(String)  # "admin", "member", "viewer"
    joined_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="memberships")
    organization = relationship("Organization", back_populates="memberships")
    
    # Composite unique constraint
    __table_args__ = (
        UniqueConstraint('user_id', 'organization_id'),
    )

class Post(Base):
    """Resource that belongs to an organization"""
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(Text)
    author_id = Column(Integer, ForeignKey("users.id"))
    organization_id = Column(Integer, ForeignKey("organizations.id"))  # CRITICAL
    
    # Relationships
    author = relationship("User")
    organization = relationship("Organization", back_populates="posts")
    
    # Index for fast tenant filtering
    __table_args__ = (
        Index('idx_organization_id', 'organization_id'),
    )
\`\`\`

**Tenant Extraction & Validation**:

\`\`\`python
from fastapi import Request, Header
from typing import Optional

async def get_current_organization(
    request: Request,
    x_organization_id: Optional[int] = Header(None),
    db: Session = Depends(get_db)
) -> Organization:
    """
    Extract organization from:
    1. Subdomain (tenant1.example.com)
    2. Header (X-Organization-ID)
    3. URL path parameter
    """
    org_id = None
    
    # Try subdomain first
    hostname = request.url.hostname
    if '.' in hostname:
        subdomain = hostname.split('.')[0]
        org = db.query(Organization).filter(
            Organization.subdomain == subdomain,
            Organization.is_active == True
        ).first()
        
        if org:
            org_id = org.id
    
    # Try header
    if not org_id and x_organization_id:
        org_id = x_organization_id
    
    # Validate organization exists
    if not org_id:
        raise HTTPException(
            status_code=400,
            detail="Organization not specified"
        )
    
    organization = db.query(Organization).filter(
        Organization.id == org_id,
        Organization.is_active == True
    ).first()
    
    if not organization:
        raise HTTPException(
            status_code=403,
            detail="Invalid organization"
        )
    
    return organization

async def get_current_user_with_org(
    current_user: User = Depends(get_current_user),
    organization: Organization = Depends(get_current_organization),
    db: Session = Depends(get_db)
) -> tuple[User, Organization, str]:
    """
    Get user, organization, and user's role in that organization
    
    Returns: (user, organization, role)
    """
    membership = db.query(OrganizationMember).filter(
        OrganizationMember.user_id == current_user.id,
        OrganizationMember.organization_id == organization.id
    ).first()
    
    if not membership:
        raise HTTPException(
            status_code=403,
            detail=f"User not member of organization {organization.name}"
        )
    
    return current_user, organization, membership.role

# Role checker for multi-tenant
def require_org_role(required_role: str):
    """
    Require specific role in current organization
    """
    async def checker(
        user_org_role: tuple = Depends(get_current_user_with_org)
    ):
        user, org, role = user_org_role
        
        role_hierarchy = {"admin": 3, "member": 2, "viewer": 1}
        
        if role_hierarchy.get(role, 0) < role_hierarchy.get(required_role, 999):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{required_role}' required in organization {org.name}"
            )
        
        return user, org, role
    
    return checker
\`\`\`

**Automatic Tenant Filtering Middleware**:

\`\`\`python
from starlette.middleware.base import BaseHTTPMiddleware
from contextvars import ContextVar

# Context variable for current organization
current_org_context: ContextVar[Optional[int]] = ContextVar('current_org', default=None)

class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware to set current organization in context
    """
    async def dispatch(self, request: Request, call_next):
        # Extract org from request
        org_id = None
        
        # From subdomain
        hostname = request.url.hostname
        if '.' in hostname:
            subdomain = hostname.split('.')[0]
            # Look up org_id from subdomain (cached)
            org_id = get_org_id_from_subdomain(subdomain)
        
        # From header
        if not org_id:
            org_id = request.headers.get('X-Organization-ID')
        
        # Set in context
        token = current_org_context.set(org_id)
        
        try:
            response = await call_next(request)
            return response
        finally:
            current_org_context.reset(token)

# Add middleware
app.add_middleware(TenantMiddleware)

# Database session with automatic filtering
class TenantSession(Session):
    """
    Session that automatically filters by current organization
    """
    def query(self, *entities, **kwargs):
        query = super().query(*entities, **kwargs)
        
        # Get current org from context
        org_id = current_org_context.get()
        
        if org_id:
            # Automatically filter by organization_id
            for entity in entities:
                if hasattr(entity, 'organization_id'):
                    query = query.filter(entity.organization_id == org_id)
        
        return query

def get_tenant_db():
    """Get database session with automatic tenant filtering"""
    db = TenantSession(bind=engine)
    try:
        yield db
    finally:
        db.close()
\`\`\`

**Endpoints with Tenant Isolation**:

\`\`\`python
@app.get("/posts")
async def list_posts(
    user_org_role: tuple = Depends(get_current_user_with_org),
    db: Session = Depends(get_tenant_db)
):
    """
    List posts in current organization only
    Automatic filtering by tenant
    """
    user, org, role = user_org_role
    
    # Query automatically filtered by org
    posts = db.query(Post).all()
    
    return posts

@app.post("/posts")
async def create_post(
    post: PostCreate,
    user_org_role: tuple = Depends(get_current_user_with_org),
    db: Session = Depends(get_tenant_db)
):
    """
    Create post in current organization
    """
    user, org, role = user_org_role
    
    # Automatically set organization_id
    new_post = Post(
        **post.dict(),
        author_id=user.id,
        organization_id=org.id  # CRITICAL
    )
    
    db.add(new_post)
    db.commit()
    
    return new_post

@app.delete("/posts/{post_id}")
async def delete_post(
    post_id: int,
    user_org_role: tuple = Depends(require_org_role("admin")),
    db: Session = Depends(get_tenant_db)
):
    """
    Only org admins can delete posts
    """
    user, org, role = user_org_role
    
    # Query automatically scoped to current org
    post = db.query(Post).filter(Post.id == post_id).first()
    
    if not post:
        raise HTTPException(status_code=404)
    
    # Double-check tenant (defense in depth)
    if post.organization_id != org.id:
        raise HTTPException(status_code=403, detail="Tenant mismatch")
    
    db.delete(post)
    db.commit()
    
    return {"deleted": post_id}
\`\`\`

**Security Considerations**:

1. **Always set organization_id**:
   - Every resource must have organization_id foreign key
   - Set on creation, never allow user to specify
   
2. **Validate tenant in every query**:
   - Use TenantSession with automatic filtering
   - Double-check organization_id before mutations
   
3. **Index organization_id**:
   - Fast tenant filtering
   - Prevents slow queries
   
4. **User-Organization membership**:
   - Validate user is member before any operation
   - Check role for authorization
   
5. **Defense in depth**:
   - Middleware sets context
   - Dependencies validate membership
   - Queries filter by org
   - Final check before mutations
   
6. **Prevent leakage**:
   - Never return organization_id in responses
   - Never expose org switching without re-authentication
   - Log all cross-tenant access attempts
   
7. **Testing**:
   - Test that user A cannot access org B's data
   - Test that changing headers doesn't bypass isolation
   - Test SQL injection attempts with org_id`,
        },
        {
            id: 3,
            question:
                'You are building a healthcare application that must comply with HIPAA regulations. Design an authorization system that implements the principle of least privilege, where doctors can only access patient records they are assigned to, nurses can view records but not edit diagnoses, and administrators can view audit logs but not patient data. Implement a comprehensive audit logging system that tracks every authorization decision (success and failure). How would you test this authorization system to ensure there are no security vulnerabilities? Provide specific test cases that would catch common authorization bugs.',
            answer: `**HIPAA-Compliant Authorization System**:

**Database Schema with Audit**:

\`\`\`python
class User(Base):
    """Healthcare staff"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    role = Column(String)  # "doctor", "nurse", "admin"
    department = Column(String)  # "cardiology", "emergency", etc.

class Patient(Base):
    """Patient records"""
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    ssn_encrypted = Column(LargeBinary)  # Encrypted SSN
    diagnosis = Column(Text)
    department = Column(String)

class PatientAssignment(Base):
    """Doctor-Patient assignment"""
    __tablename__ = "patient_assignments"
    
    doctor_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), primary_key=True)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    assigned_by = Column(Integer, ForeignKey("users.id"))

class AuthorizationAuditLog(Base):
    """HIPAA audit trail"""
    __tablename__ = "authorization_audit_logs"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String)  # "read_patient", "update_diagnosis", etc.
    resource_type = Column(String)  # "patient", "audit_log"
    resource_id = Column(Integer)
    result = Column(String)  # "success", "denied"
    reason = Column(Text)  # Why denied
    ip_address = Column(String)
    user_agent = Column(String)
\`\`\`

**Authorization Decorator with Audit**:

\`\`\`python
import logging
from functools import wraps
from fastapi import Request

logger = logging.getLogger(__name__)

def audit_access(action: str, resource_type: str):
    """
    Decorator for auditing all access attempts
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context
            request = kwargs.get('request')
            current_user = kwargs.get('current_user')
            resource_id = kwargs.get('patient_id') or kwargs.get('user_id')
            db = kwargs.get('db')
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Log successful access
                audit_log = AuthorizationAuditLog(
                    user_id=current_user.id if current_user else None,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    result="success",
                    reason="Authorized",
                    ip_address=request.client.host if request else None,
                    user_agent=request.headers.get('user-agent') if request else None
                )
                
                db.add(audit_log)
                db.commit()
                
                # Also log to application logs
                logger.info(
                    f"Authorization SUCCESS: {action} on {resource_type}:{resource_id}",
                    extra={
                        "user_id": current_user.id,
                        "action": action,
                        "resource": f"{resource_type}:{resource_id}"
                    }
                )
                
                return result
                
            except HTTPException as e:
                # Log failed access
                audit_log = AuthorizationAuditLog(
                    user_id=current_user.id if current_user else None,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    result="denied",
                    reason=e.detail,
                    ip_address=request.client.host if request else None,
                    user_agent=request.headers.get('user-agent') if request else None
                )
                
                db.add(audit_log)
                db.commit()
                
                # Log security event
                logger.warning(
                    f"Authorization DENIED: {action} on {resource_type}:{resource_id}",
                    extra={
                        "user_id": current_user.id,
                        "action": action,
                        "resource": f"{resource_type}:{resource_id}",
                        "reason": e.detail
                    }
                )
                
                raise
        
        return wrapper
    return decorator

# Authorization dependencies
async def require_assigned_patient(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Patient:
    """
    Doctor can only access patients assigned to them
    """
    # Get patient
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Check assignment
    if current_user.role == "doctor":
        assignment = db.query(PatientAssignment).filter(
            PatientAssignment.doctor_id == current_user.id,
            PatientAssignment.patient_id == patient_id
        ).first()
        
        if not assignment:
            raise HTTPException(
                status_code=403,
                detail=f"Doctor {current_user.id} not assigned to patient {patient_id}"
            )
    
    elif current_user.role == "nurse":
        # Nurses can view patients in their department
        if patient.department != current_user.department:
            raise HTTPException(
                status_code=403,
                detail=f"Nurse can only access patients in {current_user.department}"
            )
    
    elif current_user.role == "admin":
        # Admins cannot access patient data
        raise HTTPException(
            status_code=403,
            detail="Administrators cannot access patient data"
        )
    
    else:
        raise HTTPException(status_code=403, detail="Unauthorized role")
    
    return patient

# Endpoints with audit logging
@app.get("/patients/{patient_id}")
@audit_access("read_patient", "patient")
async def get_patient(
    patient_id: int,
    request: Request,
    patient: Patient = Depends(require_assigned_patient),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    View patient record (doctors and nurses only)
    """
    return {
        "id": patient.id,
        "name": patient.name,
        "diagnosis": patient.diagnosis if current_user.role == "doctor" else None,
        # Nurses cannot see diagnosis
    }

@app.put("/patients/{patient_id}/diagnosis")
@audit_access("update_diagnosis", "patient")
async def update_diagnosis(
    patient_id: int,
    diagnosis: str,
    request: Request,
    patient: Patient = Depends(require_assigned_patient),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update diagnosis (doctors only, not nurses)
    """
    if current_user.role != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Only doctors can update diagnoses"
        )
    
    patient.diagnosis = diagnosis
    db.commit()
    
    return patient

@app.get("/audit-logs")
@audit_access("read_audit_logs", "audit_log")
async def get_audit_logs(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """
    View audit logs (admins only, cannot see patient data)
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only administrators can view audit logs"
        )
    
    logs = db.query(AuthorizationAuditLog).order_by(
        AuthorizationAuditLog.timestamp.desc()
    ).offset(skip).limit(limit).all()
    
    return logs
\`\`\`

**Comprehensive Test Cases**:

\`\`\`python
import pytest

@pytest.mark.parametrize("role,patient_id,expected_status", [
    ("doctor", 1, 200),  # Assigned patient
    ("doctor", 999, 403),  # Not assigned
    ("nurse", 1, 200),  # Same department
    ("nurse", 2, 403),  # Different department
    ("admin", 1, 403),  # Admins cannot access patient data
])
def test_patient_access_control(role, patient_id, expected_status, client, db):
    """Test all role-patient access combinations"""
    # Setup: Create users and patients
    user = User(id=1, role=role, department="cardiology")
    patient1 = Patient(id=1, department="cardiology")
    patient2 = Patient(id=2, department="emergency")
    assignment = PatientAssignment(doctor_id=1, patient_id=1)
    
    db.add_all([user, patient1, patient2, assignment])
    db.commit()
    
    # Mock authentication
    app.dependency_overrides[get_current_user] = lambda: user
    
    # Test access
    response = client.get(f"/patients/{patient_id}")
    assert response.status_code == expected_status

def test_nurse_cannot_update_diagnosis(client, db):
    """Nurses can view but not edit diagnoses"""
    nurse = User(id=2, role="nurse", department="cardiology")
    patient = Patient(id=1, department="cardiology", diagnosis="Original")
    db.add_all([nurse, patient])
    db.commit()
    
    app.dependency_overrides[get_current_user] = lambda: nurse
    
    # Can view
    response = client.get("/patients/1")
    assert response.status_code == 200
    assert response.json().get("diagnosis") is None  # Nurses don't see diagnosis
    
    # Cannot update
    response = client.put("/patients/1/diagnosis", json={"diagnosis": "Updated"})
    assert response.status_code == 403
    assert "only doctors" in response.json()["detail"].lower()

def test_authorization_bypass_attempts(client, db):
    """Test common authorization bypass attempts"""
    doctor = User(id=1, role="doctor", department="cardiology")
    patient = Patient(id=2, department="emergency")
    # No assignment between doctor and patient
    db.add_all([doctor, patient])
    db.commit()
    
    app.dependency_overrides[get_current_user] = lambda: doctor
    
    # Attempt 1: Access unassigned patient
    response = client.get("/patients/2")
    assert response.status_code == 403
    
    # Attempt 2: Parameter tampering
    response = client.get("/patients/2?doctor_id=999")
    assert response.status_code == 403
    
    # Attempt 3: SQL injection in patient_id
    response = client.get("/patients/2 OR 1=1--")
    assert response.status_code in [404, 422]  # Not 200

def test_audit_log_created_on_access(client, db):
    """Every access attempt must be logged"""
    doctor = User(id=1, role="doctor")
    patient = Patient(id=1)
    assignment = PatientAssignment(doctor_id=1, patient_id=1)
    db.add_all([doctor, patient, assignment])
    db.commit()
    
    app.dependency_overrides[get_current_user] = lambda: doctor
    
    # Clear audit logs
    db.query(AuthorizationAuditLog).delete()
    db.commit()
    
    # Access patient
    client.get("/patients/1")
    
    # Check audit log created
    logs = db.query(AuthorizationAuditLog).all()
    assert len(logs) == 1
    assert logs[0].action == "read_patient"
    assert logs[0].user_id == 1
    assert logs[0].resource_id == 1
    assert logs[0].result == "success"

def test_audit_log_on_denied_access(client, db):
    """Failed access attempts must also be logged"""
    doctor = User(id=1, role="doctor")
    patient = Patient(id=2)
    # No assignment
    db.add_all([doctor, patient])
    db.commit()
    
    app.dependency_overrides[get_current_user] = lambda: doctor
    
    # Attempt unauthorized access
    response = client.get("/patients/2")
    assert response.status_code == 403
    
    # Check audit log created
    logs = db.query(AuthorizationAuditLog).filter(
        AuthorizationAuditLog.result == "denied"
    ).all()
    
    assert len(logs) == 1
    assert logs[0].reason.lower().contains("not assigned")
\`\`\`

**Additional Security Measures**:

1. **Role validation on every request**: Never trust client-provided roles
2. **Immutable audit logs**: Append-only, no deletes
3. **Real-time alerting**: Alert on suspicious patterns (many denials, unusual access times)
4. **Encrypted PHI**: Encrypt sensitive fields at rest
5. **Session timeout**: Auto-logout after inactivity
6. **MFA for privileged roles**: Require MFA for doctors and admins`,
        },
    ].map(({ id, ...q }, idx) => ({
        id: `fastapi-authz-q-${idx + 1}`,
        question: q.question,
        sampleAnswer: String(q.answer),
        keyPoints: []
    }));

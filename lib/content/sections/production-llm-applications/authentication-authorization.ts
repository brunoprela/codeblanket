export const authenticationAuthorizationContent = `
# Authentication & Authorization

## Introduction

Production LLM applications need secure authentication to identify users and authorization to control access to features and manage costs per user. This section covers API key management, OAuth2, JWT tokens, and role-based access control.

## API Key Authentication

\`\`\`python
from fastapi import FastAPI, Security, HTTPException
from fastapi.security import APIKeyHeader
import secrets

app = FastAPI()
api_key_header = APIKeyHeader (name="X-API-Key")

def generate_api_key() -> str:
    """Generate secure API key."""
    return f"sk-{secrets.token_urlsafe(32)}"

async def verify_api_key (api_key: str = Security (api_key_header)):
    """Verify API key and return user."""
    user = get_user_by_api_key (api_key)
    if not user:
        raise HTTPException (status_code=401, detail="Invalid API key")
    return user

@app.post("/generate")
async def generate (prompt: str, user=Depends (verify_api_key)):
    """Protected endpoint requiring API key."""
    return await generate_completion (prompt, user)
\`\`\`

## OAuth2 with JWT

\`\`\`python
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext (schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer (tokenUrl="token")

def create_access_token (data: dict, expires_delta: timedelta = None):
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta (minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode (to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user (token: str = Depends (oauth2_scheme)):
    """Get current user from JWT token."""
    try:
        payload = jwt.decode (token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException (status_code=401)
        return get_user (user_id)
    except JWTError:
        raise HTTPException (status_code=401)

@app.post("/token")
async def login (form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint."""
    user = authenticate_user (form_data.username, form_data.password)
    if not user:
        raise HTTPException (status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(
        data={"sub": user.id},
        expires_delta=timedelta (minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}
\`\`\`

## Role-Based Access Control

\`\`\`python
from enum import Enum
from fastapi import Depends

class UserRole (str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

class Permissions:
    """Define permissions per role."""
    
    ROLE_PERMISSIONS = {
        UserRole.FREE: {
            'max_requests_per_day': 100,
            'models': ['gpt-3.5-turbo'],
            'features': ['basic_generation']
        },
        UserRole.PRO: {
            'max_requests_per_day': 10000,
            'models': ['gpt-3.5-turbo', 'gpt-4'],
            'features': ['basic_generation', 'advanced_features', 'priority_support']
        },
        UserRole.ENTERPRISE: {
            'max_requests_per_day': float('inf'),
            'models': ['gpt-3.5-turbo', 'gpt-4', 'claude-3'],
            'features': ['all']
        }
    }
    
    @staticmethod
    def can_use_model (user_role: UserRole, model: str) -> bool:
        """Check if user can use model."""
        return model in Permissions.ROLE_PERMISSIONS[user_role]['models']
    
    @staticmethod
    def can_use_feature (user_role: UserRole, feature: str) -> bool:
        """Check if user can use feature."""
        features = Permissions.ROLE_PERMISSIONS[user_role]['features']
        return feature in features or 'all' in features

def require_role (required_role: UserRole):
    """Dependency to require specific role."""
    async def role_checker (user=Depends (get_current_user)):
        if user.role.value < required_role.value:
            raise HTTPException (status_code=403, detail="Insufficient permissions")
        return user
    return role_checker

@app.post("/admin/users")
async def create_user (user=Depends (require_role(UserRole.ADMIN))):
    """Admin-only endpoint."""
    pass
\`\`\`

## Session Management

\`\`\`python
from fastapi import FastAPI, Response, Cookie
import redis
import json

redis_client = redis.Redis()

class SessionManager:
    """Manage user sessions."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.session_ttl = 86400  # 24 hours
    
    def create_session (self, user_id: str) -> str:
        """Create new session."""
        session_id = secrets.token_urlsafe(32)
        session_data = {
            'user_id': user_id,
            'created_at': datetime.utcnow().isoformat(),
            'last_activity': datetime.utcnow().isoformat()
        }
        
        self.redis.setex(
            f"session:{session_id}",
            self.session_ttl,
            json.dumps (session_data)
        )
        
        return session_id
    
    def get_session (self, session_id: str) -> dict:
        """Get session data."""
        data = self.redis.get (f"session:{session_id}")
        if not data:
            return None
        
        session = json.loads (data)
        
        # Update last activity
        session['last_activity'] = datetime.utcnow().isoformat()
        self.redis.setex(
            f"session:{session_id}",
            self.session_ttl,
            json.dumps (session)
        )
        
        return session
    
    def delete_session (self, session_id: str):
        """Delete session (logout)."""
        self.redis.delete (f"session:{session_id}")

session_manager = SessionManager (redis_client)

@app.post("/login")
async def login (credentials: LoginRequest, response: Response):
    """Login and create session."""
    user = authenticate (credentials.username, credentials.password)
    if not user:
        raise HTTPException (status_code=401)
    
    session_id = session_manager.create_session (user.id)
    
    # Set cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=86400
    )
    
    return {"user": user}

async def get_session_user (session_id: str = Cookie(None)):
    """Get user from session cookie."""
    if not session_id:
        raise HTTPException (status_code=401)
    
    session = session_manager.get_session (session_id)
    if not session:
        raise HTTPException (status_code=401, detail="Invalid session")
    
    return get_user (session['user_id'])
\`\`\`

## Best Practices

1. **Use API keys** for service-to-service authentication
2. **Use OAuth2/JWT** for user authentication
3. **Implement role-based access** for feature gating
4. **Store API keys securely** (hashed, never in plaintext)
5. **Rotate keys regularly** and provide key management UI
6. **Use HTTPS only** for all authentication
7. **Implement rate limiting** per user/key
8. **Log authentication attempts** for security monitoring
9. **Use short-lived tokens** with refresh tokens
10. **Implement 2FA** for sensitive accounts

Proper authentication and authorization protect your LLM application and control costs by user tier.
`;

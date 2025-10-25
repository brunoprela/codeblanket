export const authenticationJwtOauth2 = {
  title: 'Authentication (JWT, OAuth2)',
  id: 'authentication-jwt-oauth2',
  content: `
# Authentication (JWT, OAuth2)

## Introduction

Authentication is the **foundation of API security**—verifying who users are before granting access. FastAPI provides excellent built-in support for modern authentication patterns including JWT (JSON Web Tokens) and OAuth2, the industry standards for API authentication.

**Why JWT + OAuth2 matters:**
- **Stateless**: No server-side session storage
- **Scalable**: Works across distributed systems
- **Standard**: OAuth2 is the internet standard (used by Google, Facebook, GitHub)
- **Secure**: Cryptographically signed tokens
- **Flexible**: Multiple authentication flows

In production, authentication solves:
- User login and registration
- API access control
- Mobile app authentication
- Third-party integrations (OAuth2)
- Microservices communication

In this section, you'll master:
- JWT token generation and validation
- Password hashing with bcrypt
- OAuth2 password flow implementation
- Refresh tokens for long-lived sessions
- Social login integration
- Security best practices
- Production patterns

### The Authentication Flow

\`\`\`
1. User sends credentials (username/password)
   ↓
2. Server verifies credentials
   ↓
3. Server generates JWT access token
   ↓
4. Client stores token
   ↓
5. Client includes token in requests (Authorization: Bearer <token>)
   ↓
6. Server validates token
   ↓
7. Server grants access
\`\`\`

---

## Password Hashing

### Secure Password Storage

\`\`\`python
"""
Password Hashing with bcrypt
"""

from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """
    Hash password using bcrypt
    
    - Generates salt automatically
    - Computationally expensive (prevents brute force)
    - Industry standard for password hashing
    """
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash
    
    Returns True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)

# Example usage
password = "SecurePassword123!"
hashed = hash_password(password)  # $2b$12$abc...

# Verify
is_valid = verify_password("SecurePassword123!", hashed)  # True
is_invalid = verify_password("WrongPassword", hashed)     # False
\`\`\`

### User Registration

\`\`\`python
"""
User Registration with Password Hashing
"""

from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy.orm import Session

app = FastAPI()

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength"""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        if not any(c in '!@#$%^&*()' for c in v):
            raise ValueError('Password must contain special character')
        return v

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user: UserRegister,
    db: Session = Depends(get_db)
):
    """
    Register new user with hashed password
    """
    # Check if username exists
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered"
        )
    
    # Check if email exists
    existing_email = db.query(User).filter(User.email == user.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
    
    # Hash password
    hashed_password = hash_password(user.password)
    
    # Create user
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user
\`\`\`

---

## JWT Tokens

### JWT Token Generation

\`\`\`python
"""
JWT Token Generation and Validation
"""

from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

# Secret key for signing tokens (use environment variable in production)
SECRET_KEY = "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token
    
    Args:
        data: Data to encode in token (user_id, username, etc.)
        expires_delta: Optional custom expiration time
    
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(user_id: int) -> str:
    """
    Create JWT refresh token (longer lived)
    """
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> dict:
    """
    Decode and validate JWT token
    
    Raises:
        JWTError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Example usage
token_data = {"sub": "123", "username": "alice"}
access_token = create_access_token(token_data)
# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Decode
decoded = decode_token(access_token)
# {"sub": "123", "username": "alice", "exp": 1234567890, ...}
\`\`\`

### Token Validation Dependency

\`\`\`python
"""
Token Validation with Dependency Injection
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

# OAuth2 scheme (extracts token from Authorization header)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_token_payload(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Extract and validate token
    
    Dependency that:
    1. Extracts token from Authorization: Bearer <token>
    2. Decodes and validates JWT
    3. Returns payload
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        
        # Check token type
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        return payload
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(
    token_payload: dict = Depends(get_token_payload),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current user from token
    
    Dependency chain:
    1. oauth2_scheme extracts token
    2. get_token_payload validates token
    3. get_current_user fetches user from database
    """
    user_id = token_payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user

# Protected endpoint
@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """
    Get current user profile
    
    Requires valid JWT token in Authorization header
    """
    return current_user
\`\`\`

---

## OAuth2 Password Flow

### Login Endpoint

\`\`\`python
"""
OAuth2 Password Flow Implementation
"""

from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

@app.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    OAuth2 password flow login
    
    Request body (form data):
    - username: str
    - password: str
    
    Returns:
    - access_token: JWT access token
    - refresh_token: JWT refresh token
    - token_type: "bearer"
    """
    # Find user by username or email
    user = db.query(User).filter(
        (User.username == form_data.username) |
        (User.email == form_data.username)
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username}
    )
    refresh_token = create_refresh_token(user.id)
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

# Alternative JSON login endpoint
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login", response_model=Token)
async def login_json(
    credentials: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    JSON-based login (alternative to OAuth2 form)
    
    Accepts JSON body instead of form data
    """
    user = db.query(User).filter(
        (User.username == credentials.username) |
        (User.email == credentials.username)
    ).first()
    
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }
\`\`\`

### Refresh Token Flow

\`\`\`python
"""
Refresh Token Implementation
"""

from fastapi import Header

def get_refresh_token_payload(
    refresh_token: str = Header(..., alias="X-Refresh-Token")
) -> dict:
    """
    Validate refresh token from header
    """
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check token type
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token expired"
            )
        
        return payload
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@app.post("/token/refresh", response_model=Token)
async def refresh_access_token(
    token_payload: dict = Depends(get_refresh_token_payload),
    db: Session = Depends(get_db)
):
    """
    Refresh access token using refresh token
    
    Headers:
    - X-Refresh-Token: <refresh_token>
    
    Returns new access token
    """
    user_id = token_payload.get("sub")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user"
        )
    
    # Create new access token
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username}
    )
    
    # Optionally rotate refresh token
    new_refresh_token = create_refresh_token(user.id)
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }
\`\`\`

---

## Security Best Practices

### Token Security

\`\`\`python
"""
Security Best Practices
"""

import secrets

# 1. Strong secret key generation
def generate_secret_key() -> str:
    """Generate cryptographically strong secret key"""
    return secrets.token_urlsafe(32)

# Use in production (environment variable)
# SECRET_KEY = os.getenv("JWT_SECRET_KEY", generate_secret_key())

# 2. Token blacklist (for logout)
from redis import Redis

redis_client = Redis(host='localhost', port=6379, decode_responses=True)

async def blacklist_token(token: str, expires_in: int):
    """
    Add token to blacklist
    
    Used for logout - prevents token reuse
    """
    redis_client.setex(f"blacklist:{token}", expires_in, "1")

async def is_token_blacklisted(token: str) -> bool:
    """Check if token is blacklisted"""
    return redis_client.exists(f"blacklist:{token}") > 0

def get_token_payload_with_blacklist(
    token: str = Depends(oauth2_scheme)
) -> dict:
    """
    Validate token and check blacklist
    """
    # Check blacklist
    if await is_token_blacklisted(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked"
        )
    
    # Validate token
    return get_token_payload(token)

@app.post("/logout")
async def logout(
    token: str = Depends(oauth2_scheme),
    token_payload: dict = Depends(get_token_payload)
):
    """
    Logout - blacklist token
    """
    # Calculate remaining TTL
    exp = token_payload.get("exp")
    ttl = exp - datetime.utcnow().timestamp()
    
    if ttl > 0:
        await blacklist_token(token, int(ttl))
    
    return {"detail": "Successfully logged out"}

# 3. Rate limiting login attempts
from collections import defaultdict
from time import time

login_attempts = defaultdict(list)

def check_login_rate_limit(username: str):
    """
    Rate limit login attempts
    
    Max 5 attempts per 15 minutes
    """
    now = time()
    attempts = login_attempts[username]
    
    # Remove old attempts (older than 15 minutes)
    attempts = [t for t in attempts if now - t < 900]
    
    if len(attempts) >= 5:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Try again in 15 minutes."
        )
    
    attempts.append(now)
    login_attempts[username] = attempts

# 4. Password reset tokens
import secrets

def create_password_reset_token(user_id: int) -> str:
    """
    Create secure password reset token
    
    Store in database with expiration
    """
    token = secrets.token_urlsafe(32)
    expire = datetime.utcnow() + timedelta(hours=1)
    
    # Store in database
    reset_token = PasswordResetToken(
        user_id=user_id,
        token=token,
        expires_at=expire
    )
    db.add(reset_token)
    db.commit()
    
    return token

@app.post("/password/reset-request")
async def request_password_reset(
    email: EmailStr,
    db: Session = Depends(get_db)
):
    """
    Request password reset
    
    Sends email with reset token
    """
    user = db.query(User).filter(User.email == email).first()
    
    # Don't reveal if user exists (security)
    if not user:
        return {"detail": "If email exists, reset link sent"}
    
    # Create reset token
    token = create_password_reset_token(user.id)
    
    # Send email (pseudo-code)
    send_email(
        to=email,
        subject="Password Reset",
        body=f"Reset link: https://example.com/reset?token={token}"
    )
    
    return {"detail": "If email exists, reset link sent"}
\`\`\`

---

## Social Login (OAuth2)

### Google OAuth2

\`\`\`python
"""
Google OAuth2 Integration
"""

from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

# Configuration
config = Config('.env')
oauth = OAuth(config)

oauth.register(
    name='google',
    client_id=config('GOOGLE_CLIENT_ID'),
    client_secret=config('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

@app.get("/auth/google")
async def login_google(request: Request):
    """
    Redirect to Google OAuth2 login
    """
    redirect_uri = request.url_for('auth_google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def auth_google_callback(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Google OAuth2 callback
    
    Creates or logs in user
    """
    # Get token from Google
    token = await oauth.google.authorize_access_token(request)
    
    # Get user info
    user_info = token.get('userinfo')
    
    if not user_info:
        raise HTTPException(status_code=400, detail="Failed to get user info")
    
    # Find or create user
    user = db.query(User).filter(User.email == user_info['email']).first()
    
    if not user:
        # Create new user
        user = User(
            email=user_info['email'],
            username=user_info['email'].split('@')[0],
            full_name=user_info.get('name'),
            profile_picture=user_info.get('picture'),
            is_verified=True,  # Email verified by Google
            oauth_provider='google',
            oauth_id=user_info['sub']
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # Create JWT tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(user.id)
    
    # Redirect to frontend with tokens
    return RedirectResponse(
        url=f"https://frontend.com/auth/callback?access_token={access_token}&refresh_token={refresh_token}"
    )
\`\`\`

---

## Testing Authentication

### Testing with Override

\`\`\`python
"""
Testing Authentication
"""

import pytest
from fastapi.testclient import TestClient

# Mock current user
def override_get_current_user():
    return User(
        id=1,
        username="testuser",
        email="test@example.com",
        is_active=True
    )

@pytest.fixture
def authenticated_client():
    """Test client with authenticated user"""
    app.dependency_overrides[get_current_user] = override_get_current_user
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()

def test_protected_endpoint(authenticated_client):
    """Test protected endpoint"""
    response = authenticated_client.get("/users/me")
    
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"

def test_login(client):
    """Test login"""
    response = client.post(
        "/token",
        data={"username": "testuser", "password": "password123"}
    )
    
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_invalid_login(client):
    """Test login with wrong password"""
    response = client.post(
        "/token",
        data={"username": "testuser", "password": "wrongpassword"}
    )
    
    assert response.status_code == 401
\`\`\`

---

## Summary

### Key Takeaways

✅ **Password hashing**: Use bcrypt, never store plain passwords  
✅ **JWT tokens**: Stateless, scalable, standard format  
✅ **OAuth2**: Industry standard, multiple flows  
✅ **Access + refresh**: Short-lived access, long-lived refresh  
✅ **Dependency injection**: Clean auth with get_current_user  
✅ **Token blacklist**: Implement logout with Redis  
✅ **Rate limiting**: Prevent brute force attacks

### Best Practices

**1. Token security**:
- Strong secret key (32+ bytes, random)
- Short expiration (15-30 minutes access, 7 days refresh)
- HTTPS only in production
- Token blacklist for logout

**2. Password security**:
- bcrypt hashing (not MD5/SHA)
- Password strength validation
- Rate limit login attempts
- Secure password reset flow

**3. Production considerations**:
- Store secret key in environment variables
- Use Redis for token blacklist
- Implement refresh token rotation
- Monitor failed login attempts
- Log all authentication events

### Next Steps

In the next section, we'll explore **Authorization & Permissions**: implementing role-based access control (RBAC), permission checks, and fine-grained authorization for your API endpoints.

**Production mindset**: Authentication is just the first step—knowing who the user is. Next, we need authorization to control what they can do.
`,
};

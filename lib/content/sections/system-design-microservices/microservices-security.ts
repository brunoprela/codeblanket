/**
 * Microservices Security Section
 */

export const microservicessecuritySection = {
  id: 'microservices-security',
  title: 'Microservices Security',
  content: `Security in microservices is more complex than monoliths due to the distributed nature. With many services communicating over the network, the attack surface increases significantly.

## Security Challenges in Microservices

**Compared to monolith**:

**Monolith**:
- Single security perimeter (firewall)
- Internal function calls (no network exposure)
- One authentication point
- Centralized authorization

**Microservices**:
- Multiple security perimeters
- All communication over network (intercept risk)
- Authentication at each service
- Distributed authorization
- More attack surface

---

## Defense in Depth

**Layered security approach** - multiple defensive layers.

\`\`\`
┌─────────────────────────────────────┐
│ Network Security (Firewalls, VPC)  │
├─────────────────────────────────────┤
│ API Gateway (Auth, Rate Limiting)   │
├─────────────────────────────────────┤
│ Service Mesh (mTLS, AuthZ Policies) │
├─────────────────────────────────────┤
│ Service-Level Security              │
├─────────────────────────────────────┤
│ Data Encryption (at rest & transit) │
└─────────────────────────────────────┘
\`\`\`

**Principle**: If one layer fails, others protect the system.

---

## Authentication & Authorization

### 1. API Gateway Authentication

**Gateway authenticates users**, services trust the gateway.

\`\`\`javascript
// API Gateway
async function authenticate (req, res, next) {
    const token = req.headers['authorization']?.replace('Bearer ', ');
    
    if (!token) {
        return res.status(401).json({ error: 'No token provided' });
    }
    
    try {
        // Verify JWT
        const decoded = jwt.verify (token, JWT_SECRET);
        
        // Add user context to headers for downstream services
        req.headers['X-User-Id'] = decoded.userId;
        req.headers['X-User-Email'] = decoded.email;
        req.headers['X-User-Roles'] = JSON.stringify (decoded.roles);
        
        next();
    } catch (error) {
        return res.status(401).json({ error: 'Invalid token' });
    }
}

app.use (authenticate);
\`\`\`

**Downstream service** trusts headers:
\`\`\`javascript
// Order Service
app.post('/orders', async (req, res) => {
    const userId = req.headers['X-User-Id']; // Trusts gateway
    const roles = JSON.parse (req.headers['X-User-Roles'] || '[]');
    
    // Authorization
    if (!roles.includes('customer')) {
        return res.status(403).json({ error: 'Forbidden' });
    }
    
    const order = await createOrder({ ...req.body, userId });
    res.json (order);
});
\`\`\`

**Security concern**: What if attacker bypasses gateway?

**Solution**: Service Mesh with mTLS + Authorization Policies.

### 2. Service-to-Service Authentication

**Problem**: How does Order Service verify Payment Service is really Payment Service?

**Solution**: Mutual TLS (mTLS)

#### Mutual TLS (mTLS)

**Both client and server** authenticate each other with certificates.

**Without mTLS**:
\`\`\`
Order Service → Payment Service
"I'm Order Service" (no proof)
\`\`\`

**With mTLS**:
\`\`\`
Order Service → Payment Service
Order Service presents certificate signed by trusted CA
Payment Service verifies certificate
Payment Service presents its certificate
Order Service verifies certificate
✅ Both authenticated
\`\`\`

**Istio automatically** handles mTLS:
\`\`\`yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: default
spec:
  mtls:
    mode: STRICT  # Require mTLS for all services
\`\`\`

**Certificates**:
- Istio provisions certificates to sidecar proxies
- Certificates rotate automatically (every 24 hours)
- Services talk to localhost (proxy handles mTLS)

### 3. OAuth 2.0 for Third-Party Access

**Use case**: Mobile app, partner APIs.

**Flow (Authorization Code)**:
\`\`\`
1. User clicks "Login with Google"
2. Redirect to Google authorization server
3. User grants permission
4. Google redirects back with authorization code
5. App exchanges code for access token
6. App uses access token to call API
\`\`\`

**Example**:
\`\`\`javascript
// API Gateway
app.get('/auth/google', (req, res) => {
    const authUrl = \`https://accounts.google.com/o/oauth2/v2/auth?
        client_id=\${GOOGLE_CLIENT_ID}
        &redirect_uri=\${REDIRECT_URI}
        &response_type=code
        &scope=openid email profile\`;
    res.redirect (authUrl);
});

app.get('/auth/google/callback', async (req, res) => {
    const { code } = req.query;
    
    // Exchange code for tokens
    const response = await axios.post('https://oauth2.googleapis.com/token', {
        code,
        client_id: GOOGLE_CLIENT_ID,
        client_secret: GOOGLE_CLIENT_SECRET,
        redirect_uri: REDIRECT_URI,
        grant_type: 'authorization_code'
    });
    
    const { access_token, id_token } = response.data;
    
    // Verify ID token and create session
    const userInfo = jwt.decode (id_token);
    const sessionToken = createJWT({ userId: userInfo.sub, email: userInfo.email });
    
    res.cookie('session', sessionToken);
    res.redirect('/dashboard');
});
\`\`\`

### 4. Authorization Patterns

#### Role-Based Access Control (RBAC)

**Users have roles**, roles have permissions.

\`\`\`javascript
const roles = {
    customer: ['read:products', 'create:order', 'read:own-orders'],
    admin: ['read:products', 'create:product', 'read:all-orders', 'delete:order'],
    support: ['read:products', 'read:all-orders', 'update:order-status']
};

function authorize (requiredPermission) {
    return (req, res, next) => {
        const userRoles = JSON.parse (req.headers['X-User-Roles'] || '[]');
        const userPermissions = userRoles.flatMap (role => roles[role] || []);
        
        if (!userPermissions.includes (requiredPermission)) {
            return res.status(403).json({ error: 'Forbidden' });
        }
        
        next();
    };
}

// Usage
app.get('/orders', authorize('read:all-orders'), async (req, res) => {
    // Only admin and support can access
    const orders = await getAllOrders();
    res.json (orders);
});
\`\`\`

#### Attribute-Based Access Control (ABAC)

**More flexible** - considers attributes (user, resource, environment).

\`\`\`javascript
function canAccessOrder (user, order, environment) {
    // User is order owner
    if (order.userId === user.id) return true;
    
    // Admin can access all
    if (user.roles.includes('admin')) return true;
    
    // Support can access during business hours
    if (user.roles.includes('support')) {
        const hour = new Date().getHours();
        return hour >= 9 && hour < 17; // 9 AM - 5 PM
    }
    
    return false;
}

app.get('/orders/:id', async (req, res) => {
    const order = await getOrder (req.params.id);
    const user = { id: req.headers['X-User-Id'], roles: JSON.parse (req.headers['X-User-Roles']) };
    
    if (!canAccessOrder (user, order, {})) {
        return res.status(403).json({ error: 'Forbidden' });
    }
    
    res.json (order);
});
\`\`\`

---

## Network Security

### 1. Network Segmentation

**Isolate services** in different network zones.

\`\`\`
┌─────────────────────────────────┐
│ Public Zone (Internet-facing)   │
│ - API Gateway                   │
│ - Web Server                    │
└──────────┬──────────────────────┘
           │ Firewall
┌──────────▼──────────────────────┐
│ Application Zone (Internal)     │
│ - Order Service                 │
│ - User Service                  │
│ - Product Service               │
└──────────┬──────────────────────┘
           │ Firewall
┌──────────▼──────────────────────┐
│ Data Zone (Restricted)          │
│ - Databases                     │
│ - Cache                         │
└─────────────────────────────────┘
\`\`\`

**Kubernetes NetworkPolicy**:
\`\`\`yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: payment-service-policy
spec:
  podSelector:
    matchLabels:
      app: payment-service
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: order-service  # Only Order Service can call Payment
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres  # Payment can only talk to its database
    ports:
    - protocol: TCP
      port: 5432
\`\`\`

**Result**: Even if attacker compromises one service, they can't access others.

### 2. API Gateway Rate Limiting

**Prevent DDoS attacks** and abuse.

\`\`\`javascript
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // 100 requests per window
    message: 'Too many requests, please try again later',
    standardHeaders: true,
    legacyHeaders: false,
    // Rate limit by user ID (if authenticated) or IP
    keyGenerator: (req) => {
        return req.headers['X-User-Id'] || req.ip;
    }
});

app.use('/api/', limiter);
\`\`\`

**Tiered rate limiting**:
\`\`\`javascript
const limits = {
    free: { windowMs: 60000, max: 10 },      // 10 req/min
    pro: { windowMs: 60000, max: 100 },      // 100 req/min
    enterprise: { windowMs: 60000, max: 1000 } // 1000 req/min
};

function getRateLimiter (req) {
    const tier = req.user?.tier || 'free';
    return rateLimit (limits[tier]);
}
\`\`\`

---

## Data Security

### 1. Encryption in Transit

**Always use TLS** for service communication.

**API Gateway to Client**: HTTPS (TLS 1.2+)
**Service to Service**: mTLS (via service mesh)

\`\`\`javascript
// HTTPS server
const https = require('https');
const fs = require('fs');

const options = {
    key: fs.readFileSync('server-key.pem'),
    cert: fs.readFileSync('server-cert.pem')
};

https.createServer (options, app).listen(443);
\`\`\`

### 2. Encryption at Rest

**Encrypt sensitive data** in database.

\`\`\`javascript
const crypto = require('crypto');

// Encryption
function encrypt (text, key) {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-gcm', Buffer.from (key, 'hex'), iv);
    
    let encrypted = cipher.update (text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const authTag = cipher.getAuthTag();
    
    return {
        encrypted,
        iv: iv.toString('hex'),
        authTag: authTag.toString('hex')
    };
}

// Decryption
function decrypt (encrypted, iv, authTag, key) {
    const decipher = crypto.createDecipheriv(
        'aes-256-gcm',
        Buffer.from (key, 'hex'),
        Buffer.from (iv, 'hex')
    );
    
    decipher.setAuthTag(Buffer.from (authTag, 'hex'));
    
    let decrypted = decipher.update (encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
}

// Store credit card
async function saveCreditCard (userId, cardNumber) {
    const encrypted = encrypt (cardNumber, ENCRYPTION_KEY);
    
    await db.creditCards.insert({
        userId,
        cardNumber: encrypted.encrypted,
        iv: encrypted.iv,
        authTag: encrypted.authTag
    });
}
\`\`\`

**Better**: Use **vault services** (HashiCorp Vault, AWS KMS) for key management.

### 3. Secrets Management

**Never hardcode secrets**. Use secret management tools.

**Kubernetes Secrets**:
\`\`\`yaml
apiVersion: v1
kind: Secret
metadata:
  name: payment-service-secrets
type: Opaque
data:
  stripe-api-key: c2stdGVzdF8xMjM0NTY3ODkw  # base64
  database-password: cGFzc3dvcmQxMjM=        # base64
\`\`\`

**HashiCorp Vault**:
\`\`\`javascript
const vault = require('node-vault')({
    endpoint: 'http://vault:8200',
    token: process.env.VAULT_TOKEN
});

// Read secret
const secret = await vault.read('secret/data/payment-service');
const stripeApiKey = secret.data.data['stripe-api-key'];
\`\`\`

---

## Input Validation & Sanitization

**Validate all inputs** to prevent injection attacks.

### SQL Injection

❌ **Vulnerable**:
\`\`\`javascript
const query = \`SELECT * FROM users WHERE email = '\${req.body.email}'\`;
// Attacker sends: email = "' OR '1'='1"
// Query becomes: SELECT * FROM users WHERE email = ' OR '1'='1'
// Returns all users!
\`\`\`

✅ **Safe** (Parameterized queries):
\`\`\`javascript
const query = 'SELECT * FROM users WHERE email = $1';
const result = await db.query (query, [req.body.email]);
\`\`\`

### NoSQL Injection

❌ **Vulnerable**:
\`\`\`javascript
const user = await db.users.findOne({ email: req.body.email });
// Attacker sends: { "email": { "$ne": null } }
// Returns first user in database!
\`\`\`

✅ **Safe** (Validate input):
\`\`\`javascript
const Joi = require('joi');

const schema = Joi.object({
    email: Joi.string().email().required()
});

const { error, value } = schema.validate (req.body);
if (error) {
    return res.status(400).json({ error: error.details[0].message });
}

const user = await db.users.findOne({ email: value.email });
\`\`\`

### Cross-Site Scripting (XSS)

**Sanitize output**:
\`\`\`javascript
const escapeHtml = require('escape-html');

app.get('/user/:id', async (req, res) => {
    const user = await getUser (req.params.id);
    
    res.send(\`
        <h1>Welcome, \${escapeHtml (user.name)}</h1>
    \`);
});
\`\`\`

---

## Audit Logging

**Log all security-relevant events**.

\`\`\`javascript
function auditLog (event) {
    logger.info({
        event: event.type,
        userId: event.userId,
        resource: event.resource,
        action: event.action,
        result: event.result,
        timestamp: new Date().toISOString(),
        ip: event.ip,
        userAgent: event.userAgent
    });
}

app.post('/orders', async (req, res) => {
    try {
        const order = await createOrder (req.body);
        
        auditLog({
            type: 'ORDER_CREATED',
            userId: req.headers['X-User-Id'],
            resource: 'order',
            action: 'create',
            result: 'success',
            ip: req.ip,
            userAgent: req.headers['user-agent']
        });
        
        res.json (order);
    } catch (error) {
        auditLog({
            type: 'ORDER_CREATE_FAILED',
            userId: req.headers['X-User-Id'],
            resource: 'order',
            action: 'create',
            result: 'failure',
            error: error.message,
            ip: req.ip
        });
        
        throw error;
    }
});
\`\`\`

**What to log**:
- Authentication attempts (success/failure)
- Authorization failures
- Data access (sensitive data)
- Configuration changes
- Service-to-service calls

**What NOT to log**:
- Passwords
- API keys
- Credit card numbers
- PII without masking

---

## Security Headers

**Add security headers** to all responses.

\`\`\`javascript
app.use((req, res, next) => {
    // Prevent clickjacking
    res.setHeader('X-Frame-Options', 'DENY');
    
    // Prevent MIME sniffing
    res.setHeader('X-Content-Type-Options', 'nosniff');
    
    // XSS Protection
    res.setHeader('X-XSS-Protection', '1; mode=block');
    
    // Content Security Policy
    res.setHeader('Content-Security-Policy', "default-src 'self'");
    
    // HSTS (force HTTPS)
    res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
    
    next();
});
\`\`\`

---

## Dependency Scanning

**Scan for vulnerabilities** in dependencies.

\`\`\`bash
# npm audit
npm audit

# Fix vulnerabilities
npm audit fix

# Snyk
snyk test
snyk monitor

# OWASP Dependency Check
dependency-check --project my-service --scan .
\`\`\`

**Automate** in CI/CD pipeline:
\`\`\`yaml
# .gitlab-ci.yml
security-scan:
  stage: test
  script:
    - npm audit
    - snyk test --severity-threshold=high
  allow_failure: false  # Fail build if vulnerabilities found
\`\`\`

---

## Container Security

### 1. Use Minimal Base Images

\`\`\`dockerfile
# ❌ Large attack surface
FROM node:18

# ✅ Minimal image
FROM node:18-alpine
\`\`\`

### 2. Run as Non-Root User

\`\`\`dockerfile
FROM node:18-alpine

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nodejs -u 1001

WORKDIR /app
COPY --chown=nodejs:nodejs . .

# Switch to non-root user
USER nodejs

CMD ["node", "server.js"]
\`\`\`

### 3. Scan Images for Vulnerabilities

\`\`\`bash
# Trivy
trivy image order-service:v1.2.3

# Clair
clairctl analyze order-service:v1.2.3
\`\`\`

---

## Best Practices

1. **Defense in depth**: Multiple security layers
2. **Zero trust**: Verify everything, trust nothing
3. **mTLS**: Encrypt all service-to-service communication
4. **Least privilege**: Minimum permissions needed
5. **Input validation**: Validate all inputs
6. **Audit logging**: Log security events
7. **Secrets management**: Use vaults, never hardcode
8. **Dependency scanning**: Regular vulnerability checks
9. **Network segmentation**: Isolate services
10. **Security headers**: Add to all responses

---

## Key Takeaways

1. **Distributed** systems have larger attack surface
2. **mTLS** authenticates services to each other
3. **API Gateway** centralizes authentication
4. **Network policies** restrict service communication
5. **Encrypt** data in transit (TLS) and at rest
6. **Validate** all inputs to prevent injection
7. **Audit log** all security events
8. **Scan** dependencies and containers for vulnerabilities`,
};

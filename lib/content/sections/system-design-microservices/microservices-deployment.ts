/**
 * Microservices Deployment Section
 */

export const microservicesdeploymentSection = {
  id: 'microservices-deployment',
  title: 'Microservices Deployment',
  content: `Deploying microservices requires orchestration, containerization, and CI/CD automation. The deployment strategy significantly impacts reliability, scalability, and development velocity.

## Containerization

**Why containers?** Consistency across environments, isolation, portability.

### Docker

**Package service as container image**:

\`\`\`dockerfile
# Dockerfile for Order Service
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy source
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=3s \\
  CMD node healthcheck.js || exit 1

EXPOSE 8080

CMD ["node", "server.js"]
\`\`\`

**Build and run**:
\`\`\`bash
docker build -t order-service:v1.2.3 .
docker run -p 8080:8080 order-service:v1.2.3
\`\`\`

**Best practices**:
- Multi-stage builds (reduce image size)
- Layer caching (faster builds)
- Non-root user (security)
- Small base images (alpine)
- Health checks

---

## Kubernetes

**De facto** orchestration platform for microservices.

### Core Concepts

**1. Pod**: Smallest deployable unit (1+ containers)
**2. Deployment**: Manages replica Pods
**3. Service**: Stable network endpoint
**4. Ingress**: External access
**5. ConfigMap/Secret**: Configuration

### Example Deployment

\`\`\`yaml
# order-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  labels:
    app: order-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
  template:
    metadata:
      labels:
        app: order-service
        version: v1.2.3
    spec:
      containers:
      - name: order-service
        image: myregistry/order-service:v1.2.3
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: order-service-secrets
              key: database-url
        - name: PAYMENT_SERVICE_URL
          value: "http://payment-service"
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: order-service
spec:
  selector:
    app: order-service
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
\`\`\`

**Deploy**:
\`\`\`bash
kubectl apply -f order-service-deployment.yaml
kubectl get pods
kubectl logs order-service-abc123
\`\`\`

---

## Deployment Strategies

### 1. Rolling Update (Default)

**Gradually** replace old pods with new ones.

\`\`\`yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Max extra pods during update
      maxUnavailable: 0  # Min available pods during update
\`\`\`

**Flow**:
\`\`\`
v1: [Pod1] [Pod2] [Pod3]
    [Pod1] [Pod2] [Pod3] [Pod4-v2]  # Create new pod
    [Pod1] [Pod2] [Pod4-v2]          # Delete old pod
    [Pod1] [Pod2] [Pod4-v2] [Pod5-v2]
    [Pod2] [Pod4-v2] [Pod5-v2]
    [Pod2] [Pod4-v2] [Pod5-v2] [Pod6-v2]
    [Pod4-v2] [Pod5-v2] [Pod6-v2]
\`\`\`

**Pros**: Zero downtime, automatic rollback
**Cons**: Mixed versions during deployment

### 2. Blue-Green Deployment

**Two identical environments**, switch traffic instantly.

\`\`\`yaml
# Blue (current)
apiVersion: v1
kind: Service
metadata:
  name: order-service
spec:
  selector:
    app: order-service
    version: blue
  ports:
  - port: 80
    targetPort: 8080
---
# Deploy Green
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
      version: green
  template:
    metadata:
      labels:
        app: order-service
        version: green
    spec:
      containers:
      - name: order-service
        image: order-service:v1.2.3
\`\`\`

**Switch traffic** (update Service selector):
\`\`\`yaml
spec:
  selector:
    app: order-service
    version: green  # Changed from blue
\`\`\`

**Pros**: Instant rollback, testing before switch
**Cons**: 2x resources, database migration challenges

### 3. Canary Deployment

**Gradually** shift traffic to new version.

Using Istio:
\`\`\`yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-service
spec:
  hosts:
  - order-service
  http:
  - route:
    - destination:
        host: order-service
        subset: v1
      weight: 90  # 90% to old version
    - destination:
        host: order-service
        subset: v2
      weight: 10  # 10% to new version (canary)
\`\`\`

**Increase gradually**: 10% → 25% → 50% → 100%

**Pros**: Low risk, real user testing
**Cons**: Requires service mesh, slow rollout

---

## CI/CD Pipeline

**Automated** build, test, deploy.

### Example Pipeline (GitLab CI)

\`\`\`yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

unit-test:
  stage: test
  script:
    - npm install
    - npm test

integration-test:
  stage: test
  script:
    - docker-compose up -d
    - npm run test:integration
    - docker-compose down

contract-test:
  stage: test
  script:
    - npm run test:pact

deploy-staging:
  stage: deploy
  script:
    - kubectl set image deployment/order-service \\
        order-service=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA \\
        -n staging
  environment:
    name: staging
  only:
    - develop

deploy-production:
  stage: deploy
  script:
    - kubectl set image deployment/order-service \\
        order-service=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA \\
        -n production
  environment:
    name: production
  when: manual  # Require approval
  only:
    - main
\`\`\`

---

## Configuration Management

**Never** hardcode config. Use environment variables or config files.

### ConfigMaps

\`\`\`yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: order-service-config
data:
  PAYMENT_SERVICE_URL: "http://payment-service"
  LOG_LEVEL: "info"
  MAX_RETRIES: "3"
\`\`\`

### Secrets

\`\`\`yaml
apiVersion: v1
kind: Secret
metadata:
  name: order-service-secrets
type: Opaque
data:
  database-url: cG9zdGdyZXM6Ly8uLi4=  # base64 encoded
  api-key: YWJjZGVm  # base64 encoded
\`\`\`

**Use in Pod**:
\`\`\`yaml
env:
- name: DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: order-service-secrets
      key: database-url
- name: PAYMENT_SERVICE_URL
  valueFrom:
    configMapKeyRef:
      name: order-service-config
      key: PAYMENT_SERVICE_URL
\`\`\`

---

## Service Mesh Deployment

**Istio automatically** injects sidecar proxies.

**Enable injection**:
\`\`\`bash
kubectl label namespace default istio-injection=enabled
\`\`\`

**Deploy service** (Istio adds proxy automatically):
\`\`\`bash
kubectl apply -f order-service-deployment.yaml

# Pod now has 2 containers: order-service + istio-proxy
kubectl get pods
NAME                             READY   STATUS
order-service-abc123             2/2     Running
\`\`\`

---

## Database Migrations

**Challenge**: Deploy new service version with database changes.

### Backward-Compatible Migrations

**Step 1**: Add new column (optional)
\`\`\`sql
ALTER TABLE orders ADD COLUMN shipping_address TEXT;
\`\`\`

**Step 2**: Deploy new service version (uses new column)

**Step 3**: Backfill data
\`\`\`sql
UPDATE orders SET shipping_address = address WHERE shipping_address IS NULL;
\`\`\`

**Step 4**: Make column required (next deployment)
\`\`\`sql
ALTER TABLE orders ALTER COLUMN shipping_address SET NOT NULL;
\`\`\`

**Step 5**: Remove old column (next deployment)
\`\`\`sql
ALTER TABLE orders DROP COLUMN address;
\`\`\`

**Key**: Each step is backward-compatible.

---

## Monitoring and Observability

**Essential for production** microservices.

### Health Checks

**Liveness probe**: Is service alive? (restart if fails)
**Readiness probe**: Is service ready for traffic? (remove from load balancer if fails)

\`\`\`javascript
// /health endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

// /ready endpoint (check dependencies)
app.get('/ready', async (req, res) => {
    try {
        await database.ping();
        await paymentService.healthCheck();
        res.json({ status: 'ready' });
    } catch (error) {
        res.status(503).json({ status: 'not ready', error: error.message });
    }
});
\`\`\`

### Metrics

**Export Prometheus metrics**:
\`\`\`javascript
const promClient = require('prom-client');
const register = new promClient.Registry();

// Metrics
const httpRequestDuration = new promClient.Histogram({
    name: 'http_request_duration_seconds',
    help: 'Duration of HTTP requests in seconds',
    labelNames: ['method', 'route', 'status_code'],
    registers: [register]
});

// Middleware
app.use((req, res, next) => {
    const end = httpRequestDuration.startTimer();
    res.on('finish', () => {
        end({ method: req.method, route: req.route?.path, status_code: res.statusCode });
    });
    next();
});

// Metrics endpoint
app.get('/metrics', async (req, res) => {
    res.set('Content-Type', register.contentType);
    res.end (await register.metrics());
});
\`\`\`

---

## Scaling

### Horizontal Pod Autoscaler

**Automatically** scale based on CPU/memory:

\`\`\`yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: order-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
\`\`\`

**Kubernetes automatically** adds/removes pods to maintain target CPU/memory.

---

## Best Practices

1. **Containerize** all services
2. **Use Kubernetes** for orchestration
3. **Automate** CI/CD pipeline
4. **Canary deployments** for low-risk rollouts
5. **Health checks** (liveness + readiness)
6. **Resource limits** (CPU, memory)
7. **ConfigMaps/Secrets** for configuration
8. **Backward-compatible** database migrations
9. **Monitor** everything (metrics, logs, traces)
10. **Autoscale** based on load

---

## Key Takeaways

1. **Containers** provide consistency and isolation
2. **Kubernetes** is standard for microservices orchestration
3. **Deployment strategies**: Rolling, Blue-Green, Canary
4. **CI/CD** automates build, test, deploy
5. **Health checks** enable automatic recovery
6. **Database migrations** must be backward-compatible
7. **Configuration** via ConfigMaps/Secrets
8. **Autoscaling** handles variable load`,
};

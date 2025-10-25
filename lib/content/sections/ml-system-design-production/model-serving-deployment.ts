export const modelServingDeployment = {
  title: 'Model Serving & Deployment',
  id: 'model-serving-deployment',
  content: `
# Model Serving & Deployment

## Introduction

**"A model in a notebook is just an experiment. A model in production is a business asset."**

Model deployment is where ML meets the real world. You've trained a great model—now you need to serve predictions to users, applications, or trading systems at scale.

**Deployment Challenges**:
- Low latency requirements (<100ms)
- High throughput (1000s requests/sec)
- 24/7 availability
- Model versioning and rollbacks
- Monitoring and debugging
- Cost optimization

This section covers production deployment patterns, from simple REST APIs to high-performance serving systems for trading.

### Deployment Options

\`\`\`
Development → Staging → Production
    ↓           ↓          ↓
 Notebook   Docker    Kubernetes
              ↓          ↓
         REST API   Model Server
                      (TensorFlow Serving, TorchServe, etc.)
\`\`\`

By the end of this section, you'll understand:
- REST API serving with FastAPI
- Containerization with Docker
- Cloud deployment (AWS, GCP, Azure)
- Model serving frameworks
- Batch vs real-time serving
- Canary deployments and rollbacks

---

## REST API Serving with FastAPI

### Basic Model API

\`\`\`python
"""
Simple Model Serving API with FastAPI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig (level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Trading Model API",
    description="Serve predictions for algorithmic trading",
    version="1.0.0"
)

# Global model variable (loaded on startup)
model = None
scaler = None

# Request/Response models
class PredictionRequest(BaseModel):
    """Request schema"""
    features: List[float] = Field(..., description="Feature vector")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.5, 0.3, -0.2, 0.1, 0.8]
            }
        }

class PredictionResponse(BaseModel):
    """Response schema"""
    prediction: float = Field(..., description="Model prediction")
    confidence: float = Field(..., description="Prediction confidence")
    model_version: str = Field(..., description="Model version")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def load_model():
    """
    Load model on startup
    """
    global model, scaler
    
    try:
        # Load model
        model_data = joblib.load('model.pkl')
        model = model_data['model']
        scaler = model_data.get('scaler')
        
        logger.info("✓ Model loaded successfully")
    
    except Exception as e:
        logger.error (f"Failed to load model: {e}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Trading Model API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict (request: PredictionRequest):
    """
    Prediction endpoint
    
    Latency target: <50ms
    """
    if model is None:
        raise HTTPException (status_code=503, detail="Model not loaded")
    
    try:
        # Convert to numpy array
        features = np.array (request.features).reshape(1, -1)
        
        # Validate input shape
        if features.shape[1] != 20:  # Expected features
            raise HTTPException(
                status_code=400,
                detail=f"Expected 20 features, got {features.shape[1]}"
            )
        
        # Scale features
        if scaler is not None:
            features = scaler.transform (features)
        
        # Predict
        prediction = model.predict (features)[0]
        
        # Confidence (for tree models)
        try:
            confidence = np.std([tree.predict (features)[0] 
                                for tree in model.estimators_])
        except:
            confidence = 0.0
        
        logger.info (f"Prediction: {prediction:.6f}")
        
        return PredictionResponse(
            prediction=float (prediction),
            confidence=float (confidence),
            model_version="1.0.0"
        )
    
    except Exception as e:
        logger.error (f"Prediction error: {e}")
        raise HTTPException (status_code=500, detail=str (e))


@app.post("/batch_predict")
async def batch_predict (requests: List[PredictionRequest]):
    """
    Batch prediction endpoint
    """
    if model is None:
        raise HTTPException (status_code=503, detail="Model not loaded")
    
    try:
        # Stack features
        features = np.array([req.features for req in requests])
        
        # Scale
        if scaler is not None:
            features = scaler.transform (features)
        
        # Predict
        predictions = model.predict (features)
        
        return {
            "predictions": predictions.tolist(),
            "count": len (predictions)
        }
    
    except Exception as e:
        logger.error (f"Batch prediction error: {e}")
        raise HTTPException (status_code=500, detail=str (e))


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

# Run with: uvicorn api:app --reload
# Test with: curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [0.5, 0.3, -0.2, 0.1, 0.8, 0.2, 0.4, -0.1, 0.6, 0.3, 0.1, 0.2, 0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.4, 0.2]}'
\`\`\`

### Production-Ready API with Error Handling

\`\`\`python
"""
Production-Ready API with Advanced Features
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from typing import Optional
import hashlib

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key authentication
API_KEYS = {
    hashlib.sha256(b"secret_key_1").hexdigest(): "user_1",
    hashlib.sha256(b"secret_key_2").hexdigest(): "user_2"
}

def verify_api_key (x_api_key: str = Header(...)):
    """
    Verify API key
    """
    api_key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    
    if api_key_hash not in API_KEYS:
        raise HTTPException (status_code=401, detail="Invalid API key")
    
    return API_KEYS[api_key_hash]


# Request ID middleware
@app.middleware("http")
async def add_request_id (request, call_next):
    """
    Add request ID for tracing
    """
    request_id = hashlib.md5(
        f"{time.time()}:{request.client.host}".encode()
    ).hexdigest()[:8]
    
    request.state.request_id = request_id
    
    response = await call_next (request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Latency tracking middleware
@app.middleware("http")
async def track_latency (request, call_next):
    """
    Track request latency
    """
    start_time = time.time()
    
    response = await call_next (request)
    
    latency = (time.time() - start_time) * 1000  # ms
    response.headers["X-Process-Time"] = f"{latency:.2f}ms"
    
    logger.info (f"Request {request.state.request_id}: {latency:.2f}ms")
    
    # Alert if slow
    if latency > 100:
        logger.warning (f"Slow request: {latency:.2f}ms")
    
    return response


@app.post("/predict")
async def predict_authenticated(
    request: PredictionRequest,
    user: str = Depends (verify_api_key)
):
    """
    Authenticated prediction endpoint
    """
    logger.info (f"Prediction request from user: {user}")
    
    # Call prediction logic
    # ... (same as before)
    
    return {"prediction": 0.005, "user": user}


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler (request, exc):
    """
    Global exception handler
    """
    logger.error (f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": getattr (request.state, 'request_id', 'unknown')
        }
    )


print("Production API example defined")
print("Features: CORS, API auth, request IDs, latency tracking")
\`\`\`

---

## Docker Containerization

### Dockerfile for Model API

\`\`\`dockerfile
# Dockerfile for Model Serving

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .
COPY model.pkl .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
\`\`\`

### docker-compose for Multi-Service Deployment

\`\`\`yaml
# docker-compose.yml

version: '3.8'

services:
  # Model API
  model-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model.pkl
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
  
  # Redis cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
  
  # Nginx load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - model-api
    restart: unless-stopped

volumes:
  redis_data:
\`\`\`

### Building and Running

\`\`\`python
"""
Docker commands for deployment
"""

commands = {
    "build": "docker build -t trading-model-api:v1.0 .",
    
    "run": "docker run -d -p 8000:8000 --name model-api trading-model-api:v1.0",
    
    "run_with_volume": """
        docker run -d \\
            -p 8000:8000 \\
            -v $(pwd)/models:/app/models \\
            -e MODEL_PATH=/app/models/model.pkl \\
            --name model-api \\
            trading-model-api:v1.0
    """,
    
    "compose_up": "docker-compose up -d",
    
    "compose_down": "docker-compose down",
    
    "compose_scale": "docker-compose up -d --scale model-api=3",
    
    "logs": "docker logs -f model-api",
    
    "shell": "docker exec -it model-api /bin/bash",
    
    "stop": "docker stop model-api && docker rm model-api"
}

print("Docker Commands:")
for name, cmd in commands.items():
    print(f"\\n{name}:")
    print(f"  {cmd}")
\`\`\`

---

## Cloud Deployment

### AWS Deployment with EC2

\`\`\`python
"""
Deploy to AWS EC2
"""

import boto3
from typing import Dict, Any

class EC2Deployer:
    """
    Deploy model API to AWS EC2
    """
    
    def __init__(self, region='us-east-1'):
        self.ec2 = boto3.client('ec2', region_name=region)
        self.region = region
    
    def launch_instance (self, config: Dict[str, Any]):
        """
        Launch EC2 instance
        """
        # User data script (runs on startup)
        user_data = """#!/bin/bash
        # Update system
        apt-get update
        apt-get install -y docker.io
        
        # Start Docker
        systemctl start docker
        systemctl enable docker
        
        # Pull and run model API
        docker pull your-registry/trading-model-api:v1.0
        docker run -d -p 80:8000 your-registry/trading-model-api:v1.0
        """
        
        # Launch instance
        response = self.ec2.run_instances(
            ImageId=config['ami_id'],  # Ubuntu AMI
            InstanceType=config.get('instance_type', 't3.medium'),
            KeyName=config['key_pair'],
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            SecurityGroupIds=[config['security_group_id']],
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': 'model-api-server'},
                    {'Key': 'Environment', 'Value': 'production'}
                ]
            }]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        
        print(f"✓ Launched instance: {instance_id}")
        
        # Wait for instance to be running
        waiter = self.ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])
        
        # Get public IP
        instance_info = self.ec2.describe_instances(
            InstanceIds=[instance_id]
        )
        
        public_ip = instance_info['Reservations'][0]['Instances'][0].get('PublicIpAddress')
        
        print(f"✓ Instance running at: {public_ip}")
        
        return {
            'instance_id': instance_id,
            'public_ip': public_ip
        }
    
    def create_load_balancer (self):
        """
        Create Application Load Balancer
        """
        elb = boto3.client('elbv2', region_name=self.region)
        
        # Create load balancer
        response = elb.create_load_balancer(
            Name='model-api-lb',
            Subnets=['subnet-xxx', 'subnet-yyy'],  # Configure
            SecurityGroups=['sg-xxx'],  # Configure
            Scheme='internet-facing',
            Type='application',
            IpAddressType='ipv4'
        )
        
        lb_arn = response['LoadBalancers'][0]['LoadBalancerArn']
        lb_dns = response['LoadBalancers'][0]['DNSName']
        
        print(f"✓ Load balancer created: {lb_dns}")
        
        return lb_arn, lb_dns


# Example usage
# deployer = EC2Deployer (region='us-east-1')
# instance = deployer.launch_instance({
#     'ami_id': 'ami-0c55b159cbfafe1f0',
#     'instance_type': 't3.medium',
#     'key_pair': 'my-key-pair',
#     'security_group_id': 'sg-xxx'
# })
\`\`\`

### AWS Lambda for Serverless

\`\`\`python
"""
Deploy model to AWS Lambda (serverless)
"""

import json
import boto3
import joblib

# Lambda handler
def lambda_handler (event, context):
    """
    AWS Lambda handler for predictions
    
    Pros:
    - No server management
    - Auto-scaling
    - Pay per request
    
    Cons:
    - Cold start latency
    - Limited to 15 min execution
    - 10GB memory limit
    """
    try:
        # Parse request
        body = json.loads (event['body'])
        features = body['features']
        
        # Load model (cached after first invocation)
        global model
        if 'model' not in globals():
            # Load from S3 or /tmp
            model = joblib.load('/tmp/model.pkl')
        
        # Predict
        import numpy as np
        features_array = np.array (features).reshape(1, -1)
        prediction = model.predict (features_array)[0]
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction': float (prediction)
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str (e)})
        }


# Deploy script
class LambdaDeployer:
    """
    Deploy to AWS Lambda
    """
    
    def __init__(self):
        self.lambda_client = boto3.client('lambda')
        self.s3_client = boto3.client('s3')
    
    def upload_model_to_s3(self, model_path, bucket, key):
        """
        Upload model to S3
        """
        self.s3_client.upload_file (model_path, bucket, key)
        print(f"✓ Uploaded model to s3://{bucket}/{key}")
    
    def create_lambda_function (self, function_name, role_arn, bucket, model_key):
        """
        Create Lambda function
        """
        # Create deployment package
        # (In practice: zip your code + dependencies)
        
        response = self.lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.10',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={
                'S3Bucket': bucket,
                'S3Key': 'lambda_deployment.zip'
            },
            Timeout=30,
            MemorySize=1024,
            Environment={
                'Variables': {
                    'MODEL_BUCKET': bucket,
                    'MODEL_KEY': model_key
                }
            }
        )
        
        function_arn = response['FunctionArn']
        
        print(f"✓ Lambda function created: {function_arn}")
        
        return function_arn


# Example
# deployer = LambdaDeployer()
# deployer.upload_model_to_s3('model.pkl', 'my-bucket', 'models/model.pkl')
# deployer.create_lambda_function('trading-model', 'arn:aws:iam::xxx', 'my-bucket', 'models/model.pkl')
\`\`\`

---

## Model Serving Frameworks

### TensorFlow Serving

\`\`\`python
"""
Deploy with TensorFlow Serving
"""

import tensorflow as tf
import numpy as np

# Save model in SavedModel format
def save_for_tf_serving (model, export_path):
    """
    Save model for TensorFlow Serving
    """
    # Convert to TF model if needed
    # ... conversion logic ...
    
    # Save
    tf.saved_model.save (model, export_path)
    
    print(f"✓ Model saved to {export_path}")
    print(f"\\nServe with:")
    print(f"docker run -p 8501:8501 \\")
    print(f"  -v {export_path}:/models/model \\")
    print(f"  -e MODEL_NAME=model \\")
    print(f"  tensorflow/serving")


# Client code
import requests

def predict_with_tf_serving (features, url="http://localhost:8501/v1/models/model:predict"):
    """
    Call TensorFlow Serving API
    """
    payload = {
        "instances": [features]
    }
    
    response = requests.post (url, json=payload)
    
    if response.status_code == 200:
        predictions = response.json()['predictions']
        return predictions[0]
    else:
        raise Exception (f"Prediction failed: {response.text}")


print("TensorFlow Serving example defined")
\`\`\`

### TorchServe

\`\`\`python
"""
Deploy PyTorch with TorchServe
"""

# Model archiver
model_archiver_cmd = """
torch-model-archiver \\
    --model-name trading_model \\
    --version 1.0 \\
    --model-file model.py \\
    --serialized-file model.pth \\
    --handler custom_handler.py \\
    --export-path model_store
"""

# Start TorchServe
torchserve_cmd = """
torchserve \\
    --start \\
    --model-store model_store \\
    --models trading_model=trading_model.mar
"""

# Custom handler
handler_code = ''
import torch
from ts.torch_handler.base_handler import BaseHandler

class TradingHandler(BaseHandler):
    """
    Custom handler for trading model
    """
    
    def preprocess (self, data):
        """
        Preprocess input
        """
        # Parse JSON
        input_data = data[0].get("body")
        features = input_data.get("features")
        
        # Convert to tensor
        tensor = torch.tensor (features, dtype=torch.float32)
        
        return tensor
    
    def inference (self, tensor):
        """
        Run inference
        """
        self.model.eval()
        
        with torch.no_grad():
            output = self.model (tensor)
        
        return output
    
    def postprocess (self, output):
        """
        Format output
        """
        prediction = output.item()
        
        return [{
            "prediction": prediction
        }]
''

print("TorchServe deployment commands:")
print(f"\\n1. Archive model:\\n{model_archiver_cmd}")
print(f"\\n2. Start server:\\n{torchserve_cmd}")
\`\`\`

---

## Canary Deployments

### Blue-Green Deployment

\`\`\`python
"""
Blue-Green Deployment Pattern
"""

class BlueGreenDeployment:
    """
    Blue-Green deployment for zero-downtime updates
    
    Process:
    1. Deploy new version (green) alongside old (blue)
    2. Test green environment
    3. Switch traffic to green
    4. Keep blue as backup
    5. Decommission blue if green stable
    """
    
    def __init__(self):
        self.environments = {
            'blue': {
                'version': '1.0',
                'url': 'http://model-v1.example.com',
                'active': True
            },
            'green': {
                'version': '1.1',
                'url': 'http://model-v2.example.com',
                'active': False
            }
        }
        self.traffic_split = {'blue': 100, 'green': 0}
    
    def deploy_green (self, new_version):
        """
        Deploy new version to green environment
        """
        print(f"Deploying version {new_version} to green...")
        
        # Deploy green (new version)
        self.environments['green']['version'] = new_version
        
        print("✓ Green deployed")
    
    def test_green (self):
        """
        Test green environment
        """
        print("Testing green environment...")
        
        # Run smoke tests
        # ... test logic ...
        
        print("✓ Green tests passed")
        
        return True
    
    def canary_release (self, green_percentage=10):
        """
        Gradual traffic shift (canary)
        """
        print(f"Shifting {green_percentage}% traffic to green...")
        
        self.traffic_split['blue'] = 100 - green_percentage
        self.traffic_split['green'] = green_percentage
        
        print(f"✓ Traffic split: {self.traffic_split}")
    
    def full_cutover (self):
        """
        Full cutover to green
        """
        print("Full cutover to green...")
        
        self.traffic_split['blue'] = 0
        self.traffic_split['green'] = 100
        
        self.environments['blue']['active'] = False
        self.environments['green']['active'] = True
        
        print("✓ Cutover complete")
    
    def rollback (self):
        """
        Rollback to blue
        """
        print("Rolling back to blue...")
        
        self.traffic_split['blue'] = 100
        self.traffic_split['green'] = 0
        
        self.environments['blue']['active'] = True
        self.environments['green']['active'] = False
        
        print("✓ Rollback complete")
    
    def swap_colors (self):
        """
        Swap blue/green labels
        """
        self.environments['blue'], self.environments['green'] = \\
            self.environments['green'], self.environments['blue']
        
        print("✓ Colors swapped (green is now blue)")


# Example deployment flow
deployment = BlueGreenDeployment()

# 1. Deploy new version
deployment.deploy_green('1.1')

# 2. Test
if deployment.test_green():
    # 3. Canary (10% traffic)
    deployment.canary_release(10)
    
    # 4. Monitor metrics...
    # If metrics good:
    deployment.canary_release(50)  # 50%
    
    # 5. Full cutover
    deployment.full_cutover()
    
    # 6. Swap colors for next deployment
    deployment.swap_colors()
else:
    print("Tests failed, keeping blue active")
\`\`\`

---

## Batch vs Real-Time Serving

### Batch Serving

\`\`\`python
"""
Batch Prediction Service
"""

import pandas as pd
from typing import List
import time

class BatchPredictionService:
    """
    Batch prediction for non-real-time use cases
    
    Use when:
    - Latency not critical (minutes/hours okay)
    - Large volumes of data
    - Cost optimization important
    """
    
    def __init__(self, model, batch_size=1000):
        self.model = model
        self.batch_size = batch_size
    
    def predict_batch (self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on batch of data
        """
        print(f"Processing {len (data)} records in batches of {self.batch_size}...")
        
        predictions = []
        
        # Process in batches
        for i in range(0, len (data), self.batch_size):
            batch = data.iloc[i:i+self.batch_size]
            
            # Predict
            batch_preds = self.model.predict (batch)
            predictions.extend (batch_preds)
            
            print(f"  Processed {min (i+self.batch_size, len (data))}/{len (data)}")
        
        # Add predictions to dataframe
        data['prediction'] = predictions
        
        return data
    
    def daily_batch_job (self, data_path, output_path):
        """
        Daily batch prediction job
        """
        print(f"\\n=== Daily Batch Job ===")
        start = time.time()
        
        # Load data
        data = pd.read_csv (data_path)
        print(f"Loaded {len (data)} records")
        
        # Predict
        results = self.predict_batch (data)
        
        # Save
        results.to_csv (output_path, index=False)
        
        duration = time.time() - start
        print(f"\\n✓ Batch job complete: {duration:.2f}s")
        print(f"  Throughput: {len (data)/duration:.0f} records/sec")


# Example
from sklearn.ensemble import RandomForestRegressor
import numpy as np

model = RandomForestRegressor (n_estimators=100)
X_train = np.random.randn(1000, 20)
y_train = np.random.randn(1000)
model.fit(X_train, y_train)

batch_service = BatchPredictionService (model, batch_size=500)

# Simulate daily job
# batch_service.daily_batch_job('data.csv', 'predictions.csv')
\`\`\`

---

## Key Takeaways

1. **REST API**: FastAPI for simple, fast model serving
2. **Docker**: Containerize for consistency and portability
3. **Cloud**: Deploy to AWS EC2, Lambda, or managed services
4. **Serving Frameworks**: TensorFlow Serving, TorchServe for production
5. **Blue-Green**: Zero-downtime deployments with canary releases
6. **Batch vs Real-Time**: Choose based on latency requirements

**Trading-Specific**:
- Real-time serving for HFT (<10ms)
- Batch for daily strategy signals
- Canary deploy to paper trading first
- Monitor prediction latency closely

**Next Steps**: With deployment handled, we'll cover model monitoring to ensure production health.
`,
};

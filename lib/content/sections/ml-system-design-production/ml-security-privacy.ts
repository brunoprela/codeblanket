export const mlSecurityPrivacy = {
  title: 'ML Security & Privacy',
  id: 'ml-security-privacy',
  content: `
# ML Security & Privacy

## Introduction

**"Security is not a product, but a process."** - Bruce Schneier

ML systems face unique security and privacy challenges beyond traditional software. Models can be stolen, poisoned, or tricked into making wrong predictions. User data must be protected while still enabling model training.

**ML-Specific Threats**:
- **Model theft**: Stealing trained models via API
- **Data poisoning**: Corrupting training data
- **Adversarial attacks**: Inputs designed to fool models
- **Privacy leaks**: Models memorizing sensitive data
- **Backdoor attacks**: Hidden malicious behavior

This section covers securing ML systems and protecting user privacy in production.

### Security & Privacy Landscape

\`\`\`
Data Collection → Training → Deployment → Inference
      ↓             ↓           ↓            ↓
  Privacy      Poisoning    Theft      Adversarial
  (GDPR)       Defense      Defense    Examples
\`\`\`

By the end of this section, you'll understand:
- Adversarial attacks and defenses
- Model extraction attacks
- Data privacy techniques (differential privacy)
- Federated learning
- Secure multi-party computation

---

## Adversarial Attacks

### Fast Gradient Sign Method (FGSM)

\`\`\`python
"""
Adversarial Attacks on ML Models
"""

import torch
import torch.nn as nn
import numpy as np

class AdversarialAttacks:
    """
    Generate adversarial examples
    """
    
    def __init__(self, model):
        self.model = model
    
    def fgsm_attack(self, X, y, epsilon=0.1):
        """
        Fast Gradient Sign Method
        
        Creates adversarial examples by:
        1. Computing loss gradient w.r.t. input
        2. Adding small perturbation in gradient direction
        
        Args:
            X: Input data
            y: True labels
            epsilon: Perturbation magnitude
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).requires_grad_(True)
        y_tensor = torch.LongTensor(y)
        
        # Forward pass
        self.model.eval()
        output = self.model(X_tensor)
        
        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y_tensor)
        
        # Backward pass to get gradient
        self.model.zero_grad()
        loss.backward()
        
        # Create adversarial example
        # Add epsilon * sign(gradient) to input
        X_grad = X_tensor.grad.data
        X_adv = X_tensor + epsilon * X_grad.sign()
        
        # Clip to valid range
        X_adv = torch.clamp(X_adv, X_tensor.min(), X_tensor.max())
        
        return X_adv.detach().numpy()
    
    def evaluate_robustness(self, X_test, y_test, epsilons=[0.0, 0.05, 0.1, 0.2]):
        """
        Evaluate model robustness to adversarial attacks
        """
        print("\\n=== Adversarial Robustness Evaluation ===\\n")
        
        for eps in epsilons:
            if eps == 0:
                # Clean accuracy
                X_adv = X_test
            else:
                # Generate adversarial examples
                X_adv = self.fgsm_attack(X_test, y_test, epsilon=eps)
            
            # Evaluate
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_adv)
                output = self.model(X_tensor)
                predictions = output.argmax(dim=1).numpy()
                accuracy = (predictions == y_test).mean()
            
            print(f"Epsilon: {eps:.2f}, Accuracy: {accuracy*100:.2f}%")
            
            if eps == 0:
                clean_acc = accuracy
            else:
                drop = (clean_acc - accuracy) * 100
                print(f"  → Accuracy drop: {drop:.2f}%")


# Example: Create simple model
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=10, num_classes=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)


# Train simple model
model = SimpleClassifier()

# Generate sample data
np.random.seed(42)
X_train = np.random.randn(1000, 10)
y_train = np.random.randint(0, 3, 1000)
X_test = np.random.randn(100, 10)
y_test = np.random.randint(0, 3, 100)

# Train model (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    
    loss.backward()
    optimizer.step()

# Test adversarial robustness
attacker = AdversarialAttacks(model)
attacker.evaluate_robustness(X_test, y_test)
\`\`\`

### Adversarial Defense

\`\`\`python
"""
Defending Against Adversarial Attacks
"""

class AdversarialDefense:
    """
    Techniques to defend against adversarial attacks
    """
    
    def adversarial_training(self, model, X_train, y_train, epochs=10, epsilon=0.1):
        """
        Adversarial Training: Train on mix of clean and adversarial examples
        
        Most effective defense technique
        """
        print("\\n=== Adversarial Training ===\\n")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        attacker = AdversarialAttacks(model)
        
        for epoch in range(epochs):
            model.train()
            
            # Clean data
            X_tensor = torch.FloatTensor(X_train)
            y_tensor = torch.LongTensor(y_train)
            
            # Generate adversarial examples
            X_adv = attacker.fgsm_attack(X_train, y_train, epsilon=epsilon)
            X_adv_tensor = torch.FloatTensor(X_adv)
            
            # Train on both clean and adversarial
            for X, y in [(X_tensor, y_tensor), (X_adv_tensor, y_tensor)]:
                optimizer.zero_grad()
                
                output = model(X)
                loss = criterion(output, y)
                
                loss.backward()
                optimizer.step()
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        print("\\n✓ Model trained on adversarial examples")
        print("  Should be more robust to attacks")
        
        return model
    
    def input_transformation(self, X, method='jpeg_compression'):
        """
        Transform inputs to remove adversarial perturbations
        
        Methods:
        - JPEG compression
        - Bit depth reduction
        - Smoothing filters
        """
        if method == 'jpeg_compression':
            # Simulate JPEG compression (removes high-frequency noise)
            return self._jpeg_compression(X)
        
        elif method == 'bit_reduction':
            # Reduce bit depth (quantization)
            return np.round(X * 10) / 10
        
        else:
            return X
    
    def _jpeg_compression(self, X):
        """Simulate JPEG compression"""
        # Simplified: just add noise and smooth
        compressed = X + np.random.randn(*X.shape) * 0.01
        return compressed
    
    def ensemble_defense(self, models, X):
        """
        Ensemble of models for robustness
        
        Adversarial examples often don't transfer across models
        """
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                output = model(X_tensor)
                pred = output.argmax(dim=1).numpy()
                predictions.append(pred)
        
        # Majority vote
        predictions = np.array(predictions)
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
        
        return ensemble_pred


# Example: Train robust model
defense = AdversarialDefense()
robust_model = SimpleClassifier()

# Train with adversarial examples
robust_model = defense.adversarial_training(
    robust_model, X_train, y_train, epochs=10, epsilon=0.1
)

# Compare robustness
print("\\n=== Comparing Standard vs Robust Model ===")

print("\\nStandard Model:")
attacker.model = model
attacker.evaluate_robustness(X_test, y_test, epsilons=[0.0, 0.1, 0.2])

print("\\nRobust Model (Adversarial Training):")
attacker.model = robust_model
attacker.evaluate_robustness(X_test, y_test, epsilons=[0.0, 0.1, 0.2])
\`\`\`

---

## Model Extraction Attacks

### Stealing Models via API

\`\`\`python
"""
Model Extraction (Model Stealing) Attack
"""

class ModelExtractionAttack:
    """
    Steal a model by querying its API
    """
    
    def __init__(self, victim_model):
        self.victim_model = victim_model
        self.query_count = 0
    
    def query_victim(self, X):
        """
        Query victim model API
        """
        self.query_count += len(X)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            output = self.victim_model(X_tensor)
            # Return probabilities
            probs = torch.softmax(output, dim=1).numpy()
        
        return probs
    
    def steal_model(self, input_dim=10, num_classes=3, num_queries=1000):
        """
        Steal model by querying and training substitute
        """
        print(f"\\n=== Model Extraction Attack ===\\n")
        
        # Generate synthetic queries
        X_queries = np.random.randn(num_queries, input_dim)
        
        # Query victim model
        print(f"Querying victim model {num_queries} times...")
        y_probs = self.query_victim(X_queries)
        y_labels = y_probs.argmax(axis=1)
        
        # Train substitute model
        print("Training substitute model...")
        substitute_model = SimpleClassifier(input_dim, num_classes)
        
        optimizer = torch.optim.Adam(substitute_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        X_tensor = torch.FloatTensor(X_queries)
        y_tensor = torch.LongTensor(y_labels)
        
        for epoch in range(50):
            substitute_model.train()
            optimizer.zero_grad()
            
            output = substitute_model(X_tensor)
            loss = criterion(output, y_tensor)
            
            loss.backward()
            optimizer.step()
        
        print(f"\\n✓ Substitute model trained with {self.query_count} queries")
        
        return substitute_model
    
    def evaluate_extraction(self, substitute_model, X_test, y_test):
        """
        Evaluate how well substitute mimics victim
        """
        # Victim predictions
        victim_probs = self.query_victim(X_test)
        victim_preds = victim_probs.argmax(axis=1)
        
        # Substitute predictions
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test)
            sub_output = substitute_model(X_tensor)
            sub_preds = sub_output.argmax(dim=1).numpy()
        
        # Agreement rate
        agreement = (victim_preds == sub_preds).mean()
        
        # Both accuracies
        victim_acc = (victim_preds == y_test).mean()
        sub_acc = (sub_preds == y_test).mean()
        
        print(f"\\n=== Extraction Success ===")
        print(f"Victim accuracy: {victim_acc*100:.2f}%")
        print(f"Substitute accuracy: {sub_acc*100:.2f}%")
        print(f"Agreement: {agreement*100:.2f}%")
        print(f"\\n{'✓' if agreement > 0.8 else '✗'} Model successfully stolen" if agreement > 0.8 else "✗ Model extraction incomplete")


# Example: Steal model
victim = model  # Our trained model
attacker = ModelExtractionAttack(victim)

# Steal model
stolen_model = attacker.steal_model(input_dim=10, num_classes=3, num_queries=1000)

# Evaluate
attacker.evaluate_extraction(stolen_model, X_test, y_test)

print("\\n⚠️  Defense: Rate limiting, query monitoring, adding noise to outputs")
\`\`\`

### Defenses Against Model Extraction

\`\`\`python
"""
Defending Against Model Extraction
"""

class ModelExtractionDefense:
    """
    Techniques to prevent model stealing
    """
    
    def __init__(self, model):
        self.model = model
        self.query_log = []
    
    def rate_limiting(self, user_id, max_queries_per_hour=1000):
        """
        Limit queries per user
        """
        from datetime import datetime, timedelta
        
        # Count recent queries
        cutoff = datetime.now() - timedelta(hours=1)
        recent_queries = [
            q for q in self.query_log
            if q['user_id'] == user_id and q['timestamp'] > cutoff
        ]
        
        if len(recent_queries) >= max_queries_per_hour:
            raise Exception(f"Rate limit exceeded: {len(recent_queries)} queries in last hour")
        
        return True
    
    def add_prediction_noise(self, predictions, epsilon=0.01):
        """
        Add noise to predictions to prevent exact extraction
        """
        noise = np.random.randn(*predictions.shape) * epsilon
        noisy_predictions = predictions + noise
        
        # Renormalize probabilities
        noisy_predictions = np.maximum(noisy_predictions, 0)
        noisy_predictions = noisy_predictions / noisy_predictions.sum(axis=1, keepdims=True)
        
        return noisy_predictions
    
    def detect_extraction_attempt(self, queries, threshold=100):
        """
        Detect potential model extraction
        
        Indicators:
        - High query volume
        - Random-looking inputs
        - Systematic exploration of input space
        """
        if len(queries) > threshold:
            # Check if inputs are random/systematic
            queries_array = np.array(queries)
            
            # Random inputs have high entropy
            mean_std = np.mean([np.std(q) for q in queries_array])
            
            if mean_std > 0.8:  # Threshold for randomness
                print("\\n🚨 ALERT: Potential model extraction detected!")
                print(f"   - {len(queries)} queries")
                print(f"   - High input entropy (random exploration)")
                return True
        
        return False
    
    def protected_predict(self, X, user_id):
        """
        Prediction with protections
        """
        # 1. Rate limiting
        self.rate_limiting(user_id)
        
        # 2. Log query
        self.query_log.append({
            'user_id': user_id,
            'timestamp': datetime.now(),
            'input_shape': X.shape
        })
        
        # 3. Make prediction
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            output = self.model(X_tensor)
            probs = torch.softmax(output, dim=1).numpy()
        
        # 4. Add noise
        probs = self.add_prediction_noise(probs, epsilon=0.01)
        
        # 5. Check for extraction attempt
        recent_queries = [q for q in self.query_log if q['user_id'] == user_id]
        self.detect_extraction_attempt([X], threshold=100)
        
        return probs


from datetime import datetime

# Example: Protected API
defender = ModelExtractionDefense(model)

# Normal usage
result = defender.protected_predict(X_test[:5], user_id='user_123')
print("\\n✓ Normal query successful")

# Attempted extraction (many queries)
print("\\nAttempting extraction with 200 queries...")
for i in range(200):
    try:
        X_query = np.random.randn(1, 10)
        result = defender.protected_predict(X_query, user_id='attacker_456')
    except Exception as e:
        print(f"\\n❌ Blocked: {e}")
        break
\`\`\`

---

## Differential Privacy

### Private Machine Learning

\`\`\`python
"""
Differential Privacy for ML
"""

class DifferentialPrivacy:
    """
    Train models with differential privacy guarantees
    
    Idea: Add noise so individual data points can't be inferred
    """
    
    def __init__(self, epsilon=1.0, delta=1e-5):
        """
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Probability of privacy violation
        """
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise_to_gradients(self, gradients, sensitivity, batch_size):
        """
        Add Gaussian noise to gradients (DP-SGD)
        
        Args:
            sensitivity: L2 sensitivity (gradient clipping bound)
        """
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        noise = np.random.normal(0, noise_scale, gradients.shape)
        
        noisy_gradients = gradients + noise
        
        return noisy_gradients
    
    def clip_gradients(self, gradients, max_norm=1.0):
        """
        Clip gradients to bound sensitivity
        """
        grad_norm = np.linalg.norm(gradients)
        
        if grad_norm > max_norm:
            gradients = gradients * (max_norm / grad_norm)
        
        return gradients
    
    def dp_sgd_step(self, model, X_batch, y_batch, optimizer, criterion, max_norm=1.0):
        """
        One DP-SGD training step
        """
        optimizer.zero_grad()
        
        # Forward pass
        X_tensor = torch.FloatTensor(X_batch)
        y_tensor = torch.LongTensor(y_batch)
        
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        
        # Backward pass
        loss.backward()
        
        # Clip and add noise to gradients
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    # Clip
                    grad_numpy = param.grad.numpy()
                    clipped_grad = self.clip_gradients(grad_numpy, max_norm)
                    
                    # Add noise
                    noisy_grad = self.add_noise_to_gradients(
                        clipped_grad,
                        sensitivity=max_norm,
                        batch_size=len(X_batch)
                    )
                    
                    param.grad = torch.FloatTensor(noisy_grad)
        
        # Update parameters
        optimizer.step()
        
        return loss.item()
    
    def train_private_model(self, model, X_train, y_train, epochs=10, batch_size=32):
        """
        Train model with differential privacy
        """
        print(f"\\n=== Training with Differential Privacy ===")
        print(f"Privacy budget: ε={self.epsilon}, δ={self.delta}\\n")
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        n_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                loss = self.dp_sgd_step(
                    model, X_batch, y_batch,
                    optimizer, criterion,
                    max_norm=1.0
                )
                
                epoch_loss += loss
            
            avg_loss = epoch_loss / n_batches
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print(f"\\n✓ Model trained with (ε={self.epsilon}, δ={self.delta})-differential privacy")
        
        return model


# Example: Train private model
dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

private_model = SimpleClassifier()
private_model = dp.train_private_model(private_model, X_train, y_train, epochs=10)

# Evaluate
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_test)
    output = private_model(X_tensor)
    predictions = output.argmax(dim=1).numpy()
    accuracy = (predictions == y_test).mean()

print(f"\\nPrivate model accuracy: {accuracy*100:.2f}%")
print("(Typically lower than non-private model due to noise)")
\`\`\`

---

## Federated Learning

### Training Without Centralizing Data

\`\`\`python
"""
Federated Learning
"""

class FederatedLearning:
    """
    Train models across decentralized data
    
    Applications:
    - Mobile devices (Google Keyboard)
    - Hospitals (medical data)
    - Financial institutions (trading data)
    """
    
    def __init__(self, num_clients=5):
        self.num_clients = num_clients
        self.global_model = None
    
    def initialize_global_model(self, input_dim=10, num_classes=3):
        """
        Initialize global model
        """
        self.global_model = SimpleClassifier(input_dim, num_classes)
        print(f"✓ Initialized global model")
    
    def client_update(self, client_model, X_local, y_local, epochs=5):
        """
        Train on local client data
        """
        optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            client_model.train()
            optimizer.zero_grad()
            
            X_tensor = torch.FloatTensor(X_local)
            y_tensor = torch.LongTensor(y_local)
            
            output = client_model(X_tensor)
            loss = criterion(output, y_tensor)
            
            loss.backward()
            optimizer.step()
        
        return client_model
    
    def federated_averaging(self, client_models):
        """
        FedAvg: Average client model parameters
        """
        # Initialize averaged parameters
        global_dict = self.global_model.state_dict()
        
        for key in global_dict.keys():
            # Average parameters from all clients
            global_dict[key] = torch.stack([
                client.state_dict()[key].float()
                for client in client_models
            ]).mean(dim=0)
        
        # Update global model
        self.global_model.load_state_dict(global_dict)
    
    def train_federated(self, client_datasets, rounds=10):
        """
        Federated learning training
        
        Args:
            client_datasets: List of (X, y) tuples for each client
        """
        print(f"\\n=== Federated Learning ===")
        print(f"Clients: {len(client_datasets)}, Rounds: {rounds}\\n")
        
        for round_num in range(rounds):
            print(f"Round {round_num+1}/{rounds}")
            
            # Each client trains locally
            client_models = []
            
            for i, (X_local, y_local) in enumerate(client_datasets):
                # Copy global model to client
                client_model = SimpleClassifier()
                client_model.load_state_dict(self.global_model.state_dict())
                
                # Train locally
                client_model = self.client_update(
                    client_model, X_local, y_local, epochs=5
                )
                
                client_models.append(client_model)
            
            # Aggregate models
            self.federated_averaging(client_models)
            
            print(f"  ✓ Aggregated {len(client_models)} client updates")
        
        print(f"\\n✓ Federated training complete")
        
        return self.global_model


# Example: Federated learning
fl = FederatedLearning(num_clients=5)
fl.initialize_global_model(input_dim=10, num_classes=3)

# Simulate client datasets (non-IID)
client_datasets = []
for i in range(5):
    # Each client has different data distribution
    X_client = np.random.randn(200, 10) + i * 0.5  # Different means
    y_client = np.random.randint(0, 3, 200)
    client_datasets.append((X_client, y_client))

# Train federated
global_model = fl.train_federated(client_datasets, rounds=10)

# Evaluate global model
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_test)
    output = global_model(X_tensor)
    predictions = output.argmax(dim=1).numpy()
    accuracy = (predictions == y_test).mean()

print(f"\\nGlobal model accuracy: {accuracy*100:.2f}%")
print("\\n✓ Privacy: Raw data never left clients")
\`\`\`

---

## Key Takeaways

1. **Adversarial Attacks**: FGSM, PGD - small perturbations fool models
   - Defense: Adversarial training most effective
2. **Model Extraction**: Attackers can steal models via API queries
   - Defense: Rate limiting, add noise, detect patterns
3. **Differential Privacy**: Add noise to protect individual data
   - Trade-off: Privacy vs accuracy
4. **Federated Learning**: Train without centralizing data
   - Good for sensitive data (medical, financial)

**Security Best Practices**:
- ✅ Monitor API usage for anomalies
- ✅ Rate limit predictions per user
- ✅ Add small noise to outputs
- ✅ Use adversarial training for robustness
- ✅ Implement differential privacy for sensitive data

**Trading-Specific**:
- Protect proprietary trading models from extraction
- Use federated learning across multiple funds
- Monitor for adversarial market manipulation
- Privacy-preserving portfolio optimization

**Next Steps**: With security covered, we'll examine real-world ML system case studies.
`,
};

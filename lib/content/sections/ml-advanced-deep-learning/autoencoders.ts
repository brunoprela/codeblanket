/**
 * Autoencoders Content
 */

export const autoencodersSection = {
  id: 'autoencoders',
  title: 'Autoencoders',
  content: `
# Autoencoders

## Introduction

**Autoencoders** are unsupervised neural networks that learn to **compress** (encode) and **reconstruct** (decode) data. The goal is to learn a compact representation that captures the most important features.

**Architecture**:
- **Encoder**: Compresses input x into latent representation z (bottleneck)
- **Decoder**: Reconstructs input from latent representation: x̂ ≈ x
- **Training objective**: Minimize reconstruction error ||x - x̂||²

**Key insight**: The bottleneck forces the network to learn **only the most important features** (dimensionality reduction).

**Applications**:
- Dimensionality reduction (alternative to PCA)
- Denoising (remove noise from images/audio)
- Anomaly detection (high reconstruction error = anomaly)
- Generative modeling (sample from latent space)
- Feature learning (use encoder for downstream tasks)

---

## Basic Autoencoder

### Architecture

**Encoder**: x → h₁ → h₂ → ... → z (latent code)

**Decoder**: z → h₂' → h₁' → ... → x̂ (reconstruction)

**Symmetric structure** (typical but not required):
- Encoder: 784 → 512 → 256 → 128 → 64 (latent)
- Decoder: 64 → 128 → 256 → 512 → 784 (mirror)

### Mathematical Formulation

**Encoder**:
\`\`\`
z = f_encoder (x; θ_enc)
\`\`\`

**Decoder**:
\`\`\`
x̂ = f_decoder (z; θ_dec)
\`\`\`

**Loss function** (reconstruction loss):
\`\`\`
L(x, x̂) = ||x - x̂||² = Σᵢ (xᵢ - x̂ᵢ)²
\`\`\`

For binary data (e.g., MNIST), use **binary cross-entropy**:
\`\`\`
L(x, x̂) = -Σᵢ [xᵢ log (x̂ᵢ) + (1-xᵢ) log(1-x̂ᵢ)]
\`\`\`

### Implementation

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Autoencoder (nn.Module):
    def __init__(self, input_dim=784, latent_dim=64):
        """
        Simple autoencoder with symmetric encoder-decoder.
        
        Args:
            input_dim: Input dimension (e.g., 28×28 = 784 for MNIST)
            latent_dim: Latent space dimension (bottleneck)
        """
        super().__init__()
        
        # Encoder: compress input to latent space
        self.encoder = nn.Sequential(
            nn.Linear (input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            # No activation on latent code (allows full range)
        )
        
        # Decoder: reconstruct input from latent space
        self.decoder = nn.Sequential(
            nn.Linear (latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # For images in [0, 1]
        )
    
    def forward (self, x):
        """
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            x_recon: (batch_size, input_dim) - Reconstructed input
            z: (batch_size, latent_dim) - Latent representation
        """
        z = self.encoder (x)
        x_recon = self.decoder (z)
        return x_recon, z


# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda (lambda x: x.view(-1))  # Flatten 28×28 to 784
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader (train_dataset, batch_size=128, shuffle=True)

# Create model
model = Autoencoder (input_dim=784, latent_dim=64)
print(f"Latent dimension: 64 (12× compression from 784)")

# Training setup
optimizer = torch.optim.Adam (model.parameters(), lr=1e-3)
criterion = nn.MSELoss()  # Reconstruction loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to (device)

# Training loop
num_epochs = 20
model.train()

for epoch in range (num_epochs):
    total_loss = 0.0
    
    for data, _ in train_loader:  # Ignore labels (unsupervised)
        data = data.to (device)
        
        # Forward pass
        x_recon, z = model (data)
        loss = criterion (x_recon, data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len (train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")

print("Training complete!")

# Visualize reconstructions
model.eval()
with torch.no_grad():
    # Get test samples
    test_data = next (iter (train_loader))[0][:8].to (device)
    reconstructions, latents = model (test_data)
    
    # Plot original vs. reconstruction
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        # Original
        axes[0, i].imshow (test_data[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)
        
        # Reconstruction
        axes[1, i].imshow (reconstructions[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('autoencoder_reconstructions.png', dpi=150, bbox_inches='tight')
    print("Saved reconstructions to autoencoder_reconstructions.png")
\`\`\`

---

## Convolutional Autoencoder

For image data, **convolutional** layers preserve spatial structure (better than fully connected).

**Architecture**:
- **Encoder**: Convolutions + pooling (or strided convolutions) to downsample
- **Decoder**: Transposed convolutions (upsampling) to reconstruct

### Implementation

\`\`\`python
class ConvAutoencoder (nn.Module):
    def __init__(self, latent_dim=128):
        """
        Convolutional autoencoder for 28×28 grayscale images.
        
        Args:
            latent_dim: Latent space dimension
        """
        super().__init__()
        
        # Encoder: 28×28 → 14×14 → 7×7 → latent
        self.encoder = nn.Sequential(
            # Input: (1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # → (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # → (64 × 7 × 7 = 3136)
            nn.Linear(64 * 7 * 7, latent_dim),  # → (latent_dim)
        )
        
        # Decoder: latent → 7×7 → 14×14 → 28×28
        self.decoder = nn.Sequential(
            # Input: (latent_dim)
            nn.Linear (latent_dim, 64 * 7 * 7),  # → (3136)
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),  # → (64, 7, 7)
            # Transposed convolution (upsampling)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # → (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # → (1, 28, 28)
            nn.Sigmoid()
        )
    
    def forward (self, x):
        """
        Args:
            x: (batch_size, 1, 28, 28)
        
        Returns:
            x_recon: (batch_size, 1, 28, 28)
            z: (batch_size, latent_dim)
        """
        z = self.encoder (x)
        x_recon = self.decoder (z)
        return x_recon, z


# Load data (keep image shape)
transform = transforms.Compose([
    transforms.ToTensor(),
    # No flattening - keep (1, 28, 28) shape
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader (train_dataset, batch_size=128, shuffle=True)

# Train convolutional autoencoder
model = ConvAutoencoder (latent_dim=128)
optimizer = torch.optim.Adam (model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ... training loop same as before ...

print("Convolutional autoencoder trained!")
\`\`\`

**Advantages of convolutional autoencoders**:
- Preserve spatial structure
- Fewer parameters (weight sharing)
- Better reconstructions for images
- Translation invariance

---

## Denoising Autoencoder (DAE)

**Idea**: Train to reconstruct **clean** data from **noisy** input.

**Training**:
1. Add noise to input: x̃ = x + ε (where ε ~ N(0, σ²))
2. Encoder: z = f (x̃)
3. Decoder: x̂ = g (z)
4. Loss: ||x - x̂||² (reconstruct ORIGINAL, not noisy version)

**Benefits**:
- Learns robust features (not just memorizing)
- Forces model to capture underlying structure
- Prevents overfitting
- Useful for actual denoising tasks

### Implementation

\`\`\`python
def add_noise (x, noise_factor=0.3):
    """Add Gaussian noise to input"""
    noisy_x = x + noise_factor * torch.randn_like (x)
    noisy_x = torch.clamp (noisy_x, 0., 1.)  # Keep in [0, 1]
    return noisy_x


# Training loop for denoising autoencoder
model = ConvAutoencoder (latent_dim=128)
optimizer = torch.optim.Adam (model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to (device)
model.train()

for epoch in range(20):
    total_loss = 0.0
    
    for data, _ in train_loader:
        data = data.to (device)
        
        # Add noise to input
        noisy_data = add_noise (data, noise_factor=0.3)
        
        # Forward pass: reconstruct CLEAN from NOISY
        x_recon, _ = model (noisy_data)
        loss = criterion (x_recon, data)  # Compare to original, not noisy
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len (train_loader)
    print(f"Epoch {epoch+1}/20: Loss = {avg_loss:.6f}")

# Visualize denoising
model.eval()
with torch.no_grad():
    test_data = next (iter (train_loader))[0][:8].to (device)
    noisy_test = add_noise (test_data, noise_factor=0.5)
    denoised, _ = model (noisy_test)
    
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for i in range(8):
        # Original
        axes[0, i].imshow (test_data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)
        
        # Noisy
        axes[1, i].imshow (noisy_test[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Noisy', fontsize=12)
        
        # Denoised
        axes[2, i].imshow (denoised[i].cpu().squeeze(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Denoised', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('denoising_autoencoder.png', dpi=150, bbox_inches='tight')
    print("Saved denoising results")
\`\`\`

---

## Variational Autoencoder (VAE)

**Problem with basic autoencoders**: Latent space has no structure - can't sample new data.

**Variational Autoencoder**: Learns a **probabilistic** latent space with smooth structure.

**Key differences**:
1. Encoder outputs **distribution parameters** (μ, σ²), not single vector
2. Sample z ~ N(μ, σ²) (reparameterization trick)
3. Loss = reconstruction + KL divergence (regularization)

### Mathematical Foundation

**Encoder (Recognition Model)**:
\`\`\`
q (z|x) = N(μ(x), σ²(x))
\`\`\`

Outputs mean μ and variance σ² for each dimension.

**Sampling (Reparameterization Trick)**:
\`\`\`
z = μ + σ ⊙ ε,  where ε ~ N(0, I)
\`\`\`

This allows backpropagation through sampling!

**Decoder (Generative Model)**:
\`\`\`
p (x|z) = Decoder (z)
\`\`\`

**Loss Function** (ELBO - Evidence Lower Bound):
\`\`\`
L = E[||x - x̂||²] + KL(q (z|x) || p (z))
     ↑                   ↑
reconstruction loss   regularization
\`\`\`

**KL divergence** (forces q (z|x) to be close to prior p (z) = N(0, I)):
\`\`\`
KL = -0.5 × Σᵢ [1 + log(σᵢ²) - μᵢ² - σᵢ²]
\`\`\`

### Implementation

\`\`\`python
class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        """
        Variational Autoencoder.
        
        Args:
            input_dim: Input dimension
            latent_dim: Latent space dimension
        """
        super().__init__()
        
        # Encoder: x → hidden → (μ, log σ²)
        self.encoder_hidden = nn.Sequential(
            nn.Linear (input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Output μ and log σ² (log for numerical stability)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: z → hidden → x
        self.decoder = nn.Sequential(
            nn.Linear (latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def encode (self, x):
        """
        Encode x to distribution parameters.
        
        Returns:
            mu: Mean of q (z|x)
            logvar: Log variance of q (z|x)
        """
        h = self.encoder_hidden (x)
        mu = self.fc_mu (h)
        logvar = self.fc_logvar (h)
        return mu, logvar
    
    def reparameterize (self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ ⊙ ε
        
        Args:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        
        Returns:
            z: (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 × log σ²)
        eps = torch.randn_like (std)     # ε ~ N(0, I)
        z = mu + std * eps              # z = μ + σ ⊙ ε
        return z
    
    def decode (self, z):
        """Decode z to reconstruction"""
        return self.decoder (z)
    
    def forward (self, x):
        """
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            x_recon: (batch_size, input_dim)
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        mu, logvar = self.encode (x)
        z = self.reparameterize (mu, logvar)
        x_recon = self.decode (z)
        return x_recon, mu, logvar


def vae_loss (x_recon, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction + β × KL divergence
    
    Args:
        x_recon: Reconstructed x
        x: Original x
        mu, logvar: Distribution parameters
        beta: Weight for KL term (β-VAE)
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (binary cross-entropy for binary data)
    recon_loss = F.binary_cross_entropy (x_recon, x, reduction='sum')
    
    # KL divergence: KL(q (z|x) || N(0, I))
    # -0.5 × Σ[1 + log σ² - μ² - σ²]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda (lambda x: x.view(-1))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader (train_dataset, batch_size=128, shuffle=True)

# Create VAE
model = VAE(input_dim=784, latent_dim=20)
optimizer = torch.optim.Adam (model.parameters(), lr=1e-3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to (device)

# Training loop
num_epochs = 20
model.train()

for epoch in range (num_epochs):
    total_loss_epoch = 0.0
    total_recon_epoch = 0.0
    total_kl_epoch = 0.0
    
    for data, _ in train_loader:
        data = data.to (device)
        
        # Forward pass
        x_recon, mu, logvar = model (data)
        loss, recon, kl = vae_loss (x_recon, data, mu, logvar, beta=1.0)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss_epoch += loss.item()
        total_recon_epoch += recon.item()
        total_kl_epoch += kl.item()
    
    # Average per sample
    n_samples = len (train_dataset)
    avg_loss = total_loss_epoch / n_samples
    avg_recon = total_recon_epoch / n_samples
    avg_kl = total_kl_epoch / n_samples
    
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")

print("VAE training complete!")

# Generate new samples
model.eval()
with torch.no_grad():
    # Sample from standard normal
    z_sample = torch.randn(16, 20).to (device)
    generated = model.decode (z_sample)
    
    # Plot generated images
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate (axes.flat):
        ax.imshow (generated[i].cpu().view(28, 28), cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('vae_generated_samples.png', dpi=150, bbox_inches='tight')
    print("Saved generated samples")
\`\`\`

### Latent Space Interpolation

VAE's smooth latent space enables **interpolation** between images:

\`\`\`python
def interpolate (model, x1, x2, num_steps=10):
    """
    Interpolate between two images in latent space.
    
    Args:
        model: Trained VAE
        x1, x2: Two input images
        num_steps: Number of interpolation steps
    
    Returns:
        interpolated: List of interpolated images
    """
    model.eval()
    with torch.no_grad():
        # Encode both images
        mu1, logvar1 = model.encode (x1.unsqueeze(0))
        mu2, logvar2 = model.encode (x2.unsqueeze(0))
        
        # Interpolate in latent space
        alphas = torch.linspace(0, 1, num_steps)
        interpolated = []
        
        for alpha in alphas:
            # Linear interpolation
            z = (1 - alpha) * mu1 + alpha * mu2
            
            # Decode
            x_interp = model.decode (z)
            interpolated.append (x_interp.cpu().squeeze())
        
        return interpolated


# Example: Interpolate between two digits
test_data = next (iter (train_loader))[0]
x1, x2 = test_data[0], test_data[10]  # Two different digits

interpolated_images = interpolate (model, x1, x2, num_steps=10)

# Visualize
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i, (ax, img) in enumerate (zip (axes, interpolated_images)):
    ax.imshow (img.view(28, 28), cmap='gray')
    ax.axis('off')
    if i == 0:
        ax.set_title('Start', fontsize=10)
    elif i == 9:
        ax.set_title('End', fontsize=10)

plt.tight_layout()
plt.savefig('vae_interpolation.png', dpi=150, bbox_inches='tight')
print("Saved interpolation")
\`\`\`

---

## Sparse Autoencoder

**Idea**: Add **sparsity constraint** on latent activations - only few neurons active for each input.

**Loss**:
\`\`\`
L = ||x - x̂||² + λ × Σ|zᵢ|  (L1 sparsity)
\`\`\`

or 

\`\`\`
L = ||x - x̂||² + λ × Σ zᵢ²  (L2 regularization)
\`\`\`

**Benefits**:
- Learns interpretable features (each neuron captures specific pattern)
- Prevents overfitting
- Better generalization

\`\`\`python
# Training with sparsity
lambda_sparse = 1e-4

for data, _ in train_loader:
    x_recon, z = model (data)
    
    # Reconstruction + sparsity
    recon_loss = criterion (x_recon, data)
    sparsity_loss = lambda_sparse * torch.sum (torch.abs (z))  # L1 sparsity
    
    loss = recon_loss + sparsity_loss
    
    # ... backprop ...
\`\`\`

---

## Applications

### 1. Dimensionality Reduction

**Alternative to PCA**: Non-linear, potentially more powerful.

\`\`\`python
# Use trained encoder for dimensionality reduction
model.eval()
with torch.no_grad():
    data_2d = []
    labels_list = []
    
    for data, labels in train_loader:
        data = data.to (device)
        _, z = model (data)  # Get latent representations
        data_2d.append (z.cpu())
        labels_list.append (labels)
    
    data_2d = torch.cat (data_2d, dim=0)
    labels_list = torch.cat (labels_list, dim=0)

# Visualize (if latent_dim = 2)
if model.latent_dim == 2:
    plt.figure (figsize=(10, 8))
    scatter = plt.scatter (data_2d[:, 0], data_2d[:, 1], 
                         c=labels_list, cmap='tab10', alpha=0.5)
    plt.colorbar (scatter)
    plt.title('Latent Space Visualization')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.savefig('latent_space.png', dpi=150, bbox_inches='tight')
\`\`\`

### 2. Anomaly Detection

**Idea**: Normal data reconstructs well, anomalies have high reconstruction error.

\`\`\`python
def detect_anomalies (model, test_loader, threshold=0.01):
    """
    Detect anomalies using reconstruction error.
    
    Args:
        model: Trained autoencoder
        test_loader: Test data
        threshold: Reconstruction error threshold
    
    Returns:
        anomalies: Boolean mask (True = anomaly)
        errors: Reconstruction errors
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to (device)
            x_recon, _ = model (data)
            
            # Compute per-sample reconstruction error
            error = torch.mean((data - x_recon) ** 2, dim=1)
            errors.append (error.cpu())
    
    errors = torch.cat (errors)
    anomalies = errors > threshold
    
    return anomalies, errors


# Example: Detect anomalies
anomalies, errors = detect_anomalies (model, test_loader, threshold=0.015)
print(f"Detected {anomalies.sum().item()} anomalies out of {len (errors)} samples")

# Visualize reconstruction errors
plt.figure (figsize=(10, 6))
plt.hist (errors.numpy(), bins=50)
plt.axvline (x=0.015, color='r', linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Reconstruction Errors')
plt.legend()
plt.savefig('anomaly_detection.png', dpi=150, bbox_inches='tight')
\`\`\`

---

## Discussion Questions

1. **Why does the bottleneck in an autoencoder force the model to learn important features?**
   - Consider information capacity and compression

2. **In denoising autoencoders, why do we add noise during training but not during testing?**
   - Think about what the model learns vs. what we want to use it for

3. **VAE uses KL divergence to regularize the latent space. What happens if we remove this term?**
   - Consider the structure of the latent space and ability to generate

4. **For anomaly detection, why might autoencoders work better than supervised classifiers?**
   - Think about what data is available and what's being learned

5. **Sparse autoencoders encourage few active neurons. How does this relate to the way biological neurons encode information?**
   - Consider efficiency and interpretability

---

## Key Takeaways

- **Autoencoders** learn compressed representations through encoder-decoder architecture with reconstruction objective
- **Bottleneck** forces learning only most important features (dimensionality reduction)
- **Convolutional autoencoders** preserve spatial structure for image data
- **Denoising autoencoders** learn robust features by reconstructing clean from noisy
- **Variational autoencoders (VAEs)** learn structured probabilistic latent space for generation
- **Reparameterization trick** enables backpropagation through sampling in VAEs
- **VAE loss** = reconstruction + KL divergence (forces latent space to be standard normal)
- **Sparse autoencoders** add sparsity constraint for interpretable features
- **Applications**: Dimensionality reduction, denoising, anomaly detection, generation, feature learning
- **Limitation**: Reconstruction quality often lower than GANs for generation

---

## Practical Tips

1. **Start simple**: Basic autoencoder before trying VAE

2. **Latent dimension**: Too small (under-fitting), too large (no compression). Experiment!

3. **For images**: Use convolutional layers, not fully connected

4. **VAE training**: Balance reconstruction and KL (try β-VAE with β ≠ 1)

5. **Denoising**: Noise level matters (0.2-0.5 typical), validate on clean data

6. **Anomaly detection**: Set threshold using validation data (normal examples)

7. **Visualization**: Plot reconstructions, latent space, loss curves to understand training

---

## Further Reading

- ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114) - Kingma & Welling, 2013 (VAE)
- ["Tutorial on Variational Autoencoders"](https://arxiv.org/abs/1606.05908) - Doersch, 2016
- ["β-VAE"](https://openreview.net/forum?id=Sy2fzU9gl) - Higgins et al., 2017
- [PyTorch VAE Examples](https://github.com/pytorch/examples/tree/master/vae)

---

*Next Section: Generative Adversarial Networks (GANs) - Learn how two networks compete to generate realistic synthetic data!*
`,
};

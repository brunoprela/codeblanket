/**
 * Generative Adversarial Networks (GANs) Content
 */

export const generativeAdversarialNetworksSection = {
  id: 'generative-adversarial-networks',
  title: 'Generative Adversarial Networks (GANs)',
  content: `
# Generative Adversarial Networks (GANs)

## Introduction

**Generative Adversarial Networks (GANs)**, introduced by Ian Goodfellow in 2014, are a class of generative models that learn through an **adversarial process** between two neural networks.

**The Two Players**:

1. **Generator (G)**: Creates fake data from random noise
   - Goal: Fool the discriminator into thinking fakes are real

2. **Discriminator (D)**: Classifies data as real or fake
   - Goal: Correctly identify real vs. generated data

**Training**: Generator and discriminator compete in a **minimax game**:
- Generator tries to maximize discriminator's error
- Discriminator tries to minimize its error

**Analogy**: Counterfeiter (generator) vs. police (discriminator)
- Counterfeiter makes fake money, tries to fool police
- Police learn to spot fakes
- Over time, counterfeiter gets better, police get better
- Eventually: Perfect counterfeit money!

**Applications**:
- Image generation (faces, landscapes, artwork)
- Style transfer and image-to-image translation
- Super-resolution (enhance image quality)
- Data augmentation
- Drug discovery (generate molecular structures)

---

## Mathematical Foundation

### Minimax Objective

\`\`\`
min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
                       ↑                        ↑
                    real data              fake data
\`\`\`

**Discriminator's perspective** (maximize V):
- Wants D(real) → 1 (log D(x) → 0, maximized)
- Wants D(fake) → 0 (log(1-D(G(z))) → 0, maximized)

**Generator's perspective** (minimize V):
- Wants D(G(z)) → 1 (log(1-D(G(z))) → -∞, minimized)
- Makes discriminator think fakes are real

### Training Algorithm

**Alternate between**:

1. **Train Discriminator** (k steps):
   - Sample real data: {x₁, ..., x_m} ~ p_data
   - Sample noise: {z₁, ..., z_m} ~ p_z
   - Generate fakes: {G(z₁), ..., G(z_m)}
   - Update D to maximize:
     \`\`\`
     L_D = Σᵢ [log D(xᵢ) + log(1 - D(G(zᵢ)))]
     \`\`\`

2. **Train Generator** (1 step):
   - Sample noise: {z₁, ..., z_m} ~ p_z
   - Update G to minimize (or maximize alternative):
     \`\`\`
     L_G = Σᵢ log(1 - D(G(zᵢ)))  (original)
     L_G = -Σᵢ log D(G(zᵢ))      (non-saturating alternative, better gradients)
     \`\`\`

---

## Basic GAN Implementation

\`\`\`python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 100  # Noise dimension
img_size = 28  # MNIST images
channels = 1  # Grayscale
batch_size = 128
lr = 2e-4
num_epochs = 50

# Generator: noise → image
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        """
        Generator network: z (noise) → fake image
        
        Args:
            latent_dim: Dimension of input noise vector
            img_shape: Output image shape (channels, height, width)
        """
        super().__init__()
        
        self.img_shape = img_shape
        self.img_dim = channels * img_size * img_size  # 1 × 28 × 28 = 784
        
        self.model = nn.Sequential(
            # Input: (batch, latent_dim)
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, self.img_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        """
        Args:
            z: (batch_size, latent_dim) - Random noise
        
        Returns:
            img: (batch_size, channels, height, width) - Generated image
        """
        img_flat = self.model(z)
        img = img_flat.view(img_flat.size(0), *self.img_shape)
        return img


# Discriminator: image → real/fake probability
class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        """
        Discriminator network: image → probability(real)
        
        Args:
            img_shape: Input image shape
        """
        super().__init__()
        
        self.img_dim = channels * img_size * img_size
        
        self.model = nn.Sequential(
            # Input: (batch, img_dim)
            nn.Linear(self.img_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability in [0, 1]
        )
    
    def forward(self, img):
        """
        Args:
            img: (batch_size, channels, height, width)
        
        Returns:
            validity: (batch_size, 1) - Probability that input is real
        """
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# Initialize models
generator = Generator(latent_dim)
discriminator = Discriminator()

# Loss function
adversarial_loss = nn.BCELoss()  # Binary cross-entropy

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = generator.to(device)
discriminator = discriminator.to(device)
adversarial_loss = adversarial_loss.to(device)

# Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

dataloader = DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

# Training loop
print("Starting GAN training...")

for epoch in range(num_epochs):
    d_losses = []
    g_losses = []
    
    for i, (real_imgs, _) in enumerate(dataloader):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        
        # Labels for real and fake
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Real images
        real_pred = discriminator(real_imgs)
        d_loss_real = adversarial_loss(real_pred, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(z)
        fake_pred = discriminator(fake_imgs.detach())  # Detach to avoid training G
        d_loss_fake = adversarial_loss(fake_pred, fake_labels)
        
        # Total discriminator loss
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_imgs = generator(z)
        
        # Generator wants discriminator to think fakes are real
        fake_pred = discriminator(gen_imgs)
        g_loss = adversarial_loss(fake_pred, real_labels)  # Fool discriminator
        
        g_loss.backward()
        optimizer_G.step()
        
        # Track losses
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
    
    # Print progress
    avg_d_loss = sum(d_losses) / len(d_losses)
    avg_g_loss = sum(g_losses) / len(g_losses)
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
    
    # Generate sample images every 5 epochs
    if (epoch + 1) % 5 == 0:
        generator.eval()
        with torch.no_grad():
            z_sample = torch.randn(16, latent_dim, device=device)
            gen_samples = generator(z_sample).cpu()
            
            # Denormalize [-1, 1] → [0, 1]
            gen_samples = (gen_samples + 1) / 2
            
            # Plot
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for idx, ax in enumerate(axes.flat):
                ax.imshow(gen_samples[idx].squeeze(), cmap='gray')
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'gan_samples_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
        
        generator.train()

print("Training complete!")
\`\`\`

---

## Training Dynamics and Challenges

### Training Instability

**Problem**: GAN training is notoriously unstable.

**Common issues**:

1. **Mode collapse**: Generator produces limited variety (e.g., only one digit)
   - Loss: G finds one example that fools D, produces only that
   - Solution: Mini-batch discrimination, unrolled GANs

2. **Vanishing gradients**: When D is too good, G receives no gradient signal
   - If D(G(z)) ≈ 0, then log(1-D(G(z))) ≈ 0, gradient → 0
   - Solution: Use non-saturating loss: -log D(G(z))

3. **Oscillation**: D and G continuously overpower each other
   - Loss oscillates without converging
   - Solution: Careful learning rate tuning, gradient penalties

### Nash Equilibrium

**Goal**: Reach Nash equilibrium where neither player can improve by changing strategy alone.

At equilibrium:
- Generator produces samples indistinguishable from real data: p_g = p_data
- Discriminator outputs 0.5 for all inputs (can't tell real from fake)

**In practice**: Perfect equilibrium rarely achieved, but models learn useful representations.

---

## Deep Convolutional GAN (DCGAN)

**DCGAN** (2015) introduced architectural guidelines that dramatically improved stability:

**Key principles**:
1. Replace pooling with strided convolutions (discriminator) and fractional-strided convolutions (generator)
2. Use batch normalization in both networks (except generator output and discriminator input)
3. Remove fully connected hidden layers
4. Use ReLU in generator (Tanh for output)
5. Use LeakyReLU in discriminator

### Implementation

\`\`\`python
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        """
        DCGAN Generator: noise → image using transposed convolutions
        
        Architecture: Linear → Reshape → ConvTranspose2d (upsample) → Image
        """
        super().__init__()
        
        self.init_size = 7  # Initial spatial size (7×7 for 28×28 output)
        self.latent_dim = latent_dim
        
        # Linear projection and reshape
        self.fc = nn.Linear(latent_dim, 128 * self.init_size ** 2)
        
        # Convolutional layers (upsample)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            
            # 7×7 → 14×14
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 14×14 → 28×28
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final conv
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        """
        Args:
            z: (batch, latent_dim)
        
        Returns:
            img: (batch, channels, 28, 28)
        """
        # Project and reshape
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        
        # Convolutional blocks
        img = self.conv_blocks(out)
        return img


class DCGANDiscriminator(nn.Module):
    def __init__(self, channels=1):
        """
        DCGAN Discriminator: image → real/fake using convolutions
        
        Architecture: ConvDownsample → Flatten → Linear → Sigmoid
        """
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            # Input: (1, 28, 28)
            
            # 28×28 → 14×14
            nn.Conv2d(channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            # 14×14 → 7×7
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            # 7×7 → 3×3
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
        )
        
        # Flatten and classify
        self.adv_layer = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        """
        Args:
            img: (batch, channels, 28, 28)
        
        Returns:
            validity: (batch, 1)
        """
        out = self.conv_blocks(img)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity


# Use DCGAN architectures
generator = DCGANGenerator(latent_dim=100)
discriminator = DCGANDiscriminator()

# Training same as before...
\`\`\`

---

## Advanced GAN Variants

### 1. Conditional GAN (cGAN)

**Idea**: Condition generation on additional information (e.g., class label).

**Architecture**:
- Generator: G(z, y) where y is condition (e.g., digit class)
- Discriminator: D(x, y) checks if x matches condition y

\`\`\`python
# Conditional Generator
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            # ... rest of generator ...
        )
    
    def forward(self, z, labels):
        # Concatenate noise and label embedding
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat([z, label_emb], dim=1)
        img = self.model(gen_input)
        return img
\`\`\`

**Usage**: Generate specific digit by providing label: \`G(z, label = 7)\` → image of "7"

### 2. Wasserstein GAN (WGAN)

**Problem**: Original GAN loss can lead to vanishing gradients.

**Solution**: Use Wasserstein distance (Earth Mover's distance) instead of JS divergence.

**Loss**:
\`\`\`
L_D = E[D(real)] - E[D(fake)]  (maximize difference)
L_G = -E[D(fake)]              (maximize D output on fakes)
\`\`\`

**Constraint**: D must be 1-Lipschitz (enforce with weight clipping or gradient penalty)

**Advantages**:
- More stable training
- Meaningful loss metric (correlates with quality)
- Less mode collapse

### 3. Progressive GAN

**Idea**: Start with low resolution, progressively add layers for higher resolution.

**Training**:
1. Start: 4×4 images
2. Add layers: 8×8 → 16×16 → 32×32 → ... → 1024×1024
3. Each stage trains to stability before adding next layer

**Result**: High-resolution images (1024×1024) with stable training.

### 4. StyleGAN

**Idea**: Control different levels of style (coarse to fine) in generated images.

**Architecture**:
- Mapping network: z → w (intermediate latent space)
- Synthesis network: Progressive generation with style injection at each resolution
- Adaptive Instance Normalization (AdaIN) applies style

**Features**:
- Style mixing: Combine coarse style from one image, fine style from another
- High-quality face generation
- Fine control over attributes

---

## Evaluating GANs

**Challenge**: No single metric captures generation quality perfectly.

### Inception Score (IS)

\`\`\`
IS = exp(E_x [KL(p(y|x) || p(y))])
\`\`\`

- Higher is better (more diverse and confident predictions)
- Range: 1 to num_classes
- Problem: Only measures diversity within classes, not realism

### Fréchet Inception Distance (FID)

**Most popular metric**: Measures distance between real and generated image distributions in Inception network feature space.

\`\`\`python
from scipy.linalg import sqrtm

def calculate_fid(real_features, fake_features):
    """
    Calculate FID score between real and fake features.
    
    Args:
        real_features: Features from real images (n_real, feature_dim)
        fake_features: Features from fake images (n_fake, feature_dim)
    
    Returns:
        fid: FID score (lower is better)
    """
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)
    
    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = np.dot(diff, diff) + np.trace(sigma_real + sigma_fake - 2*covmean)
    
    return fid
\`\`\`

**Lower FID = better** (0 = perfect match to real data)

---

## Applications

### 1. Image-to-Image Translation (pix2pix, CycleGAN)

**Task**: Transform image from one domain to another (e.g., sketch → photo, summer → winter).

**pix2pix**: Supervised (paired training data)
- Input: Sketch, Output: Photo
- Loss: Adversarial + L1 reconstruction

**CycleGAN**: Unsupervised (unpaired data)
- Two GANs: A → B and B → A
- Cycle consistency: A → B → A should equal A

### 2. Super-Resolution

**Task**: Enhance low-resolution image to high-resolution.

**SRGAN**: GAN for photo-realistic super-resolution
- Generator: Upsamples 4× (e.g., 64×64 → 256×256)
- Discriminator: Classifies high-res real vs. generated
- Perceptual loss: Feature matching in VGG network

### 3. Text-to-Image Generation

**Task**: Generate image from text description.

**StackGAN**: Multi-stage refinement
- Stage 1: Low-res image from text (64×64)
- Stage 2: High-res refinement (256×256)
- Conditioning: Text encoded with RNN

---

## Discussion Questions

1. **Why is training GANs more difficult than training standard neural networks?**
   - Consider the adversarial nature and equilibrium requirements

2. **Mode collapse means the generator produces limited variety. Why doesn't the discriminator reject these samples?**
   - Think about what happens during training dynamics

3. **In Wasserstein GAN, why must the discriminator (critic) be 1-Lipschitz continuous?**
   - Consider the theoretical foundation of Wasserstein distance

4. **Conditional GANs can generate specific outputs (e.g., digit 7). How does this compare to VAEs for controlled generation?**
   - Compare explicit conditioning vs. latent space navigation

5. **GANs don't have an explicit likelihood. How does this affect their use compared to VAEs or likelihood-based models?**
   - Consider evaluation, density estimation, and anomaly detection

---

## Key Takeaways

- **GANs** train generator and discriminator in adversarial game to produce realistic synthetic data
- **Minimax objective**: Generator minimizes what discriminator maximizes
- **Training dynamics**: Alternate training, careful balancing, aim for Nash equilibrium
- **Challenges**: Mode collapse, vanishing gradients, training instability
- **DCGAN principles**: Strided convolutions, batch normalization, LeakyReLU dramatically improve stability
- **Conditional GANs**: Control generation with additional information (class labels, text)
- **Wasserstein GAN**: More stable training using Wasserstein distance
- **Evaluation**: FID score most common (lower better), Inception Score for diversity
- **Applications**: Image generation, style transfer, super-resolution, domain adaptation, data augmentation
- **Advantage over VAEs**: Higher quality, sharper images; **Disadvantage**: Harder to train, no likelihood

---

## Practical Tips

1. **Start with DCGAN architecture**: Proven stable guidelines

2. **Learning rates**: Lower for both G and D (2e-4 typical), use Adam with β₁=0.5

3. **Balance G and D**: Train D more if G dominates (k=2-5 D steps per G step)

4. **Monitor both losses**: Should fluctuate but stay roughly balanced

5. **Check samples frequently**: Loss alone doesn't indicate quality

6. **Use non-saturating loss**: -log D(G(z)) instead of log(1-D(G(z)))

7. **Label smoothing**: Use 0.9 instead of 1 for real labels (regularization)

8. **Add noise to discriminator inputs**: Helps prevent overconfident D

---

## Further Reading

- ["Generative Adversarial Networks"](https://arxiv.org/abs/1406.2661) - Goodfellow et al., 2014 (Original GAN)
- ["DCGAN"](https://arxiv.org/abs/1511.06434) - Radford et al., 2015
- ["Wasserstein GAN"](https://arxiv.org/abs/1701.07875) - Arjovsky et al., 2017
- ["Progressive GAN"](https://arxiv.org/abs/1710.10196) - Karras et al., 2017
- ["StyleGAN"](https://arxiv.org/abs/1812.04948) - Karras et al., 2018

---

*Next Section: Graph Neural Networks (GNNs) - Learn how to apply deep learning to graph-structured data!*
`,
};

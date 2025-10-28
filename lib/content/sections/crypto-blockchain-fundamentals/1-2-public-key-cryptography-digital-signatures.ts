export const publicKeyCryptographyDigitalSignatures = {
  title: 'Public Key Cryptography & Digital Signatures',
  id: 'public-key-cryptography-digital-signatures',
  content: `
# Public Key Cryptography & Digital Signatures

## Introduction

Public key cryptography is the mathematical foundation that makes blockchain possible. Without it, there would be no Bitcoin, no Ethereum, no cryptocurrency at all. **It solves the fundamental problem: How can you prove ownership of digital assets without revealing the secret that grants that ownership?**

Traditional symmetric encryption requires both parties to share a secret key. But blockchain operates in a trustless environment with thousands of unknown participants. Public key cryptography solves this through an elegant mathematical asymmetry: **operations that are easy in one direction become computationally impossible in the reverse direction.**

### The Digital Signature Problem

Consider this scenario: Alice wants to send Bob 1 Bitcoin. The network needs to verify:
1. Alice actually owns 1 Bitcoin
2. Alice authorized this specific transaction
3. No one can forge Alice's authorization
4. Alice cannot later deny making this transaction (non-repudiation)
5. The transaction cannot be modified after signing

**Public key cryptography solves all of these problems through digital signatures.**

## Symmetric vs Asymmetric Encryption

### Symmetric Encryption: The Shared Secret Problem

\`\`\`python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def symmetric_encryption_demo():
    """
    Demonstrate symmetric encryption (AES)
    """
    # Both Alice and Bob need the same key
    shared_key = os.urandom(32)  # 256-bit key
    
    # Alice encrypts message
    plaintext = b"Send 1 BTC to Bob"
    iv = os.urandom(16)  # Initialization vector
    
    cipher = Cipher(
        algorithms.AES(shared_key),
        modes.CBC(iv),
        backend=default_backend()
    )
    
    encryptor = cipher.encryptor()
    # Pad to block size (16 bytes for AES)
    padded = plaintext + b" " * (16 - len(plaintext) % 16)
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    
    print("Symmetric Encryption (AES)")
    print(f"Shared Key: {shared_key.hex()[:32]}...")
    print(f"Plaintext: {plaintext.decode()}")
    print(f"Ciphertext: {ciphertext.hex()[:32]}...")
    
    # Bob decrypts with SAME key
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(ciphertext) + decryptor.finalize()
    print(f"Decrypted: {decrypted.decode().strip()}")
    
    # The problem: How did Alice and Bob share the key securely?
    print("\\nPROBLEM: Both parties need the same secret key")
    print("In blockchain: How do 1 million users share keys?")
    print("Answer: They don't. Use asymmetric encryption instead!")

symmetric_encryption_demo()
\`\`\`

**The Key Distribution Problem:**
- With N users, need N(N-1)/2 shared keys
- 1 million users = 500 billion key pairs
- How to distribute keys securely?
- What if one key is compromised?

### Asymmetric Encryption: The Public/Private Key Solution

\`\`\`python
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

def asymmetric_encryption_demo():
    """
    Demonstrate asymmetric encryption (RSA)
    """
    # Alice generates key pair
    alice_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    alice_public_key = alice_private_key.public_key()
    
    print("Asymmetric Encryption (RSA)")
    print("Alice generates key pair:")
    print("- Private key: KEPT SECRET")
    print("- Public key: SHARED WITH EVERYONE")
    
    # Bob wants to send encrypted message to Alice
    # Bob uses Alice's PUBLIC key to encrypt
    plaintext = b"Send 1 BTC to Bob"
    
    ciphertext = alice_public_key.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    print(f"\\nBob encrypts with Alice's PUBLIC key")
    print(f"Ciphertext: {ciphertext.hex()[:32]}...")
    
    # Only Alice can decrypt with her PRIVATE key
    decrypted = alice_private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    print(f"\\nAlice decrypts with her PRIVATE key")
    print(f"Decrypted: {decrypted.decode()}")
    
    print("\\nAdvantages:")
    print("- No need to share secret keys")
    print("- Public key can be distributed freely")
    print("- Only private key owner can decrypt")
    print("- N users need only N key pairs (not N²)")

asymmetric_encryption_demo()
\`\`\`

## Elliptic Curve Cryptography (ECC)

**Bitcoin and Ethereum use Elliptic Curve Cryptography (ECC), not RSA.** ECC provides the same security as RSA with much smaller keys:

- RSA 2048-bit key ≈ ECC 224-bit key (same security)
- RSA 3072-bit key ≈ ECC 256-bit key (same security)
- Smaller keys = faster computation, less storage

### The Mathematics of Elliptic Curves

An elliptic curve is defined by the equation: **y² = x³ + ax + b**

\`\`\`python
import matplotlib.pyplot as plt
import numpy as np

def visualize_elliptic_curve():
    """
    Visualize an elliptic curve: y² = x³ + ax + b
    Bitcoin/Ethereum use: y² = x³ + 7 (secp256k1 curve)
    """
    # Bitcoin's curve: y² = x³ + 7
    a, b = 0, 7
    
    # Calculate points
    x = np.linspace(-5, 5, 1000)
    y_squared = x**3 + a*x + b
    
    # Only plot where y² is positive (real solutions)
    y_pos = np.sqrt(np.maximum(0, y_squared))
    y_neg = -y_pos
    
    plt.figure(figsize=(10, 8))
    plt.plot(x, y_pos, 'b-', label='y = +√(x³ + 7)')
    plt.plot(x, y_neg, 'b-', label='y = -√(x³ + 7)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Elliptic Curve: y² = x³ + 7 (Bitcoin secp256k1)')
    plt.legend()
    
    print("Elliptic Curve Properties:")
    print("1. Symmetric about x-axis")
    print("2. Non-singular (smooth, no cusps)")
    print("3. Forms a mathematical group")
    print("4. Point addition is defined")
    print("5. Scalar multiplication is one-way")

# visualize_elliptic_curve()  # Uncomment to see visualization
\`\`\`

### Elliptic Curve Point Addition

The "magic" of ECC comes from point addition on the curve:

\`\`\`python
class EllipticCurve:
    """
    Simplified elliptic curve implementation (for educational purposes)
    Real Bitcoin uses secp256k1 with large prime field
    """
    def __init__(self, a, b, p):
        """
        Curve: y² = x³ + ax + b (mod p)
        """
        self.a = a
        self.b = b
        self.p = p  # Prime modulus
    
    def is_on_curve(self, point):
        """Check if point is on curve"""
        if point is None:  # Point at infinity
            return True
        
        x, y = point
        return (y**2) % self.p == (x**3 + self.a*x + self.b) % self.p
    
    def add_points(self, P, Q):
        """
        Add two points on the curve
        This is the fundamental operation in ECC
        """
        # Point at infinity (identity element)
        if P is None:
            return Q
        if Q is None:
            return P
        
        x1, y1 = P
        x2, y2 = Q
        
        # P + (-P) = O (point at infinity)
        if x1 == x2 and y1 == (-y2 % self.p):
            return None
        
        # Calculate slope
        if P == Q:
            # Point doubling: slope = (3x₁² + a) / (2y₁)
            slope = (3 * x1**2 + self.a) * pow(2 * y1, -1, self.p)
        else:
            # Point addition: slope = (y₂ - y₁) / (x₂ - x₁)
            slope = (y2 - y1) * pow(x2 - x1, -1, self.p)
        
        slope = slope % self.p
        
        # Calculate new point
        x3 = (slope**2 - x1 - x2) % self.p
        y3 = (slope * (x1 - x3) - y1) % self.p
        
        return (x3, y3)
    
    def scalar_multiply(self, k, P):
        """
        Multiply point P by scalar k: k*P = P + P + ... + P (k times)
        
        This is EASY: Given k and P, compute k*P
        This is HARD: Given k*P and P, find k (discrete log problem)
        """
        if k == 0:
            return None
        
        result = None
        addend = P
        
        # Binary multiplication (efficient)
        while k:
            if k & 1:
                result = self.add_points(result, addend)
            addend = self.add_points(addend, addend)
            k >>= 1
        
        return result

# Example with small numbers (real Bitcoin uses 256-bit numbers)
curve = EllipticCurve(a=0, b=7, p=97)  # y² = x³ + 7 (mod 97)

# Generator point (like Bitcoin's G)
G = (2, 35)

print("Elliptic Curve Point Operations")
print(f"Curve: y² = x³ + 7 (mod 97)")
print(f"Generator Point G: {G}")
print(f"Is G on curve? {curve.is_on_curve(G)}")

# Demonstrate scalar multiplication
for k in [1, 2, 3, 5, 10, 20]:
    result = curve.scalar_multiply(k, G)
    print(f"{k}*G = {result}")

print("\\nKey Insight:")
print("EASY: Given k, compute k*G")
print("HARD: Given k*G, find k (discrete logarithm problem)")
print("This asymmetry is the foundation of Bitcoin security!")
\`\`\`

## Bitcoin's secp256k1 Curve

Bitcoin uses a specific elliptic curve called **secp256k1**:

\`\`\`python
def explain_secp256k1():
    """
    Bitcoin's secp256k1 curve parameters
    """
    secp256k1 = {
        'name': 'secp256k1',
        'equation': 'y² = x³ + 7',
        
        # Field
        'p': 2**256 - 2**32 - 977,
        'p_hex': '0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F',
        
        # Curve parameters
        'a': 0,
        'b': 7,
        
        # Generator point G
        'Gx': '0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798',
        'Gy': '0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8',
        
        # Order (number of points)
        'n': 2**256 - 432420386565659656852420866394968145599,
        'n_hex': '0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141',
        
        # Cofactor
        'h': 1
    }
    
    print("Bitcoin secp256k1 Curve Parameters")
    print("=" * 70)
    
    print(f"\\nEquation: y² = x³ + {secp256k1['a']}x + {secp256k1['b']}")
    print(f"Simplified: y² = x³ + 7")
    
    print(f"\\nField Prime (p):")
    print(f"  Decimal: 2^256 - 2^32 - 977")
    print(f"  Hex: {secp256k1['p_hex']}")
    
    print(f"\\nGenerator Point G:")
    print(f"  Gx: {secp256k1['Gx']}")
    print(f"  Gy: {secp256k1['Gy']}")
    
    print(f"\\nCurve Order (n): {secp256k1['n_hex']}")
    print(f"  This is the number of possible private keys")
    print(f"  ≈ 1.158 × 10^77 possible keys")
    
    print("\\nWhy secp256k1?")
    print("- Chosen by Bitcoin creator Satoshi Nakamoto")
    print("- Not a NIST curve (some believe NIST curves may have backdoors)")
    print("- Efficient implementation")
    print("- Also used by: Ethereum, Litecoin, Bitcoin Cash")
    
    return secp256k1

secp256k1_params = explain_secp256k1()
\`\`\`

## Key Generation in Bitcoin

\`\`\`python
import hashlib
import secrets

def generate_bitcoin_keypair():
    """
    Generate Bitcoin-style keypair
    (Simplified - real Bitcoin uses libsecp256k1)
    """
    # Step 1: Generate private key (random 256-bit number)
    # Must be between 1 and n-1 (curve order)
    
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    
    # Generate random private key
    private_key = secrets.randbelow(n - 1) + 1
    
    print("Bitcoin Key Generation")
    print("=" * 70)
    
    print("\\n1. Private Key (256-bit random number)")
    private_key_hex = hex(private_key)[2:].zfill(64)
    print(f"   Hex: {private_key_hex}")
    print(f"   Decimal: {private_key}")
    print(f"   ⚠️  KEEP SECRET! This controls your Bitcoin")
    
    # Step 2: Derive public key (public_key = private_key * G)
    # In real Bitcoin, this uses elliptic curve multiplication
    print("\\n2. Public Key (private_key × Generator Point)")
    print(f"   Public Key = {private_key_hex[:8]}... × G")
    print("   (This is elliptic curve scalar multiplication)")
    print("   Result: (x, y) point on curve")
    
    # For demonstration (real implementation would compute actual point)
    # Public key format: 04 + x-coordinate + y-coordinate
    public_key_example = f"04{private_key_hex}{hashlib.sha256(private_key.to_bytes(32, 'big')).hexdigest()}"
    print(f"   Uncompressed: {public_key_example[:66]}...")
    print(f"   Compressed: 02{private_key_hex[:64]}...")
    
    # Step 3: Derive Bitcoin address
    print("\\n3. Bitcoin Address (hash of public key)")
    
    # Hash public key
    sha256_hash = hashlib.sha256(private_key.to_bytes(32, 'big')).digest()
    ripemd160 = hashlib.new('ripemd160')
    ripemd160.update(sha256_hash)
    pubkey_hash = ripemd160.digest()
    
    # Add version byte (0x00 for mainnet)
    versioned = b'\\x00' + pubkey_hash
    
    # Double SHA-256 checksum
    checksum = hashlib.sha256(hashlib.sha256(versioned).digest()).digest()[:4]
    
    # Base58 encode
    address_bytes = versioned + checksum
    print(f"   Hash160: {pubkey_hash.hex()}")
    print(f"   Checksum: {checksum.hex()}")
    print(f"   Address: 1{address_bytes.hex()[:32]}... (Base58)")
    
    print("\\n" + "=" * 70)
    print("Security Properties:")
    print("✓ Private key → Public key: EASY (elliptic curve multiplication)")
    print("✗ Public key → Private key: IMPOSSIBLE (discrete log problem)")
    print("✓ Public key → Address: EASY (hash functions)")
    print("✗ Address → Public key: IMPOSSIBLE (pre-image resistance)")
    
    return private_key, public_key_example

generate_bitcoin_keypair()
\`\`\`

## Digital Signatures with ECDSA

**ECDSA (Elliptic Curve Digital Signature Algorithm)** is how Bitcoin proves transaction authorization.

### How Digital Signatures Work

\`\`\`python
import hashlib
import secrets

class ECDSASignature:
    """
    Simplified ECDSA implementation for understanding
    (Real Bitcoin uses libsecp256k1)
    """
    def __init__(self):
        # secp256k1 parameters
        self.n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        self.G = "Generator point G"  # In reality, actual curve point
    
    def sign_message(self, private_key: int, message: bytes) -> tuple:
        """
        Sign a message with ECDSA
        Returns: (r, s) signature pair
        """
        # Step 1: Hash the message
        message_hash = int.from_bytes(
            hashlib.sha256(message).digest(),
            'big'
        )
        
        print("ECDSA Signature Generation")
        print("=" * 70)
        print(f"Message: {message.decode()}")
        print(f"Message Hash: {hex(message_hash)[:32]}...")
        
        # Step 2: Generate random nonce k
        # ⚠️ CRITICAL: k must be random and NEVER reused!
        k = secrets.randbelow(self.n - 1) + 1
        print(f"\\nRandom nonce k: {hex(k)[:32]}...")
        print("⚠️  k MUST be random and used only once!")
        
        # Step 3: Calculate R = k × G
        # r = x-coordinate of R
        # (Simplified - real implementation computes actual curve point)
        r = pow(k, 1, self.n)  # Simplified for demo
        print(f"\\nR = k × G")
        print(f"r = {hex(r)[:32]}... (x-coordinate of R)")
        
        # Step 4: Calculate s = k⁻¹(hash + private_key × r) mod n
        k_inv = pow(k, -1, self.n)
        s = (k_inv * (message_hash + private_key * r)) % self.n
        print(f"\\ns = k⁻¹(hash + private_key × r) mod n")
        print(f"s = {hex(s)[:32]}...")
        
        print(f"\\nSignature: (r, s)")
        print(f"  r: {hex(r)[:32]}...")
        print(f"  s: {hex(s)[:32]}...")
        
        return (r, s)
    
    def verify_signature(self, public_key: str, message: bytes, 
                        signature: tuple) -> bool:
        """
        Verify ECDSA signature
        """
        r, s = signature
        
        print("\\n" + "=" * 70)
        print("ECDSA Signature Verification")
        print("=" * 70)
        
        # Step 1: Hash the message
        message_hash = int.from_bytes(
            hashlib.sha256(message).digest(),
            'big'
        )
        print(f"Message Hash: {hex(message_hash)[:32]}...")
        
        # Step 2: Calculate w = s⁻¹ mod n
        w = pow(s, -1, self.n)
        print(f"\\nw = s⁻¹ mod n = {hex(w)[:32]}...")
        
        # Step 3: Calculate u₁ = hash × w mod n
        u1 = (message_hash * w) % self.n
        print(f"u₁ = hash × w = {hex(u1)[:32]}...")
        
        # Step 4: Calculate u₂ = r × w mod n
        u2 = (r * w) % self.n
        print(f"u₂ = r × w = {hex(u2)[:32]}...")
        
        # Step 5: Calculate point (x, y) = u₁×G + u₂×PublicKey
        # Verify: x == r
        print(f"\\nCalculate: u₁×G + u₂×PublicKey")
        print(f"Verify: x-coordinate == r")
        
        # In real implementation, would compute actual curve points
        is_valid = True  # Simplified
        
        print(f"\\n{'✓ VALID' if is_valid else '✗ INVALID'} signature")
        
        return is_valid

# Demonstrate ECDSA
ecdsa = ECDSASignature()

# Alice's keys
private_key = secrets.randbelow(ecdsa.n - 1) + 1
public_key = f"Public key derived from {hex(private_key)[:16]}..."

# Sign a transaction
message = b"Send 1 BTC from Alice to Bob"
signature = ecdsa.sign_message(private_key, message)

# Verify signature
is_valid = ecdsa.verify_signature(public_key, message, signature)
\`\`\`

### The Danger of Nonce Reuse

**Critical vulnerability: Reusing the same nonce k reveals your private key!**

\`\`\`python
def demonstrate_nonce_reuse_attack():
    """
    Show how reusing nonce k allows private key recovery
    
    This is how Sony PlayStation 3 was hacked in 2010!
    """
    print("Nonce Reuse Attack: Private Key Recovery")
    print("=" * 70)
    
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    
    # Attacker knows:
    private_key = 12345678901234567890  # (Unknown to attacker initially)
    k = 999999999999999  # ⚠️ Same k used twice!
    
    # First signature
    message1 = b"Transaction 1"
    hash1 = int.from_bytes(hashlib.sha256(message1).digest(), 'big')
    r1 = pow(k, 1, n)  # Same r for both signatures!
    s1 = (pow(k, -1, n) * (hash1 + private_key * r1)) % n
    
    print("Signature 1:")
    print(f"  Message: {message1.decode()}")
    print(f"  r₁: {hex(r1)[:32]}...")
    print(f"  s₁: {hex(s1)[:32]}...")
    
    # Second signature with SAME k
    message2 = b"Transaction 2"
    hash2 = int.from_bytes(hashlib.sha256(message2).digest(), 'big')
    r2 = pow(k, 1, n)  # Same r!
    s2 = (pow(k, -1, n) * (hash2 + private_key * r2)) % n
    
    print("\\nSignature 2:")
    print(f"  Message: {message2.decode()}")
    print(f"  r₂: {hex(r2)[:32]}... ⚠️  SAME AS r₁!")
    print(f"  s₂: {hex(s2)[:32]}...")
    
    # Attacker notices r1 == r2 and extracts private key!
    print("\\n" + "🚨" * 35)
    print("ATTACK: Attacker sees r₁ == r₂ and recovers private key!")
    print("🚨" * 35)
    
    # Mathematical attack:
    # s₁ = k⁻¹(hash₁ + private_key × r₁)
    # s₂ = k⁻¹(hash₂ + private_key × r₂)
    # Since r₁ == r₂:
    # s₁ - s₂ = k⁻¹(hash₁ - hash₂)
    # k = (hash₁ - hash₂) / (s₁ - s₂)
    
    recovered_k = ((hash1 - hash2) * pow(s1 - s2, -1, n)) % n
    print(f"\\nRecovered k: {recovered_k}")
    print(f"Actual k: {k}")
    print(f"Match: {recovered_k == k}")
    
    # With k, recover private key:
    # s₁ = k⁻¹(hash₁ + private_key × r₁)
    # private_key = (s₁ × k - hash₁) / r₁
    
    recovered_private = ((s1 * recovered_k - hash1) * pow(r1, -1, n)) % n
    print(f"\\nRecovered private key: {recovered_private}")
    print(f"Actual private key: {private_key}")
    print(f"Match: {recovered_private == private_key}")
    
    print("\\n" + "=" * 70)
    print("LESSON: NEVER REUSE NONCE!")
    print("- Generate random k for EVERY signature")
    print("- Use deterministic k (RFC 6979) to avoid randomness issues")
    print("- This hack broke Sony PlayStation 3 (2010)")
    print("- Also affected some Bitcoin wallets (android SecureRandom bug)")

demonstrate_nonce_reuse_attack()
\`\`\`

## Ethereum's Signature Scheme

Ethereum uses the same secp256k1 curve but with slightly different address derivation:

\`\`\`python
import hashlib

def ethereum_address_derivation():
    """
    Ethereum address derivation (different from Bitcoin)
    """
    print("Ethereum Address Derivation")
    print("=" * 70)
    
    # Step 1: Generate private key (same as Bitcoin)
    private_key = secrets.randbelow(2**256)
    print(f"\\n1. Private Key: {hex(private_key)[:32]}...")
    
    # Step 2: Derive public key (private_key × G)
    print("\\n2. Public Key = private_key × G")
    print("   (Elliptic curve multiplication on secp256k1)")
    
    # Simplified public key (real implementation computes curve point)
    public_key_x = hashlib.sha256(private_key.to_bytes(32, 'big')).digest()
    public_key_y = hashlib.sha256(public_key_x).digest()
    
    # Ethereum uses uncompressed public key (without 0x04 prefix for hashing)
    public_key = public_key_x + public_key_y
    print(f"   x: {public_key_x.hex()[:32]}...")
    print(f"   y: {public_key_y.hex()[:32]}...")
    
    # Step 3: Keccak-256 hash of public key
    print("\\n3. Keccak-256 Hash of Public Key")
    import hashlib
    # Note: Ethereum uses Keccak-256, not SHA3-256
    # (Python's sha3_256 is NIST SHA-3, different from Ethereum's Keccak)
    keccak_hash = hashlib.sha3_256(public_key).digest()
    print(f"   Keccak-256: {keccak_hash.hex()[:32]}...")
    
    # Step 4: Take last 20 bytes as address
    address = keccak_hash[-20:]
    print("\\n4. Ethereum Address = Last 20 bytes of hash")
    print(f"   Address: 0x{address.hex()}")
    
    print("\\n" + "=" * 70)
    print("Differences from Bitcoin:")
    print("- Uses Keccak-256 instead of SHA-256 + RIPEMD-160")
    print("- No Base58 encoding (uses hex)")
    print("- No checksum in address (use EIP-55 checksumming)")
    print("- 20 bytes (40 hex chars) vs Bitcoin's 25 bytes")

ethereum_address_derivation()
\`\`\`

## Ed25519: Solana's Choice

Solana uses **Ed25519** instead of secp256k1:

\`\`\`python
from cryptography.hazmat.primitives.asymmetric import ed25519

def ed25519_demo():
    """
    Demonstrate Ed25519 signatures (used by Solana)
    """
    print("Ed25519 Signatures (Solana)")
    print("=" * 70)
    
    # Generate keypair
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    
    print("\\n1. Key Generation")
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    print(f"   Private Key (32 bytes): {private_bytes.hex()[:32]}...")
    print(f"   Public Key (32 bytes): {public_bytes.hex()[:32]}...")
    
    # Sign message
    message = b"Solana transaction data"
    signature = private_key.sign(message)
    
    print("\\n2. Signature")
    print(f"   Message: {message.decode()}")
    print(f"   Signature (64 bytes): {signature.hex()[:32]}...")
    
    # Verify
    try:
        public_key.verify(signature, message)
        print("\\n3. Verification: ✓ VALID")
    except:
        print("\\n3. Verification: ✗ INVALID")
    
    print("\\n" + "=" * 70)
    print("Ed25519 vs secp256k1:")
    print("\\nAdvantages of Ed25519:")
    print("+ Faster signature generation")
    print("+ Faster signature verification")
    print("+ No nonce needed (deterministic)")
    print("+ Smaller signatures (64 bytes vs 71-73 bytes)")
    print("+ Easier to implement securely")
    
    print("\\nWhy Bitcoin uses secp256k1:")
    print("- Historical choice (2009)")
    print("- Ed25519 not widely known at the time")
    print("- Changing would require hard fork")
    print("- secp256k1 is still very secure")
    
    print("\\nWhy Solana uses Ed25519:")
    print("- Modern choice (2020)")
    print("- Better performance for high-throughput blockchain")
    print("- Simpler implementation")

# Note: Need to import serialization
from cryptography.hazmat.primitives import serialization
ed25519_demo()
\`\`\`

## Security Considerations

### Private Key Security

\`\`\`python
def private_key_security_best_practices():
    """
    Best practices for private key management
    """
    print("Private Key Security Best Practices")
    print("=" * 70)
    
    security_rules = {
        '1. Generation': {
            'DO': [
                'Use cryptographically secure random number generator',
                'Use os.urandom() or secrets module in Python',
                'Ensure sufficient entropy (256 bits for secp256k1)'
            ],
            'DON\'T': [
                'Use time-based seeds',
                'Use predictable random() function',
                'Generate from passwords without KDF',
                'Use brain wallets (human-chosen phrases)'
            ]
        },
        
        '2. Storage': {
            'DO': [
                'Encrypt private keys at rest',
                'Use hardware wallets for large amounts',
                'Use key derivation (BIP32 HD wallets)',
                'Split keys (Shamir Secret Sharing)'
            ],
            'DON\'T': [
                'Store in plaintext',
                'Email to yourself',
                'Store in cloud without encryption',
                'Screenshot on phone',
                'Store on internet-connected computer'
            ]
        },
        
        '3. Usage': {
            'DO': [
                'Use deterministic nonces (RFC 6979)',
                'Verify signature after creation',
                'Use well-audited libraries (libsecp256k1)',
                'Air-gap for high-value transactions'
            ],
            'DON\'T': [
                'Reuse nonces',
                'Sign untrusted data',
                'Implement crypto yourself',
                'Use unaudited libraries'
            ]
        }
    }
    
    for category, rules in security_rules.items():
        print(f"\\n{category}")
        print("-" * 40)
        print("DO:")
        for rule in rules['DO']:
            print(f"  ✓ {rule}")
        print("\\nDON'T:")
        for rule in rules['DON\\'T']:
            print(f"  ✗ {rule}")
    
    print("\\n" + "=" * 70)
    print("Historical Breaches Due to Bad Key Management:")
    print("- Mt. Gox (2014): $450M, hot wallet compromise")
    print("- Bitfinex (2016): $72M, multi-sig vulnerability")
    print("- Parity (2017): $150M, library self-destruct")
    print("- Poly Network (2021): $600M, private key compromise")

private_key_security_best_practices()
\`\`\`

## Practical Implementation: Bitcoin Transaction Signing

\`\`\`python
import hashlib

class BitcoinTransaction:
    """
    Simplified Bitcoin transaction signing
    """
    def __init__(self, from_address: str, to_address: str, amount: float):
        self.from_address = from_address
        self.to_address = to_address
        self.amount = amount
    
    def serialize(self) -> bytes:
        """Serialize transaction for signing"""
        return f"{self.from_address}{self.to_address}{self.amount}".encode()
    
    def sign(self, private_key: int) -> tuple:
        """
        Sign transaction with private key
        Returns (r, s) signature
        """
        # Step 1: Serialize transaction
        tx_bytes = self.serialize()
        
        # Step 2: Double SHA-256 hash
        hash1 = hashlib.sha256(tx_bytes).digest()
        hash2 = hashlib.sha256(hash1).digest()
        tx_hash = int.from_bytes(hash2, 'big')
        
        print("Transaction Signing")
        print("=" * 70)
        print(f"From: {self.from_address}")
        print(f"To: {self.to_address}")
        print(f"Amount: {self.amount} BTC")
        print(f"\\nTx Hash: {hash2.hex()[:32]}...")
        
        # Step 3: Sign with ECDSA
        n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        k = secrets.randbelow(n - 1) + 1
        
        r = pow(k, 1, n)
        s = (pow(k, -1, n) * (tx_hash + private_key * r)) % n
        
        print(f"\\nSignature:")
        print(f"  r: {hex(r)[:32]}...")
        print(f"  s: {hex(s)[:32]}...")
        
        return (r, s)
    
    def verify(self, public_key: str, signature: tuple) -> bool:
        """Verify transaction signature"""
        # Verification logic (simplified)
        print("\\nVerification: ✓ VALID")
        return True

# Create and sign transaction
tx = BitcoinTransaction(
    from_address="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    to_address="1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
    amount=0.5
)

# Sign with private key
private_key = secrets.randbelow(2**256)
signature = tx.sign(private_key)

# Verify
public_key = "derived_from_private_key"
tx.verify(public_key, signature)
\`\`\`

## Summary

Public key cryptography enables blockchain through mathematical one-way functions:

1. **Asymmetric Encryption**: Public key encrypts, private key decrypts
2. **Elliptic Curves**: Efficient cryptography with small keys (256 bits)
3. **secp256k1**: Bitcoin's curve (y² = x³ + 7)
4. **ECDSA**: Digital signature algorithm for proving ownership
5. **Ed25519**: Alternative used by Solana (faster, simpler)
6. **Key Security**: Critical - compromise means total loss of funds

**Key Equations**:
- Public Key = Private Key × G (easy direction)
- Private Key = Public Key ÷ G (impossible - discrete log problem)
- Signature: (r, s) proves knowledge of private key without revealing it

**Security Properties**:
- ✓ Private → Public: Easy
- ✗ Public → Private: Impossible
- ✓ Sign with private key: Easy
- ✗ Forge signature without private key: Impossible
- ⚠️ Nonce reuse: Catastrophic - reveals private key!

In the next section, we'll explore **Distributed Consensus Fundamentals** and how thousands of nodes agree on a single truth without central authority.

## Further Reading

- **"Mastering Bitcoin"** by Andreas Antonopoulos: Chapter on Keys and Addresses
- **FIPS 186-4**: Digital Signature Standard (DSS)
- **SEC 2**: Recommended Elliptic Curve Domain Parameters
- **RFC 6979**: Deterministic Usage of ECDSA
- **BIP 32**: Hierarchical Deterministic Wallets

## Practice Exercises

1. Generate a Bitcoin keypair and derive the address
2. Implement ECDSA signature generation and verification
3. Demonstrate nonce reuse attack
4. Compare secp256k1 vs Ed25519 performance
5. Build a simple transaction signing system
`,
};

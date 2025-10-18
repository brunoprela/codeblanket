/**
 * Password Strength Validator
 * Problem ID: intermediate-password-manager
 * Order: 21
 */

import { Problem } from '../../../types';

export const intermediate_password_managerProblem: Problem = {
  id: 'intermediate-password-manager',
  title: 'Password Strength Validator',
  difficulty: 'Medium',
  description: `Create a comprehensive password strength validator and generator.

**Validation Criteria:**
- Minimum length (8+ characters)
- Contains uppercase and lowercase
- Contains numbers
- Contains special characters
- Not a common password
- Calculate strength score (0-100)

**Features:**
- Validate password strength
- Generate strong passwords
- Suggest improvements
- Check against common passwords`,
  examples: [
    {
      input: 'validate_password("P@ssw0rd123")',
      output: 'Strong (score: 85)',
    },
  ],
  constraints: [
    'Check all criteria',
    'Calculate numeric score',
    'Provide specific feedback',
  ],
  hints: [
    'Use regex for pattern matching',
    'Award points for each criterion',
    'Check against common password list',
  ],
  starterCode: `import re
import random
import string

class PasswordValidator:
    """
    Validate and score password strength.
    
    Examples:
        >>> validator = PasswordValidator()
        >>> result = validator.validate("MyP@ssw0rd")
        >>> print(result['score'])
    """
    
    # Common passwords to check against
    COMMON_PASSWORDS = {
        'password', '123456', '12345678', 'qwerty', 'abc123',
        'monkey', '1234567', 'letmein', 'trustno1', 'dragon'
    }
    
    def __init__(self, min_length=8):
        """
        Initialize validator.
        
        Args:
            min_length: Minimum password length
        """
        self.min_length = min_length
    
    def validate(self, password):
        """
        Validate password and return detailed results.
        
        Args:
            password: Password to validate
            
        Returns:
            Dict with score, strength, and suggestions
            
        Examples:
            >>> validator.validate("P@ssw0rd123")
            {'score': 85, 'strength': 'Strong', 'passed': True, ...}
        """
        pass
    
    def _calculate_score(self, password):
        """Calculate password strength score (0-100)."""
        pass
    
    def _get_strength_level(self, score):
        """Convert score to strength level."""
        pass
    
    def _get_suggestions(self, password):
        """Get list of improvements."""
        pass
    
    def generate_password(self, length=12, use_special=True):
        """
        Generate a strong random password.
        
        Args:
            length: Password length
            use_special: Include special characters
            
        Returns:
            Generated password string
            
        Examples:
            >>> validator.generate_password(16)
            'K9@mPz!vXcQ#w8Rt'
        """
        pass


# Test
validator = PasswordValidator()

test_passwords = [
    "weak",
    "password123",
    "MyPassword1",
    "Str0ng!P@ss",
    "C0mpl3x!P@ssw0rd#2024"
]

print("Password Strength Analysis:")
print("=" * 70)

for pwd in test_passwords:
    result = validator.validate(pwd)
    print(f"\\nPassword: {pwd}")
    print(f"Score: {result['score']}/100")
    print(f"Strength: {result['strength']}")
    print(f"Passed: {result['passed']}")
    
    if result['suggestions']:
        print("Suggestions:")
        for suggestion in result['suggestions']:
            print(f"  - {suggestion}")

# Generate strong passwords
print("\\n\\nGenerated Strong Passwords:")
print("=" * 70)
for i in range(5):
    pwd = validator.generate_password(16)
    result = validator.validate(pwd)
    print(f"{pwd} (Score: {result['score']})")


# Test helper function (for automated testing)
def test_password_validator(password):
    """Test function for PasswordValidator - implement the class methods above first!"""
    try:
        validator = PasswordValidator()
        result = validator.validate(password)
        return result['passed']
    except:
        return None  # Return None if methods not yet implemented
`,
  testCases: [
    {
      input: ['Str0ng!P@ss'],
      expected: true,
      functionName: 'test_password_validator',
    },
  ],
  solution: `import re
import random
import string

class PasswordValidator:
    COMMON_PASSWORDS = {
        'password', '123456', '12345678', 'qwerty', 'abc123',
        'monkey', '1234567', 'letmein', 'trustno1', 'dragon',
        'password123', 'password1', 'admin', 'welcome', 'login'
    }
    
    def __init__(self, min_length=8):
        self.min_length = min_length
    
    def validate(self, password):
        score = self._calculate_score(password)
        strength = self._get_strength_level(score)
        suggestions = self._get_suggestions(password)
        
        return {
            'score': score,
            'strength': strength,
            'passed': score >= 60,
            'suggestions': suggestions,
            'criteria': {
                'length': len(password) >= self.min_length,
                'uppercase': bool(re.search(r'[A-Z]', password)),
                'lowercase': bool(re.search(r'[a-z]', password)),
                'numbers': bool(re.search(r'[0-9]', password)),
                'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)),
                'not_common': password.lower() not in self.COMMON_PASSWORDS
            }
        }
    
    def _calculate_score(self, password):
        score = 0
        
        # Length (max 30 points)
        if len(password) >= self.min_length:
            score += min(30, (len(password) - self.min_length + 1) * 5)
        
        # Character variety (max 40 points)
        if re.search(r'[a-z]', password):
            score += 10
        if re.search(r'[A-Z]', password):
            score += 10
        if re.search(r'[0-9]', password):
            score += 10
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 10
        
        # Complexity bonus (max 20 points)
        unique_chars = len(set(password))
        score += min(20, unique_chars)
        
        # Penalty for common passwords
        if password.lower() in self.COMMON_PASSWORDS:
            score = min(score, 30)
        
        # Penalty for repeated characters
        if re.search(r'(.)\\1{2,}', password):
            score -= 10
        
        return max(0, min(100, score))
    
    def _get_strength_level(self, score):
        if score >= 80:
            return 'Very Strong'
        elif score >= 60:
            return 'Strong'
        elif score >= 40:
            return 'Moderate'
        elif score >= 20:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def _get_suggestions(self, password):
        suggestions = []
        
        if len(password) < self.min_length:
            suggestions.append(f"Increase length to at least {self.min_length} characters")
        
        if not re.search(r'[a-z]', password):
            suggestions.append("Add lowercase letters")
        
        if not re.search(r'[A-Z]', password):
            suggestions.append("Add uppercase letters")
        
        if not re.search(r'[0-9]', password):
            suggestions.append("Add numbers")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            suggestions.append("Add special characters (!@#$%^&*)")
        
        if password.lower() in self.COMMON_PASSWORDS:
            suggestions.append("Avoid common passwords")
        
        if re.search(r'(.)\\1{2,}', password):
            suggestions.append("Avoid repeated characters")
        
        if len(password) < 12:
            suggestions.append("Consider using 12+ characters for better security")
        
        return suggestions
    
    def generate_password(self, length=12, use_special=True):
        # Ensure we have at least one of each required type
        chars = []
        
        # Add required characters
        chars.append(random.choice(string.ascii_lowercase))
        chars.append(random.choice(string.ascii_uppercase))
        chars.append(random.choice(string.digits))
        
        if use_special:
            special_chars = '!@#$%^&*(),.?":{}|<>'
            chars.append(random.choice(special_chars))
        
        # Fill remaining with random mix
        pool = string.ascii_letters + string.digits
        if use_special:
            pool += '!@#$%^&*(),.?":{}|<>'
        
        remaining_length = length - len(chars)
        chars.extend(random.choices(pool, k=remaining_length))
        
        # Shuffle to avoid predictable pattern
        random.shuffle(chars)
        
        return ''.join(chars)


# Test helper function (for automated testing)
def test_password_validator(password):
    """Test function for PasswordValidator."""
    validator = PasswordValidator()
    result = validator.validate(password)
    return result['passed']`,
  timeComplexity: 'O(n) where n is password length',
  spaceComplexity: 'O(1)',
  order: 21,
  topic: 'Python Intermediate',
};

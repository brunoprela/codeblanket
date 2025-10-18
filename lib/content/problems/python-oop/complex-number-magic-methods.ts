/**
 * Complex Number with Magic Methods
 * Problem ID: complex-number-magic-methods
 * Order: 14
 */

import { Problem } from '../../../types';

export const complex_number_magic_methodsProblem: Problem = {
  id: 'complex-number-magic-methods',
  title: 'Complex Number with Magic Methods',
  difficulty: 'Medium',
  category: 'python-oop',
  description: `Create a \`ComplexNumber\` class that implements magic methods for arithmetic operations, comparison, and string representation.

Implement these magic methods:
- \`__init__(real, imag)\`: Initialize with real and imaginary parts
- \`__str__()\`: Return user-friendly string like "3 + 4i"
- \`__repr__()\`: Return developer string like "ComplexNumber(3, 4)"
- \`__add__(other)\`: Add two complex numbers
- \`__sub__(other)\`: Subtract complex numbers
- \`__mul__(other)\`: Multiply complex numbers
- \`__eq__(other)\`: Check equality
- \`__abs__()\`: Return magnitude (distance from origin)

**Examples:**
\`\`\`python
c1 = ComplexNumber(3, 4)
c2 = ComplexNumber(1, 2)

print(c1)           # "3 + 4i"
print(repr(c1))     # "ComplexNumber(3, 4)"
print(c1 + c2)      # "4 + 6i"
print(c1 * c2)      # "-5 + 10i"  (3+4i)*(1+2i) = 3 + 6i + 4i + 8i² = -5 + 10i
print(abs(c1))      # 5.0  (sqrt(3² + 4²))
print(c1 == ComplexNumber(3, 4))  # True
\`\`\`

**Constraints:**
- Handle negative imaginary parts correctly in \`__str__\`
- \`__abs__\` should return a float`,
  starterCode: `class ComplexNumber:
    def __init__(self, real, imag):
        """Initialize complex number with real and imaginary parts."""
        pass
    
    def __str__(self):
        """Return user-friendly string representation."""
        pass
    
    def __repr__(self):
        """Return developer-friendly representation."""
        pass
    
    def __add__(self, other):
        """Add two complex numbers."""
        pass
    
    def __sub__(self, other):
        """Subtract complex numbers."""
        pass
    
    def __mul__(self, other):
        """Multiply complex numbers."""
        pass
    
    def __eq__(self, other):
        """Check equality."""
        pass
    
    def __abs__(self):
        """Return magnitude."""
        pass`,
  testCases: [
    {
      input: [
        ['ComplexNumber', 3, 4],
        ['ComplexNumber', 1, 2],
        ['add'],
        ['str'],
      ],
      expected: '4 + 6i',
    },
    {
      input: [['ComplexNumber', 3, 4], ['abs']],
      expected: 5.0,
    },
    {
      input: [
        ['ComplexNumber', 3, 4],
        ['ComplexNumber', 1, 2],
        ['multiply'],
        ['str'],
      ],
      expected: '-5 + 10i',
    },
  ],
  hints: [
    'For __str__, handle negative imaginary with f"{real} - {abs(imag)}i"',
    'For __mul__, use (a+bi)*(c+di) = (ac-bd) + (ad+bc)i',
    'For __abs__, use sqrt(real² + imag²)',
    'Always check isinstance(other, ComplexNumber) in operations',
  ],
  solution: `import math

class ComplexNumber:
    def __init__(self, real, imag):
        """Initialize complex number with real and imaginary parts."""
        self.real = real
        self.imag = imag
    
    def __str__(self):
        """Return user-friendly string representation."""
        if self.imag >= 0:
            return f"{self.real} + {self.imag}i"
        else:
            return f"{self.real} - {abs(self.imag)}i"
    
    def __repr__(self):
        """Return developer-friendly representation."""
        return f"ComplexNumber({self.real}, {self.imag})"
    
    def __add__(self, other):
        """Add two complex numbers."""
        if not isinstance(other, ComplexNumber):
            return NotImplemented
        return ComplexNumber(self.real + other.real, self.imag + other.imag)
    
    def __sub__(self, other):
        """Subtract complex numbers."""
        if not isinstance(other, ComplexNumber):
            return NotImplemented
        return ComplexNumber(self.real - other.real, self.imag - other.imag)
    
    def __mul__(self, other):
        """Multiply complex numbers: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i"""
        if not isinstance(other, ComplexNumber):
            return NotImplemented
        real_part = self.real * other.real - self.imag * other.imag
        imag_part = self.real * other.imag + self.imag * other.real
        return ComplexNumber(real_part, imag_part)
    
    def __eq__(self, other):
        """Check equality."""
        if not isinstance(other, ComplexNumber):
            return False
        return self.real == other.real and self.imag == other.imag
    
    def __abs__(self):
        """Return magnitude: sqrt(real² + imag²)"""
        return math.sqrt(self.real ** 2 + self.imag ** 2)


# Test
c1 = ComplexNumber(3, 4)
c2 = ComplexNumber(1, 2)
print(c1 + c2)      # 4 + 6i
print(c1 * c2)      # -5 + 10i
print(abs(c1))      # 5.0`,
  timeComplexity: 'O(1) for all operations',
  spaceComplexity: 'O(1)',
  order: 14,
  topic: 'Python Object-Oriented Programming',
};

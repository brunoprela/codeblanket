/**
 * Differentiation Rules Section
 */

export const differentiationrulesSection = {
  id: 'differentiation-rules',
  title: 'Differentiation Rules',
  content: `
# Differentiation Rules

## Introduction

Calculus provides powerful rules that make differentiation efficient. These rules are essential for backpropagation, where we compute millions of derivatives.

## Product Rule & Quotient Rule

**Product Rule**: (uv)' = u'v + uv'  
**Quotient Rule**: (u/v)' = (u'v - uv')/v²

\`\`\`python
import numpy as np
import sympy as sp

# Product rule example
x = sp.Symbol('x')
f = x**2 * sp.sin (x)
print(f"d/dx[x² · sin (x)] = {sp.diff (f, x)}")

# Quotient rule example
g = sp.sin (x) / x
print(f"d/dx[sin (x)/x] = {sp.simplify (sp.diff (g, x))}")
\`\`\`

## Chain Rule - Heart of Backpropagation

**Rule**: (f∘g)'(x) = f'(g (x)) · g'(x)

\`\`\`python
# Example: d/dx[sin (x²)] = cos (x²) · 2x
f = sp.sin (x**2)
print(f"Chain rule: {sp.diff (f, x)}")

# Backpropagation example
class TwoLayerNet:
    def forward (self, x):
        self.x = x
        self.z1 = np.dot (self.W1, x)
        self.a1 = sigmoid (self.z1)
        self.z2 = np.dot (self.W2, self.a1)
        return self.z2
    
    def backward (self, dL_dout):
        # Chain rule layer by layer
        dL_dz2 = dL_dout
        dL_dW2 = np.outer (dL_dz2, self.a1)
        dL_da1 = np.dot (self.W2.T, dL_dz2)
        dL_dz1 = dL_da1 * sigmoid_derivative (self.z1)
        dL_dW1 = np.outer (dL_dz1, self.x)
        return dL_dW1, dL_dW2
\`\`\`

## Common Derivatives Reference

\`\`\`python
derivatives = {
    "x^n": "n·x^(n-1)",
    "e^x": "e^x",
    "ln (x)": "1/x",
    "sin (x)": "cos (x)",
    "cos (x)": "-sin (x)",
    "tan (x)": "sec²(x)",
    "arcsin (x)": "1/√(1-x²)",
    "arctan (x)": "1/(1+x²)"
}

for func, deriv in derivatives.items():
    print(f"d/dx[{func}] = {deriv}")
\`\`\`

## Summary

The chain rule is fundamental to backpropagation. Master it!
`,
};

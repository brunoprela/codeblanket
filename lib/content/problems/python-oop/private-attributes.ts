/**
 * Private Attributes (Name Mangling)
 * Problem ID: oop-private-attributes
 * Order: 25
 */

import { Problem } from '../../../types';

export const private_attributesProblem: Problem = {
  id: 'oop-private-attributes',
  title: 'Private Attributes (Name Mangling)',
  difficulty: 'Easy',
  description: `Use name mangling for pseudo-private attributes.

**Name mangling:**
- __attribute becomes _ClassName__attribute
- Prevents accidental access
- Not truly private
- Convention: _ prefix for internal

This tests:
- Name mangling
- Encapsulation
- Private attributes`,
  examples: [
    {
      input: '__private_var',
      output: 'Name mangled to _Class__private_var',
    },
  ],
  constraints: ['Use double underscore prefix', 'Understand mangling'],
  hints: [
    '__var for name mangling',
    '_var for internal use',
    'Not truly private',
  ],
  starterCode: `class BankAccount:
    """Bank account with private balance"""
    def __init__(self, initial_balance):
        self.__balance = initial_balance  # Name mangled
    
    def deposit(self, amount):
        """Public method to deposit"""
        if amount > 0:
            self.__balance += amount
    
    def withdraw(self, amount):
        """Public method to withdraw"""
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return True
        return False
    
    def get_balance(self):
        """Public method to get balance"""
        return self.__balance


def test_private():
    """Test private attributes"""
    account = BankAccount(100)
    
    # Use public methods
    account.deposit(50)
    account.withdraw(30)
    
    # Get balance
    balance = account.get_balance()
    
    # Try direct access (would fail in real code)
    # account.__balance  # AttributeError
    
    return balance
`,
  testCases: [
    {
      input: [],
      expected: 120,
      functionName: 'test_private',
    },
  ],
  solution: `class BankAccount:
    def __init__(self, initial_balance):
        self.__balance = initial_balance
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return True
        return False
    
    def get_balance(self):
        return self.__balance


def test_private():
    account = BankAccount(100)
    account.deposit(50)
    account.withdraw(30)
    balance = account.get_balance()
    return balance`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 25,
  topic: 'Python Object-Oriented Programming',
};

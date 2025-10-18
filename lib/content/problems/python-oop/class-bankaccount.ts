/**
 * Bank Account Class
 * Problem ID: class-bankaccount
 * Order: 1
 */

import { Problem } from '../../../types';

export const class_bankaccountProblem: Problem = {
  id: 'class-bankaccount',
  title: 'Bank Account Class',
  difficulty: 'Easy',
  description: `Implement a \`BankAccount\` class with proper encapsulation and methods.

The class should:
- Store account holder name and balance (private)
- Provide deposit() and withdraw() methods
- Prevent negative balance with withdraw()
- Implement __str__ for readable output
- Use properties for controlled access

**Requirements:**
- Balance should be private (_balance)
- Withdraw should return True/False for success
- Deposit should only accept positive amounts`,
  examples: [
    {
      input: 'account = BankAccount("Alice", 1000); account.withdraw(200)',
      output: 'True, balance becomes 800',
    },
    {
      input: 'account.withdraw(2000)',
      output: 'False, insufficient funds',
    },
  ],
  constraints: [
    'Balance must be non-negative',
    'Deposit must be positive',
    'Use encapsulation (private attributes)',
  ],
  hints: [
    'Use _balance for private attribute',
    'Check balance before withdrawing',
    'Return True/False to indicate success',
  ],
  starterCode: `class BankAccount:
    """
    Bank account with deposit and withdraw operations.
    """
    
    def __init__(self, name, initial_balance=0):
        """
        Initialize account.
        
        Args:
            name: Account holder name
            initial_balance: Starting balance (default 0)
        """
        # Your code here
        pass
    
    def deposit(self, amount):
        """
        Deposit money into account.
        
        Args:
            amount: Amount to deposit
            
        Raises:
            ValueError: If amount is not positive
        """
        # Your code here
        pass
    
    def withdraw(self, amount):
        """
        Withdraw money from account.
        
        Args:
            amount: Amount to withdraw
            
        Returns:
            True if successful, False if insufficient funds
        """
        # Your code here
        pass
    
    @property
    def balance(self):
        """Get current balance."""
        # Your code here
        pass
    
    def __str__(self):
        """String representation."""
        # Your code here
        pass


# Test
account = BankAccount("Alice", 1000)
print(account)  # BankAccount(Alice, balance=1000)
account.deposit(500)
print(account.balance)  # 1500
print(account.withdraw(200))  # True
print(account.balance)  # 1300
print(account.withdraw(2000))  # False


def test_bank_account(name, initial_balance, deposit_amount, withdraw_amount):
    """Test function for BankAccount class."""
    account = BankAccount(name, initial_balance)
    account.deposit(deposit_amount)
    result = account.withdraw(withdraw_amount)
    if not result:
        return False
    return account.balance
`,
  testCases: [
    {
      input: ['Alice', 1000, 500, 200],
      expected: 1300,
      functionName: 'test_bank_account',
    },
    {
      input: ['Bob', 100, 0, 200],
      expected: false, // withdraw fails
      functionName: 'test_bank_account',
    },
  ],
  solution: `class BankAccount:
    def __init__(self, name, initial_balance=0):
        self.name = name
        self._balance = initial_balance
    
    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
    
    def withdraw(self, amount):
        if amount > self._balance:
            return False
        self._balance -= amount
        return True
    
    @property
    def balance(self):
        return self._balance
    
    def __str__(self):
        return f"BankAccount({self.name}, balance={self._balance})"


def test_bank_account(name, initial_balance, deposit_amount, withdraw_amount):
    """Test function for BankAccount class."""
    account = BankAccount(name, initial_balance)
    account.deposit(deposit_amount)
    result = account.withdraw(withdraw_amount)
    if not result:
        return False
    return account.balance`,
  timeComplexity: 'O(1) for all operations',
  spaceComplexity: 'O(1)',
  order: 1,
  topic: 'Python Object-Oriented Programming',
};

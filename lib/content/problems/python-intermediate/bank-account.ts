/**
 * Bank Account with File Persistence
 * Problem ID: intermediate-bank-account
 * Order: 9
 */

import { Problem } from '../../../types';

export const intermediate_bank_accountProblem: Problem = {
  id: 'intermediate-bank-account',
  title: 'Bank Account with File Persistence',
  difficulty: 'Medium',
  description: `Create a BankAccount class that persists transactions to a file.

**Features:**
- Deposit and withdraw money
- Check balance
- View transaction history
- Save/load state from JSON file
- Prevent overdrafts

**Transaction Format:**
\`\`\`python
{
    "timestamp": "2024-01-15 10:30:45",
    "type": "deposit",
    "amount": 100.00,
    "balance": 1100.00
}
\`\`\``,
  examples: [
    {
      input: 'account.deposit(100)',
      output: 'New balance: 1100.00',
    },
  ],
  constraints: [
    'Prevent negative balance',
    'Track all transactions',
    'Persist to file',
  ],
  hints: [
    'Use datetime for timestamps',
    'Store transactions as list',
    'Use JSON for persistence',
  ],
  starterCode: `import json
from datetime import datetime

class InsufficientFundsError(Exception):
    """Raised when withdrawal exceeds balance."""
    pass


class BankAccount:
    """Bank account with file persistence."""
    
    def __init__(self, account_number, initial_balance=0, filename=None):
        """
        Initialize bank account.
        
        Args:
            account_number: Account identifier
            initial_balance: Starting balance
            filename: Optional file for persistence
        """
        self.account_number = account_number
        self.balance = initial_balance
        self.transactions = []
        self.filename = filename or f"account_{account_number}.json"
        # Note: load() intentionally not called in starter to avoid file errors
    
    def deposit(self, amount):
        """
        Deposit money into account.
        
        Args:
            amount: Amount to deposit
            
        Raises:
            ValueError: If amount is negative
        """
        # TODO: Implement deposit logic
        # - Validate amount is positive
        # - Add to balance
        # - Add transaction
        # - Save to file
        pass
    
    def withdraw(self, amount):
        """
        Withdraw money from account.
        
        Args:
            amount: Amount to withdraw
            
        Raises:
            ValueError: If amount is negative
            InsufficientFundsError: If balance is insufficient
        """
        # TODO: Implement withdrawal logic
        # - Validate amount is positive
        # - Check sufficient funds
        # - Subtract from balance
        # - Add transaction
        # - Save to file
        pass
    
    def get_balance(self):
        """Get current balance."""
        return self.balance
    
    def get_transactions(self):
        """Get transaction history."""
        return self.transactions
    
    def save(self):
        """Save account state to file."""
        # TODO: Implement save logic
        # - Create dict with account data
        # - Write to JSON file
        pass
    
    def load(self):
        """Load account state from file."""
        # TODO: Implement load logic
        # - Read from JSON file
        # - Update balance and transactions
        # - Handle FileNotFoundError
        pass
    
    def _add_transaction(self, trans_type, amount):
        """Add transaction to history."""
        # TODO: Implement transaction logging
        # - Create transaction dict with timestamp, type, amount, balance
        # - Append to transactions list
        pass


# Test helper function (for automated testing)
def test_bank_account(initial_balance, deposit_amount):
    """Test function for BankAccount - implement the class methods above first!"""
    try:
        account = BankAccount("test123", initial_balance, "test_account.json")
        account.deposit(deposit_amount)
        return account.get_balance()
    except:
        return None  # Return None if methods not yet implemented
`,
  testCases: [
    {
      input: [1000, 500],
      expected: 1500,
      functionName: 'test_bank_account',
    },
  ],
  solution: `import json
from datetime import datetime

class InsufficientFundsError(Exception):
    """Raised when withdrawal exceeds balance."""
    pass


class BankAccount:
    def __init__(self, account_number, initial_balance=0, filename=None):
        self.account_number = account_number
        self.balance = initial_balance
        self.transactions = []
        self.filename = filename or f"account_{account_number}.json"

        self.load()

    def deposit(self, amount):
        if amount < 0:
            raise ValueError("Deposit amount must be positive")

        self.balance += amount
        self._add_transaction("deposit", amount)
        self.save()

    def withdraw(self, amount):
        if amount < 0:
            raise ValueError("Withdrawal amount must be positive")

        if amount > self.balance:
            raise InsufficientFundsError(
                f"Insufficient funds: balance {self.balance:.2f}, "
                f"withdrawal {amount:.2f}"
            )

        self.balance -= amount
        self._add_transaction("withdrawal", amount)
        self.save()

    def get_balance(self):
        return self.balance

    def get_transactions(self):
        return self.transactions

    def save(self):
        data = {
            "account_number": self.account_number,
            "balance": self.balance,
            "transactions": self.transactions
        }
        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                self.balance = data.get("balance", self.balance)
                self.transactions = data.get("transactions", [])
        except FileNotFoundError:
            # New account
            pass

    def _add_transaction(self, trans_type, amount):
        transaction = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": trans_type,
            "amount": amount,
            "balance": self.balance
        }
        self.transactions.append(transaction)


def test_bank_account(initial_balance, deposit_amount):
    """Test function for BankAccount."""
    account = BankAccount("test123", initial_balance, "test_account.json")
    account.deposit(deposit_amount)
    return account.get_balance()`,
  timeComplexity: 'O(1) for deposit/withdraw, O(n) for save',
  spaceComplexity: 'O(t) where t is number of transactions',
  order: 9,
  topic: 'Python Intermediate',
};

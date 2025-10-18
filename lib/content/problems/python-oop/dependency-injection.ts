/**
 * Dependency Injection
 * Problem ID: oop-dependency-injection
 * Order: 34
 */

import { Problem } from '../../../types';

export const dependency_injectionProblem: Problem = {
  id: 'oop-dependency-injection',
  title: 'Dependency Injection',
  difficulty: 'Medium',
  description: `Use dependency injection for loose coupling.

**Pattern:**
- Dependencies passed to class
- Not created internally
- Easier testing
- More flexible

This tests:
- Dependency injection
- Loose coupling
- Design principles`,
  examples: [
    {
      input: 'Pass dependencies via constructor',
      output: "Class doesn't create dependencies",
    },
  ],
  constraints: ['Inject dependencies', "Don't create internally"],
  hints: [
    'Pass via __init__',
    'Store as attributes',
    'Use interfaces/protocols',
  ],
  starterCode: `class EmailService:
    """Service to send emails"""
    def send(self, to, message):
        return f"Email sent to {to}: {message}"


class SMSService:
    """Service to send SMS"""
    def send(self, to, message):
        return f"SMS sent to {to}: {message}"


class NotificationManager:
    """Manages notifications using injected service"""
    def __init__(self, notification_service):
        # Dependency injected
        self.service = notification_service
    
    def notify(self, user, message):
        """Send notification using injected service"""
        return self.service.send(user, message)


def test_dependency_injection():
    """Test dependency injection"""
    # Inject email service
    email_service = EmailService()
    manager1 = NotificationManager(email_service)
    result1 = manager1.notify("alice@example.com", "Hello")
    
    # Inject SMS service (same manager interface)
    sms_service = SMSService()
    manager2 = NotificationManager(sms_service)
    result2 = manager2.notify("555-1234", "Hi")
    
    return len(result1) + len(result2)
`,
  testCases: [
    {
      input: [],
      expected: 70,
      functionName: 'test_dependency_injection',
    },
  ],
  solution: `class EmailService:
    def send(self, to, message):
        return f"Email sent to {to}: {message}"


class SMSService:
    def send(self, to, message):
        return f"SMS sent to {to}: {message}"


class NotificationManager:
    def __init__(self, notification_service):
        self.service = notification_service
    
    def notify(self, user, message):
        return self.service.send(user, message)


def test_dependency_injection():
    email_service = EmailService()
    manager1 = NotificationManager(email_service)
    result1 = manager1.notify("alice@example.com", "Hello")
    
    sms_service = SMSService()
    manager2 = NotificationManager(sms_service)
    result2 = manager2.notify("555-1234", "Hi")
    
    return len(result1) + len(result2)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 34,
  topic: 'Python Object-Oriented Programming',
};

export const SignalsHooksMultipleChoice = {
  title: 'Signals & Hooks - Multiple Choice Questions',
  questions: [
    {
      question:
        'When should you use transaction.on_commit() with Django signals?',
      options: [
        'A) When you want the signal to execute faster',
        'B) When performing side effects like sending emails',
        'C) When you need to validate data before saving',
        'D) When implementing signal receivers for multiple models',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) When performing side effects like sending emails**

Use \`transaction.on_commit()\` to ensure side effects only happen after the database transaction successfully commits. This prevents sending emails or calling external APIs for operations that might be rolled back.

\`\`\`python
@receiver(post_save, sender=Order)
def send_confirmation(sender, instance, created, **kwargs):
    if created:
        # Only send email after transaction commits
        transaction.on_commit(
            lambda: send_email(instance.id)
        )
\`\`\`

Without this, you might send confirmation emails for orders that fail to save.
      `,
    },
    {
      question: 'What is the main disadvantage of using signals in Django?',
      options: [
        'A) Signals are slower than direct method calls',
        'B) Signals make code harder to debug and trace',
        'C) Signals cannot access request context',
        'D) Signals only work with built-in Django models',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) Signals make code harder to debug and trace**

Signals create "action at a distance" - code execution happens implicitly when models are saved/deleted, making it harder to trace program flow and debug issues.

\`\`\`python
# Hard to know this triggers email sending
article.save()  # Where does the email get sent?

# vs explicit
article.save()
send_article_notification(article)  # Clear and traceable
\`\`\`

Use signals for cross-app communication and decoupling, but prefer explicit calls for core business logic.
      `,
    },
    {
      question:
        'How do you prevent infinite loops when a signal handler modifies and saves the sender instance?',
      options: [
        'A) Use pre_save instead of post_save',
        'B) Use update() instead of save() to avoid triggering signals',
        'C) Add a flag to track if the signal has already fired',
        'D) Disable signals temporarily with a context manager',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) Use update() instead of save() to avoid triggering signals**

\`update()\` performs database updates without triggering model signals, preventing infinite loops.

\`\`\`python
@receiver(post_save, sender=Article)
def update_counter(sender, instance, **kwargs):
    # ❌ This creates infinite loop
    # instance.view_count += 1
    # instance.save()  # Triggers post_save again!
    
    # ✅ This doesn't trigger signals
    Article.objects.filter(id=instance.id).update(
        view_count=F('view_count') + 1
    )
\`\`\`
      `,
    },
    {
      question:
        'Which signal should you use to perform cleanup before a model instance is deleted?',
      options: [
        'A) post_delete',
        'B) pre_delete',
        'C) pre_save',
        'D) m2m_changed',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) pre_delete**

\`pre_delete\` fires before the delete operation, allowing you to access related data before it's removed.

\`\`\`python
@receiver(pre_delete, sender=Article)
def cleanup_files(sender, instance, **kwargs):
    # Access files before article is deleted
    if instance.image:
        instance.image.delete()
    
    # Log deletion with full data still available
    AuditLog.objects.create(
        action='DELETE',
        data=instance.to_dict()
    )
\`\`\`

\`post_delete\` fires after deletion, when the instance no longer exists in the database.
      `,
    },
    {
      question:
        'What happens if you raise an exception inside a post_save signal handler?',
      options: [
        'A) The save operation is rolled back',
        'B) The exception is caught and logged automatically',
        'C) The exception propagates and the save completes',
        'D) Only the signal handler fails, the save succeeds',
      ],
      correctAnswer: 2,
      explanation: `
**Correct Answer: C) The exception propagates and the save completes**

In \`post_save\`, the model has already been saved to the database. If an exception is raised, it propagates up but doesn't undo the save (unless you're in an explicit transaction).

\`\`\`python
@receiver(post_save, sender=Article)
def process_article(sender, instance, **kwargs):
    if some_condition:
        raise ValueError("Processing failed")
    # Article is already saved!

# To make it atomic:
with transaction.atomic():
    article.save()  # Will rollback if signal raises exception
\`\`\`

For \`pre_save\`, raising an exception prevents the save. For \`post_save\`, wrap in transaction.atomic() if you need rollback capability.
      `,
    },
  ],
};

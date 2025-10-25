import { MultipleChoiceQuestion } from '@/lib/types';

export const sessionManagementMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-session-mc-1',
    question: 'What is the recommended value for autocommit in sessionmaker?',
    options: [
      'True (auto-commit after each statement)',
      'False (manual commits)',
      'Depends on use case',
      'Not specified (use default)',
    ],
    correctAnswer: 1,
    explanation:
      "autocommit=False (manual commits) is strongly recommended for production. With autocommit=False, you explicitly control transaction boundaries with session.commit(), making behavior predictable. With autocommit=True, each statement auto-commits, losing transaction semantics (can't rollback multiple operations together). Example problem: transfer money requires two UPDATEs in same transaction - autocommit=True commits each separately (money lost if crash between them). Always use autocommit=False and explicit session.commit().",
  },
  {
    id: 'sql-session-mc-2',
    question:
      'When does an object transition from PENDING to PERSISTENT state?',
    options: [
      'When session.add() is called',
      'When session.flush() or session.commit() is called',
      'When session.close() is called',
      'Immediately when object is created',
    ],
    correctAnswer: 1,
    explanation:
      'Objects transition from PENDING to PERSISTENT when session.flush() or session.commit() is executed. At this point, SQLAlchemy sends INSERT to database and object receives its primary key. PENDING: added to session but not in database (no PK). PERSISTENT: in session AND in database (has PK). You can verify: user = User(); session.add(user); print(user.id) # None (PENDING); session.flush(); print(user.id) # 1 (PERSISTENT).',
  },
  {
    id: 'sql-session-mc-3',
    question: 'What is the purpose of session.expire_all()?',
    options: [
      'Deletes all objects from database',
      'Closes the session',
      'Marks all objects as stale; they will reload from DB on next access',
      'Commits all pending changes',
    ],
    correctAnswer: 2,
    explanation:
      'session.expire_all() marks all objects in the session as stale (expired). On next attribute access, SQLAlchemy queries the database to reload fresh values. This is crucial for: (1) Memory management in batch processing - clears the identity map, (2) Ensuring fresh data after external changes, (3) Preventing stale data in long-lived sessions. Common pattern: for i in range(100000): process(); if i % 1000 == 0: session.commit(); session.expire_all() # Free memory.',
  },
  {
    id: 'sql-session-mc-4',
    question:
      "Why should you avoid creating a global session that's reused across requests?",
    options: [
      'It improves performance',
      'Causes thread safety issues, stale data, and connection leaks',
      "SQLAlchemy doesn't support it",
      "It's the recommended pattern",
    ],
    correctAnswer: 1,
    explanation:
      'Global sessions cause critical production issues: (1) Thread safety: Session is not thread-safe. Multiple requests accessing same session causes race conditions. (2) Stale data: Session caches objects. Request A sees old data modified by Request B. (3) Connection leaks: Never closed, holds database connection forever, exhausts pool. (4) Memory growth: Holds all queried objects in memory indefinitely. Correct pattern: One session per request, created fresh and closed after. Use dependency injection or context managers.',
  },
  {
    id: 'sql-session-mc-5',
    question: 'What happens to objects when session.close() is called?',
    options: [
      'Objects are deleted from database',
      'Objects become detached; connection returned to pool',
      'Session commits all changes',
      'Session rolls back all changes',
    ],
    correctAnswer: 1,
    explanation:
      'session.close() transitions all objects to DETACHED state and returns the database connection to the pool. Objects still exist in Python memory but are no longer tracked by session. Accessing lazy-loaded relationships raises DetachedInstanceError. Changes are NOT automatically committed or rolled back - you must call commit() or rollback() before close(). Best practice: Use context manager (with get_db_session()) to ensure proper commit/rollback/close sequence.',
  },
];

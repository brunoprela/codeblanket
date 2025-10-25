import { MultipleChoiceQuestion } from '@/lib/types';

export const definingModelsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-models-mc-1',
    question:
      'What is the purpose of the `back_populates` parameter in SQLAlchemy relationships?',
    options: [
      'It creates a backward foreign key',
      'It establishes bidirectional relationship navigation',
      'It enables cascade deletes',
      'It improves query performance',
    ],
    correctAnswer: 1,
    explanation:
      'back_populates establishes bidirectional relationship navigation between two models. If User has posts = relationship("Post", back_populates="user") and Post has user = relationship("User", back_populates="posts"), SQLAlchemy keeps both sides synchronized. When you set post.user = user, SQLAlchemy automatically adds post to user.posts. Without back_populates, you would manually manage both sides. It does not create foreign keys (Column does that), enable cascades (cascade parameter), or affect performance.',
  },
  {
    id: 'sql-models-mc-2',
    question:
      'When should you use UUID as a primary key instead of auto-incrementing integers?',
    options: [
      'Always, UUIDs are always better',
      'Never, integers are always faster',
      'For distributed systems where IDs are generated across multiple nodes',
      'Only for user authentication tables',
    ],
    correctAnswer: 2,
    explanation:
      'UUIDs are ideal for distributed systems because they can be generated independently on any node without coordination (no central sequence). This prevents bottlenecks and race conditions. Trade-offs: UUIDs are 16 bytes (vs 4 bytes for int), slightly slower for lookups, and random UUIDs cause index fragmentation. Integers are smaller, faster, and sequential (better for clustering). Use UUIDs when: distributed system, microservices, want client-side ID generation, need to prevent ID enumeration. Use integers when: single database, performance critical, prefer smaller storage.',
  },
  {
    id: 'sql-models-mc-3',
    question:
      'What does the `cascade="all, delete-orphan"` parameter do in a relationship?',
    options: [
      'Deletes parent when child is deleted',
      'Deletes children when parent is deleted, and children removed from parent',
      'Prevents deletion of parent if children exist',
      'Sets child foreign key to NULL',
    ],
    correctAnswer: 1,
    explanation:
      '"all, delete-orphan" cascades all operations (save, update, delete, merge, refresh) to children, AND deletes children when they are removed from the parent relationship. Example: user.posts has cascade="all, delete-orphan". When you delete user, all posts are deleted (cascade="delete"). When you remove a post from user.posts (post is now orphaned), it is also deleted (delete-orphan). This enforces strong ownership: children cannot exist without parent. Use for composition relationships where child\'s lifecycle is tied to parent.',
  },
  {
    id: 'sql-models-mc-4',
    question:
      'In a self-referential relationship (parent-child), what does the `remote_side` parameter specify?',
    options: [
      'Which column is the foreign key',
      'Which side of the relationship is the "many" side',
      'Which column represents the remote (parent) side',
      'Which database table is remote',
    ],
    correctAnswer: 2,
    explanation:
      'In self-referential relationships, SQLAlchemy cannot automatically determine which column is the "remote" (parent) side versus the "local" (child) side. remote_side=[id] specifies that the id column is the remote side (parent). Example: parent = relationship("Category", remote_side=[id], back_populates="children") means "join where this object\'s id matches the foreign key from the other side." Without remote_side, SQLAlchemy raises an error because both sides reference the same table.',
  },
  {
    id: 'sql-models-mc-5',
    question:
      'What is the difference between an association table and an association object for many-to-many relationships?',
    options: [
      'No difference, they are the same thing',
      'Association table is for simple M2M, association object adds extra columns',
      'Association object is faster',
      'Association table requires more code',
    ],
    correctAnswer: 1,
    explanation:
      'Association table: Simple Table() definition, no extra columns beyond foreign keys. Used with secondary parameter: relationship("Tag", secondary=post_tags). Cannot store extra data like timestamps or who created the association. Association object: Full class with extra columns (role, created_at, etc.). Provides access to metadata: for assoc in user.group_associations: print(assoc.role, assoc.joined_at). More flexible but requires more code. Use association table for simple M2M. Use association object when you need extra data on the relationship.',
  },
];

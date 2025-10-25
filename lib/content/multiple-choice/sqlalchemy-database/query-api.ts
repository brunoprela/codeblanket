import { MultipleChoiceQuestion } from '@/lib/types';

export const queryApiMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-query-mc-1',
    question:
      'What is the correct way to fetch User entities in SQLAlchemy 2.0?',
    options: [
      'session.query(User).all()',
      'session.execute(select(User)).scalars().all()',
      'session.select(User).all()',
      'User.query.all()',
    ],
    correctAnswer: 1,
    explanation:
      'SQLAlchemy 2.0 uses session.execute(select(User)).scalars().all(). The select() construct creates the statement, execute() runs it and returns Result, scalars() extracts the first column (User entities), and all() returns a list. session.query() is legacy 1.x API (still works but not recommended). Options 3 and 4 are invalid—session has no select method, User has no query attribute.',
  },
  {
    id: 'sql-query-mc-2',
    question: 'What does scalars() do in query result processing?',
    options: [
      'Scales the query to multiple databases',
      'Extracts the first column from Row objects',
      'Converts results to scalar values (int, str)',
      'Multiplies all numeric results by a factor',
    ],
    correctAnswer: 1,
    explanation:
      'scalars() extracts the first column from Result Row objects. session.execute(select(User)) returns Result[Row] where each Row is tuple-like. scalars() transforms it to Result[User], allowing direct iteration over entities. Without scalars(), you get Row objects: for row in result: user = row[0]. With scalars(): for user in result.scalars(): directly access user. Essential for entity queries.',
  },
  {
    id: 'sql-query-mc-3',
    question:
      'What is the difference between scalar_one() and scalar_one_or_none()?',
    options: [
      'scalar_one() returns one row, scalar_one_or_none() returns multiple',
      'scalar_one() raises if not found, scalar_one_or_none() returns None',
      'They are aliases for the same method',
      'scalar_one() is for integers, scalar_one_or_none() is for strings',
    ],
    correctAnswer: 1,
    explanation:
      'scalar_one() raises NoResultFound if zero rows, MultipleResultsFound if more than one. Use when you expect exactly one result. scalar_one_or_none() returns None if zero rows, raises MultipleResultsFound if more than one. Use when result is optional. Example: user = session.execute(select(User).where(User.id == 1)).scalar_one() - crashes if not found. user = session.execute(select(User).where(User.id == 1)).scalar_one_or_none() - returns None if not found.',
  },
  {
    id: 'sql-query-mc-4',
    question:
      'When using LIMIT and OFFSET for pagination, what happens at scale?',
    options: [
      'Performance remains constant regardless of offset size',
      'OFFSET requires scanning and discarding rows, causing linear degradation',
      'Database automatically optimizes large offsets',
      'Larger offsets are faster due to caching',
    ],
    correctAnswer: 1,
    explanation:
      'OFFSET requires the database to scan and discard rows before returning results. OFFSET 1000000 LIMIT 20 scans 1,000,000 rows, discards them, then returns 20. This causes linear degradation: page 1 (10ms), page 1000 (500ms), page 10000 (5s). Solution: Use cursor-based pagination (WHERE id > last_id) for constant-time seeks, or limit max pages for traditional pagination. Production apps typically cap pagination at 100-1000 pages.',
  },
  {
    id: 'sql-query-mc-5',
    question: 'What does GROUP BY require in the SELECT clause?',
    options: [
      'Only aggregate functions',
      'Only grouped columns',
      'Either grouped columns or aggregate functions',
      'All columns from the table',
    ],
    correctAnswer: 2,
    explanation:
      'SELECT with GROUP BY can only include: (1) columns in GROUP BY clause, or (2) aggregate functions. SELECT user_id, COUNT(*) FROM posts GROUP BY user_id - valid (user_id is grouped). SELECT user_id, title FROM posts GROUP BY user_id - invalid (title not grouped or aggregated). SELECT user_id, COUNT(*), MAX(created_at) FROM posts GROUP BY user_id - valid (COUNT and MAX are aggregates). This ensures deterministic results—without this rule, which title would be selected for each user_id?',
  },
];

import { MultipleChoiceQuestion } from '@/lib/types';

export const requestResponseModelsPydanticMultipleChoice = [
  {
    id: 'fastapi-pydantic-mc-1',
    question: 'What is the purpose of response_model in a FastAPI endpoint?',
    options: [
      'To validate the incoming request data',
      'To automatically filter and serialize the response, excluding fields not in the model',
      'To create database migrations',
      'To generate frontend TypeScript types',
    ],
    correctAnswer: 1,
    explanation:
      'response_model automatically filters and serializes response data to match the specified model. This is crucial for security and consistency: if you return a User database object with password_hash field but response_model=UserResponse (without password_hash), FastAPI automatically excludes password_hash from the response. It also handles: validation of returned data, serialization to JSON, OpenAPI documentation generation. Example: @app.get("/users/{id}", response_model=UserResponse) ensures only UserResponse fields are returned, even if the database model has more fields.',
  },
  {
    id: 'fastapi-pydantic-mc-2',
    question:
      'What is the difference between a Field with ... (Ellipsis) and None as default?',
    options: [
      'They are identical',
      '... makes the field required; None makes it optional with no default',
      'None is faster to validate',
      '... is deprecated in Pydantic v2',
    ],
    correctAnswer: 1,
    explanation:
      '... (Ellipsis) marks a field as required—must be provided, no default value. Example: name: str = Field(..., min_length=3) → name is required. None makes a field optional—can be omitted, defaults to None. Example: bio: Optional[str] = None → bio is optional. Missing required fields raise ValidationError: "field required". Note: Field default vs None: age: int = 0 (optional with default), age: int = None (invalid for int, should be Optional[int] = None). Always use Field(...) for required fields with constraints.',
  },
  {
    id: 'fastapi-pydantic-mc-3',
    question: 'When should you use @validator vs @root_validator in Pydantic?',
    options: [
      '@validator for single fields, @root_validator for cross-field validation',
      '@root_validator is faster and should always be used',
      '@validator is deprecated in Pydantic v2',
      'They are interchangeable',
    ],
    correctAnswer: 0,
    explanation:
      '@validator validates a single field in isolation. Use for: field-specific checks (password strength, username format), each_item=True for validating list items. Example: @validator("email") def check_email(cls, v): if not "@" in v: raise ValueError("Invalid email"). @root_validator validates entire model with access to all fields. Use for: cross-field validation (password == password_confirm), business rules involving multiple fields (shipping address required if has_physical_items). Example: @root_validator def check_passwords(cls, values): if values.get("password") != values.get("password_confirm"): raise ValueError("Passwords don\'t match"). Performance: @validator runs first (per field), @root_validator runs after all field validators pass.',
  },
  {
    id: 'fastapi-pydantic-mc-4',
    question: 'Why is Pydantic v2 significantly faster than v1?',
    options: [
      'It uses more caching',
      'It has fewer features',
      'Core validation logic is rewritten in Rust (pydantic-core)',
      'It only validates required fields',
    ],
    correctAnswer: 2,
    explanation:
      'Pydantic v2 uses pydantic-core, a Rust-based validation library, replacing Python-based validators. Rust advantages: Compiled to machine code (vs interpreted Python), no GIL limitations (Global Interpreter Lock), memory efficient (no Python object overhead), vectorized operations. Result: 5-17x faster validation, especially for complex models. Example benchmark: validating 10,000 simple models: v1 ~300ms, v2 ~50ms (6x faster). The Python API remains the same for backward compatibility—only the internal validation engine changed. Other v2 improvements: better error messages, improved JSON Schema generation, strict mode for exact type matching.',
  },
  {
    id: 'fastapi-pydantic-mc-5',
    question: 'What does orm_mode = True do in Pydantic Config?',
    options: [
      'Enables database query optimization',
      'Allows creating Pydantic models from ORM objects (like SQLAlchemy models)',
      'Automatically creates database tables',
      'Validates foreign key constraints',
    ],
    correctAnswer: 1,
    explanation:
      'orm_mode = True enables Pydantic to create instances from ORM objects that use attribute access instead of dict access. SQLAlchemy/Django ORM models use: user.username (attribute), not user["username"] (dict). With orm_mode: UserResponse.from_orm(db_user) works, converting SQLAlchemy model to Pydantic model. Without orm_mode: only UserResponse(**user.dict()) or UserResponse(**dict(user)) works. Example: class UserResponse(BaseModel): id: int; username: str; class Config: orm_mode = True. db_user = session.query(User).first() # SQLAlchemy model, response = UserResponse.from_orm(db_user) # Works! This is essential for FastAPI + SQLAlchemy integration, enabling: return db_user with response_model=UserResponse (FastAPI calls from_orm internally).',
  },
];

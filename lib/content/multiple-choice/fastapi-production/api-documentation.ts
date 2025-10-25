export const apiDocumentationMultipleChoice = {
  title: 'API Documentation - Multiple Choice',
  id: 'api-documentation-mc',
  questions: [
    {
      id: 1,
      question:
        "What is the primary advantage of FastAPI's automatic OpenAPI documentation generation?",
      options: [
        'Documentation stays in sync with code automatically, reducing maintenance burden and preventing drift',
        'It generates better documentation than manual writing',
        'It works offline without internet connection',
        'It supports more programming languages than other tools',
      ],
      correctAnswer: 0,
      explanation:
        'Auto-generated documentation from code ensures docs never drift out of sync. When you change endpoint signature, response model, or validation rules, documentation updates automatically. Manual docs require separate updates (often forgotten). FastAPI generates docs from Python type hints, Pydantic models, and docstrings—single source of truth.',
    },
    {
      id: 2,
      question:
        'How do you add examples to Pydantic models for better API documentation?',
      options: [
        'Use Field(example=...) for individual fields or Config.schema_extra for complete model examples',
        'Examples are automatically generated from field types',
        'Add examples in a separate JSON file',
        'Examples cannot be added to Pydantic models',
      ],
      correctAnswer: 0,
      explanation:
        'Pydantic supports examples via Field(example="value") for individual fields or Config.schema_extra={"example": {...}} for complete model examples. These appear in OpenAPI/Swagger UI, helping developers understand expected data format.',
    },
    {
      id: 3,
      question: 'What is the purpose of tags in FastAPI API documentation?',
      options: [
        'Tags organize endpoints into logical groups in the documentation UI for better navigation and discoverability',
        'Tags are used for access control and permissions',
        'Tags improve API performance',
        "Tags are only for internal use and don't appear in documentation",
      ],
      correctAnswer: 0,
      explanation:
        'Tags group related endpoints together in Swagger UI/ReDoc, making documentation easier to navigate. Example: "users", "authentication", "posts" tags organize endpoints by domain. Tags appear as sections in docs with descriptions and external links.',
    },
    {
      id: 4,
      question:
        'Why should deprecated endpoints be clearly marked in API documentation?',
      options: [
        'To warn developers the endpoint will be removed, allowing them to migrate to newer versions before breaking changes',
        'Deprecated endpoints are automatically faster',
        'It improves SEO for API documentation',
        "Deprecated endpoints don't need documentation",
      ],
      correctAnswer: 0,
      explanation:
        'Marking endpoints as deprecated (deprecated=True) provides advance notice of upcoming removal, preventing breaking changes from surprising API consumers. Include migration timeline, replacement endpoint, and migration guide links.',
    },
    {
      id: 5,
      question:
        'What tool can generate client SDKs from FastAPI OpenAPI specifications?',
      options: [
        'OpenAPI Generator can create client libraries in 40+ languages from the OpenAPI JSON schema',
        'Client SDKs must be manually written for each language',
        'FastAPI automatically generates all client SDKs',
        'Only JavaScript clients can be auto-generated',
      ],
      correctAnswer: 0,
      explanation:
        'OpenAPI Generator (openapi-generator-cli) generates client SDKs in Python, TypeScript, Go, Java, Ruby, PHP, etc. from /openapi.json. Ensures clients stay in sync with API and reduces manual SDK maintenance.',
    },
  ],
};

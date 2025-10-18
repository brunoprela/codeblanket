/**
 * Quiz questions for API Documentation section
 */

export const apidocumentationQuiz = [
  {
    id: 'docs-d1',
    question:
      'Design comprehensive API documentation for a payment processing API. Include authentication, error handling, webhooks, and code examples.',
    sampleAnswer: `Complete API documentation structure:

**1. Getting Started**: API keys, authentication, test vs production
**2. Authentication**: Bearer tokens, API key management
**3. Core Resources**: Payments, customers, refunds
**4. Code Examples**: JavaScript, Python, Ruby, cURL for each endpoint
**5. Error Handling**: Complete error code reference
**6. Webhooks**: Event types, payload examples, verification
**7. Rate Limiting**: Limits per plan, retry strategies
**8. Testing**: Test cards, sandbox environment
**9. SDKs**: Client libraries for popular languages
**10. Changelog**: Version history and migrations`,
    keyPoints: [
      'Clear getting started guide with authentication setup',
      'Comprehensive error code reference with solutions',
      'Code examples in multiple languages for each endpoint',
      'Webhook documentation with signature verification',
      'Test environment and sample data for development',
    ],
  },
  {
    id: 'docs-d2',
    question:
      'Your API has grown to 50+ endpoints and documentation is becoming hard to navigate. How would you organize and improve it?',
    sampleAnswer: `Documentation organization strategy:

**1. Categorization**: Group by resource (Users, Orders, Products)
**2. Search**: Full-text search across all docs
**3. Navigation**: Sidebar with collapsible sections
**4. Versioning**: Separate docs for each API version
**5. Tutorials**: Step-by-step guides for common workflows
**6. Reference**: Searchable endpoint reference
**7. Changelog**: What changed in each version
**8. Status Page**: Real-time API status
**9. Postman Collection**: Importable API collection
**10. SDK Docs**: Separate documentation for client libraries`,
    keyPoints: [
      'Group endpoints by resource type with clear navigation',
      'Add full-text search for quick discovery',
      'Provide both tutorials (workflows) and reference (endpoints)',
      'Version documentation separately for clarity',
      'Include Postman collections for easy testing',
    ],
  },
  {
    id: 'docs-d3',
    question:
      'Compare auto-generated documentation (Swagger) vs hand-written documentation (GitBook). When would you use each?',
    sampleAnswer: `Comparison:

**Auto-Generated (Swagger/OpenAPI)**:
- Pros: Always up-to-date, interactive, generated from code
- Cons: Limited customization, technical focus, no tutorials
- Use when: Fast-moving API, technical audience, need accuracy

**Hand-Written (GitBook/Docusaurus)**:
- Pros: Custom design, tutorials, narratives, examples
- Cons: Can become outdated, manual maintenance
- Use when: Need tutorials, marketing-focused, complex workflows

**Hybrid Approach** (Best):
- OpenAPI for API reference (auto-generated)
- GitBook for guides, tutorials, concepts
- Link between them
- Example: Stripe uses both

**Decision Matrix**:
- Internal APIs → Swagger (auto-generated)
- External APIs → Hybrid (reference + guides)
- Simple APIs → Swagger only
- Complex APIs → Hybrid with extensive guides`,
    keyPoints: [
      'Auto-generated ensures accuracy and stays up-to-date',
      'Hand-written provides better storytelling and tutorials',
      'Hybrid approach combines benefits of both',
      'Use auto-generated for reference, hand-written for guides',
      'Most successful APIs use hybrid documentation strategy',
    ],
  },
];

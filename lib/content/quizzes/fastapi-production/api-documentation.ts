export const apiDocumentationQuiz = {
  title: 'API Documentation - Discussion Questions',
  id: 'api-documentation-quiz',
  questions: [
    {
      id: 1,
      question:
        'Design comprehensive API documentation for a payment processing API that handles: credit card charges, refunds, webhooks, and subscription management. The documentation must include request/response examples for success and error cases, authentication requirements, rate limits, idempotency keys, and webhook signature verification. Show how you would organize this in FastAPI with tags, examples, and security schemes.',
      answer: `Complete payment API documentation with FastAPI showing organization, examples, security schemes, and error documentation for production payment processing API.`,
    },
    {
      id: 2,
      question:
        'Compare auto-generated API documentation (FastAPI/OpenAPI) versus manually written documentation (like Stripe or Twilio). What are the advantages and disadvantages of each approach? When would you supplement auto-generated docs with additional documentation? Design a documentation strategy that combines both approaches.',
      answer: `Analysis of auto-generated vs manual documentation with hybrid approach combining FastAPI OpenAPI generation with supplementary guides, tutorials, and SDKs.`,
    },
    {
      id: 3,
      question:
        'Design a documentation strategy for a public API that supports multiple programming languages (Python, JavaScript, Go, Ruby). The strategy should include: generating client SDKs from OpenAPI spec, versioning documentation, deprecation notices, migration guides, and code examples in each language. Implement the OpenAPI customization needed to generate high-quality SDKs.',
      answer: `Multi-language API documentation strategy with OpenAPI spec customization for SDK generation, versioning, and comprehensive developer experience across languages.`,
    },
  ],
};

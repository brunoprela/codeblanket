/**
 * API Governance Section
 */

export const apigovernanceSection = {
  id: 'api-governance',
  title: 'API Governance',
  content: `API governance ensures consistency, quality, and maintainability across all APIs in an organization. It defines standards, processes, and best practices.

## Why API Governance?

- **Consistency**: Same patterns across all APIs
- **Quality**: Maintain high standards
- **Security**: Enforce security practices
- **Discoverability**: Catalog all APIs
- **Compliance**: Meet regulatory requirements
- **Developer experience**: Predictable, well-documented APIs

## API Design Standards

### **REST API Conventions**

Document standards for your organization:

\`\`\`markdown
# API Design Standards

## Naming Conventions
- Use plural nouns: /users not /user
- Use kebab-case: /user-profiles not /userProfiles
- No trailing slashes: /users not /users/
- Resource nesting max 2 levels: /users/123/orders/456 ❌

## HTTP Methods
- GET: Retrieve resources
- POST: Create resources
- PUT: Replace entire resource
- PATCH: Partial update
- DELETE: Remove resource

## Response Codes
- 200: Success
- 201: Created
- 400: Client error (validation)
- 401: Unauthorized
- 403: Forbidden
- 404: Not found
- 500: Server error

## Pagination
- Use cursor-based for scale
- Parameters: \`limit\`, \`after\`
- Response includes: \`hasMore\`, \`endCursor\`

## Error Format
\`\`\`json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid email address",
    "details": [
      {
        "field": "email",
        "message": "Must be valid email"
      }
    ]
  }
}
\`\`\`

## Authentication
- Use Bearer tokens (JWT)
- Include \`Authorization: Bearer TOKEN\` header
- Tokens expire in 1 hour
\`\`\`

## API Catalog

Central registry of all APIs:

\`\`\`javascript
// api-catalog.json
{
  "apis": [
    {
      "name": "User API",
      "version": "v2",
      "baseUrl": "https://api.example.com/v2",
      "openApiSpec": "https://api.example.com/v2/openapi.json",
      "owner": "identity-team@example.com",
      "status": "active",
      "documentation": "https://docs.example.com/apis/users",
      "slackChannel": "#user-api-support"
    },
    {
      "name": "Payment API",
      "version": "v1",
      "baseUrl": "https://api.example.com/v1/payments",
      "openApiSpec": "https://api.example.com/v1/payments/openapi.json",
      "owner": "payments-team@example.com",
      "status": "deprecated",
      "deprecationDate": "2024-12-31",
      "migrationGuide": "https://docs.example.com/migrations/payments-v1-v2"
    }
  ]
}
\`\`\`

## API Lifecycle

Define clear stages:

\`\`\`
1. Design → OpenAPI spec review
2. Development → Code review, tests
3. Review → API review board approval
4. Alpha → Internal testing
5. Beta → Limited external access
6. GA → General availability
7. Deprecated → Sunset timeline announced
8. Sunset → API removed
\`\`\`

## API Review Process

**API Review Checklist**:

- ✅ Follows naming conventions
- ✅ Uses correct HTTP methods and status codes
- ✅ Implements authentication
- ✅ Rate limiting configured
- ✅ Error handling consistent
- ✅ OpenAPI spec complete
- ✅ Documentation written
- ✅ Tests written (80%+ coverage)
- ✅ Security review passed
- ✅ Load testing completed
- ✅ Monitoring configured

## Automated Governance

### **Linting OpenAPI Specs**

\`\`\`javascript
// .spectral.yaml
rules:
  operation-success-response:
    description: Operations must have a success response
    severity: error
    given: $.paths.*[get,post,put,patch,delete]
    then:
      field: responses
      function: truthy
  
  no-$ref-siblings:
    description: $ref cannot have siblings
    severity: error
  
  operation-tags:
    description: Operations must have tags
    severity: warn
  
  path-params-defined:
    description: Path parameters must be defined
    severity: error

// Run linter
$ spectral lint openapi.yaml
\`\`\`

### **Pre-commit Hooks**

\`\`\`javascript
// .husky/pre-commit
#!/bin/sh

# Lint OpenAPI specs
npm run lint:openapi

# Run tests
npm run test

# Check API design standards
npm run check:api-standards
\`\`\`

## Versioning Policy

**Semantic Versioning Rules**:

- **MAJOR**: Breaking changes (field removal, type change)
- **MINOR**: New features (new endpoints, optional fields)
- **PATCH**: Bug fixes (no API changes)

**Deprecation Policy**:
- Announce 6 months before sunset
- Support N-1 versions (current + previous)
- Provide migration guide
- Email all affected clients

## Best Practices

1. **Design-first approach**: OpenAPI spec before code
2. **Consistent patterns**: Same conventions across APIs
3. **Automated checks**: Lint specs, enforce standards
4. **API catalog**: Central registry
5. **Review process**: API review board approval
6. **Versioning policy**: Clear rules
7. **Deprecation timeline**: 6-12 months notice
8. **Documentation**: Always up-to-date
9. **Monitoring**: Track usage, errors, performance
10. **Security**: Regular security reviews`,
};

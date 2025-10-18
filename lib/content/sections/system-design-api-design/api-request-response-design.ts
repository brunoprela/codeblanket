/**
 * API Request/Response Design Section
 */

export const apirequestresponsedesignSection = {
  id: 'api-request-response-design',
  title: 'API Request/Response Design',
  content: `Designing clear, consistent request and response structures is crucial for API usability. Well-designed APIs are intuitive, efficient, and handle edge cases gracefully.

## Request Structure Best Practices

### **URL Structure**
\`\`\`
https://api.example.com/v1/resources/123?filter=value
\`\`\`

### **Pagination Strategies**

**Offset-Based**:
\`\`\`
GET /api/users?page=2&limit=20
\`\`\`

**Cursor-Based (Recommended for Scale)**:
\`\`\`
GET /api/users?cursor=eyJpZCI6MTIzfQ&limit=20
\`\`\`

**Key Differences**:
- Offset: Simple, can jump to any page, but slow at high offsets and inconsistent with real-time data
- Cursor: Fast, consistent results, but can't jump to arbitrary page

### **Filtering and Sorting**

\`\`\`
GET /api/users?role=admin&status=active
GET /api/users?age[gte]=18&age[lte]=65
GET /api/users?sort=created_at:desc
\`\`\`

### **Field Selection**

\`\`\`
GET /api/users/123?fields=id,name,email
\`\`\`

Reduces bandwidth and improves performance for mobile clients.

### **Error Response Standards**

\`\`\`json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {
        "field": "email",
        "message": "Email must be valid"
      }
    ]
  }
}
\`\`\``,
};

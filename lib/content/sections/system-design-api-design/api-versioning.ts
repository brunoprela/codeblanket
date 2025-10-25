/**
 * API Versioning Strategies Section
 */

export const apiversioningSection = {
  id: 'api-versioning',
  title: 'API Versioning Strategies',
  content: `API versioning enables evolving APIs while maintaining backward compatibility. Proper versioning prevents breaking changes from disrupting existing clients.

## Why Version APIs?

- **Backward compatibility**: Don't break existing clients
- **Gradual migration**: Give clients time to upgrade
- **Testing**: Test new versions before migrating
- **Deprecation**: Sunset old versions gracefully
- **Feature releases**: Ship features incrementally

## Versioning Strategies

### **1. URL Path Versioning** (Most Common)

Version in URL path:

\`\`\`
https://api.example.com/v1/users
https://api.example.com/v2/users
\`\`\`

**Implementation**:

\`\`\`javascript
// v1 routes
app.use('/v1/users', require('./routes/v1/users'));

// v2 routes
app.use('/v2/users', require('./routes/v2/users'));
\`\`\`

**Pros**:
- Clear, visible in URL
- Easy to route
- Works with HTTP caching
- Simple for clients

**Cons**:
- Pollutes URL namespace
- Version in every request

**Use when**: Public APIs, major version changes

### **2. Header Versioning**

Version in custom header:

\`\`\`http
GET /users HTTP/1.1
Host: api.example.com
Accept-Version: v2
\`\`\`

**Implementation**:

\`\`\`javascript
app.use('/users', (req, res, next) => {
  const version = req.headers['accept-version'] || 'v1';
  
  if (version === 'v2') {
    return require('./routes/v2/users')(req, res, next);
  }
  
  return require('./routes/v1/users')(req, res, next);
});
\`\`\`

**Pros**:
- Clean URLs
- Easy to add new versions
- RESTful (resource stays same)

**Cons**:
- Not visible in URL
- Harder to cache
- Requires header support

**Use when**: Internal APIs, semantic versioning

### **3. Query Parameter Versioning**

Version in query string:

\`\`\`
https://api.example.com/users?version=2
\`\`\`

**Pros**:
- Optional (default version)
- Easy for testing

**Cons**:
- Pollutes query params
- Not RESTful
- Caching issues

**Use when**: Rarely (not recommended)

### **4. Content Negotiation (Accept Header)**

Version via media type:

\`\`\`http
GET /users HTTP/1.1
Host: api.example.com
Accept: application/vnd.example.v2+json
\`\`\`

**Pros**:
- RESTful
- Follows HTTP standards
- Multiple versions per resource

**Cons**:
- Complex to implement
- Less discoverable
- Requires understanding of media types

**Use when**: Hypermedia APIs, academic correctness

## Breaking vs Non-Breaking Changes

### **Non-Breaking Changes** (Same Version)

Safe changes that don't require version bump:

- ✅ Adding new endpoints
- ✅ Adding optional parameters
- ✅ Adding new fields to responses
- ✅ Making required fields optional
- ✅ Adding new error codes
- ✅ Relaxing validation

**Example**:

\`\`\`javascript
// v1: Original
{
  "id": "123",
  "name": "John"
}

// v1: After adding field (non-breaking)
{
  "id": "123",
  "name": "John",
  "email": "john@example.com"  // NEW (clients ignore)
}
\`\`\`

### **Breaking Changes** (New Version Required)

Changes that break existing clients:

- ❌ Removing endpoints
- ❌ Removing fields from responses
- ❌ Renaming fields
- ❌ Changing field types
- ❌ Adding required parameters
- ❌ Changing error codes
- ❌ Tightening validation

**Example**:

\`\`\`javascript
// v1
{
  "full_name": "John Doe"  // Old field name
}

// v2 (breaking: renamed field)
{
  "name": "John Doe"  // New field name
}
\`\`\`

## Version Migration Strategy

### **1. Deprecation Notice**

Warn users before removing versions:

\`\`\`javascript
app.use('/v1/users', (req, res, next) => {
  res.set({
    'X-API-Deprecated': 'true',
    'X-API-Sunset': '2024-12-31',  // When it will be removed
    'X-API-Migration-Guide': 'https://docs.example.com/v1-to-v2'
  });
  
  next();
});
\`\`\`

### **2. Parallel Run**

Support multiple versions simultaneously:

\`\`\`javascript
// Both versions available
app.use('/v1/users', v1UsersHandler);
app.use('/v2/users', v2UsersHandler);

// Migrate data between versions
app.get('/v1/users/:id', async (req, res) => {
  const user = await getUser (req.params.id);
  
  // Convert v2 data to v1 format
  res.json({
    full_name: user.name  // Map new field to old
  });
});
\`\`\`

### **3. Sunset Period**

Give clients time to migrate:

\`\`\`
Day 0: Announce v2, deprecate v1
Day 90: Warning headers on v1
Day 180: v1 returns 410 Gone for new clients
Day 270: v1 fully decommissioned
\`\`\`

### **4. Monitor Usage**

Track version adoption:

\`\`\`javascript
const versionUsage = new prometheus.Counter({
  name: 'api_version_usage_total',
  help: 'API calls per version',
  labelNames: ['version', 'endpoint']
});

app.use((req, res, next) => {
  const version = req.path.split('/')[1];  // Extract version from path
  versionUsage.labels (version, req.path).inc();
  next();
});
\`\`\`

## Semantic Versioning

Use semantic versioning (MAJOR.MINOR.PATCH):

\`\`\`
v2.1.3
│ │ │
│ │ └─ PATCH: Bug fixes, no API changes
│ └─── MINOR: New features, backward compatible
└───── MAJOR: Breaking changes
\`\`\`

**Examples**:
- v1.0.0 → v1.1.0: Added new endpoint (minor)
- v1.1.0 → v1.1.1: Fixed bug (patch)
- v1.1.1 → v2.0.0: Renamed field (major)

## Best Practices

1. **Use URL path versioning**: Most straightforward
2. **Version major changes only**: Don't version every change
3. **Default to latest stable**: For convenience
4. **Deprecation warnings**: Give advance notice
5. **Sunset timeline**: 6-12 months for migrations
6. **Document changes**: Clear migration guides
7. **Monitor usage**: Track version adoption
8. **Support N-1 versions**: Current + previous major version
9. **Test all versions**: Automated tests for each
10. **Semantic versioning**: Clear version meaning`,
};

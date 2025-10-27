/**
 * Drawing Effective Architecture Diagrams Section
 */

export const architecturediagramsSection = {
  id: 'architecture-diagrams',
  title: 'Drawing Effective Architecture Diagrams',
  content: `Visual communication is crucial in system design interviews. A well-drawn architecture diagram can convey complex systems clearly and demonstrate your design thinking.

## Why Diagrams Matter

### **Benefits:**
- **Shared understanding** between you and interviewer
- **Identify missing components** visually
- **Easier to discuss** specific parts
- **Shows communication skills** critical for senior roles

---

## Core Components to Draw

### **1. Client/User**
- Mobile app icon 📱
- Web browser icon 🖥️
- Label: "iOS/Android" or "Web Client"

### **2. Load Balancer**
- Box labeled "Load Balancer" or "LB"
- Shows traffic distribution

### **3. Application Servers**
- Multiple boxes: "API Server 1", "API Server 2", "API Server N"
- Or single box with "API Servers (N instances)"

### **4. Databases**
- Cylinder shape (traditional) or rectangle
- Label type: "PostgreSQL", "Cassandra", "MongoDB"
- Show primary/replica if relevant

### **5. Cache**
- Box labeled "Redis" or "Memcached"
- Often shown between app servers and database

### **6. Message Queue**
- Box labeled "Kafka", "RabbitMQ", "SQS"
- Shows async processing

### **7. Object Storage**
- Box labeled "S3", "Blob Storage"
- For media files

### **8. CDN**
- Cloud icon or box labeled "CloudFront", "Akamath", "CDN"
- Shows content delivery

---

## Drawing Conventions

### **Arrows:**
- **→ Solid arrow**: Request/response (synchronous)
- **⇢ Dashed arrow**: Async communication
- **⟷ Double arrow**: Bidirectional communication
- **Number arrows**: Show flow sequence (1, 2, 3...)

### **Grouping:**
- Draw box around related components
- Label: "Region 1", "Data Center", "Microservices"

### **Replication:**
- Multiple boxes or "×3" notation
- Primary ← → Replica arrows

---

## Example: Twitter Architecture Diagram

\`\`\`
            [Mobile / Web Clients]
        ↓
    [Load Balancer]
        ↓
    [API Servers] ←→[Redis Cache]
        ↓                ↓
    [Write Path][Read Path]
        ↓                ↓
    [Message Queue][Cassandra]
        ↓           (Timeline DB)
    [Fanout Workers]
        ↓
    [Cassandra]
        (Tweets DB)
        ↓
    [S3] →[CDN]
        (Media)
        \`\`\`

---

## Step-by-Step Drawing Process

### **Step 1: Start with Client**
Draw user/client at top or left

### **Step 2: Add Entry Point**
Load balancer as first component users hit

### **Step 3: Application Layer**
API servers handling business logic

### **Step 4: Data Layer**
Databases, caches, storage

### **Step 5: Add Auxiliary Services**
Message queues, background workers, CDN

### **Step 6: Label Everything**
Clear names for each component

### **Step 7: Add Arrows**
Show data flow with numbered sequence

---

## Common Patterns

### **Pattern 1: Basic Web App**
\`\`\`
    Client → LB → App Servers → DB
                   ↓
    Cache
        \`\`\`

### **Pattern 2: Read-Heavy System**
\`\`\`
    Client → LB → App → Cache(90 % hits)
                ↓
            DB Replicas (reads)
                ↓
            DB Primary (writes)
        \`\`\`

### **Pattern 3: Microservices**
\`\`\`
    Client → API Gateway →[User Service]
                     →[Post Service]
                     →[Feed Service]
                          ↓
    [Message Queue]
        \`\`\`

---

## Interview Tips

### **✅ Do:**
- Start simple, add complexity incrementally
- Label every component clearly
- Number the data flow (1, 2, 3...)
- Use whiteboard space efficiently
- Ask "Should I add more detail here?"

### **❌ Don't:**
- Draw everything at once (overwhelming)
- Use unclear abbreviations (what's "KC"?)
- Forget to label arrows
- Draw too small (hard to see)
- Erase and redraw constantly (shows poor planning)

---

## Practice Exercise

**Task:** Draw architecture for Instagram photo upload

**Components needed:**1. Mobile client
2. Load balancer
3. Upload service
4. Image processing service
5. Object storage (S3)
6. Metadata database
7. CDN
8. Cache

**Flow:**1. User uploads photo from mobile app
2. Goes through load balancer
3. Upload service receives image
4. Stores original in S3
5. Publishes to message queue
6. Image processing service creates thumbnails
7. Stores thumbnails in S3
8. Saves metadata in database
9. CDN caches images for fast delivery

**Practice drawing this in 5 minutes!**`,
};

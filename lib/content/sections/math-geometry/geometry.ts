/**
 * Coordinate Geometry Section
 */

export const geometrySection = {
  id: 'geometry',
  title: 'Coordinate Geometry',
  content: `**Essential Geometric Concepts:**

**1. Distance Formula:**
- Euclidean distance: sqrt((x2-x1)² + (y2-y1)²)
- Manhattan distance: |x2-x1| + |y2-y1|

**2. Area Calculations:**
- **Triangle**: 0.5 × base × height
- **Rectangle**: width × height
- **Circle**: π × r²
- **Polygon** (Shoelace formula):
\`\`\`python
def polygon_area (points):
    n = len (points)
    area = 0
    for i in range (n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs (area) / 2
\`\`\`

**3. Angle Calculations:**
- Dot product: a·b = |a||b|cos(θ)
- Cross product: Determines orientation
- Slope: (y2-y1)/(x2-x1)

**4. Point-Line Relationships:**
- Point on line segment
- Shortest distance from point to line
- Line intersection

**5. Convex Hull:**
- Find smallest convex polygon containing all points
- Graham scan: O(n log n)
- Jarvis march: O(nh) where h = hull size`,
};

/**
 * Matrix Manipulation Section
 */

export const matrixoperationsSection = {
  id: 'matrix-operations',
  title: 'Matrix Manipulation',
  content: `**Common Matrix Operations:**

**1. Matrix Traversal Patterns:**
- Row by row: Standard nested loops
- Column by column: Reverse loop order
- Diagonal: i+j or i-j patterns
- Spiral: Layer by layer from outside in
- Snake: Alternate row directions

**2. Matrix Rotation:**
- **90° Clockwise**: Transpose + Reverse each row
- **90° Counter-clockwise**: Reverse each row + Transpose
- **180°**: Reverse rows + Reverse columns

**3. Matrix Transpose:**
- Swap matrix[i][j] with matrix[j][i]
- Only need to process upper triangle

**4. Matrix Spiral:**
- Process layer by layer
- Track top, bottom, left, right boundaries
- Move in order: right → down → left → up

**5. In-Place Operations:**
- Many matrix problems require O(1) space
- Use clever swapping and boundary tracking`,
};

/**
 * Common BFS Patterns Section
 */

export const patternsSection = {
  id: 'patterns',
  title: 'Common BFS Patterns',
  content: `**Pattern 1: Level-by-Level Processing**
\`\`\`python
def level_order (root):
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        level_size = len (queue)  # Key: capture size
        
        for _ in range (level_size):
            node = queue.popleft()
            level.append (node.val)
            # Add children
        
        result.append (level)
\`\`\`

**Pattern 2: Shortest Path / Minimum Steps**
\`\`\`python
def min_steps (start, end):
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        state, steps = queue.popleft()
        if state == end:
            return steps
        
        for next_state in get_neighbors (state):
            if next_state not in visited:
                visited.add (next_state)
                queue.append((next_state, steps + 1))
\`\`\`

**Pattern 3: Multi-Source BFS**
\`\`\`python
def multi_source (grid, sources):
    queue = deque (sources)  # Start from all sources
    visited = set (sources)
    
    while queue:
        node = queue.popleft()
        for neighbor in get_neighbors (node):
            if neighbor not in visited:
                visited.add (neighbor)
                queue.append (neighbor)
\`\`\`

**Pattern 4: Bidirectional BFS**
\`\`\`python
def bidirectional_bfs (start, end):
    """Meet in the middle - faster for long paths"""
    if start == end:
        return 0
    
    front = {start}
    back = {end}
    distance = 0
    
    while front and back:
        distance += 1
        
        # Expand smaller frontier
        if len (front) > len (back):
            front, back = back, front
        
        next_front = set()
        for node in front:
            for neighbor in get_neighbors (node):
                if neighbor in back:
                    return distance
                if neighbor not in visited:
                    next_front.add (neighbor)
        
        front = next_front
\`\`\`

**Pattern 5: State Space BFS**
\`\`\`python
def min_moves (start_state):
    """BFS on implicit graph of states"""
    queue = deque([(start_state, 0)])
    visited = {start_state}
    
    while queue:
        state, moves = queue.popleft()
        
        if is_goal (state):
            return moves
        
        for next_state in generate_next_states (state):
            if next_state not in visited:
                visited.add (next_state)
                queue.append((next_state, moves + 1))
\`\`\``,
};

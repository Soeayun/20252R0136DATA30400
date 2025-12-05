import os
from collections import defaultdict

# Load hierarchy
DATA_DIR = "Amazon_products"
HIERARCHY_PATH = os.path.join(DATA_DIR, "class_hierarchy.txt")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.txt")

# Load class IDs
id2class = {}
with open(CLASSES_PATH, 'r') as f:
    for line in f:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) == 2:
                class_id, class_name = parts
                id2class[int(class_id)] = class_name

# Load edges
edges = []
with open(HIERARCHY_PATH, 'r') as f:
    for line in f:
        if line.strip():
            parent, child = map(int, line.strip().split('\t'))
            edges.append((parent, child))

# Build parent dictionary
parents_dict = defaultdict(list)
children_dict = defaultdict(list)
all_nodes = set()

for parent, child in edges:
    parents_dict[child].append(parent)
    children_dict[parent].append(child)
    all_nodes.add(parent)
    all_nodes.add(child)

# Find root nodes (nodes that have no parents and exist in the hierarchy)
root_nodes = []
for node in all_nodes:
    if node not in parents_dict:  # No parent = root
        root_nodes.append(node)

print(f"Found {len(root_nodes)} root nodes: {sorted(root_nodes)}")

# BFS from each root to find levels
from collections import deque

level_dict = {}

for root in root_nodes:
    queue = deque([(root, 0)])
    if root not in level_dict:  # Only process if not already assigned
        level_dict[root] = 0
    
    while queue:
        node, level = queue.popleft()
        for child in children_dict.get(node, []):
            if child not in level_dict:  # First time seeing this node
                level_dict[child] = level + 1
                queue.append((child, level + 1))
            else:
                # Node already has a level, take the minimum (closest to root)
                level_dict[child] = min(level_dict[child], level + 1)

# Count nodes at each level
level_counts = defaultdict(int)
for node, level in level_dict.items():
    level_counts[level] += 1

# Print results
print("=" * 50)
print("Hierarchy Level Statistics")
print("=" * 50)
for level in sorted(level_counts.keys()):
    print(f"Level {level}: {level_counts[level]} classes")

print(f"\nTotal classes: {len(id2class)}")
print(f"Classes in hierarchy: {len(level_dict)}")

# Find max depth
max_level = max(level_dict.values()) if level_dict else 0
print(f"Maximum depth: {max_level}")

# Show some examples at each level
print("\n" + "=" * 50)
print("Example Classes at Each Level")
print("=" * 50)

for level in range(min(5, max_level + 1)):
    nodes_at_level = [node for node, l in level_dict.items() if l == level]
    print(f"\nLevel {level} ({len(nodes_at_level)} classes, showing first 10):")
    for node in sorted(nodes_at_level)[:10]:
        class_name = id2class.get(node, "Unknown")
        print(f"  - ID {node}: {class_name}")

import random
import numpy as np
import re

def grid_to_label_ap(grid, ap_dict, ap_id):
    # Convert 2D slice to a label map with radius 0
    label_map = np.zeros(grid.shape, dtype= np.int8)
    label_map_viz = np.zeros(grid.shape, dtype= np.int8)
    pattern = r'reach\(object_(\d+)\)'
    num_ap = len(ap_dict)
    for key in ap_dict.keys():
        # Extract Object ID
        match = re.search(pattern, key)
        if match:
            object_id = int(match.group(1))
        else:
            object_id = -1
            
        if object_id >= 0:
            # Update values
            position = int(ap_id[ap_dict[key]])
            label_map[grid == object_id] = 1 << (num_ap - position - 1) 
            label_map_viz[grid == object_id] = label_map[grid == object_id] * 50 # To visualize in rviz

    return label_map, label_map_viz

def generate_label_map_radius(label_map, label_map_viz, radius=1):
    neighbor_label_map = np.zeros_like(label_map, dtype=np.int8)
    neighbor_label_map_viz = np.zeros_like(label_map, dtype=np.int8)
    rows, cols = label_map.shape

    # Direction vectors within the radius
    directions = [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1) if not (dr == 0 and dc == 0)]
    
    for row in range(rows):
        for col in range(cols):
            current_label = label_map[row][col]
            current_label_viz = label_map_viz[row][col]
            # Sum the neighbor labels within the radius
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                # Check if the neighbor is within bounds
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    neighbor_label = label_map[new_row][new_col]
                    neighbor_label_viz = label_map_viz[new_row][new_col]
                    # OR operation with the neighbor to include its bits
                    current_label |= neighbor_label
                    current_label_viz |= neighbor_label_viz
            neighbor_label_map[row][col] = current_label
            neighbor_label_map_viz[row][col] = current_label_viz
            
    return neighbor_label_map, neighbor_label_map_viz


def generate_label_map(semantic_map, ap_dict, ap_id, radius=1):
    
    label_map, label_map_viz = grid_to_label_ap(semantic_map, ap_dict, ap_id)
    label_map_radius, label_map_radius_viz = generate_label_map_radius(label_map, label_map_viz, radius)
    return label_map_radius, label_map_radius_viz

if __name__ == "__main__":
    # Test Label map
    radius = 2
    id_list = [72, 69,85, 45, 81, 62, 33, 12]
    test_array = np.random.choice(id_list, size=(10,10), replace=True)
    #semantic_slice = np.random.randint(0,73, size=(10,10), dtype=np.int8)
    ap_dict = {'reach(object_69)': 'p1', 'reach(object_62)': 'p2', 'reach(object_72)': 'p3'}
    print(f"\nLabel Map with Radius {radius}:")
    label_map = generate_label_map(test_array, ap_dict, radius)
    print(label_map)


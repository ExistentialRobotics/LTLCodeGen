import numpy as np
import cv2
import re
import math

def grid_to_label_ap(grid, ap_dict, ap_id, radius):
    """
    Converts 2D semantic map to label map with radius 0 and also
    finds the radius at each location. REturns label map and radius map
    """  
    label_map = np.zeros(grid.shape, dtype= np.int8)
    label_map_viz = np.zeros(grid.shape, dtype= np.int8)
    pattern = re.compile(r'reach\(object_(\d+)\)')
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
            ap_name = ap_dict[key]
            if ap_name not in ap_id:
                continue
            position = int(ap_id[ap_name])
            label_map[grid == object_id] = 1 << (num_ap - position - 1)
            label_map_viz[grid == object_id] = label_map[grid == object_id] * 50 # To visualize in rviz

    
    free_mask = (grid == -128).astype(np.uint8) 
    min_region_size = 10
    alpha = 1.2 # Alpha always greater than 1, adjust parameter for  controlling label radius
    # Connected component labeling
    num_labels, labels = cv2.connectedComponents(free_mask, connectivity=4)

    # Remove small regions
    cleaned_free_mask = np.zeros_like(free_mask)
    for lbl in range(1, num_labels):  # label 0 is background
        region = (labels == lbl)
        if np.sum(region) >= min_region_size:
            cleaned_free_mask[region] = 1

    # Create radius map
    object_mask = (cleaned_free_mask == 0).astype(np.uint8)
    radius_map = cv2.distanceTransform(object_mask, cv2.DIST_L2, 5)
    radius_map += (alpha *radius)
    radius_map[grid == -1] = 0
    
    return label_map, label_map_viz, radius_map

def generate_label_map_radius(label_map, label_map_viz, radius_map):
    """
    Given a 2D label map and radius map apply the label for all cells
    within the radius
    """
    neighbor_label_map = np.zeros_like(label_map, dtype=np.int8)
    neighbor_label_map_viz = np.zeros_like(label_map, dtype=np.int8)
    rows, cols = label_map.shape
    
    for row in range(rows):
        for col in range(cols):
            current_label = label_map[row][col]
            current_label_viz = label_map_viz[row][col]
            radius = int(math.ceil(radius_map[row][col]))
            # Direction vectors within the radius
            directions = [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1) if not (dr == 0 and dc == 0)]
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


def generate_label_map(semantic_map, ap_dict, ap_id, radius=1.0):
    
    label_map, label_map_viz, radius_map = grid_to_label_ap(semantic_map, ap_dict, ap_id, radius)
    label_map_radius, label_map_radius_viz = generate_label_map_radius(label_map, label_map_viz, radius_map)
    return label_map_radius, label_map_radius_viz

if __name__ == "__main__":
    # Test Label map
    radius = 2
    id_list = [72, 69,85, 45, 81, 62, 33, 12]
    test_array = np.random.choice(id_list, size=(10,10), replace=True)
    ap_dict = {'reach(object_69)': 'p1', 'reach(object_62)': 'p2', 'reach(object_72)': 'p3'}
    ap_id = {'p1' : 0, 'p2' : 1, 'p3' : 2}
    print(f"\nLabel Map with Radius {radius}:")
    label_map = generate_label_map(test_array, ap_dict, ap_id, radius)
    print(label_map)


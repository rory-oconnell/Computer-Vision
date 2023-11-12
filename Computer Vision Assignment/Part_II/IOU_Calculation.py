SIZE = 200
sum = 0

def calculate_iou_from_centers(center_a, center_b, SIZE):
    # Calculate the top-left and bottom-right coordinates of each box from the center points
    boxa = [center_a[0] - SIZE // 2, center_a[1] - SIZE // 2,
            center_a[0] + SIZE // 2, center_a[1] + SIZE // 2]
    boxb = [center_b[0] - SIZE // 2, center_b[1] - SIZE // 2,
            center_b[0] + SIZE // 2, center_b[1] + SIZE // 2]

    x1_min, y1_min, x1_max, y1_max = boxa
    x2_min, y2_min, x2_max, y2_max = boxb
    
    # Calculate the coordinates of the intersection rectangle
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    # Calculate the area of the intersection
    inter_width = max(0, x_inter_max - x_inter_min)
    inter_height = max(0, y_inter_max - y_inter_min)
    intersection_area = inter_width * inter_height
    
    # Calculate the areas of the bounding boxes
    boxa_area = (x1_max - x1_min) * (y1_max - y1_min)
    boxb_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate the area of union
    union_area = boxa_area + boxb_area - intersection_area
    
    # Calculate and return IOU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

# List of center points
centers = [
(1224, 490),(1228, 491),
(2562, 555),(2554, 560),
(223, 2739),(291, 2697),
(3742, 2762),(3709, 2680),
]

# Calculate IoU for each pair of centers
for i in range(0, len(centers), 2):
    center_a = centers[i]
    center_b = centers[i + 1]
    iou = calculate_iou_from_centers(center_a, center_b, SIZE)
    print(f"The IoU of the bounding boxes centered at {center_a} and {center_b} is: {iou}")
    sum += iou

print(f"The average IoU is: {sum / 4}")
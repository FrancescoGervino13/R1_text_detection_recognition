sensor_msgs/Image image
int32[] bounding_boxes  # Stores a flatten list of (x1, y1, x2, y2)
float32[] x_axes  # Stores the x-axis coordinates for each bounding box
float32[] y_axes  # Stores the y-axis coordinates for each bounding box
float32[] normal_vectors  # Stores the normal vectors for each bounding box
float32[] dimensions  # Stores the dimensions for each bounding box
float32[] centres  # Stores the centers for each bounding box
int32[] inliers_list  # Stores the number of inliers for each bounding box

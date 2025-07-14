import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
import numpy as np
import cv2
import torch
import message_filters
from rclpy.callback_groups import ReentrantCallbackGroup

from tf2_ros import TransformListener, Buffer

from sensor_msgs.msg import PointField
import math
from tf2_ros import TransformStamped
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

import open3d as o3d
import pyransac3d as pyrsc
import os

from paddleocr import PaddleOCR
import time

def get_plane_features(points_np: np.ndarray) :
    """
    Fits a plane to the input point cloud (Nx3), and returns:
    - Normal vector
    - Centre of inliers
    - Bounding box dimensions (width, height) in the plane
    """
    assert points_np.shape[1] == 3, "Input must be Nx3"

    # Fit plane using RANSAC
    plane = pyrsc.Plane()
    best_eq, best_inliers = plane.fit(points_np, thresh=0.01)

    # Extract inlier points
    inlier_points = points_np[best_inliers]

    inlier_pcd = o3d.geometry.PointCloud()
    inlier_pcd.points = o3d.utility.Vector3dVector(inlier_points)

    # Compute Oriented Bounding Box (OBB)
    obb = inlier_pcd.get_oriented_bounding_box()

    centre = np.asarray(obb.center)
    obb_axes = np.asarray(obb.R)
    obb_extent = np.asarray(obb.extent)
    x_axis = obb_axes[:, 0] * obb_extent[0]  # X-axis (width)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = obb_axes[:, 1] * obb_extent[1]  # Y-axis (height)
    y_axis = y_axis / np.linalg.norm(y_axis)
    normal_vector = obb_axes[:, 2] * obb_extent[2]  # Z-axis (depth)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    dim1, dim2, _ = obb.extent  # You can ignore depth (thickness)
    
    view_direction = np.array([0, 0, 1], dtype=np.float32)
    dot_product = np.dot(normal_vector, view_direction)
    if abs(dot_product) < 0.1 :
        if np.sign(normal_vector[0]) == np.sign(inlier_points[0][0]) :
            x_axis = -x_axis
            normal_vector = -normal_vector
    elif dot_product > 0 :
        x_axis = -x_axis
        normal_vector = -normal_vector

    if y_axis[1] > 0 :
        x_axis = -x_axis
        y_axis = -y_axis

    orientation = np.argmax(inlier_points.max(axis=0)-inlier_points.min(axis=0))
    if orientation == 1 :
        # Swap x and y axes if the orientation is along the y-axis
        x_axis, y_axis = y_axis, x_axis
        bbox_dims = np.array([dim2, dim1], dtype=np.float32)
    else :
        bbox_dims = np.array([dim1, dim2], dtype=np.float32)

    return normal_vector, centre, bbox_dims, x_axis, y_axis

def quaternion_matrix(quaternion):  #Copied from https://github.com/ros/geometry/blob/noetic-devel/tf/src/tf/transformations.py#L1515
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    # epsilon for testing whether a number is close to zero
    _EPS = np.finfo(float).eps * 4.0

    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def from_tf_to_matrix(tf : TransformStamped):
     ## Convert tf2 transform to np array components
    transform_pose_np = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
    transform_quat_np = np.array([tf.transform.rotation.x, tf.transform.rotation.y,
                                        tf.transform.rotation.z, tf.transform.rotation.w])
    transform_np = quaternion_matrix(transform_quat_np)
    transform_np[0:3, -1] = transform_pose_np

    return transform_np

def xyzrgb_array_to_pointcloud2(points, colors, stamp, frame_id, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points and a synched array of color values.
    '''

    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp

    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize
    fields = [PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyzrgb')]
    nbytes = 6
    xyzrgb = np.array(np.hstack([points, colors/255]), dtype=np.float32)
    msg = PointCloud2(header=header, 
                        height = 1, 
                        width= points.shape[0], 
                        fields=fields, 
                        is_dense= False, 
                        is_bigedian=False, 
                        point_step=(itemsize * nbytes), 
                        row_step = (itemsize * nbytes * points.shape[0]), 
                        data=xyzrgb.tobytes())

    return msg

def xyz_array_to_pointcloud2(points, stamp, frame_id):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points (without color).
    '''
    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp

    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize
    fields = [PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyz')]
    nbytes = 3  # xyz only

    xyz = np.array(points, dtype=dtype)

    msg = PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        fields=fields,
        is_dense=False,
        is_bigendian=False,
        point_step=itemsize * nbytes,
        row_step=itemsize * nbytes * points.shape[0],
        data=xyz.tobytes()
    )

    return msg

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.ita_ocr = PaddleOCR(
                lang="it", # Specify Italian recognition model with the lang parameter
                use_doc_orientation_classify=False, # Disable document orientation classification model
                use_doc_unwarping=False, # Disable text image unwarping model
                use_textline_orientation=False, # Disable text line orientation classification model
            )   
        self.get_logger().info('Detection Node is running...')
        self.camera_info_available = False
        self.calib_mat = None
        self.callback_group = ReentrantCallbackGroup()
        self.folder_name = "first_floor2"
        # Create the directory if it doesn't exist
        os.makedirs(self.folder_name, exist_ok=True)

        self.debug_mode = True

        self. img_number = 0
        self.save_path = f"obb_debug{self.img_number}.png"

        img_topic="/cer/realsense_repeater/color_image"
        depth_topic="/cer/realsense_repeater/depth_image"

        self.tf_buffer = Buffer()
        self.tf_sub = TransformListener(self.tf_buffer, self)
        # Subscribe to the image topic
        self.img_sub = message_filters.Subscriber(self, Image, img_topic, callback_group=self.callback_group)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic, callback_group=self.callback_group)
        self.tss = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.depth_sub], 1, slop=0.3)
        self.camera_info_sub = self.create_subscription(CameraInfo, "/cer/realsense_repeater/camera_info", self.camera_info_callback, 10, callback_group=self.callback_group)        
        self.tss.registerCallback(self.image_callback)
    
    def camera_info_callback(self, msg):
        """
        Saves the calib matrix of the camera intrinsic parameters
        """
        if not self.camera_info_available:
            self.calib_mat = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            self.camera_info_available = True

    def image_callback(self, img_msg, depth_msg):
        """Callback function to process incoming images."""
        if not self.camera_info_available:
            self.get_logger().error("Camera info not available yet.")
            return
        try:           
            tf = self.tf_buffer.lookup_transform("map", 
                                                img_msg.header.frame_id,
                                                img_msg.header.stamp
                                                )
            transform = from_tf_to_matrix(tf)
            # Convert ROS Image message to NumPy array (raw byte data)
            cv_image = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
            cv_depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)

            ## Convert depth from ros2 to OpenCv
            depth = cv_depth.astype(np.float16)        
            
            start = time.time()
            result = self.ita_ocr.predict(cv_image)
            bboxes = []
            texts = []
            scores = []

            for i in range(len(result[0]['dt_polys'])) :
                if result[0]['rec_scores'][i] > 0.8 :
                    bboxes.append(result[0]['dt_polys'][i].tolist())
                    texts.append(result[0]['rec_texts'][i])
                    scores.append(result[0]['rec_scores'][i])
                
            if len(bboxes) > 0 :
                self.indexes, _ = self.merge(bboxes)
                text_list = []
                for index in self.indexes :
                    text = ""
                    point_cloud = []
                    for i in index :  
                        text += texts[i] + " "
                        point_cloud.append(self.project_depth_bboxes_pc_torch(depth, bboxes[i], self.calib_mat, min_depth=0.2, max_depth=6.0, depth_factor=1.0, downsampling_factor=10.0))
                    normal_vector, centre, dimensions, x_axis, y_axis = get_plane_features(np.concatenate(point_cloud, axis=0))
                    normal_vector, centre, x_axis, y_axis = self.to_robot_frame(normal_vector, centre, x_axis, y_axis, transform)
                    text_list.append(text.strip())
                    with open('/home/fgervino-iit.local/visual-language-navigation/mmocr_ros/text_rec/mmocr_ros/recognized_texts.txt', 'a') as f:
                        f.write(f"[{text.strip()}] + {centre.tolist()} + {dimensions.tolist()} + {x_axis.tolist()} + {y_axis.tolist()} + {normal_vector.tolist()}\n")
                
            end = time.time()
            print(f"OCR completed in {end - start:.2f} seconds")
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # Option 1: Suppress completely (no logs)
            return

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}") 

    def project_depth_bboxes_pc_torch(self, depth, bbox, calib_matrix, min_depth = 0.2, max_depth = 6.0, depth_factor=1.0, downsampling_factor=10.0):
        """
        Creates the 3D pointcloud, in camera frame, from the depth and alignes the clip features and RGB color for each 3D point.
        Uses tensors to speed up the process. Uses GPU

        :param depth: matrix of shape (W , H), depth image from the camera
        :param bbox: bounding box of the text in the image, in the form of a list of 4 points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        :param calib_matrix: matrix of shape (3, 3) containing the intrinsic parameters of the camera in matrix form
        :param min_depth: (float) filters out the points below this Z distance: must be positive
        :param max_depth: (float) filters out the points above this Z distance:  must be positive
        :param depth_factor: (float) scale factor for the depth image (it divides the depth z values)
        :param downsample_factor: (float) how much to reduce the number of points extracted from depth
        :return: numpy array of shape (N, 3) of 3D points, numpy array of shape (N, F) containing aligned CLIP features to each 3D point, numpy array of shape (N, 3) of aligned RGB color for each point, 
        """

        fx = calib_matrix[0, 0]
        fy = calib_matrix[1, 1]
        cx = calib_matrix[0, 2]
        cy = calib_matrix[1, 2]

        if min_depth < 0.0:
              min_depth = 0.2
        if max_depth < 0.0:
              max_depth = 6.0

        #depth = torch.tensor(list(depth), device='cuda').type(torch.float32)
        depth = torch.from_numpy(depth).to(device='cuda', dtype=torch.float32)

        H, W = depth.shape

        # 1. Filter depth by valid range
        valid_mask = (depth > min_depth) & (depth < max_depth)

        # 2. Create polygon mask for quadrilateral
        mask_np = np.zeros((H, W), dtype=np.uint8)
        quad_np = np.array(bbox, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask_np, [quad_np], 1)
        poly_mask = torch.from_numpy(mask_np).bool().to(depth.device)

        # 3. Combine masks: only points inside quad AND valid depth range
        final_mask = poly_mask & valid_mask

        # 4. Get pixel coordinates inside mask
        ys, xs = torch.where(final_mask)

        if ys.numel() == 0:
            # No valid points
            return torch.empty((0, 3), device=depth.device)

        # 5. Gather depth values at those coordinates
        depth_vals = depth[ys, xs].float()

        # 6. Compute 3D pointcloud coordinates
        xs = xs.float()
        ys = ys.float()

        xx = (xs - cx) * depth_vals / fx
        yy = (ys - cy) * depth_vals / fy
        zz = depth_vals / depth_factor

        point_cloud = torch.stack((xx, yy, zz), dim=1)  # Nx3
        
        return point_cloud.cpu().numpy()
        
    def to_robot_frame(self, normal_vector, centre, x_axis, y_axis, transform_matrix) :
        # Get normal vector and centre
        tf_torch = torch.tensor(transform_matrix, dtype=torch.float32, device="cuda")
        x_axis = tf_torch[:3, :3] @ torch.tensor(x_axis, dtype=torch.float32, device=tf_torch.device)
        y_axis = tf_torch[:3, :3] @ torch.tensor(y_axis, dtype=torch.float32, device=tf_torch.device)
        normal_vector = tf_torch[:3, :3] @ torch.tensor(normal_vector, dtype=torch.float32, device=tf_torch.device)
        centre = torch.tensor(centre, dtype=torch.float32, device=tf_torch.device)
        centre = tf_torch[:3, :3] @ centre + tf_torch[:3, 3]

        return normal_vector.cpu().numpy(), centre.cpu().numpy(), x_axis.cpu().numpy(), y_axis.cpu().numpy()

    def crop(self, bboxes, delta=0, x_limit = 639, y_limit = 479) :
        """ Modify the bounding boxes from tuples of 4 coordinates to tuples of 2 points (top-left and bottom-right) widen by delta."""
        coords = []
        for bbox in bboxes:     
            x = [point[0] for point in bbox]
            y = [point[1] for point in bbox]
            x_tl = max(round(min(x))-delta,0); y_tl = max(round(min(y))-delta,0)
            x_br = min(round(max(x))+delta,x_limit); y_br = min(round(max(y))+delta,y_limit)
            coords.append([[x_tl,y_tl], [x_br,y_br]])

        return coords
    
    def merge(self, boxes):
        boxes = self.crop(boxes)

        # returns true if the two boxes overlap
        def overlap(source, target):
            tl1, br1 = source
            tl2, br2 = target
            if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
                return False
            if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
                return False
            return True

        # returns all overlapping boxes
        def getAllOverlaps(boxes, bounds, index):
            overlaps = []
            for a in range(len(boxes)):
                if a != index:
                    if overlap(bounds, boxes[a]):
                        overlaps.append(a)
            return overlaps

        merge_margin = 15
        merge_index = []
        box_indices = [[i] for i in range(len(boxes))]

        finished = False
        while not finished:
            finished = True
            index = len(boxes) - 1
            while index >= 0:
                curr = boxes[index]
                tl = curr[0][:]
                br = curr[1][:]
                tl[0] -= merge_margin
                tl[1] -= merge_margin
                br[0] += merge_margin
                br[1] += merge_margin

                overlaps = getAllOverlaps(boxes, [tl, br], index)

                if len(overlaps) > 0:
                    overlaps.append(index)
                    overlaps = list(set(overlaps))
                    overlaps.sort()
                    # Merge all indices
                    merged_indices = []
                    for ind in overlaps:
                        merged_indices.extend(box_indices[ind])
                    # Remove duplicates and sort
                    merged_indices = sorted(set(merged_indices))
                    # Get all corners for merged box
                    con = []
                    for ind in overlaps:
                        tl, br = boxes[ind]
                        con.append([tl])
                        con.append([br])
                    con = np.array(con)
                    x, y, w, h = cv2.boundingRect(con)
                    w -= 1
                    h -= 1
                    merged = [[x, y], [x + w, y + h]]

                    # Remove boxes and indices in reverse order
                    for ind in sorted(overlaps, reverse=True):
                        del boxes[ind]
                        del box_indices[ind]
                    boxes.append(merged)
                    box_indices.append(merged_indices)

                    finished = False
                    break
                index -= 1

        # Add remaining indices (boxes that were never merged)
        merge_index = box_indices

        return merge_index, boxes

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
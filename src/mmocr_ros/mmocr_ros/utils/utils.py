from std_msgs.msg import Header
from sensor_msgs.msg import PointField, PointCloud2
import numpy as np
import math
from tf2_ros import TransformStamped
import torch

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


def project_depth_bboxes_pc_torch(depth, bboxes, cv_image, calib_matrix, transform_matrix = None, min_depth = 0.2, max_depth = 6.0, depth_factor=1.0, downsampling_factor=10.0, colored = True):
        """
        Creates the 3D pointcloud, in camera frame, from the depth and alignes the clip features and RGB color for each 3D point.
        Uses tensors to speed up the process. Uses GPU

        :param depth: matrix of shape (W , H), depth image from the camera
        :param bboxes: list of bounding boxes, each bbox is a list of 4 points [x1 y1 x2 y2 x3 y3 x4 y4] in the image
        :param calib_matrix: matrix of shape (3, 3) containing the intrinsic parameters of the camera in matrix form
        :param transform: matrix of shape (4, 4), homogeneous, containing the transformation from the camera frame to a desired frame. If None the points will not be transformed.
        :param min_depth: (float) filters out the points below this Z distance: must be positive
        :param max_depth: (float) filters out the points above this Z distance:  must be positive
        :param depth_factor: (float) scale factor for the depth image (it divides the depth z values)
        :param downsample_factor: (float) how much to reduce the number of points extracted from depth
        :param colored: (bool) flag that if true returns the aligned color array, otherwise it will be empty
        :return: numpy array of shape (N, 3) of 3D points, numpy array of shape (N, F) containing aligned CLIP features to each 3D point, numpy array of shape (N, 3) of aligned RGB color for each point
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

        #intrisics = [[fx, 0.0, cx],
        #             [0.0, fy, cy],
        #             [0.0, 0.0, 1.0 / depth_factor]]
        #intrisics = torch.tensor(list(intrisics), device='cuda').type(torch.float32)

        # filter depth coords based on z distance
        vv, uu = torch.where((depth > min_depth) & (depth < max_depth))

        pointclouds = []
        colors = []
        if transform_matrix is not None:
            tf_torch = torch.tensor(transform_matrix, dtype=torch.float32, device="cuda")

        # filter depth coords based on bboxes
        for bbox in bboxes:
            x_min, y_min = bbox[0]
            x_max, y_max = bbox[1]
            #x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            #x_min = round(max(min(x1, x2, x3, x4),0)); y_min = round(max(min(y1, y2, y3, y4),0))
            #x_max = round(min(max(x1, x2, x3, x4),639)); y_max = round(min(max(y1, y2, y3, y4),479))
            mask_u = (uu >= x_min) & (uu <= x_max)
            mask_v = (vv >= y_min) & (vv <= y_max)
            uu_bbox = uu[mask_u & mask_v]
            vv_bbox = vv[mask_v & mask_u]
            coords = torch.stack((vv_bbox, uu_bbox), dim=1)
            vv_bbox = coords[:, 0]
            uu_bbox = coords[:, 1]
            xx_bbox = (uu_bbox - cx) * depth[vv_bbox, uu_bbox] / fx
            yy_bbox = (vv_bbox - cy) * depth[vv_bbox, uu_bbox] / fy
            zz_bbox = depth[vv_bbox, uu_bbox] / depth_factor
            pointcloud_bbox = torch.cat((xx_bbox.unsqueeze(1), yy_bbox.unsqueeze(1), zz_bbox.unsqueeze(1)), 1)
            if transform_matrix is not None:
                ones = torch.ones(pointcloud_bbox.shape[0], 1, dtype=torch.float32 ,device="cuda")
                hom_pointcloud_torch = torch.cat((pointcloud_bbox, ones), dim=1)
                pointcloud_bbox = (tf_torch @ hom_pointcloud_torch.T).T[:, :3]

            pointclouds.append(pointcloud_bbox.cpu().numpy())
            if colored :
                colors.append(cv_image[vv_bbox.cpu().numpy(), uu_bbox.cpu().numpy()])
        
        return bboxes, pointclouds, colors

def load_config(file_path):
    config = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Remove any leading/trailing whitespace, including newline characters
            line = line.strip()
            if line and '=' in line:
                # Split the line into key and value
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config
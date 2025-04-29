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

from mmocr_interfaces.msg import ImageBoundingBoxes
from mmocr.apis import MMOCRInferencer

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

def crop(bboxes, delta=0, x_limit = 639, y_limit = 479) :
        """ Modify the bounding boxes from tuples of 8 elements([x1 y1 x2 y2 x3 y3 x4 y4]) to [[x_top_left, y_top_left],[x_bottom_right, y_bottom_right]] and also it widens the box by delta pixels"""
        coords = []
        for bbox in bboxes:     
            x = [bbox[0],bbox[2],bbox[4],bbox[6]]
            y = [bbox[1],bbox[3],bbox[5],bbox[7]]
            x_tl = max(round(min(x))-delta,0); y_tl = max(round(min(y))-delta,0)
            x_br = min(round(max(x))+delta,x_limit); y_br = min(round(max(y))+delta,y_limit)
            coords.append([[x_tl,y_tl], [x_br,y_br]])

        return coords
    
def merge(boxes) : 
    
    # tuplify
    def tup(point):
        return (point[0], point[1])

    # returns true if the two boxes overlap
    def overlap(source, target):
        # unpack points
        tl1, br1 = source
        tl2, br2 = target

        # checks
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

    # go through the boxes and start merging
    merge_margin = 15

    # this is gonna take a long time
    finished = False
    highlight = [[0,0], [1,1]]
    points = [[[0,0]]]
    while not finished:
        # set end con
        finished = True

        # loop through boxes
        index = len(boxes) - 1
        while index >= 0:
            # grab current box
            curr = boxes[index]

            # add margin
            tl = curr[0][:]
            br = curr[1][:]
            tl[0] -= merge_margin
            tl[1] -= merge_margin
            br[0] += merge_margin
            br[1] += merge_margin

            # get matching boxes
            overlaps = getAllOverlaps(boxes, [tl, br], index)
            
            # check if empty
            if len(overlaps) > 0:
                # combine boxes
                # convert to a contour
                con = []
                overlaps.append(index)
                for ind in overlaps:
                    tl, br = boxes[ind]
                    con.append([tl])
                    con.append([br])
                con = np.array(con)

                # get bounding rect
                x,y,w,h = cv2.boundingRect(con)

                # stop growing
                w -= 1
                h -= 1
                merged = [[x,y], [x+w, y+h]]

                # highlights
                highlight = merged[:]
                points = con

                # remove boxes from list
                overlaps.sort(reverse = True)
                for ind in overlaps: del boxes[ind]
                boxes.append(merged)

                # set flag
                finished = False
                break

            # increment
            index -= 1
    cv2.destroyAllWindows()

    return boxes

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.get_logger().info('Detection Node is running...')
        self.detector = MMOCRInferencer(det='dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015', device='cuda:0')
        self.camera_info_available = False
        self.calib_mat = None
        self.callback_group = ReentrantCallbackGroup()

        self.debug_mode = True

        img_topic="/cer/realsense_repeater/color_image"
        depth_topic="/cer/realsense_repeater/depth_image"
        # Subscribe to the image topic
        self.img_sub = message_filters.Subscriber(self, Image, img_topic, callback_group=self.callback_group)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic, callback_group=self.callback_group)
        self.tss = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.depth_sub], 1, slop=0.3)
        self.camera_info_sub = self.create_subscription(CameraInfo, "/cer/realsense_repeater/camera_info", self.camera_info_callback, 10, callback_group=self.callback_group)        
        self.tss.registerCallback(self.image_callback)

        # Publish the image with the bounding boxes
        self.publisher = self.create_publisher(ImageBoundingBoxes, 'image_and_bboxes', 10)
        if self.debug_mode:
            self.debug_publisher = self.create_publisher(PointCloud2, 'debug_pointcloud_bb', 10)
        else:
            self.point_cloud_publisher = self.create_publisher(PointCloud2, 'pointcloud_bb', 10)

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
            # Convert ROS Image message to NumPy array (raw byte data)
            cv_image = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
            cv_depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)

            ## Convert depth from ros2 to OpenCv
            depth = cv_depth.astype(np.float16)
            
            # Show topic's image
            '''
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Image", cv_image_bgr)
            cv2.waitKey(1)            
            '''
            
            new_message = ImageBoundingBoxes()
            result = self.detector(cv_image)

            # Extract bounding boxes
            predictions = result['predictions'][0]  # List of detected bounding boxes and the respective certainty

            bboxes = predictions['det_polygons']
            scores = predictions['det_scores']
            # Filter out low-confidence detections
            bboxes = [bbox for bbox, score in zip(bboxes, scores) if score > 0.8]

            if len(bboxes) > 0 :
                point_clouds = []
                delta = 5
                bboxes = crop(bboxes, delta, 639, 479)
                bboxes = merge(bboxes)
                #print(bboxes)
                _, pc_list, colors = self.project_depth_bboxes_pc_torch(depth, bboxes, cv_image, self.calib_mat, min_depth=0.2, max_depth=6.0, depth_factor=1.0, downsampling_factor=10.0)
                if self.debug_mode:
                    for pc, color in zip(pc_list, colors):
                        msg = xyzrgb_array_to_pointcloud2(pc, color, stamp=self.get_clock().now().to_msg(), frame_id=depth_msg.header.frame_id)
                        msg.header.stamp = depth_msg.header.stamp
                        self.debug_publisher.publish(msg)
                        point_clouds.append(msg)
                else :
                    # Create a PointCloud2 message
                    msg = xyz_array_to_pointcloud2(pc_list, stamp=depth_msg.header.stamp, frame_id=depth_msg.header.frame_id)
                    msg.header.stamp = depth_msg.header.stamp
                    self.point_cloud_publisher.publish(msg)
                
                new_message.bounding_boxes = [x for box in bboxes for pair in box for x in pair]
                new_message.point_cloud_list = point_clouds
                #print(new_message.bounding_boxes)
                new_message.image = img_msg

                # Publish results
                self.publisher.publish(new_message)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}") 

    def project_depth_bboxes_pc_torch(self, depth, bboxes, cv_image, calib_matrix, min_depth = 0.2, max_depth = 6.0, depth_factor=1.0, downsampling_factor=10.0):
        """
        Creates the 3D pointcloud, in camera frame, from the depth and alignes the clip features and RGB color for each 3D point.
        Uses tensors to speed up the process. Uses GPU

        :param depth: matrix of shape (W , H), depth image from the camera
        :param bboxes: list of bounding boxes, each bbox is a list of 4 points [x1 y1 x2 y2 x3 y3 x4 y4] in the image
        :param calib_matrix: matrix of shape (3, 3) containing the intrinsic parameters of the camera in matrix form
        :param min_depth: (float) filters out the points below this Z distance: must be positive
        :param max_depth: (float) filters out the points above this Z distance:  must be positive
        :param depth_factor: (float) scale factor for the depth image (it divides the depth z values)
        :param downsample_factor: (float) how much to reduce the number of points extracted from depth
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
            pointclouds.append(pointcloud_bbox.cpu().numpy())
            if self.debug_mode :
                colors.append(cv_image[vv_bbox.cpu().numpy(), uu_bbox.cpu().numpy()])
        
        return bboxes, pointclouds, colors


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
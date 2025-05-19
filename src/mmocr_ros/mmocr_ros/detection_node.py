import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import numpy as np
import message_filters
from rclpy.callback_groups import ReentrantCallbackGroup
from tf2_ros import TransformListener, Buffer
from utils.utils import xyzrgb_array_to_pointcloud2, xyz_array_to_pointcloud2, from_tf_to_matrix, project_depth_bboxes_pc_torch
from utils.image_utils import crop, merge

from mmocr_interfaces.msg import ImageBoundingBoxes
from mmocr.apis import MMOCRInferencer

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.get_logger().info('Detection Node is running...')
        self.detector = MMOCRInferencer(det='dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015', device='cuda:0')
        self.camera_info_available = False
        self.calib_mat = None
        self.callback_group = ReentrantCallbackGroup()

        self.declare_parameters(namespace='', parameters=[
            ('img_topic', "/cer/realsense_repeater/color_image"),
            ('depth_topic', "/cer/realsense_repeater/depth_image"),
            ('info_topic', "/cer/realsense_repeater/camera_info"),
            ('detection_confidence_threshold', 0.8),
            ('debug_pc', True)
        ])
        img_topic = self.get_parameter("img_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("info_topic").value
        self.debug_mode = self.get_parameter("debug_pc").value
        self.confidence_threshold = self.get_parameter('detection_confidence_threshold').value

        self.tf_buffer = Buffer()
        self.tf_sub = TransformListener(self.tf_buffer, self)
        # Subscribe to the image topic
        self.img_sub = message_filters.Subscriber(self, Image, img_topic, callback_group=self.callback_group)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic, callback_group=self.callback_group)
        self.tss = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.depth_sub], 1, slop=0.3)
        self.camera_info_sub = self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, 10, callback_group=self.callback_group)        
        self.tss.registerCallback(self.image_callback)

        # Publish the image with the bounding boxes
        self.publisher = self.create_publisher(ImageBoundingBoxes, 'image_and_bboxes', 10)
        if self.debug_mode:
            self.debug_publisher = self.create_publisher(PointCloud2, 'debug_pointcloud_bb', 10)
        else:
            self.point_cloud_publisher = self.create_publisher(PointCloud2, 'pointcloud_bb', 10)

    def camera_info_callback(self, msg : CameraInfo):
        """
        Saves the calib matrix of the camera intrinsic parameters
        """
        if not self.camera_info_available:
            self.calib_mat = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            self.camera_info_available = True

    def image_callback(self, img_msg : Image, depth_msg : Image):
        """Callback function to process incoming images."""
        if not self.camera_info_available:
            self.get_logger().error("Camera info not available yet.")
            return
        try:
            tf = self.tf_buffer.lookup_transform("map", 
                                                img_msg.header.frame_id,
                                                img_msg.header.stamp)
            transform = from_tf_to_matrix(tf)
            # Convert ROS Image message to NumPy array (raw byte data)
            cv_image = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
            cv_depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)

            ## Convert depth from ros2 to OpenCv
            depth = cv_depth.astype(np.float16)
            
            # Show topic's image
            #cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            #cv2.imshow("Image", cv_image_bgr)
            #cv2.waitKey(1)            
            
            new_message = ImageBoundingBoxes()
            result = self.detector(cv_image)

            # Extract bounding boxes
            predictions = result['predictions'][0]  # List of detected bounding boxes and the respective certainty

            bboxes = predictions['det_polygons']
            scores = predictions['det_scores']
            # Filter out low-confidence detections
            bboxes = [bbox for bbox, score in zip(bboxes, scores) if score > self.confidence_threshold]

            if len(bboxes) > 0 :
                point_clouds = []
                delta = 5
                bboxes = crop(bboxes, delta, img_msg.width - 1, img_msg.height - 1)
                bboxes = merge(bboxes)
                #print(bboxes)
                _, pc_list, colors = project_depth_bboxes_pc_torch(depth, bboxes, cv_image, self.calib_mat, transform_matrix=transform, min_depth=0.2, max_depth=6.0, depth_factor=1.0, downsampling_factor=10.0, colored=True)
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



def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
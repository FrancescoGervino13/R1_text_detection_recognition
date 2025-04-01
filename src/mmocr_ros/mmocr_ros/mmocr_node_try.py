import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2

from mmocr2.tools.det_rec_ros2 import det_rec

class MmoCRNode(Node):
    def __init__(self):
        super().__init__('mmocr_node')
        #self.bridge = CvBridge()
        self.get_logger().info('MMOCR Node is running...')

        # Subscribe to the image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        """Callback function to process incoming images."""
        try:
            # Convert ROS Image message to NumPy array (raw byte data)
            y_max = msg.height
            x_max = msg.width
            cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            '''
            cv2.imshow("Cropped Image", cv_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
            # Call the MMOCR function with the OpenCV image
            self.call_mmocr_function(cv_image,x_max,y_max)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def call_mmocr_function(self, image,x_max,y_max):
        """Run the MMOCR script using the image array."""
        try:
            recognized_texts, bboxes = det_rec(image,x_max,y_max)
            print(recognized_texts, bboxes)
            #print(f"Time spent: {time}")
        except Exception as e:
            self.get_logger().error(f"Error calling MMOCR function: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = MmoCRNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
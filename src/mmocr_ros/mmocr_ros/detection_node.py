import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2

from mmocr_interfaces.msg import ImageBoundingBoxes
from mmocr.apis import MMOCRInferencer

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.get_logger().info('Detection Node is running...')
        self.detector = MMOCRInferencer(det='dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015')

        # Subscribe to the image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            1
        )

        # Publish the image with the bounding boxes
        self.publisher = self.create_publisher(ImageBoundingBoxes, 'image_and_bboxes', 10)

    def image_callback(self, msg):
        """Callback function to process incoming images."""
        try:
            # Convert ROS Image message to NumPy array (raw byte data)
            cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

            '''
            # Show topic's image
            cv2.imshow("Cropped Image", cv_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''

            new_message = ImageBoundingBoxes()
            result = self.detector(cv_image)

            # Extract bounding boxes
            predictions = result['predictions'][0]  # List of detected bounding boxes and the respective certainty

            bboxes = predictions['det_polygons']
            if len(bboxes) > 0 :
                #print(bboxes)
                new_message.bounding_boxes = [coord for box in bboxes for coord in box]
                #print(new_message.bounding_boxes)
                new_message.image = msg

                # Publish results
                self.publisher.publish(new_message)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")            


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
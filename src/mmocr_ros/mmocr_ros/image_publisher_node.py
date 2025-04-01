import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher_node')
        
        # Publisher to the 'image_raw' topic
        self.publisher = self.create_publisher(Image, 'image', 10)
        self.get_logger().info('Image Publisher Node is running...')
        
        # Create CvBridge instance
        self.bridge = CvBridge()
        
        # Path to folder containing images
        self.image_folder = '/home/fgervino-iit.local/Images4R1'
        
        # Get list of images in the folder
        self.image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))]
        
        # Timer to publish images at 1 Hz
        self.timer = self.create_timer(5, self.publish_image)
        self.index = 0  # To iterate through images

    def publish_image(self):
        # Get the next image file
        if self.index >= len(self.image_files):
            self.index = 0  # Reset to loop through the images again

        image_path = os.path.join(self.image_folder, self.image_files[self.index])

        # Read the image using OpenCV
        cv_image = cv2.imread(image_path)

        if cv_image is None:
            self.get_logger().error(f'Failed to read image {image_path}')
            return

        # Convert OpenCV image to ROS message (sensor_msgs/Image)
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        
        # Publish the image
        self.publisher.publish(ros_image)
        self.get_logger().info(f'Publishing image: {self.image_files[self.index]}')

        # Increment the index to move to the next image
        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
from rclpy.qos import QoSProfile

class ImageSubscriber(Node):

    def __init__(self):
        # Initialize the node
        super().__init__('image_subscriber_node')
        
        # Define the subscriber to the /camera/image topic
        self.subscription = self.create_subscription(
            Image,  # The message type
            '/image',  # The topic name (change this to match your image topic)
            self.image_callback,  # The callback function when a message is received
            QoSProfile(depth=10)  # Quality of Service settings
        )
        
        # To make sure we don’t miss any messages in case the node is slow
        self.subscription  # Prevent unused variable warning

    def image_callback(self, msg: Image):
        # Convert ROS Image message to numpy array
        image_data = np.frombuffer(msg.data, dtype=np.uint8)
        
        # Reshape the image data according to its width, height, and channels
        image = image_data.reshape((msg.height, msg.width, 3))  # For RGB images
        
        # Show the image using OpenCV
        cv2.imshow("Received Image", image)
        cv2.waitKey(1)  # Update the window

def main(args=None):
    # Initialize the ROS2 Python client library (rclpy)
    rclpy.init(args=args)

    # Create an instance of the ImageSubscriber class
    image_subscriber = ImageSubscriber()

    # Spin the node to keep it alive and receiving messages
    rclpy.spin(image_subscriber)

    # Shutdown ROS2 when done
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import subprocess
import base64
import json
import re

class MmoCRNode(Node):
    def __init__(self):
        super().__init__('mmocr_node')
        self.bridge = CvBridge()
        self.get_logger().info('MMOCR Node is running...')

        # Subscribe to the image topic
        self.subscription = self.create_subscription(
            Image,
            'image',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        """Callback function to process incoming images."""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Call the MMOCR function with the OpenCV image
            self.call_mmocr_function(cv_image)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def call_mmocr_function(self, image):
        """Run the MMOCR script using the image array."""
        try:
            _, buffer = cv2.imencode('.jpg', image)  # Encode image to JPEG format
            image_bytes = buffer.tobytes()  # Convert the image buffer to bytes
            base64_image = base64.b64encode(image_bytes).decode('utf-8')  # Encode bytes to base64 string

            # Pass the image array directly to the MMOCR script
            result = subprocess.run(
                [
                    '/home/fgervino-iit.local/anaconda3/envs/open-mmlab/bin/python',
                    '/home/fgervino-iit.local/mmocr/tools/final_rec.py',
                    base64_image
                ],
                capture_output=True, text=True
            )

            # Now capture only the JSON output and ignore prints
            clean_output = result.stdout.strip()  # Remove leading/trailing whitespace
        
            if clean_output:
                self.get_logger().info(f'MMOCR Output: {clean_output}')

            # Extract JSON using regex
            match = re.search(r'(\{.*\})', clean_output, re.DOTALL)
            if match:
                json_output = match.group(1)
                print(json_output)

            # Parse JSON output
            try:
                response = json.loads(json_output)
                recognized_text = response.get("recognized_text", [])
                bboxes = response.get("bounding_boxes", [])
                self.get_logger().info(f'Recognized Text: {recognized_text}')
                self.get_logger().info(f'Bounding Boxes: {bboxes}')
                return recognized_text  # Return the list of strings

            except json.JSONDecodeError:
                self.get_logger().error("Failed to parse MMOCR output as JSON.")
                return []

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
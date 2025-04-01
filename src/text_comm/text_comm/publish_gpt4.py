import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TextPublisher(Node):
    def __init__(self):
        super().__init__('text_publisher')
        self.publisher_ = self.create_publisher(String, 'chat_input', 10)
        self.words = ["table", "computer", "chair", "bed"]  # List of words to publish
        self.index = 0
        self.timer = self.create_timer(3.0, self.publish_message)  # Publish every second

    def publish_message(self):
        msg = String()
        msg.data = self.words[self.index]  # Get word from the list
        self.get_logger().info(f"Publishing: {msg.data}")
        self.publisher_.publish(msg)
        self.index = (self.index + 1) % len(self.words)  # Cycle through words

def main(args=None):
    rclpy.init(args=args)
    node = TextPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
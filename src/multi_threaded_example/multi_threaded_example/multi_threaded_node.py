import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

class MultiThreadedNode(Node):
    def __init__(self):
        super().__init__('multi_threaded_node')
        self.create_subscription(String, 'topic1', self.callback1, 10)
        self.create_subscription(String, 'topic2', self.callback2, 10)
        self.get_logger().info('Node with multiple subscriptions is ready.')

    def callback1(self, msg):
        self.get_logger().info(f"Callback 1: {msg.data}")

    def callback2(self, msg):
        self.get_logger().info(f"Callback 2: {msg.data}")

def main():
    rclpy.init()

    node = MultiThreadedNode()

    # Use a MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)  # Specify 4 threads
    executor.add_node(node)

    try:
        executor.spin()  # Allows multiple callbacks to run concurrently
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

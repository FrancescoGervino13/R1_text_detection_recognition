#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class MyNode(Node) :
    # Constructor
    def __init__(self) :
        # Node's name
        super().__init__("first_node")
        self.get_logger().info("Hello from ROS2")

def main(args=None) : 
    rclpy.init(args=args)
    node = MyNode()


    # The node is created here (inside the program, it's not the whole of it)

    # Destroy the node
    rclpy.shutdown()

if __name__ == 'main'  :
    main()
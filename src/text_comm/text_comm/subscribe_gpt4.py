import rclpy
from rclpy.node import Node
from std_msgs.msg import String
#from openai import OpenAI
from openai import AzureOpenAI

AZURE_API_KEY="94ec2d76955a48488952c81f0d591e94"
AZURE_ENDPOINT="https://iitlines-swecentral1.openai.azure.com/"

class GPT4Chatbot(Node):
    def __init__(self):
        super().__init__('gpt4_chatbot')

        self.subscription = self.create_subscription(
            String,
            '/chat_input',   # Topic where user sends text
            self.chat_callback,
            10
        )

        self.publisher = self.create_publisher(String, '/chat_output', 10)

        # Conversation history
        prompt = """
        Given a list of objects, Guess only one type of room that could contain those objects.
        """
        self.messages = [{"role": "system", "content": prompt}]

        self.client = AzureOpenAI(
            azure_endpoint=f"{AZURE_ENDPOINT}", #do not add "/openai" at the end here because this will be automatically added by this SDK
            api_key=AZURE_API_KEY,
            api_version="2024-10-21"
        )

        self.object_list = ""

    def chat_callback(self, msg):
        """Handles incoming chat messages and sends them to GPT-4."""
        user_message = msg.data
        self.get_logger().info(f"Received message: {user_message}")
        self.object_list = self.object_list + ", " + user_message

        # Add user message to conversation history
        self.messages.append({"role": "user", "content": self.object_list})

        # Call GPT-4
        gpt_response = self.ask_gpt4()

        # Publish the response
        if gpt_response:
            response_msg = String()
            response_msg.data = gpt_response
            self.publisher.publish(response_msg)
            self.get_logger().info(f"GPT-4 Response: {gpt_response}")

    def ask_gpt4(self):
        """Sends messages to GPT-4 and gets a response."""
        try:
            response = self.client.chat.completions.create(
                model="hsp-Vocalinteraction_gpt4o",
                messages=self.messages
            )
            bot_reply = response.choices[0].message.content

            # Store GPT-4 response in conversation history
            self.messages.append({"role": "assistant", "content": bot_reply})
            return bot_reply
        except Exception as e:
            self.get_logger().error(f"Error calling GPT-4: {e}")
            return "Error processing your request."

def main(args=None):
    rclpy.init(args=args)
    chatbot_node = GPT4Chatbot()
    rclpy.spin(chatbot_node)
    chatbot_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
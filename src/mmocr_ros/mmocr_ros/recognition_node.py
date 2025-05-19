import rclpy
from rclpy.node import Node
from openai import AzureOpenAI
import numpy as np
from utils.utils import load_config
from utils.image_utils import encode_image_array
import json

from mmocr_interfaces.msg import ImageBoundingBoxes
from mmocr_interfaces.msg import AlignedTextsClouds

class RecognitionNode(Node):
    def __init__(self):
        super().__init__('text_recognition_node')
        self.declare_parameters(
            namespace = '',
            parameters = [
                ('config_path', '/home/user1/config.env'),      # Config Path for AZURE chatGPT keys
                ('chatgpt_model_name', 'hsp-Vocalinteraction_gpt4o'),
                ('img_topic_name', 'image_and_bboxes'),         # Input of type ImageBoundingBoxes
                ('result_topic_name', 'aligned_texts_clouds')   # Output of type AlignedTextsClouds
            ])
        openai_config = load_config(self.get_parameter('config_path').value)
        self.client = AzureOpenAI(
            azure_endpoint = f"{openai_config['AZURE_ENDPOINT']}",
            api_key = openai_config['AZURE_API_KEY'],
            api_version = "2024-10-21"
            )
        input_topic_name = self.get_parameter('img_topic_name').value
        result_topic_name = self.get_parameter('result_topic_name').value
        self.chatgpt_model_name = self.get_parameter('chatgpt_model_name').value
        self.subscriber = self.create_subscription(ImageBoundingBoxes, input_topic_name, self.callback, 10)
        self.publisher = self.create_publisher(AlignedTextsClouds, result_topic_name, 10)
        self.get_logger().info(f'{self.get_name()} Node is running...')

    def callback(self, msg : ImageBoundingBoxes):
        image = np.frombuffer(msg.image.data, dtype=np.uint8).reshape(msg.image.height, msg.image.width, 3)
        bounding_boxes = msg.bounding_boxes
        # Reshape the flattened bounding boxes into groups of 8 (each representing a bounding box)
        bboxes = np.array(bounding_boxes).reshape(-1, 4).tolist()

        recognized_texts, bboxes = self.crop_and_recognise(image, bboxes)
        if len(recognized_texts) > 0:
            msg_out = AlignedTextsClouds()
            msg_out.point_cloud_list = msg.point_cloud_list
            msg_out.text_list = recognized_texts
            self.publisher.publish(msg_out)


        self.get_logger().info(f"----------------------------")
        for i in range(len(recognized_texts)):
            #if recognized_texts[i] != "no text" and recognized_texts[i] != "\"no text\"" and recognized_texts[i] != "" :
            self.get_logger().info(f"Recognized text: {recognized_texts[0]}")
            #self.get_logger().info(f"Bounding box: {bboxes[i]}")
            # Show the cropped image (debug)
            #cv2.imshow(recognized_texts[i], image[bboxes[i][0][1]:bboxes[i][1][1], bboxes[i][0][0]:bboxes[i][1][0]])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    
    def recognise_text(self, cropped_image):
        """Sends cropped image to GPT-4 Vision client for text recognition."""
        try:
            base64_image = encode_image_array(cropped_image)
            # TODO why there is a fixed percentage in the prompt?
            messages = [
                    {"role": "system", "content": 
                    """
                    Extract text from the provided image. If you are not at least 90 percent sure about what is written, output this precise message: "no text"
                    The output will be a python list with the recognised texts.
                    For example: ["text 1", "text 2", "text 3"].
                    """},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Here is an image. Please extract any visible text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ]

            response = self.client.chat.completions.create(
                    model = self.chatgpt_model_name,
                    messages = messages,
                    temperature=0.1
            )

            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error processing image: {str(e)}"

    def crop_and_recognise(self, image, bboxes):
        """Takes text bounding boxes, crops them, and recognises text."""

        recognized_texts = []
        aligned_bboxes = []

        for bbox in bboxes: 
            x1 = bbox[0]; y1 = bbox[1]
            x2 = bbox[2]; y2 = bbox[3]

            # Crop the image
            cropped_img = image[y1:y2, x1:x2]

            # Recognise text in cropped region
            recognized_text = self.recognise_text(cropped_img)
            # Filter out the not recognized text
            if recognized_text.lower() == "no text" and recognized_text.lower() == "['no text']" and recognized_text.lower() == "[no text]":
                self.get_logger().info(f"No text recognized")
                continue
            else:
                # The output of ChatGPT is a list if there's any text
                # TODO add try-catch ?
                text_list = json.loads(recognized_text)
                print(text_list)
                if len(text_list) == 0:
                    continue
                text = ""
                # Concatenate the text of this bbox
                for i in range(len(text_list)):
                    if i == 0:
                        text = text_list[0]
                    else:
                        text = text + " " + text_list[i]
                recognized_texts.append(text)
                aligned_bboxes.append(bbox)

            ## Show the cropped image
            #cv2.imshow("Cropped Image", cropped_img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            return recognized_texts, aligned_bboxes


def main(args=None):
    rclpy.init(args = args)
    node = RecognitionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

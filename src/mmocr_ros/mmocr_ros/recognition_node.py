import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from openai import AzureOpenAI
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

from mmocr_interfaces.msg import ImageBoundingBoxes

# Add keys for openAI
AZURE_API_KEY=""
AZURE_ENDPOINT=""

class RecognitionNode(Node):
    def __init__(self):
        super().__init__('recognition_node')
        self.client = AzureOpenAI(
            azure_endpoint=f"{AZURE_ENDPOINT}",
            api_key=AZURE_API_KEY,
            api_version="2024-10-21"
            )
        self.get_logger().info('Recognition Node is running...')
        self.subscriber = self.create_subscription(ImageBoundingBoxes, 'image_and_bboxes', self.callback, 10)

    def callback(self, msg):
        x_max = msg.image.width; y_max = msg.image.height
        image = np.frombuffer(msg.image.data, dtype=np.uint8).reshape(msg.image.height, msg.image.width, 3)
        bounding_boxes = msg.bounding_boxes
        delta = 5
        #print(bounding_boxes)
        # Reshape the flattened bounding boxes into groups of 8 (each representing a bounding box)
        bboxes = np.array(bounding_boxes).reshape(-1, 8).tolist()
        #print(bboxes)
        bboxes = self.crop(bboxes, delta, x_max, y_max)
        bboxes = self.merge(bboxes)

        recognized_texts, bboxes = self.crop_and_recognise(image,bboxes)
        self.get_logger().info(f"Recognized text: {recognized_texts}")
        self.get_logger().info(f"Bounding box: {bboxes}")

        '''
        # Show the cropped image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
    
    def crop(self, bboxes, delta=0, x_limit = 639, y_limit = 479) :
        """ Modify the bounding boxes from tuples of 8 elements([x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 x7 y7 x8 y8]) to [[x_top_left y, y_top_left],[x_bottom_right, y_bottom_right]] and also it widens the box by delta pixels"""
        coords = []
        for bbox in bboxes:     
            x = [bbox[0],bbox[2],bbox[4],bbox[6]]
            y = [bbox[1],bbox[3],bbox[5],bbox[7]]
            x_tl = max(round(min(x))-delta,0); y_tl = max(round(min(y))-delta,0)
            x_br = min(round(max(x))+delta,x_limit); y_br = min(round(max(y))+delta,y_limit)
            coords.append([[x_tl,y_tl], [x_br,y_br]])

        return coords
    
    def merge(self, boxes) : 
        
        # tuplify
        def tup(point):
            return (point[0], point[1])

        # returns true if the two boxes overlap
        def overlap(source, target):
            # unpack points
            tl1, br1 = source
            tl2, br2 = target

            # checks
            if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
                return False
            if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
                return False
            return True

        # returns all overlapping boxes
        def getAllOverlaps(boxes, bounds, index):
            overlaps = []
            for a in range(len(boxes)):
                if a != index:
                    if overlap(bounds, boxes[a]):
                        overlaps.append(a)
            return overlaps

        # go through the boxes and start merging
        merge_margin = 15

        # this is gonna take a long time
        finished = False
        highlight = [[0,0], [1,1]]
        points = [[[0,0]]]
        while not finished:
            # set end con
            finished = True

            # loop through boxes
            index = len(boxes) - 1
            while index >= 0:
                # grab current box
                curr = boxes[index]

                # add margin
                tl = curr[0][:]
                br = curr[1][:]
                tl[0] -= merge_margin
                tl[1] -= merge_margin
                br[0] += merge_margin
                br[1] += merge_margin

                # get matching boxes
                overlaps = getAllOverlaps(boxes, [tl, br], index)
                
                # check if empty
                if len(overlaps) > 0:
                    # combine boxes
                    # convert to a contour
                    con = []
                    overlaps.append(index)
                    for ind in overlaps:
                        tl, br = boxes[ind]
                        con.append([tl])
                        con.append([br])
                    con = np.array(con)

                    # get bounding rect
                    x,y,w,h = cv2.boundingRect(con)

                    # stop growing
                    w -= 1
                    h -= 1
                    merged = [[x,y], [x+w, y+h]]

                    # highlights
                    highlight = merged[:]
                    points = con

                    # remove boxes from list
                    overlaps.sort(reverse = True)
                    for ind in overlaps: del boxes[ind]
                    boxes.append(merged)

                    # set flag
                    finished = False
                    break

                # increment
                index -= 1
        cv2.destroyAllWindows()

        return boxes

    def recognise_text(self, cropped_image):
            """Sends cropped image to GPT-4 Vision for text recognition."""

            def encode_image_array(image_array):
                """Converts a NumPy image (OpenCV format) to base64."""
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                buffer = BytesIO()
                pil_image.save(buffer, format="JPEG")
                buffer.seek(0)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            try:
                base64_image = encode_image_array(cropped_image)

                messages = [
                    {"role": "system", "content": 
                    """
                    Extract text from the provided image. If you are not at least 90 percent sure about the text, output "no text"
                    The output will be a python list with the recognised texts.
                    For example: ["text 1", "text 2", "text 3"].
                    """},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Here is an image. Please extract any visible text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ]

                response = self.client.chat.completions.create(
                    model="hsp-Vocalinteraction_gpt4o",
                    messages=messages
                )

                return response.choices[0].message.content
            
            except Exception as e:
                return f"Error processing image: {str(e)}"

    def crop_and_recognise(self, image, bboxes):
        """Takes text bounding boxes, crops them, and recognises text."""

        recognized_texts = []

        for i in range(len(bboxes)) : 
            x1, y1 = bboxes[i][0]
            x2, y2 = bboxes[i][1]

            # Crop the image
            cropped_img = image[y1:y2, x1:x2]

            # Recognise text in cropped region
            recognized_text = self.recognise_text(cropped_img)
            recognized_texts.append(recognized_text)

            '''
            # Show the cropped image
            cv2.imshow("Cropped Image", cropped_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''

        return recognized_texts, bboxes

def main(args=None):
    rclpy.init(args=args)
    node = RecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
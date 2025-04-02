from openai import AzureOpenAI
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from mmocr.apis import MMOCRInferencer

from merging import Merge

AZURE_API_KEY=""
AZURE_ENDPOINT=""

class TextInImage:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=f"{AZURE_ENDPOINT}",
            api_key=AZURE_API_KEY,
            api_version="2024-10-21"
        )

    def encode_image_array(self, image_array):
        """Converts a NumPy image (OpenCV format) to base64."""
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def recognize_text(self, cropped_image):
        """Sends cropped image to GPT-4 Vision for text recognition."""
        try:
            base64_image = self.encode_image_array(cropped_image)

            messages = [
                {"role": "system", "content": 
                """
                Extract text from the provided image.
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

def scale_polygon(bbox, scale_factor):
    """
    Scales a polygon outward from its centroid while maintaining its shape.

    bbox: List of (x, y) points forming the polygon
    scale_factor: How much to increase the size (e.g., 2.0 means double)

    Returns: Scaled bounding box as a NumPy array
    """
    box_np = np.array(bbox, np.float32).reshape(-1, 2)  # Convert to NumPy array
    centroid = np.mean(box_np, axis=0)  # Compute the centroid (center)
    
    # Compute vectors from centroid to each point
    vectors = box_np - centroid

    # Scale vectors outward
    expanded_box = centroid + vectors * scale_factor  # Multiply by scale factor

    return expanded_box.astype(np.int32)

def crop_and_recognize(image_path, detector, text_recognizer, scale_factor=0):
    """Detects text bounding boxes, crops them, and recognizes text."""
    
    # Perform inference
    result = detector(image_path)

    # Load original image
    image = cv2.imread(image_path)

    # Extract bounding boxes
    predictions = result['predictions'][0]  
    bboxes = predictions['det_polygons']

    #recognized_texts = []

    for bbox in bboxes:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Reshape back to OpenCV format if needed
        expanded_box = scale_polygon(bbox, scale_factor)

        # Fill the mask with the region inside the bbox
        cv2.fillPoly(mask, [expanded_box], 255)

        # Extract the region
        cropped_region = cv2.bitwise_and(image, image, mask=mask)

        cv2.imshow(f"Cropped Region", cropped_region)
        cv2.waitKey(0)  # Waits for key press before moving to next cropped image
        cv2.destroyAllWindows()

        # Recognize text in cropped region
        recognized_text = text_recognizer.recognize_text(cropped_region)
        print(recognized_text)
        #recognized_texts.append(recognized_text)

    #return recognized_texts

# Initialize Text Recognition and Detector
text_recognizer = TextInImage()
detector = MMOCRInferencer(det='dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015')
scale_factor = 2

# Process multiple images
for k in range(1, 15):
    image_path = f"/text_dec_and_rec/resized{k}.jpg"
    crop_and_recognize(image_path, detector, text_recognizer, scale_factor)
    #texts = crop_and_recognize(image_path, detector, text_recognizer, delta)
    #print(f"Texts for {k}:", texts)

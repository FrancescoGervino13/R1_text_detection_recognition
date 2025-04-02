from openai import AzureOpenAI
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from mmocr.apis import MMOCRInferencer

from merge_test import crop

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

def scale_polygon(bbox, scale_factor) :
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

def crop_and_recognize(image, detector, text_recognizer, delta, x_limit, y_limit):
    """Detects text bounding boxes, crops them, and recognizes text."""

    bboxes, scores = crop([image], detector, delta, x_limit, y_limit)

    for i in range(len(bboxes)) : 
        x1, y1 = bboxes[i][0]
        x2, y2 = bboxes[i][1]

        # Crop the image
        cropped_img = image[y1:y2, x1:x2]

        # Recognize text in cropped region
        recognized_text = text_recognizer.recognize_text(cropped_img)
        print(recognized_text)
        print(scores[i])

        # Show the cropped image
        cv2.imshow("Cropped Image", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #return recognized_texts

# Initialize Text Recognition and Detector
text_recognizer = TextInImage()
detector = MMOCRInferencer(det='dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015')
delta = 5; x_limit = 639; y_limit = 479

# Process multiple images
for k in range(1,15) :
    image_path = f"/text_dec_and_rec/resized{k}.jpg"
    # Load original image
    image = cv2.imread(image_path)
    crop_and_recognize(image, detector, text_recognizer, delta, x_limit, y_limit)
    #texts = crop_and_recognize(image_path, detector, text_recognizer, delta)
    #print(f"Texts for {k}:", texts)

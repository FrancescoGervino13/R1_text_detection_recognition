import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmocr.apis import MMOCRInferencer

from merging import merge

def crop(image, detector, delta=0, x_limit = 639, y_limit = 479) :

    result = detector(image)

    # Extract bounding boxes
    predictions = result['predictions'][0]  # List of detected bounding boxes
    bboxes = predictions['det_polygons']
    scores = predictions['det_scores']

    coords = []
    # Draw bounding boxes on the image
    for bbox in bboxes:     
        x = [bbox[0],bbox[2],bbox[4],bbox[6]]
        y = [bbox[1],bbox[3],bbox[5],bbox[7]]
        x_tl = max(round(min(x))-delta,0); y_tl = max(round(min(y))-delta,0)
        x_br = min(round(max(x))+delta,x_limit); y_br = min(round(max(y))+delta,y_limit)
        coords.append([[x_tl,y_tl], [x_br,y_br]])

    bboxes, scores = merge(image,coords,scores)

    return bboxes, scores
import numpy as np
import cv2
from PIL import Image as PIL_image
from io import BytesIO
import base64

def crop(bboxes, delta=0, x_limit = 639, y_limit = 479) :
        """ Modify the bounding boxes from tuples of 8 elements([x1 y1 x2 y2 x3 y3 x4 y4]) to [[x_top_left, y_top_left],[x_bottom_right, y_bottom_right]] and also it widens the box by delta pixels"""
        coords = []
        for bbox in bboxes:     
            x = [bbox[0],bbox[2],bbox[4],bbox[6]]
            y = [bbox[1],bbox[3],bbox[5],bbox[7]]
            x_tl = max(round(min(x))-delta,0); y_tl = max(round(min(y))-delta,0)
            x_br = min(round(max(x))+delta,x_limit); y_br = min(round(max(y))+delta,y_limit)
            coords.append([[x_tl,y_tl], [x_br,y_br]])

        return coords

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

def merge(boxes): 
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

def encode_image_array(image_array):
    """Converts a NumPy image (OpenCV format) to base64."""
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    pil_image = PIL_image.fromarray(image_rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
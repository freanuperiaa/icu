import os

import darknet
import cv2


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    # Colored labels dictionary
    color_dict = {
        'face_shield' : [206, 209, 108],
        'no face_shield' : [245, 239, 243],
        "face_mask": [0, 255, 255], 
        "no face_mask": [0, 255, 255], 
        'person' : [50, 205, 50]
    }
    

    for label, confidence, bbox in detections:
        x, y, w, h = (bbox[0],
              bbox[1],
              bbox[2],
              bbox[3])
        name_tag = label
        for name_key, color_val in color_dict.items():
            if name_key == name_tag:
                color = color_val 
                xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                cv2.rectangle(img, pt1, pt2, color, 1)
                cv2.putText(img,
                            detections[0][0] +
                            " [" + str((confidence * 100, 2))[2:7] + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)
    return img


class Darknet:

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.darknet_image = darknet.make_image(self.frame_width, self.frame_height, 3)
        
        # files for masks and shields 
        # change directories below accordingly
        configPath = "./cw/yolov4-tiny-3l-obj.cfg"
        weightPath = "./cw/yolov4-tiny-3l-obj_bestAPR27.weights"
        metaPath = "./cw/obj.data"

        # files for pretrained model
        # change directories below accordingly
        configPath2 = "./cfg/yolov4-tiny-3l.cfg"
        weightPath2 = "./cw/yolov4-tiny.weights"
        metaPath2 = "./cfg/coco.data"


        if not os.path.exists(configPath):                              # Checks whether file exists otherwise return ValueError
            raise ValueError("Invalid config path `" +
                            os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                            os.path.abspath(metaPath)+"`")
        if not os.path.exists(configPath2):
            raise ValueError("Invalid config path `" +
                            os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath2):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath2):
            raise ValueError("Invalid data file path `" +
                            os.path.abspath(metaPath)+"`")

        network, class_names, class_colors = darknet.load_network(configPath, metaPath, weightPath, batch_size=1)
        self.network = network
        self.class_names = class_names

        network, class_names, class_colors = darknet.load_network(configPath2, metaPath2, weightPath2, batch_size=1)
        self.network_pretrained = network
        self.class_names_extended = class_names

    def predict(self, image):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=0.25)
        frame_with_detections = cvDrawBoxes(detections, frame_resized)

        # preds = darknet.detect_image(self.network_pretrained, self.class_names_extended, self.darknet_image, thresh=0.25)
        # print(preds)
        return frame_with_detections, detections


import os

import darknet
import cv2


class Darknet:

    def __init__(self, frame_width, frame_height):
        # load pretrained model 
        configPath = "./cfg/yolov4.cfg"
        weightPath = "./yolov4.weights"
        metaPath = "./cfg/coco.data"

        # custom trained models
        # configPath = "./cfg/yolov4-tiny-custom-mask.cfg"                                 # Path to cfg
        # weightPath = "./mask.weights"                                 # Path to weights
        # metaPath = "./cfg/mask.data" 

        if not os.path.exists(configPath):                              # Checks whether file exists otherwise return ValueError
            raise ValueError("Invalid config path `" +
                            os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                            os.path.abspath(metaPath)+"`")

        network, class_names, class_colors = darknet.load_network(configPath,  metaPath, weightPath, batch_size=1)
        self.network = network
        self.class_names = class_names
        self.class_colors = class_colors
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.darknet_image = darknet.make_image(self.frame_width, self.frame_height, 3)

    def predict(self, image):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

        return darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=0.25)

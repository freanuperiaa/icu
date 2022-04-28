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

def cvDrawBoxesSocDis(detections, img):
    """
    :param:
    detections = total detections in one frame
    img = image from detect_image method of darknet
    :return:
    img with bbox
    """
    #================================================================
    # 3.1 Purpose : Filter out Persons class from detections and get 
    #           bounding box centroid for each person detection.
    #================================================================
    if len(detections) > 0:  						# At least 1 detection in the image and check detection presence in a frame  
        centroid_dict = dict() 						# Function creates a dictionary and calls it centroid_dict
        objectId = 0								# We inialize a variable called ObjectId and set it to 0
        for detection in detections:				# In this if statement, we filter all the detections for persons only
            # Check for the only person name tag 
            name_tag = str(detection[0].decode())   # Coco file has string of all the names
            if name_tag == 'person':                
                x, y, w, h = detection[2][0],\
                            detection[2][1],\
                            detection[2][2],\
                            detection[2][3]      	# Store the center points of the detections
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox            
                # Append center point of bbox for persons detected.
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) # Create dictionary of tuple with 'objectId' as the index center points and bbox
                objectId += 1 #Increment the index for each detection
    #=================================================================#
    
    #=================================================================
    # 3.2 Purpose : Determine which person bbox are close to each other
    #=================================================================            	
        red_zone_list = [] # List containing which Object id is in under threshold distance condition. 
        red_line_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # Get all the combinations of close detections, #List of multiple items - id1 1, points 2, 1,3
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]  	# Check the difference between centroid x: 0, y :1
            distance = is_close(dx, dy) 			# Calculates the Euclidean distance
            if distance < 75.0:						# Set our social distance threshold - If they meet this condition then..
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)       #  Add Id to a list
                    red_line_list.append(p1[0:2])   #  Add points to the list
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)		# Same for the second id 
                    red_line_list.append(p2[0:2])
        
        for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
            if idx in red_zone_list:   # if id is in red zone list
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2) # Create Red bounding boxes  #starting point, ending point size of 2
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # Create Green bounding boxes
		#=================================================================#

		#=================================================================
    	# 3.3 Purpose : Display Risk Analytics and Show Risk Indicators
    	#=================================================================        
        text = "People at Risk: %s" % str(len(red_zone_list)) 			# Count People at Risk
        location = (10,25)												# Set the location of the displayed text
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)  # Display Text

        for check in range(0, len(red_line_list)-1):					# Draw line between nearby bboxes iterate through redlist items
            start_point = red_line_list[check] 
            end_point = red_line_list[check+1]
            check_line_x = abs(end_point[0] - start_point[0])   		# Calculate the line coordinates for x  
            check_line_y = abs(end_point[1] - start_point[1])			# Calculate the line coordinates for y
            if (check_line_x < 75) and (check_line_y < 25):				# If both are We check that the lines are below our threshold distance.
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)   # Only above the threshold lines are displayed. 
        #=================================================================#
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
        detections_persons = darknet.detect_image(self.network_pretrained, self.class_names_extended, self.darknet_image, thresh=0.25)

        image = cvDrawBoxesSocDis(detections_persons, frame_resized)
        image = cvDrawBoxes(detections, image)

        return image, detections

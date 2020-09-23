# import required packages
import cv2
import argparse
import numpy as np

precaution = ''
prediction = ''




classes_path="sk-classes.txt"
config_path="sk-yolo.cfg"
weights_path="sk-10000.weights"


conf_threshold = 0.5
nms_threshold = 0.4
scale = 0.00392
classes = None
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
print(COLORS.ndim)
print(type(COLORS))
net = cv2.dnn.readNet(weights_path, config_path)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    print(color)
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
def detect(frame):
    class_ids = []
    confidences = []
    boxes = []
   
    image=frame
    Width = image.shape[1]
    Height = image.shape[0]
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                print(class_ids)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print("indices",indices)
    if len(indices) == 0:
        print("No disease detected")
        cv2.putText(image, "No disease detected", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    return image

    
    
    
capture = cv2.VideoCapture(0)
while True:
    _,frame = capture.read()
    #frame=cv2.imread("2.jpg")
    pred=detect(frame)
    cv2.imshow("object detection", pred)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
capture.release()


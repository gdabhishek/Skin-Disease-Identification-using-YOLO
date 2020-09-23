import numpy as np
import cv2
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os

root = Tk()
root.title("Skin Disease Identification")
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)
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
#print(COLORS.ndim)
#print(COLORS)
net = cv2.dnn.readNet(weights_path, config_path)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    #print(color)
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
                #print(class_ids)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print("indices",indices)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    return image

def openfn():
    filename = filedialog.askopenfilename(title='open')  
    return filename
def open_img():
    x = openfn()
    #test_image = image.load_img(x)
    #test_image = image.img_to_array(test_image)
    img1=cv2.imread(x)
    img = detect(img1)
    #print(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(250, 250))
    img = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(img)))
    global panel
    panel.pack_forget()
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()
panel=Label(root)
btn = Button(root, text='Select Skin Image', command=open_img).pack()

root.mainloop()

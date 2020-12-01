import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

path_dir = 'IMG_test'
dir_path = os.listdir(path_dir)
print(dir_path)

config_path = 'yolov3.cfg'
weight_path = 'yolov3.weights'
class_path = 'yolov3.txt'
size_1 = (416, 416)
size_2 = (608, 608)
size_0 = (320, 320)
conf_threshold = 0.2
nms_threshold = 0.4
count_loop = 0
space_slot = []

def spots():
    loc = 0


def callback(foo):
    pass

def drawRectangle(img,circle_center, a, b, c, d, text):
    sub_img = img[b:b + d, a:a + c]
    text_x = int(a+ c/4)
    text_y = int(b+d/1.5)
    for i in range(len(circle_center)):
        x_circle,y_circle = circle_center[i]
        if x_circle in range(a,a+c+1) and y_circle in range(b,b+d+1):
            cv2.rectangle(img, (a, b), (a + c, b + d), (0, 0, 255), 2)
            cv2.putText(img,f"{text}",(text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            if text in space_slot:
                space_slot.remove(text)
            return
        else:
            cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 0), 2)
            cv2.putText(img,f"{text}",(text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            if text not in space_slot:
                space_slot.append(text)

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = (0, 0, 255)

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    # cv2.putText(img, "%.2f" % confidence, (x, y_plus_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_circle_predict(img, class_id, confidence, x, y, w,h):
    # if w>=h:
    cv2.circle(img,(x+round(w/2),y+round(h/2)),1,(0,0,255),2)
    # else:
    #     cv2.circle(img,(x+round(w/2),y+round(h/2)),round(w/4),(0,0,255),2)
#
# cv2.namedWindow("Threshold")
# cv2.createTrackbar('Confidence_threshold', 'Threshold', 1, 10, callback)
# cv2.createTrackbar('NMS_threshold', 'Threshold', 1, 100, callback)

with open('ROI/rois_video.csv', 'r', newline='') as inf:
    csvr = csv.reader(inf)
    rois = list(csvr)

for image_dir_path in dir_path:
    image_path = f'{path_dir}/{image_dir_path}'
    # image_path = 'base/2020-07-22-b1/ch01_00000000000000500.jpg'
    count_loop+=1

    image = cv2.imread(image_path)
    image = cv2.resize(image,dsize=None,fy=0.8,fx=0.8)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 1/255
    # scale = 1
    classes = None

    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNetFromDarknet(config_path, weight_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # net = cv2.dnn.readNet(weight_path, config_path)

    blob = cv2.dnn.blobFromImage(image, scale, size_2, (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))
    # while True:
    image_copy = image.copy()
    class_ids = []
    confidences = []
    boxes = []
    center_circle = []

    # conf_threshold = cv2.getTrackbarPos('Confidence_threshold', 'Threshold')
    # nms_threshold = cv2.getTrackbarPos('NMS_threshold', 'Threshold')
    # conf_threshold/=10
    # nms_threshold/=100

    # Thực hiện xác định bằng HOG và SVM

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    count = 0
    for i in indices:
        i = i[0]
        if class_ids[i] == 2:
            count+=1
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            # draw_prediction(image_copy, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
            draw_circle_predict(image_copy, class_ids[i], confidences[i], round(x), round(y), w, h)
            center_circle.append([x+w/2,y+h/2])
    # print(center_circle)
    rois = [[int(float(j)) for j in i] for i in rois]
    for i in range(len(rois)):
        drawRectangle(image_copy,center_circle,rois[i][0], rois[i][1], rois[i][2], rois[i][3], rois[i][4])
    cv2.putText(image_copy, f"Number of cars: {count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(image_copy, f"Free slot: {sorted(space_slot)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("object detection", image_copy)
    # cv2.imwrite(f"Image_extract_22_c9/Predict_image{count_loop}.jpg",image_copy)
    print(count_loop)
    if cv2.waitKey() & 0xff == ord('q'):
        pass


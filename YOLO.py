import time
import cv2
import argparse
import numpy as np

# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True,
#                 help='path to input image')
# ap.add_argument('-c', '--config', required=True,
#                 help='path to yolo config file')
# ap.add_argument('-w', '--weights', required=True,
#                 help='path to yolo pre-trained weights')
# ap.add_argument('-cl', '--classes', required=True,
#                 help='path to text file containing class names')
# args = ap.parse_args()

image_path = 'data_b1.jpg'
config_path = 'yolov3.cfg'
weight_path = 'yolov3.weights'
class_path = 'yolov3.txt'
size_1 = (416,416)
size_2 = (608,608)
size_3 = (320,320)
size_resize = 0.5

def callback(foo):
  pass

cv2.namedWindow("Threshold")
cv2.createTrackbar('Confidence_threshold', 'Threshold', 1, 10, callback)
cv2.createTrackbar('NMS_threshold', 'Threshold', 1, 100, callback)

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = (0,0,255)

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, "%.2f"%confidence, (x , y_plus_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(img, label, (x , y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


image = cv2.imread(image_path)
image = cv2.resize(image,dsize=None,fy=size_resize,fx=size_resize)

Width = image.shape[1]
Height = image.shape[0]
scale = 1/255
# scale = 1
classes = None

with open(class_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromDarknet(config_path, weight_path)
# net = cv2.dnn.readNet(weight_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

blob = cv2.dnn.blobFromImage(image, scale, size_2, (0, 0, 0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))
while True:
    image_copy = image.copy()
    class_ids = []
    confidences = []
    boxes = []
    # conf_threshold = 0.25
    # nms_threshold = 0.2
    conf_threshold = cv2.getTrackbarPos('Confidence_threshold', 'Threshold')
    nms_threshold = cv2.getTrackbarPos('NMS_threshold', 'Threshold')
    conf_threshold/=10
    nms_threshold/=100

    # Thực hiện xác định bằng HOG và SVM
    start = time.time()

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
                x = center_x - w * size_resize
                y = center_y - h * size_resize
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    count = 0
    for i in indices:
        i = i[0]
        if class_ids[i] == 2:
            count += 1
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image_copy, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    cv2.putText(image_copy, f"Number of cars: {count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("object detection", image_copy)
    cv2.imwrite("Predict_image.jpg",image_copy)


    end = time.time()
    # print("YOLO Execution time: " + str(end-start))

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    # cv2.destroyAllWindows()

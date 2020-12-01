import time
import cv2
import numpy as np
import csv

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

video_path = 'Khare_testvideo_01.mp4'
config_path = 'yolov3.cfg'
weight_path = 'yolov3.weights'
class_path = 'yolov3.txt'
size_0 = (320,320)
size_1 = (416,416)
size_2 = (608,608)
conf_threshold = 0.1
nms_threshold = 0.5
count_loop = 0
space_slot = []
center_circle = []

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

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

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = (0,0,255)

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, "%.2f"%confidence, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_circle_predict(img, class_id, confidence, x, y, w,h):
    # if w>=h:
    cv2.circle(img,(x+round(w/2),y+round(h/2)),1,(0,0,255),2)

cap = cv2.VideoCapture(video_path)
video_output = cv2.VideoWriter('check_slot_608.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1536,864))
net = cv2.dnn.readNetFromDarknet(config_path, weight_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open('ROI/rois_video.csv', 'r', newline='') as inf:
    csvr = csv.reader(inf)
    rois = list(csvr)

while True:
    center_circle = []
    ret,image = cap.read()
    # cv2.imwrite("Khare_testvideo_01.jpg",image)
    image = cv2.resize(image,dsize=None,fy=0.8,fx=0.8)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 1/255

    classes = None

    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # net = cv2.dnn.readNet(weight_path, config_path)

    blob = cv2.dnn.blobFromImage(image, scale, size_2, (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

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
            if w+h<500:
                # draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
                draw_circle_predict(image, class_ids[i], confidences[i], round(x), round(y), w, h)
                center_circle.append([x + w / 2, y + h / 2])

    rois = [[int(float(j)) for j in i] for i in rois]
    for i in range(len(rois)):
        drawRectangle(image, center_circle, rois[i][0], rois[i][1], rois[i][2], rois[i][3], rois[i][4])
    cv2.putText(image, f"So xe la: {count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(image, f"Free slot: {sorted(space_slot)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    video_output.write(image)
    cv2.imshow("object detection", image)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    # cv2.destroyAllWindows()

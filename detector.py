import cv2
import csv

frame = cv2.imread('base/2020-07-22-c9/ch01_00000000000000500.jpg')
frame = cv2.resize(frame,dsize=None,fy=0.5,fx=0.5)

class spots:
    loc = 0

def drawRectangle(img, a, b, c, d, text):
    sub_img = img[b:b + d, a:a + c]
    text_x = int(a+ c/2)
    text_y = b+d
    # if pix in range(min, max):
    cv2.rectangle(img, (a, b), (a + c, b + d), (0, 0, 255), 3)
    cv2.putText(img,f"{text}",(text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    # else:
    #     cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 0), 3)
    #     spots.loc += 1
    #     cv2.putText(img,f"{text}",(text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

with open('ROI/rois.csv', 'r', newline='') as inf:
    csvr = csv.reader(inf)
    rois = list(csvr)

rois = [[int(float(j)) for j in i] for i in rois]

for i in range(len(rois)):
    drawRectangle(frame, rois[i][0], rois[i][1], rois[i][2], rois[i][3], rois[i][4])

cv2.imshow("Image",frame)
cv2.waitKey()
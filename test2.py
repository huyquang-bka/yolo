import cv2

video_path = 'Khare_testvideo_01.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret,img = cap.read()
    cv2.imwrite('IMG_test/Img_1.jpg', img)
    cv2.imshow("Img",img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
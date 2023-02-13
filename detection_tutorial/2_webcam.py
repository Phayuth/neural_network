import cv2

capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 420)

while True:
    success, img = capture.read()
    cv2.imshow("Image",img)
    cv2.waitKey(1)
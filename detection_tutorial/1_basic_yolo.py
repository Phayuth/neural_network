from ultralytics import YOLO
import cv2

model = YOLO('./weight/yolov8n.pt')
model.info()
pred = model('./dataset/2.png', show=True)
cv2.waitKey(0)









# https://www.youtube.com/watch?v=WgPbbWmnXJ8
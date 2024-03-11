import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import ultralytics
import cv2


class CVYOLOSimpleDetection:

    def __init__(self, torchModelPath) -> None:
        self.model = ultralytics.YOLO(torchModelPath)

    def detect(self, img):
        result = self.model(img, stream=True, conf=0.5)
        for r in result:
            for eachbox in r.boxes:
                xmin, ymin, xmax, ymax = eachbox.xyxy[0]
                confident = eachbox.conf[0]
                classesNumber = int(eachbox.cls[0].item())
                classesName = self.model.names[classesNumber]

                cv2.putText(img, text=classesName + str(round(confident.item(), 2)), org=(int(xmin), int(ymin + (ymax - ymin) / 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, color=(255, 0, 255))
                cv2.rectangle(img, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)), color=(255, 0, 255), thickness=3)

    def detect_yolo_annoted(self, img):
        results = self.model(img)
        return results[0].plot()


if __name__ == "__main__":
    from camera.v4l2camera import CVCamera

    cam = CVCamera(4)

    torchModelPath = "./datasave/neural_weight/yolov8x-seg.pt"
    cvc = CVYOLOSimpleDetection(torchModelPath)

    while cam.capture.isOpened():
        success, img = cam.capture.read()
        imgshow = cvc.detect(img)

        cv2.imshow("Apple mask only", imgshow)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.capture.release()
    cv2.destroyAllWindows()

import winsound
import imutils
from inference import Detect_model
from PIL import Image
import numpy as np
import cv2
import time
import threading


class detect_person(object):
    def __init__(self):
        self.model = Detect_model()
        self.width = 720
        self.static = self.load_static_img()

    def inference_video(self, show_plan=False):
        capture = cv2.VideoCapture(0)
        fps = 0.0
        count_nobady = 0  # 没有人的连续帧数计数
        while True:
            t1 = time.time()
            ref, frame = capture.read()  # shape is (480, 640, 3)
            frame = imutils.resize(frame, width=self.width)  # shape is (540, 720, 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame, flag = self.model.detect_image(frame)
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fps = (fps + (1. / (time.time() - t1))) / 2
            if show_plan:
                if flag:
                    self.buzzer_warning()
                    count_nobady = 0
                    frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("video", frame)
                else:
                    count_nobady += 1
                    if count_nobady == 1:
                        static_frame = frame
                    cv2.imshow("video", static_frame)
                if cv2.waitKey(1) & 0xff == ord("q"):
                    capture.release()
                    break
            else:
                if flag:
                    self.buzzer_warning()
                    frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("video", frame)
                else:
                    cv2.imshow("video", self.static)
                if cv2.waitKey(1) & 0xff == ord("q"):
                    capture.release()
                    break
        capture.release()

    def alarm(self):
        winsound.Beep(1000, 200)

    def buzzer_warning(self):
        make_sound = threading.Thread(target=self.alarm, args=())
        make_sound.start()

    def load_static_img(self, img_path="./model_data/static_img.jpg"):
        static_img = cv2.imread(img_path)
        static_img = imutils.resize(static_img, width=self.width)
        return static_img


if __name__ == "__main__":
    model = detect_person()
    model.inference_video()

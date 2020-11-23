import numpy as np
import imutils
import cv2

cap = cv2.VideoCapture("./model_data/1.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1080)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

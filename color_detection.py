import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
writer = cv.VideoWriter('color_detection_video.mp4', cv.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

while True:
    _, img = cap.read()
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = cv.medianBlur(hsv, 7)

    L_limit = np.array([98, 50, 50])
    U_limit = np.array([139, 255, 255])

    b_mask = cv.inRange(hsv, L_limit, U_limit)
    blue = cv.bitwise_and(img, img, mask=b_mask)
    writer.write(blue)
    cv.imshow('Result', blue)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
writer.release()
cv.destroyAllWindows()

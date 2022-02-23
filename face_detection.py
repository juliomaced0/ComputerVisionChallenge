import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
writer = cv.VideoWriter('face_detection_video.mp4', cv.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

while True:
    _, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

    writer.write(img)
    cv.imshow('img', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
writer.release()
cv.destroyAllWindows()

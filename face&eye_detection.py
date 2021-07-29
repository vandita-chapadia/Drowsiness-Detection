import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 2.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame,"Face",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(250,250,250),1)

    eyes = eye_cascade.detectMultiScale(gray, 2.3, 4)
    for (x1, y1, w1, h1) in eyes:
        cv2.rectangle(frame, (x1, y1), (x1+ w1, y1 + h1), (255, 250, 34), 2)
        cv2.putText(frame, "Eyes", (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (250, 250, 250), 1)

    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
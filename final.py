# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
from datetime import datetime


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

yawns = 0
total=0
yawn_status = False

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:, 1])


def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:, 1])


def mouth_open(image):
    landmarks = get_landmarks(image)

    if landmarks == "error":
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance


def sound_alarm(path):
    playsound.playsound(path)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="alarm.wav",
                help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 24


COUNTER = 0
ALARM_ON = False


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=670)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    rects = detector(gray, 0)
    image_landmarks, lip_distance = mouth_open(frame)

    prev_yawn_status = yawn_status

    if lip_distance > 40:
        yawn_status = True

        cv2.putText(frame, "Yawning", (50, 380),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

        output_text = " Yawn Count: " + str(yawns + 1)

        cv2.putText(frame, output_text, (30, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 127), 2)

    else:
        yawn_status = False

    if prev_yawn_status == True and yawn_status == False:
        yawns += 1

    faces = face_cascade.detectMultiScale(gray, 2.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #cv2.putText(frame, "Face", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (250, 250, 250), 1)

    eyes = eye_cascade.detectMultiScale(gray, 2.3, 4)
    for (x1, y1, w1, h1) in eyes:
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 0), 2)
        #cv2.putText(frame, "Eyes", (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (250, 250, 250), 1)


    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)


        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)


        ear = (leftEAR + rightEAR) / 2.0


        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            total += 1



            if COUNTER >= EYE_AR_CONSEC_FRAMES:



                if not ALARM_ON:
                    ALARM_ON = True


                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                                   args=(args["alarm"],))
                        t.deamon = True
                        t.start()


                    cv2.putText(frame, "DROWSINESS ALERT!", (270, 380),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        else:

            COUNTER = 0
            ALARM_ON = False


        cv2.putText(frame, "EAR: {:.2f}".format(ear), (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame,"Eyes Blink Count:"+str(total), (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, str(datetime.now()), (450,500),cv2.FONT_HERSHEY_SIMPLEX , .5, (255,255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


cv2.destroyAllWindows()
vs.stop()



import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_detector.detectMultiScale(frame_grayscale)

    # 1.7 => higher it is higher it is blurr
    # 20 => minNeighbour

    for(x, y, w, h) in face:

        the_face = frame[y:y+h, x:x+w]
        # the_face = (x, y, w, h)

        # face is made black and white
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile = smile_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=20)

        eyes = eye_detector.detectMultiScale(face_grayscale, 1.1, 30)
        # there scles are needed to be tried and error

        for(x_, y_, w_, h_) in eyes:
            cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (0, 255, 0), 5)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)

        # smile loop is nested to run only inside the face
        # for(x_, y_, w_, h_) in smile:
        #      cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (0, 255, 0), 5)
        #      now instead of drawing square we write smiling

        if len(smile) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=5,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 0, 0))
        else:
            cv2.putText(frame, 'Why so serious', (x, y+h+40),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(255, 0, 255))

    # for(x, y, w, h) in smile:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

    cv2.imshow('smile detected', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
print("Code Completed")

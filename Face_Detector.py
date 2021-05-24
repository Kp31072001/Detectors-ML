vieimport cv2

from random import randrange

# loaded some pretarined data
traind_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# photo is chosen to detects fixed photo
# img = cv2.imread('friends.jpg')

# capturing webcam
webcam = cv2.VideoCapture('video.mp4')
# the 0 => is for default webcam
# inplace of 0 we can put 'filename' for local videos


# loop for taking all frame of webcam
while True:
    # to read the frame , it return true thing successful... is a boolean if successful or not
    successful_frame_read, frame = webcam.read()

    # gray scalling images no rgb only one value
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect.faces
    face_coordinates = traind_face_data.detectMultiScale(grayscaled_img)

    for(x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (randrange(128, 255), randrange(128, 255), randrange(128, 255)), 10)

    cv2.imshow('Kankan face detector ', frame)
    key = cv2.waitKey(1)
    # it waits 1 ms

    # while loop braks for 'Q' or 'q'
    if key == 81 or key == 113:
        break

# Release the web cam
webcam.release()

print("code Completed")


"""
# Detect.faces
face_coordinates = traind_face_data.detectMultiScale(grayscaled_img)
# it returns x, y as the top left of the rectangle and w, h as the width and height of the rectangle
# detectMultiscle does detect all small and big
# face coordinated gives coordinate of the rectangle sarrounding the face


# Now we have to draw the rectangle and using original color img

# this is for more than one image
for(x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h),
                  (randrange(128, 255), randrange(128, 255), randrange(128, 255)), 10)

# for one face
# cv2.rectangle(img, (148, 182), (148+577, 182+577), (0, 0, 255), 2)

# print(face_coordinates)

# displaying image
cv2.imshow('Kankan face detector ', img)
# it waits with the image otherwise it will open for split of a sec
cv2.waitKey()
"""

print("Code Completed")

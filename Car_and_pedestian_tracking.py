import cv2
from random import randrange


# pre-trained cars
classifier_file_car = 'cars.xml'
classifer_file_ped = 'pedestrian.xml'

# img_file = 'car.jpg'
# img = cv2.imread(img_file)


car_tracker = cv2.CascadeClassifier(classifier_file_car)
ped_tracker = cv2.CascadeClassifier(classifer_file_ped)


video = cv2.VideoCapture('testCam3.mp4')

while True:

    successful_frame_read, frame = video.read()

    if successful_frame_read:

        black_n_white = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        break

    car_coordinate = car_tracker.detectMultiScale(black_n_white)
    ped_coordinate = ped_tracker.detectMultiScale(black_n_white)

    for(x, y, w, h) in car_coordinate:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 0, randrange(128, 256)), 5)

    for(x, y, w, h) in ped_coordinate:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, randrange(128, 256), 0), 5)

    cv2.imshow('Kanakans car detector', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break


print("Code Completed")

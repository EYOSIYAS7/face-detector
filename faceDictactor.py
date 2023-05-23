import cv2

trained_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Capturing form the webcam
webCam = cv2.VideoCapture(0)

while True:

# reading a single frame at a time
    successfully_frame_read, frame = webCam.read()
    grayScaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # this will return a list of detected face coordinates
    faceCordinates = trained_face.detectMultiScale(grayScaled)



    for (x,y,w,h) in faceCordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("face detection app", frame)

    key = cv2.waitKey(1) 
# press esc or Q TO exit
    if key ==81 or key==113 or key == 27:
        break

webCam.release() 

print("code completed ")

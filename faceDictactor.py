import cv2

trained_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
trained_smile = cv2.CascadeClassifier("haarcascade_smile.xml")
# Capturing form the webcam
webCam = cv2.VideoCapture(0)

while True:

# reading a single frame at a time
    successfully_frame_read, frame = webCam.read()
    grayScaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # this will return a list of detected face coordinates and smile coordinates in the face 
    faceCordinates = trained_face.detectMultiScale(grayScaled)
    smileCoordinates= trained_smile.detectMultiScale(grayScaled)


    for (x,y,w,h) in faceCordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        # extracting the face from the frame by slicing
        theFace = frame[y:y+h, x:x+w]
        grayFace = cv2.cvtColor(theFace, cv2.COLOR_BGR2GRAY)
        smileCoordinates= trained_smile.detectMultiScale(grayFace,scaleFactor=1.7,minNeighbors=20)
       
        if(len(smileCoordinates)>0):
            cv2.putText(frame, "smiling", (x, y+h+80), fontScale=3, fontFace= cv2.FONT_HERSHEY_COMPLEX_SMALL , color=(0,0,0))

    cv2.imshow("face detection app", frame)

    key = cv2.waitKey(1) 
# press esc or Q TO exit
    if key ==81 or key==113 or key == 27:
        break

webCam.release() 
cv2.destroyAllWindows()

print("code completed ")

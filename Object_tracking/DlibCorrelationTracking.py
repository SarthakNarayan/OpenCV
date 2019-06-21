# dlib correlation tracker
import cv2
import numpy as np
import dlib
import imutils

cap = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\IITB Internship\VIDEOS\1080P\ONLY CIRCULAR LOW DENSITY.mp4")
tracking = False
tracker = dlib.correlation_tracker()
# Read until video is completed
while(cap.isOpened()):
    ret, frame = cap.read()
    # frame from BGR to RGB ordering (dlib needs RGB ordering)
    #frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if ret == True:
        if tracking:
            tracker.update(rgb)
            pos = tracker.get_position()

        		# unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
            
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(60) & 0xFF
        if key == ord("s"):
            tracking = True
            initBoundingBox = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
            (x,y,w,h) = initBoundingBox
            (startX, startY, endX, endY) = (x,y,x+w,y+h)
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)

        if key == ord("q"):
                break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()
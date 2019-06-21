# Keep the downloaded files in the same directory as the python script
# Creates Problem when two similar particles come close to each other

import cv2
tracker = cv2.TrackerGOTURN_create()   
cap = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\IITB Internship\VIDEOS\1080P\ONLY CIRCULAR LOW DENSITY.mp4")

tracker = cv2.TrackerGOTURN_create()

while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (600, 600), interpolation = cv2.INTER_LINEAR)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == 2555904:
                ret, frame = cap.read()
        if key == ord("s"):
                break
initBoundingBox = cv2.selectROI("Frame", frame, fromCenter=False,
                                       showCrosshair=True)
tracker.init(frame,initBoundingBox)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, (600, 600), interpolation = cv2.INTER_LINEAR)
        
        timer = cv2.getTickCount()
        # Update tracker
        (success, initBoundingBox) = tracker.update(frame)
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        if success:
        # Tracking success
            p1 = (int(initBoundingBox[0]), int(initBoundingBox[1]))
            p2 = (int(initBoundingBox[0] + initBoundingBox[2]), int(initBoundingBox[1] + initBoundingBox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, "GOTURN Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()

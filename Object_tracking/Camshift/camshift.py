import cv2
import numpy as np

# 0/1 for idle 1 for spyder
cap = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\IITB Internship\VIDEOS\1080P\ONLY CIRCULAR LOW DENSITY.mp4")
while True:
        ret, frame = cap.read()
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0) & 0xFF
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        if key == 2555904:
                ret, frame = cap.read()
        if key == ord("s"):
                break
        
initBoundingBox = cv2.selectROI("Frame", frame, fromCenter=False,
                                       showCrosshair=True)


# To get the object to track
(x,y,w,h) = initBoundingBox
#img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


# initialize the termination criteria for cam shift, indicating
# a maximum of ten iterations or movement by a least one pixel
# along with the bounding box of the ROI
# Here 10 is the iterations the more you increase it the accuracy increases at the cost of accuracy
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 100)

roi = frame[y:y+h,x:x+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# adjust according to the hsv values obtained from trackbar
mask = cv2.inRange(hsv_roi, np.array((55., 28.,118.)), np.array((255.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

while True:
    _ , frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    
    ret, track_window = cv2.CamShift(dst, initBoundingBox, termination)
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(frame,[pts],True, 255,2)
    
    cv2.imshow('img2',img2)
    cv2.imshow('mask',mask)
    k = cv2.waitKey(0) & 0xff
    if key == 2555904:
        ret, frame = cap.read()
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()

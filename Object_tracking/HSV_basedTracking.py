import cv2
import numpy as np

# to start the camera and use webcam
cap = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\IITB Internship\VIDEOS\1080P\ONLY CIRCULAR LOW DENSITY.mp4")

#naming a window for object tracking
#cv2.namedWindow('SettingHSV')

#defining the nothing function
def nothing():
    pass

#creating trackbars
cv2.createTrackbar('h1w','SettingHSV',0,255,nothing)
cv2.createTrackbar('s1w','SettingHSV',0,255,nothing)
cv2.createTrackbar('v1w','SettingHSV',0,255,nothing)
cv2.createTrackbar('h2w','SettingHSV',0,255,nothing)
cv2.createTrackbar('s2w','SettingHSV',0,255,nothing)
cv2.createTrackbar('v2w','SettingHSV',0,255,nothing)

while True:
        ret, frame = cap.read()
        blur = cv2.GaussianBlur(frame, (11, 11), 20)
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

        h1w=cv2.getTrackbarPos('h1w','SettingHSV')
        s1w=cv2.getTrackbarPos('s1w','SettingHSV')
        v1w=cv2.getTrackbarPos('v1w','SettingHSV')
        h2w=cv2.getTrackbarPos('h2w','SettingHSV')
        s2w=cv2.getTrackbarPos('s2w','SettingHSV')
        v2w=cv2.getTrackbarPos('v2w','SettingHSV')

        lower_blue = np.array([55,28,118])
        upper_blue = np.array([255,255,255])

        #thresholding the image to get only white colors by creating a mask
        mask = cv2.inRange(hsv,lower_blue,upper_blue)

        #performing morphological operations on mask
        kernelforerosion = np.ones((5,5) , np.uint8)
        erodedmask = cv2.erode(mask,kernelforerosion,iterations = 1)

        #using bitwiseand to do the mask
        result = cv2.bitwise_and(blur,blur,mask = erodedmask)

        #for finding and drawing contours
        contours,_ = cv2.findContours(erodedmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #frameswithcontours = cv2.drawContours(result,contours,-1,(0,255,0),4)
        #frameswithcontours = cv2.drawContours(frame,contours,-1,(0,255,0),4)

        print(len(contours))
        #getting the centoid of the image
        if(len(contours)==2):
            cnt = contours[0]
            M = cv2.moments(cnt)
            if (M['m00']!=0):
                cx0 = int(M['m10']/M['m00'])
                cy0 = int(M['m01']/M['m00'])

                #drawing a circle at the centroid of the shape
                cv2.circle(result,(cx0,cy0),5,(0,0,255),-1)
                cv2.circle(frame,(cx0,cy0),5,(0,0,255),-1)
            cnt = contours[1]
            M = cv2.moments(cnt)
            if (M['m00']!=0):
                cx1 = int(M['m10']/M['m00'])
                cy1 = int(M['m01']/M['m00'])

                #drawing a circle at the centroid of the shape
                cv2.circle(result,(cx1,cy1),5,(255,0,255),-1)
                cv2.circle(frame,(cx1,cy1),5,(0,0,255),-1)
            if (cy1<cy0):
                frameswithcontours = cv2.drawContours(frame,contours[1],-1,(0,255,0),4)
                print("cy1<cy0")
            else:
                frameswithcontours = cv2.drawContours(frame,contours[0],-1,(0,255,0),4)
                print("cy1>cy0")

        #displaying the video
        #cv2.imshow('SettingHSV',mask)
        cv2.imshow('resultant',result)
        cv2.imshow('frame',frame)
        
        key = cv2.waitKey(1) & 0xFF
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        if key == 2555904:
                ret, frame = cap.read()
        if key == ord("q"):
                break
cap.release()
cv2.destroyAllWindows()

    
    
    

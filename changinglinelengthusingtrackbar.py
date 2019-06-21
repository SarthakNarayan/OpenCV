import cv2
import numpy as np
def nothing(x):
    pass
# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')
# create trackbars for color change
cv2.createTrackbar('x','image',0,255,nothing)
cv2.createTrackbar('y','image',0,255,nothing)
cv2.createTrackbar('t','image',10,255,nothing)
# create switch for ON/OFF functionality
x,y,t=0,0,10
while(1):
    cv2.line(img,(0,0),(x,y),(0,255,0),t)
    x1,y1,t1=x,y,t
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    print(k)
    if k == 27:
        break
# get current positions of four trackbars
    x = cv2.getTrackbarPos('x','image')
    y = cv2.getTrackbarPos('y','image')
    t = cv2.getTrackbarPos('t','image')
    if (x1>x) | (y1>y) | (t1>t):
        img = np.zeros((300,512,3), np.uint8)
     
cv2.destroyAllWindows()

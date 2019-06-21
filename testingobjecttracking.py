import cv2
import numpy as np
cap = cv2.VideoCapture(0)
cnt=[0]
while(1):
    _, frame = cap.read()
    img2gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(img2gray,60,255,cv2.THRESH_BINARY_INV)
    threshoriginal=thresh
    cv2.GaussianBlur(thresh,(5,5),10)
    kernel=np.ones((5,5),np.uint8)
    erosion=cv2.erode(thresh,kernel,iterations=1)
    dilation=cv2.dilate(thresh,kernel,iterations=1)
    _,contours,_=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt=contours[0]
    hull=cv2.convexHull(cnt)
    cv2.drawContours(frame,[hull],-1,(0,255,255),3)
    cv2.drawContours(frame,contours,-1,(0,255,0),3)
    x,y,w,h=cv2.boundingRect(cnt)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow('thresh',thresh)
    cv2.imshow('frame',frame)
    #cv2.imshow('erosion',erosion)
    #cv2.imshow('dilation',dilation)
    #cv2.imshow('threshoriginal',threshoriginal)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
img = cv2.imread('C:\Users\Lenovo\Desktop\open cv programsandimages\images\standard_test_images\lena_color_512.tif',1)
grayscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret1, thresh = cv2.threshold(grayscale,230,255,cv2.THRESH_BINARY_INV)
_,contours,_=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img1=cv2.drawContours(img,contours,-1,(0,255,0),2)
cnt=contours[0]
M=cv2.moments(cnt)
cx=int(M['m10']/M['m00'])
cy=int(M['m10']/M['m00'])
img1=cv2.circle(img1,(cx,cy),5,(0,0,255),-1) #can use same variable again instead of
cv2.imshow('show',img)                       #creating a new one like in this case img1    
cv2.imshow('perfect',thresh)
cv2.imshow('contours',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


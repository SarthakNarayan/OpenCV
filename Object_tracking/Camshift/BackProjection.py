import cv2
import numpy as np

kernel = np.ones((11,11),np.uint8)
# image where we have to look for the object
image = cv2.imread(r"C:\Users\Lenovo\Desktop\PyImageSearch\Image_Classifier_SSD\object-detection-deep-learning\images\example_01.jpg")
hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#initBoundingBox is the object or region of object we need to find
initBoundingBox = cv2.selectROI("image", image, fromCenter=False,
                                       showCrosshair=True)
(x,y,w,h) = initBoundingBox
roi = image[x:x+w,y:y+h]
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

# normalize histogram and apply backprojection
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsv_image],[0,1],roi_hist,[0,180,0,256],1)
copy = dst.copy()

# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# You can play with disc or kernel
cv2.filter2D(copy,-1,disc,copy)

ret,thresh = cv2.threshold(copy,80,255,0)
#thresh = cv2.dilate(thresh , kernel , iterations = 1) # dilation
# play with kernel
thresh = cv2.morphologyEx(thresh ,cv2.MORPH_CLOSE, kernel )

# Merges all the 3 color channels in this case we have a binary image but that image should 
# have the same color dimensions as the image. image has 3 color channels so should thresh
# hence we merge it with itself
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(image,thresh)

cv2.imshow("image" , image)
cv2.imshow("res" , res)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

image = cv2.imread(r"C:\Users\Lenovo\Desktop\PyImageSearch\BlobDetector\2b48ced88b8165515cfa935ee7143574.jpg")
#image = cv2.imread(r"C:\Users\Lenovo\Desktop\PyImageSearch\Evolution-of-coins-in-india.jpg")
copy = image.copy()
image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 10
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1
 
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(image)
image_with_keypoints = cv2.drawKeypoints(copy , keypoints ,np.array([]), (0,0,255) , cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("image" , image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
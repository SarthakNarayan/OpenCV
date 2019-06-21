import cv2
import numpy as np

image_to_be_pasted = cv2.imread(r"C:\Users\Lenovo\Desktop\PyImageSearch\Homography\first-image.jpg")
destination_image = cv2.imread(r"C:\Users\Lenovo\Desktop\PyImageSearch\Homography\times-square-768x512.jpg")

points = []
center = (0,0)
counter = 0
# Top left , Top Right ,  Bottom Right ,Bottom Left 
# Dont mess the order
def getPoints(event,x,y,flags,param):
    global center , counter
    if (counter<4):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            center = (x,y)
            print(center)
            counter+=1
            points.append(center)

cv2.imshow("destination image" , destination_image)
# important mouse call back should be after imshow otherwise it wont work
cv2.setMouseCallback("destination image",getPoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(points)

width , height = image_to_be_pasted.shape[0] , image_to_be_pasted.shape[1]

# Top left , Top Right ,  Bottom Right ,Bottom Left 
source_points = np.array([[0,0] ,[width-1 , 0] , [width-1 , height-1] , [0 , height-1]] ,dtype = float)

points = np.array(points)
h, status = cv2.findHomography(source_points, points)
new_image = cv2.warpPerspective(image_to_be_pasted, h, (height,width))
cv2.fillConvexPoly(destination_image, points.astype(int), 0, 16)

print(new_image.shape)

new_image = cv2.resize(new_image ,(destination_image.shape[1] , destination_image.shape[0]) ,cv2.INTER_LINEAR)
destination_image = new_image + destination_image

cv2.imshow("source image" , destination_image)
cv2.imshow("new image" , new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


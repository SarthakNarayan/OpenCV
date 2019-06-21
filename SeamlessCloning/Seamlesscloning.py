import cv2
import numpy as np
from imutils import build_montages

destination_image = cv2.imread(r"C:\Users\Lenovo\Desktop\PyImageSearch\SeamlessCloning\maxresdefault.jpg")
center = (0,0)
def getCenter(event,x,y,flags,param):
    global center
    if event == cv2.EVENT_LBUTTONDBLCLK:
        center = (x,y)

cv2.imshow("destination_image" , destination_image)
cv2.setMouseCallback("destination_image",getCenter)
print("The center is" , center)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_to_be_pasted = cv2.imread(r"C:\Users\Lenovo\Desktop\PyImageSearch\SeamlessCloning\download (1).jfif")
box = cv2.selectROI("Selector", image_to_be_pasted, fromCenter=False, showCrosshair=True)
(x,y,w,h) = box
roi = image_to_be_pasted[y:y+h , x:x+w]
mask = 255*np.ones(roi.shape, roi.dtype)

output1 = cv2.seamlessClone(roi, destination_image, mask, center, cv2.MIXED_CLONE)
output2 = cv2.seamlessClone(roi, destination_image, mask, center, cv2.NORMAL_CLONE)
# Mixed works much better than Normal

images = [output1 , output2]
montage = build_montages(images, (500, 500), (2, 1))[0]

cv2.imshow("1st mixed 2nd normal" , montage)
cv2.waitKey(0)
cv2.destroyAllWindows()

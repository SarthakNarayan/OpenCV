import numpy as np
import cv2


img = cv2.imread(r"C:\Users\Lenovo\Desktop\PyImageSearch\Test.png")
box = cv2.selectROI("Selector", img, fromCenter=False,
                            showCrosshair=True)
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = box
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
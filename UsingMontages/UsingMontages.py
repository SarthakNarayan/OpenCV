from imutils import build_montages
import cv2
import numpy as np

image1 = cv2.imread(r"example_01.jpg")
image1 = cv2.resize(image1, (200, 200), interpolation = cv2.INTER_LINEAR)
image2 = cv2.imread(r"example_02.jpg")
image2 = cv2.resize(image2, (200, 200), interpolation = cv2.INTER_LINEAR)
image3 = cv2.imread(r"example_03.jpg")
image3 = cv2.resize(image3, (200, 200), interpolation = cv2.INTER_LINEAR)
image4 = cv2.imread(r"example_04.jpg")
image4 = cv2.resize(image4, (200, 200), interpolation = cv2.INTER_LINEAR)

images = [image1 , image2 , image3 , image4]
images = np.array(images)
# First argument is the array containing the images
# 2nd is the size of the images. Make sure they are of the same size
# 3rd is the number of rows and columns required
montage = build_montages(images, (200, 200), (2, 2))[0]

cv2.imshow("Montage" , montage)
cv2.waitKey(0)
cv2.destroyAllWindows()
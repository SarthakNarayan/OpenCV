# Will run only in spyder
import cv2
from skimage.measure import compare_ssim
import numpy as np

kernel = np.ones((5,5) , np.uint8)

image1 = cv2.imread(r"everydayok-6-spot-the-difference-between-two-little-piglets.jpg",0)
image2 = cv2.imread(r"Part1.jpg",0)
image1 = cv2.GaussianBlur(image1 , (15,15) , 5)
image2 = cv2.GaussianBlur(image2 , (15,15) , 5)
image1 = cv2.resize(image1, (594, 817), interpolation = cv2.INTER_LINEAR)
(score, diff) = compare_ssim(image1, image2, full=True)

#The score  represents the structural similarity index between the two input images.
# value can fall into the range [-1, 1] with a value of one being a “perfect match”.
#The diff  image contains the actual image differences between the two input images that we wish to visualize.
#The difference image is currently represented as a floating point data type in the range [0, 1].

diff = (diff * 255).astype("uint8")
ret , thresh = cv2.threshold(diff,150,255,cv2.THRESH_BINARY_INV)
opening = cv2.morphologyEx(thresh ,cv2.MORPH_OPEN, kernel )
closing = cv2.morphologyEx(thresh ,cv2.MORPH_CLOSE, kernel )
cv2.imshow("image difference" , diff)
cv2.imshow("image " , closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

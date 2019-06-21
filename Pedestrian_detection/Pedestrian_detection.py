import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
path = r"C:\Users\Lenovo\Desktop\PyImageSearch\Pedestrian_detection\pedestrian-detection\images" 
ap.add_argument("-i", "--image", help="path to images directory",
                default = "\person_029.bmp")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image_path = path + args["image"]
image = cv2.imread(image_path) 
orig = image.copy()

# You can resize the image for faster prediction
# I am not doing it in this example

(rects, weights) = hog.detectMultiScale(image, winStride=(3, 2),
		padding=(8, 8), scale=1.01)

# fOR MORE DETAILS ABOUT hog.detectMultiScale
# REFER https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/


# The scale is for image pyramids 
# By changing the scaling we can increase the speed of the descriptor
# By increasing the scaling value we can do this but if we make it too large
# accuracy might reduce
# A sliding window of step size 4*4 is used


print("No of people found",rects.shape[0])
print("The weight is/are",weights)

for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

########################################################
        # APPLY NON MAXIMUM SUPRESSION
########################################################
 
cv2.imshow("Original Image", orig)       
cv2.waitKey(0)
cv2.destroyAllWindows()


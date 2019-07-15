import cv2
import numpy as np
image = cv2.imread(r"C:\Users\Lenovo\Desktop\GITHUB\OpenCV\K Means Segmentation\3 colors1.jfif")
image_copy = image.copy()

print("Original shape" , image.shape)
# for feeding image to K means clustering we need to reshape it
image_copy = image_copy.reshape((-1,3))
# Convert to float
image_copy = np.float32(image_copy)
print("New shape " , image_copy.shape)

# defining the criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 10 , 1.0)
no_clusters = 5

retval , labels , centers = cv2.kmeans(image_copy , no_clusters , None , criteria ,
                                         10 , cv2.KMEANS_RANDOM_CENTERS)

print("Shape of centers" , centers.shape)
print("Shape of labels" , labels.shape)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
print("Segmented data shape" , segmented_data.shape)
segmented_image = segmented_data.reshape(image.shape)
labels_rearranged = labels.reshape((image.shape[0] , image.shape[1]))
print("Labels Rearranged shape" , labels_rearranged.shape)

masked_image = image.copy()
# for cluster value == can be changed to 0,1,2
masked_image[labels_rearranged==2] = [0,0,0]

cv2.imshow("image" , image)
# Segmented image will show us the color available
cv2.imshow("segmented_image" , segmented_image)
# Masked image will remove one of those colors
cv2.imshow("masked image" , masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
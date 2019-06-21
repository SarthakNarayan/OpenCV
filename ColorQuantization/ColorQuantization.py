from sklearn.cluster import KMeans
import numpy as np
import cv2

image = cv2.imread(r"C:\Users\Lenovo\Desktop\PyImageSearch\Image_Classifier_SSD\object-detection-deep-learning\images\example_02.jpg")
(h, w) = image.shape[:2]
 
# convert the image from the RGB color space to the L*a*b*
# color space -- since we will be clustering using k-means
# which is based on the euclidean distance, we'll use the
# L*a*b* color space where the euclidean distance implies
# perceptual meaning
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
 
# reshape the image into a feature vector so that k-means
# can be applied because k means accepts a 2 dimensional array
image = image.reshape((image.shape[0] * image.shape[1], 3))
 
# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
clt = KMeans(n_clusters = 10)
labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]
 
# reshape the feature vectors to images
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))
 
# convert from L*a*b* to RGB
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
 
# display the images and wait for a keypress
cv2.imshow("image", np.hstack([image, quant]))
cv2.waitKey(0)
cv2.destroyAllWindows()
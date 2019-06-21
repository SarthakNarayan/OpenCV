import numpy as np
import cv2

image = cv2.imread("example_01.jpg")

# initialize OpenCV's objectness saliency detector and set the path
# to the input model files
saliency = cv2.saliency.ObjectnessBING_create()
saliency.setTrainingPath(r"C:\Users\Lenovo\Desktop\PyImageSearch\SaliencyDetection\saliency-detection\objectness_trained_model")

# compute the bounding box predictions used to indicate saliency
(success, saliencyMap) = saliency.computeSaliency(image)
numDetections = saliencyMap.shape[0]

# Here 10 is the max number of detections we want
# loop over the detections
for i in range(0, min(numDetections, 10)):
	# extract the bounding box coordinates
	(startX, startY, endX, endY) = saliencyMap[i].flatten()
	
	# randomly generate a color for the object and draw it on the image
	output = image.copy()
	color = np.random.randint(0, 255, size=(3,))
	color = [int(c) for c in color]
	cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)

	# show the output image
	cv2.imshow("Image", output)
	cv2.waitKey(0)

cv2.destroyAllWindows()

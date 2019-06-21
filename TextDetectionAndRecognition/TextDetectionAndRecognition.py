''' East Text detector for text detection and tesseract v4 for recognition
		The text detector gives us the bounding box which we then give to the
		tesseract ocr for recognition. '''
'''
OCR Engine modes: (--oem)
	0    Legacy engine only.
	1    Neural nets LSTM engine only.
	2    Legacy + LSTM engines.
	3    Default, based on what is available.

Page segmentation modes: (--psm)
	0    Orientation and script detection (OSD) only.
	1    Automatic page segmentation with OSD.
	2    Automatic page segmentation, but no OSD, or OCR.
	3    Fully automatic page segmentation, but no OSD. (Default)
	4    Assume a single column of text of variable sizes.
	5    Assume a single uniform block of vertically aligned text.
	6    Assume a single uniform block of text.
	7    Treat the image as a single text line.
	8    Treat the image as a single word.
	9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
			 bypassing hacks that are Tesseract-specific.
'''

from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
cap = cv2.VideoCapture(1)

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(r"C:\Users\Lenovo\Desktop\PyImageSearch\EASTTextDetector\opencv-text-detection\frozen_east_text_detection.pb")

W = int(cap.get(3))
H = int(cap.get(4))

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (320, 320)
rW = W / float(newW)
rH = H / float(newH)

(H, W) = (320, 320)

def gamma_transform(image , gamma = 2):
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

		return cv2.LUT(image, table)
roi = 0
while True:
		_ , frame = cap.read()
		#frame = cv2.flip(frame , 1)
		frame = gamma_transform(frame)
		orig = frame.copy()
		frame = cv2.resize(frame, (320, 320), interpolation = cv2.INTER_LINEAR)
		blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)
		

		# grab the number of rows and columns from the scores volume, then
		# initialize our set of bounding box rectangles and corresponding
		# confidence scores
		(numRows, numCols) = scores.shape[2:4]
		rects = []
		confidences = []


		# loop over the number of rows
		for y in range(0, numRows):
			# extract the scores (probabilities), followed by the geometrical
			# data used to derive potential bounding box coordinates that
			# surround text
			scoresData = scores[0, 0, y]
			xData0 = geometry[0, 0, y]
			xData1 = geometry[0, 1, y]
			xData2 = geometry[0, 2, y]
			xData3 = geometry[0, 3, y]
			anglesData = geometry[0, 4, y]
		
			# loop over the number of columns
			for x in range(0, numCols):
				# if our score does not have sufficient probability, ignore it
				if scoresData[x] < 0.5:
					continue
		
				# compute the offset factor as our resulting feature maps will
				# be 4x smaller than the input image
				(offsetX, offsetY) = (x * 4.0, y * 4.0)
		
				# extract the rotation angle for the prediction and then
				# compute the sin and cosine
				angle = anglesData[x]
				cos = np.cos(angle)
				sin = np.sin(angle)
		
				# use the geometry volume to derive the width and height of
				# the bounding box
				h = xData0[x] + xData2[x]
				w = xData1[x] + xData3[x]
		
				# compute both the starting and ending (x, y)-coordinates for
				# the text prediction bounding box
				endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
				endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
				startX = int(endX - w)
				startY = int(endY - h)
		
				# add the bounding box coordinates and probability score to
				# our respective lists
				rects.append((startX, startY, endX, endY))
				confidences.append(scoresData[x])
		
		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		boxes = non_max_suppression(np.array(rects), probs=confidences)
		
		# loop over the bounding boxes
		for (startX, startY, endX, endY) in boxes:
			# scale the bounding box coordinates based on the respective
			# ratios
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)
		
			# draw the bounding box on the image
			cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
			roi = orig[startY:endY , startX:endX ]        
			config = (" -l eng --oem 1 --psm 7")
			text = pytesseract.image_to_string(roi , config=config)
			print(text)
		# show the output image
		cv2.imshow("Text Detection", orig)
		cv2.imshow("Roi" , roi)
		if cv2.waitKey(25) & 0xFF == ord('q'):
						break

cap.release()
cv2.destroyAllWindows()
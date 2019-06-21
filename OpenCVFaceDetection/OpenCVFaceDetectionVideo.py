import cv2
import numpy as np

protxtpath = r"C:\Users\Lenovo\Desktop\PyImageSearch\OpenCVFaceDetection\deep-learning-face-detection\deploy.prototxt.txt"
modelpath = r"C:\Users\Lenovo\Desktop\PyImageSearch\OpenCVFaceDetection\deep-learning-face-detection\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protxtpath, modelpath)

cap = cv2.VideoCapture(1)
while True:
    _ , frame = cap.read()
    frame = cv2.flip(frame , 1)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        #extract the confidence (i.e., probability) associated with the
        #prediction
        confidence = detections[0, 0, i, 2]
 
    	# filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.6:
		    # compute the (x, y)-coordinates of the bounding box for the
		    # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
 
		    # draw the bounding box of the face along with the associated
		    # probability
            text =  "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.imshow("Frame" , frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

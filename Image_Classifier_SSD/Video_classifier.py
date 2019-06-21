import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input image" ,
                default = r"C:\Users\Lenovo\Desktop\PyImageSearch\Image_Classifier\object-detection-deep-learning\images\example_01.jpg")  
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# to generate bounding boxes of different colours
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# If u see the shape you will see an array of 21*3
print("Shape of the colors",COLORS.shape)

# Loading the model
prototxtpath = "MobileNetSSD_deploy.prototxt.txt"
modelpath = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxtpath,modelpath)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
cap = cv2.VideoCapture(1)

while True:
    
    ret, image = cap.read()
    # we are getting the height and width of the image
    h, w = image.shape[0] , image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
    	(300, 300), 127.5)
    
    print("The shape of blob" , blob.shape)
    
    net.setInput(blob)
    detections = net.forward()
    print("The shape of detections are",detections.shape)

    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
    
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            index_of_class = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            print(box.shape)
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[index_of_class], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          COLORS[index_of_class], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[index_of_class], 2)
    
    # show the output image
    cv2.imshow("Output", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

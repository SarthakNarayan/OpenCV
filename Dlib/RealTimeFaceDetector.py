# import the necessary packages
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np

def gamma_transform(image , gamma = 2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

cap = cv2.VideoCapture(1)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Lenovo\Desktop\PyImageSearch\Dlib\shape_predictor_68_face_landmarks.dat")

while True:
    # load the input image, resize it, and convert it to grayscale
    _ , image = cap.read()
    image = cv2.flip(image , 1)
    image = imutils.resize(image, width=500)
    image = gamma_transform(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
    	# determine the facial landmarks for the face region, then
    	# convert the facial landmark (x, y)-coordinates to a NumPy
    	# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
    	# convert dlib's rectangle to a OpenCV-style bounding box
    	# [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    	# show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
    		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    	# loop over the (x, y)-coordinates for the facial landmarks
    	# and draw them on the image
        # To get any particular part change shape like for mouth
        # for (x, y) in shape[48:69]: 
        '''
        The mouth can be accessed through points [48, 68].
        The right eyebrow through points [17, 22].
        The left eyebrow through points [22, 27].
        The right eye using [36, 42].
        The left eye with [42, 48].
        The nose using [27, 35].
        And the jaw via [0, 17].
        
        remember it is 1 indexed and python is 0

        '''
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
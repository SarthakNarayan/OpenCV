import imutils
import cv2
import numpy as np

print("[INFO] loading style transfer model...")
modelPath = r"C:\Users\Lenovo\Desktop\PyImageSearch\NeuralStyleTransfer\neural-style-transfer\models\instance_norm"
modelName = r"starry_night.t7"
modelPath = modelPath + "\\" + modelName
net = cv2.dnn.readNetFromTorch(modelPath)

def gamma_transform(image , gamma = 2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

cap = cv2.VideoCapture(1)
while True:
    _ , image = cap.read()
    image = cv2.flip(image , 1)
    image = gamma_transform(image)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    
    # construct a blob from the image, set the input, and then perform a
    # forward pass of the network
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()

    # reshape the output tensor, add back in the mean subtraction, and
    # then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)

    # show the images
    cv2.imshow("Input ", image)
    cv2.imshow("Output", output)
    if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
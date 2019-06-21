import numpy as np
import cv2

# load the class labels from disk
rows = open(r"C:\Users\Lenovo\Desktop\PyImageSearch\Understanding_Blob_function\blob-from-images\synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(r"C:\Users\Lenovo\Desktop\PyImageSearch\Understanding_Blob_function\blob-from-images\bvlc_googlenet.prototxt",
	r"C:\Users\Lenovo\Desktop\PyImageSearch\Understanding_Blob_function\blob-from-images\bvlc_googlenet.caffemodel")

imagePath = r"C:\Users\Lenovo\Desktop\PyImageSearch\Image_Classifier\object-detection-deep-learning\images\example_01.jpg"

# (1) load the first image from disk, (2) pre-process it by resizing
# it to 224x224 pixels, and (3) construct a blob that can be passed
# through the pre-trained network
image = cv2.imread(imagePath)
resized = cv2.resize(image, (224, 224))
blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (104, 117, 123))
print("First Blob: {}".format(blob.shape))

# set the input to the pre-trained deep learning network and obtain
# the output predicted probabilities for each of the 1,000 ImageNet
# classes
net.setInput(blob)
preds = net.forward()

# out of the 1000 classes we need 1
# shape of preds[0] is 1000
print(preds[0].shape)

# avgDists = np.array([1, 8, 6, 9, 4])
# idx = avgDists.argsort()[::-1][:n]
# For getting n numbers sorted in ascending order in our example we take the first number only
# since we want the hisghest in descending order
# if u want to sort in ascending order upto n numbers
# idx = avgDists.argsort()[:n]

# sort the probabilities (in descending) order, grab the index of the
# top predicted label, and draw it on the input image
idx = np.argsort(preds[0])[::-1][0]
text = "Label: {}, {:.2f}%".format(classes[idx],
	preds[0][idx] * 100)
cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 0, 255), 2)

top5preds = np.argsort(preds[0])[::-1][:5]
for i in range(0,len(top5preds)):
    index = top5preds[i]
    print("Class {} , Probability {}".format(classes[index],preds[0][index] * 100))
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


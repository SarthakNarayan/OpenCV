# run using projects
# Works really well when there is a single image in a big background
import cv2

image = cv2.imread(r"example_03.jpg")
saliency0 = cv2.saliency.StaticSaliencySpectralResidual_create()
(success0, saliencyMap0) = saliency0.computeSaliency(image)
saliencyMap0 = (saliencyMap0 * 255).astype("uint8")

saliency1 = cv2.saliency.StaticSaliencyFineGrained_create()
(success1, saliencyMap1) = saliency1.computeSaliency(image)
saliencyMap1 = (saliencyMap1 * 255).astype("uint8")

# if we would like a *binary* map that we could process for contours,
# compute convex hull's, extract bounding boxes, etc., we can
# additionally threshold the saliency map
threshMap = cv2.threshold(saliencyMap1.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Static Saliency Spectral Residual", saliencyMap0)
cv2.imshow("Static Saliency Fine Grained", saliencyMap1)
cv2.imshow("Thresh", threshMap)
cv2.imshow("input image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

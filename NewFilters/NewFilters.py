import cv2
from imutils import build_montages

image = cv2.imread(r"C:\Users\Lenovo\Desktop\PyImageSearch\Image_Classifier_SSD\object-detection-deep-learning\images\example_05.jpg")
images = []
for i in range(5):
    images.append(image)
'''
Edge preserving filters. It takes value RECURS_FILTER ( Recursive Filtering ) = 1 and 
NORMCONV_FILTER ( Normalized Convolution ) = 2. Using RECURS_FILTER option is about 
3.5x faster than NORMCONV_FILTER. The NORMCONV_FILTER produces sharpening of the edges 
and it good for applications like stylizations. When sharpening is not desirable, 
and speed is important RECURS_FILTER should be used.
'''
'''
In edge preserving filters there are two competing objectives — a) smooth the image b) don’t smooth the edges / color boundaries.
Smoothing filters sigma_s controls the size of the neighborhood, and sigma_r (for sigma_range) controls 
the how dissimilar colors within the neighborhood will be averaged. A larger sigma_r results in large regions of constant color.
'''
images[1] = cv2.edgePreservingFilter(images[1], flags=1, sigma_s=60, sigma_r=0.4)
images[2] = cv2.detailEnhance(images[2], sigma_s=10, sigma_r=0.15)
dst_gray, images[3] = cv2.pencilSketch(images[3], sigma_s=60, sigma_r=0.07, shade_factor=0.05)
images[4] = cv2.stylization(images[4], sigma_s=60, sigma_r=0.07)

montages = []
for i in range(5):
    montages.append(images[i])
montage = build_montages(montages, (400, 400), (3, 2))[0]

cv2.imshow("montage" ,montage)
cv2.waitKey(0)
cv2.destroyAllWindows()
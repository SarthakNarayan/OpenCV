# There is a flaw in this method i.e. when sum of two becomes same
import numpy as np
import cv2

def LeftTop(data):
    lowest = data[0][0][0] + data[0][0][1]
    index = 0
    for i in range(1,len(data)):
        lsum = data[i][0][0] + data[i][0][1]
        if lsum<lowest:
            lowest = lsum
            index = i
    return index

def RightBottom(data):
    maximum = data[0][0][0] + data[0][0][1]
    index = 0
    for i in range(1,len(data)):
        msum = data[i][0][0] + data[i][0][1]
        if msum>maximum:
            maximum = msum
            index = i
    return index

def LeftBottom(data):
    lowest = data[0][0][0] - data[0][0][1]
    index = 0
    for i in range(1,len(data)):
        lsum = data[i][0][0] - data[i][0][1]
        if lsum<lowest:
            lowest = lsum
            index = i
    return index

def RightTop(data):
    maximum = data[0][0][0] - data[0][0][1]
    index = 0
    for i in range(1,len(data)):
        msum = data[i][0][0] - data[i][0][1]
        if msum>maximum:
            maximum = msum
            index = i
    return index

def four_point_transform(image, tl , br ,tr ,bl):
    rect = np.zeros((4, 2), dtype = "float32")
    rect[0] = tl
    rect[3] =br
    rect[1] =tr
    rect[2] =bl
    widthA= np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    print(rect)
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[0, maxHeight - 1],
		[maxWidth - 1, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
    return warped

#image = cv2.imread(r"C:\Users\Lenovo\Desktop\PyImageSearch\PerspectiveTransform\IMG_20190525_212521.jpg")
image = cv2.imread("omr_test_01.png")
image = cv2.resize(image, (900, 600), interpolation = cv2.INTER_LINEAR)
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray,15,50,500)
blur = cv2.Canny(blur,50,200)
# for spyder
#_ , contours , _ = cv2.findContours(blur , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
# for idle
contours , _ = cv2.findContours(blur , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key=cv2.contourArea)
#cv2.drawContours(image , cnts[-1], -1 , (255,0,0) , 4)
c = cnts[-1]
peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.02* peri, True)
cv2.drawContours(image , [approx], -1 , (255,0,0) , 4)

LT = LeftTop(approx)
LTP = approx[LT][0]
RB = RightBottom(approx)
RBP = approx[RB][0]
RT = RightTop(approx)
RTP = approx[RT][0]
LB = LeftBottom(approx)
LBP = approx[LB][0]

cv2.circle(image,([LBP][0][0],[LBP][0][1]),4,(255,255,0),-1)
warped = four_point_transform(image , LTP , RBP , RTP , LBP)

cv2.imshow("Original Image" , image)
cv2.imshow("Edge Image" , blur)
cv2.imshow("warped" , warped)
cv2.waitKey(0) 
cv2.destroyAllWindows()

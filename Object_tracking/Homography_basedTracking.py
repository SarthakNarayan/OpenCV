import cv2
import numpy as np

# Performs very well not in my case
# Will only run with projects environment of Anaconda
#cap = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\IITB Internship\VIDEOS\1080P\ONLY CIRCULAR LOW DENSITY.mp4")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0) & 0xFF
    if key == 2555904:
        ret, frame = cap.read()
    if key == ord("s"):
            break

initBoundingBox = cv2.selectROI("Frame", frame, fromCenter=False,
                                       showCrosshair=True)

(x,y,w,h) = initBoundingBox
img = frame[y:y+h , x:x+w]
img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

## Features
sift = cv2.xfeatures2d.SIFT_create()

# For computing the key points and the descriptors 
# We pass none since we dont have a mask
kp_image, desc_image = sift.detectAndCompute(img, None)
## Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

img = cv2.drawKeypoints(img , kp_image , img)
while True:
    _, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
    good_points = []
    
    # m corresponds to matches of roi and n corresponds to matches of video
    for m, n in matches:
        if m.distance < 0.8*n.distance:
            good_points.append(m)
    #img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
#    # Homography
    if len(good_points) > 0:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        # Perspective transform
        (h, w) = img.shape[:2]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        try:
            dst = cv2.perspectiveTransform(pts, matrix)
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            cv2.imshow("Homography", homography)
        except Exception as e:
            print(str(e))
    else:
        cv2.imshow("Homography", grayframe)
#    #cv2.imshow("Image", img)
    #cv2.imshow("grayFrame", grayframe)
    #cv2.imshow("img3", img3)
    #cv2.imshow("image" , img)
    key = cv2.waitKey(1) & 0xFF
    if key == 2555904:
        ret, frame = cap.read()
    if key == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()

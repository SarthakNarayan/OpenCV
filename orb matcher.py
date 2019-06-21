import cv2

image1 = cv2.imread(r'E:\onsem project\image12\testimage1.jpg',0)
image2 = cv2.imread(r'E:\onsem project\images\train\person1\image27.png',0)

orb = cv2.ORB_create()

kp1 , des1 = orb.detectAndCompute(image1 , None)
kp2 , des2 = orb.detectAndCompute(image2 , None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(image1,kp1,image2,kp2,matches[:5],None,flags=2)

cv2.imshow('matching' , img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
def nothing(x) :
    if x==1:
        print("READY FOR PHOTOSHOOT PRESS ESCAPE")
    else :
        print("PLEASE TURN TRACKBAR TO 1 TO CLICK YOUR PHOTO")
cap=cv2.VideoCapture(0)
cv2.namedWindow('capturephoto')
cv2.createTrackbar('click','capturephoto',0,1,nothing)
while(1):
    click=cv2.getTrackbarPos('click','capturephoto')
    ret,frame = cap.read()
    cv2.imshow('capturephoto',frame)
    if cv2.waitKey(5) & 0xFF == 27 :
         break
if click==1 :
    img=frame
    cap.release()
    cv2.destroyWindow('capturephoto')
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyWindow('image')
else :
    cap.release()
    cv2.destroyWindow('capturephoto')

    
        

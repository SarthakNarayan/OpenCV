import cv2
import numpy as np
cap = cv2.VideoCapture(1)

# for documents apply adaptive thresholding in the end
count = 0
width = int(cap.get(3))
height = int(cap.get(4))
enter = True
# Order of clicking TL TR BL BR very immportant
def get_inputs(event , x , y , flags , param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if enter == True:
            global count
            value = (x,y)
            values.append(value)
            print(x,y)
            count = count + 1

values = []
cv2.namedWindow("Original Frame")
cv2.setMouseCallback("Original Frame" , get_inputs)


while True:
    _ , frame = cap.read()
    if count == 4:
        enter = False
        pts1 = np.float32([[values[0]] ,[values[1]] ,[values[2]] ,[values[3]] ])
        # New points
        pts2 = np.float32([[0,0] , [width,0], [0,height], [width,height]])
        print("performing perspective transform")
        # pts1 are the initial points and pts 2 are the final point where we want the warped object
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        result = cv2.warpPerspective(frame , matrix , (width,height))
        cv2.imshow("Perspective Transform" , result)
        
    cv2.imshow("Original Frame",frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
    
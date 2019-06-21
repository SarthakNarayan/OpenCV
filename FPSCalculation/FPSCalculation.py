import cv2
import time

cap = cv2.VideoCapture(1)

while True:
    start_time = time.time()
    ret , frame = cap.read()
    # Body
    frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    for i in range(100):
        pass
    #time.sleep(0.5)
    # body end

    cv2.imshow("image", frame)
    end_time = time.time()
    try:
        fps = 1/(end_time - start_time)
        print(fps)
    except Exception as e :
        print(e)
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

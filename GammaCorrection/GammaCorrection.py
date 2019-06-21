import cv2
import numpy as np
cap = cv2.VideoCapture(1)

def gamma_transform(image , gamma = 2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)
while True:
    _ , frame = cap.read()
    frame = cv2.flip(frame , 1)
    copy = frame.copy()
    copy = gamma_transform(copy)
    cv2.imshow("origibal frame",frame)
    cv2.imshow("with gamma correction",copy)
    if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
    

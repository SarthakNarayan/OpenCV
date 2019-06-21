import cv2

face_cascade = cv2.CascadeClassifier(r'E:\onsem project\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while(True):
    ret , img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.imwrite(r'E:\onsem project\image12\random.jpg',roi_color)
    
    cv2.imshow('haar cascade' , img)   
        
    if cv2.waitKey(30) == 27 & 0xFF:
        break

cap.release()
cv2.destroyAllWindows()    


import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread(r"C:\Users\Lenovo\Desktop\PyImageSearch\UsingTesseract\TextData.png")
config = ('-l eng --oem 1 --psm 3')
text = pytesseract.image_to_string(image, config=config)
cv2.imshow("image" , image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(text)
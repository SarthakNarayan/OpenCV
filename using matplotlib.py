import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('C:\Users\Lenovo\Desktop\download.JPG',1)
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(RGB_img)
plt.show()

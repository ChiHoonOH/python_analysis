import numpy as np
print(np.array([[1,2,3],[4,5,6]]))

import cv2

imageFile = './data/pizza2.jpg'
imageFile2 = './data/lena.jpg'
img1  = cv2.imread(imageFile)    # cv2.IMREAD_COLOR
img2 = cv2.imread(imageFile2)   # cv2.IMREAD_GRAYSCALE
print(img1.shape)
print(img2.shape)
img1 = img1.reshape(262144,)
img2 = img2.reshape(262144,)

print(np.ndarray((2,262144), buffer=np.array([img1,img2])).shape)
# cv2.imshow('Lena color',img)
cv2.imshow('Lena grayscale',img2)
print(img1.shape)
print(img2.shape)
cv2.waitKey()
cv2.destroyAllWindows()

# with open('./data/lena.jpg') as fp:
#     fp.read()
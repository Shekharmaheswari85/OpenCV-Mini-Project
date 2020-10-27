import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/scan.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 120, 255, 1)

corners = cv2.goodFeaturesToTrack(canny, 4, 0.5, 50)

cv2.imshow("canny", canny)
# cv2.imshow('image', image)
# cv2.waitKey()
cv2.imshow("Original", image)
cv2.waitKey(0)

# Cordinates of the 4 corners of the original image
points_A = np.float32([[320, 15], [700, 215], [85, 610], [530, 780]])

# Cordinates of the 4 corners of the desired output
# We use a ratio of an A4 Paper 1 : 1.41
points_B = np.float32([[0, 0], [420, 0], [0, 594], [420, 594]])

# Use the two sets of four points to compute
# the Perspective Transformation matrix, M
M = cv2.getPerspectiveTransform(points_A, points_B)

warped = cv2.warpPerspective(image, M, (420, 594))

cv2.imshow("warpPerspective", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
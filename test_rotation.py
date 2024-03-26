import cv2
import numpy as np


def find_angle(image_path):
    # Зчитуємо зображення
    image = cv2.imread(image_path)
    image = cv2.copyMakeBorder(
        image,
        top=15,
        bottom=15,
        left=15,
        right=15,
        value=[179, 171, 182],
        borderType=cv2.BORDER_CONSTANT
    )
    red_channel = image[:, :, 2]  # Red channel index is 2
    mask = red_channel > 230
    image[mask] = [255, 255, 255]

    # Перетворюємо зображення у відтінки сірого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Знаходимо контури у зображенні
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Вибираємо найдовший контур (найбільшу площу)
    largest_contour = max(contours, key=cv2.contourArea)

    # Апроксимуємо контур з 4 кутами
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    # Визначаємо кут повороту
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        angle = cv2.minAreaRect(approx)[-1]

        if angle < 30:
            angle = -angle
        else:
            angle = 90 - angle

        return angle

    print("ERROR. ANGLE NOT FOUND")
    return 0


image_path = "/home/nikita-goncharov/Desktop/Failed_checks/in_20240322074447767612_003.jpg"

# Знаходження кута повороту
angle = find_angle(image_path)
print("Кут повороту зображення:", angle)

# import cv2 as cv
# from math import atan2, cos, sin, sqrt, pi
# import numpy as np
#
# def getOrientation(pts, img):
#     ## [pca]
#     # Construct a buffer used by the pca analysis
#     sz = len(pts)
#     data_pts = np.empty((sz, 2), dtype=np.float64)
#     for i in range(data_pts.shape[0]):
#         data_pts[i, 0] = pts[i, 0, 0]
#         data_pts[i, 1] = pts[i, 0, 1]
#
#     # Perform PCA analysis
#     mean = np.empty((0))
#     mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
#
#     # Store the center of the object
#     cntr = (int(mean[0, 0]), int(mean[0, 1]))
#     ## [pca]
#
#     ## [visualization]
#     # Draw the principal components
#     cv.circle(img, cntr, 3, (255, 0, 255), 2)
#     p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
#     p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
#
#     angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
#     ## [visualization]
#     return angle
#
#
# # Load the image
# img = cv.imread("/home/nikita-goncharov/Desktop/Failed_checks/in_20240322074447767612_003.jpg")
#
# img = cv.copyMakeBorder(
#     img,
#     top=15,
#     bottom=15,
#     left=15,
#     right=15,
#     value=[179, 171, 182],
#     borderType=cv.BORDER_CONSTANT
# )
# red_channel = img[:, :, 2]  # Red channel index is 2
# mask = red_channel > 230
# img[mask] = [255, 255, 255]
#
# # Convert image to grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # Convert image to binary
# _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# # Find all the contours in the thresholded image
# contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#
# contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[3], reverse=True)
#
# for i, c in enumerate(contours):
#     # Calculate the area of each contour
#     area = cv.contourArea(c)
#     # Ignore contours that are too small or too large
#     # Draw each contour only for visualisation purposes
#     cv.drawContours(img, contours, i, (0, 0, 255), 2)
#
#     # Find the orientation of each shape
#     print(getOrientation(c, img))
#     break
#
# cv.imshow('Output Image', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

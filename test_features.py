import numpy as np
# import argparse
import cv2
from parser_cabala import ParserChecks

#
# data, error,  log = ParserChecks("chance/in_20240116164153423024_01.jpg").get_result()
# print(data, error)

# "chance/in_20240125202603311331_003.jpg"


img = cv2.imread("Rotated.jpg")

contrast = .99  # (0-127)
brightness = 40  # (0-100)
contrasted_img = cv2.addWeighted(
    img,
    contrast,
    img, 0,
    brightness
)
# blured_img_main = cv2.bilateralFilter(contrasted_img, 1, 75, 75)
contrasted_img = cv2.cvtColor(contrasted_img, cv2.COLOR_BGR2GRAY)
image = cv2.threshold(contrasted_img, 235, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite("contrasted.jpg", image)
# cv2.waitKey(0)


# src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# canny_output = cv2.Canny(src_gray, 100, 200)
# contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# contours = [cv2.boundingRect(cnt) for cnt in contours]
# contours_areas = [cnt[2]*cnt[3] for cnt in contours]
# max_area_index = np.argmax(contours_areas)
#
# print("CHECK AREA: ", contours_areas[max_area_index])
# print(contours[max_area_index])



# "chance/in_20240116164153423024_001.jpg"
# "chance/in_20240125184438502913_046.jpg")


def rotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    print(angle)
    if angle < 30:
        angle = -angle
    else:
        angle = 90-angle

    print(angle)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # cv2.imwrite("Rotated.jpg", rotated)

    src_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    canny_output = cv2.Canny(src_gray, 100, 200)
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 1000 and w > 500:
            print(x, y, h, w)
            print("AREA: ", cv2.contourArea(cnt))
            cropped = rotated[
                y:y+h,
                x:x+w
            ]
            # cv2.imwrite("Cropped.jpg", cropped)
    return rotated

# rotate(img)



# # Display the result
# cv2.imshow('Result', img)
# cv2.waitKey(0)



# img2 = cv2.imread("../log/parser/debug_img/in_20240207144432952070_006/img_wb_blured_03.jpg", cv2.IMREAD_GRAYSCALE)
# line = cv2.imread("../line.png", cv2.IMREAD_GRAYSCALE)
#
# res = cv2.matchTemplate(img2, line, cv2.TM_CCOEFF_NORMED)
# y, x = np.unravel_index(res.argmax(), res.shape)
#
# print(res)
# print(x, y)

















# **********************************************************
# import cv2
# import numpy as np
#
# # Load the image
# image = cv2.imread('your_image.jpg')
#
# # Extract the red channel
# red_channel = image[:, :, 2]  # Red channel index is 2
#
# # Create a mask where red channel values are greater than 200
# mask = red_channel > 200
#
# # Replace pixels where mask is True with white
# image[mask] = [255, 255, 255]
#
# # Display the result
# cv2.imshow('Result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# **********************************************************


















#
# ret, thresh = cv2.threshold(rotated, 150, 255, cv2.THRESH_BINARY)
# # visualize the binary image
# cv2.imshow('Binary image', thresh)
# cv2.waitKey(0)

# lower = [1, 0, 20]
# upper = [60, 40, 220]
#
# # create NumPy arrays from the boundaries
# lower = np.array(lower, dtype="uint8")
# upper = np.array(upper, dtype="uint8")
#
# # find the colors within the specified boundaries and apply
# # the mask
# mask = cv2.inRange(rotated, lower, upper)
# output = cv2.bitwise_and(rotated, rotated, mask=mask)
#
# ret,thresh = cv2.threshold(mask, 40, 255, 0)
# # if (cv2.__version__[0] > 3):
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# # else:
#     # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# if len(contours) != 0:
#     # draw in blue the contours that were founded
#     cv2.drawContours(output, contours, -1, 255, 3)
#
#     # find the biggest countour (c) by the area
#     c = max(contours, key = cv2.contourArea)
#     x,y,w,h = cv2.boundingRect(c)
#
#     # draw the biggest contour (c) in green
#     cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
#
# # show the images
# cv2.imshow("Result", rotated)
# cv2.waitKey(0)
# cv2.imshow("Result", np.hstack([rotated, output]))
#
# cv2.waitKey(0)


# gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
# contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# for contour in contours:
#     # area = cv2.contourArea(contour)
#     cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)
#     cv2.imshow("", gray)
#     cv2.waitKey(0)

# cv2.waitKey(0)


# import cv2
# import numpy as np
# from skimage import io
# # from skimage.transform import rotate
# from skimage.color import rgb2gray
# from deskew import determine_skew
# from matplotlib import pyplot as plt
#
# def rotate(img, angle, center=None, scale=1.0):
#     h, w = img.shape[:2]
#
#     if center is None:
#         center = (w / 2, h / 2)
#
#     # Perform the rotation
#     M = cv2.getRotationMatrix2D(center, angle, scale)
#     rotated = cv2.warpAffine(img, M, (w, h))
#
#     return rotated
#
# def deskew(_img):
#     image = io.imread(_img)
#     grayscale = rgb2gray(image)
#     angle = determine_skew(grayscale)
#     print(angle)
#     rotated = rotate(image, angle) #  * 255
#     cv2.imwrite("test.jpg", rotated)  # .astype(np.uint8)
#     # cv2.waitKey(0)
#     return rotated.astype(np.uint8)
#
#
# def display_avant_apres(_original):
#     plt.subplot(1, 2, 1)
#     plt.imshow(io.imread(_original))
#     plt.subplot(1, 2, 2)
#     plt.imshow(deskew(_original))
#
#
# display_avant_apres('chance/in_20240125184438502913_046.jpg')


# import cv2
# from pyzbar import pyzbar

# image = cv2.imread('chance/in_20240125184438502913_046.jpg')


# Read input
# color = cv2.imread('chance/in_20240125184438502913_046.jpg', cv2.IMREAD_COLOR)
# color = cv2.resize(color, (0, 0), fx=0.15, fy=0.15)
# # RGB to gray
# gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('gray.png', gray)
# # cv2.imwrite('output/thresh.png', thresh)
# # Edge detection
# edges = cv2.Canny(gray, 100, 200, apertureSize=3)
# # Save the edge detected image
# cv2.imwrite('edges.png', edges)
#





#
# def rotate(img, angle, center=None, scale=1.0):
#     h, w = img.shape[:2]
#
#     if center is None:
#         center = (w / 2, h / 2)
#
#     # Perform the rotation
#     M = cv2.getRotationMatrix2D(center, angle, scale)
#     rotated = cv2.warpAffine(img, M, (w, h))
#
#     return rotated
#















#
# decoded_objects = pyzbar.decode(image, symbols=[pyzbar.ZBarSymbol.QRCODE])
# l_top, l_bottom, r_bottom, r_top = decoded_objects[0].polygon
#
# if 2 < abs(l_bottom.y - r_bottom.y) < 50:
#     if (l_bottom.y - r_bottom.y) > 0:
#         image = rotate(image, -1)
#     else:
#         image = rotate(image, 1)


# decoded_objects = pyzbar.decode(image, symbols=[pyzbar.ZBarSymbol.QRCODE])
# print(decoded_objects[0])  # .data.decode("utf-8"))


# cv2.imwrite("test_rotation.jpg", image)
# cv2.imshow("", image)
# cv2.waitKey(0)



# TODO:
# Придумать относительно чего поворачивать чек если он сосканирован под углом
# Придумать как находить блок символа даже если он плохо пропечатан(дата, сумма, game_id, код?)

# # cv2.imshow('', image)
# # cv2.waitKey(0)
#
# # Iterate through each pixel
# for y in range(image.shape[0]):
#     for x in range(image.shape[1]):
#         # Check if red channel value is less than 200
#         if image[y, x, 2] > 200:  # Red channel (BGR order)
#             # Fill pixel with white color
#             image[y, x] = [255, 255, 255]  # White color (BGR order)
#
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# wb_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
#
# # decoded_objects = pyzbar.decode(wb_image, symbols=[pyzbar.ZBarSymbol.I25])
# # print(decoded_objects[0].data.decode("utf-8"))
#
# # wb_image = cv2.GaussianBlur(wb_image, (3, 3), 0)
# #
# # wb_image = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)[1]
#
# cv2.imshow('Modified Image', wb_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()